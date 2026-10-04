"""미래 시점 전월세가 분위수 예측 모델 학습·평가.

    python -m src.train_model              # horizon 데이터가 없으면 생성 후 학습
    python -m src.train_model --rebuild    # horizon 데이터 강제 재생성

구성
1) Baseline: as-of 시점 비교주택 90일 median (없으면 동 → 구 median) × 학습셋 오차 분포
2) Model   : LightGBM quantile regression (q10/q50/q90), target = log(가격)
3) 두 방법 모두 calib 구간(2025 H2)에서 horizon별 CQR(conformalized quantile regression)로
   80% 범위를 보정 → 같은 신뢰도에서 정확도/범위 폭을 공정하게 비교
4) 평가: test(2026-01-01 ~ 09-15), horizon별 / 구별
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config
from . import model_config as mc
from .horizon_dataset import build_horizon_datasets, feature_columns
from .report import setup_logging

log = logging.getLogger("preprocess")
Q_LO, Q_MID, Q_HI = mc.QUANTILES


# ------------------------------------------------------------------ data
def load_or_build(rebuild: bool) -> dict[str, pd.DataFrame]:
    paths = {k: mc.HORIZON_DATA_DIR / f"horizon_{k}.parquet" for k in mc.RENT_TYPES}
    if not rebuild and all(p.exists() for p in paths.values()):
        return {k: pd.read_parquet(p) for k, p in paths.items()}
    log.info("[1] horizon 데이터 생성 (horizons=%s, 신고지연=%dd)", mc.HORIZONS, mc.REPORTING_LAG_DAYS)
    hist = pd.read_parquet(config.PROCESSED_DATA_DIR / "all_transactions.parquet")
    data = build_horizon_datasets(hist)
    for k, d in data.items():
        d.to_parquet(paths[k], index=False)
        log.info("  %s: %s rows -> %s", k, f"{len(d):,}", paths[k].name)
    return data


def prepare_categories(df: pd.DataFrame) -> dict[str, list]:
    cats = {}
    for c in mc.CATEGORICAL_FEATURES:
        levels = sorted(df[c].dropna().astype(str).unique().tolist())
        df[c] = pd.Categorical(df[c].astype("object"), categories=levels)
        cats[c] = levels
    return cats


# ------------------------------------------------------------------ metrics
def interval_metrics(y, lo, mid, hi) -> dict:
    y, lo, mid, hi = map(np.asarray, (y, lo, mid, hi))
    ape = np.abs(mid - y) / y
    pin = lambda q, p: np.mean(np.maximum(q * (y - p), (q - 1) * (y - p)))
    return {
        "n": int(len(y)),
        "MAPE_%": round(100 * float(ape.mean()), 2),
        "MdAPE_%": round(100 * float(np.median(ape)), 2),
        "MAE": round(float(np.abs(mid - y).mean()), 1),
        "coverage_80": round(float(((y >= lo) & (y <= hi)).mean()), 3),
        "rel_width_median": round(float(np.median((hi - lo) / mid)), 3),
        "pinball_avg": round(float(np.mean([pin(Q_LO, lo), pin(Q_MID, mid), pin(Q_HI, hi)])), 2),
    }


def metrics_table(df: pd.DataFrame, by: list[str] | None, prefix: str) -> pd.DataFrame:
    def f(g):
        return pd.Series(interval_metrics(g["target"], g[f"{prefix}_lo"], g[f"{prefix}_mid"], g[f"{prefix}_hi"]))
    if not by:
        return f(df).to_frame().T
    return df.groupby(by, observed=True).apply(f, include_groups=False).reset_index()


# ------------------------------------------------------------------ conformal
def fit_conformal(calib: pd.DataFrame, prefix: str, alpha: float = 1 - mc.TARGET_COVERAGE) -> dict[int, float]:
    """CQR: 비적합 점수 E = max(lo - y, y - hi) (log 공간)의 (1-α) 분위수를 horizon별로 계산."""
    q = {}
    for h, g in calib.groupby("horizon_days"):
        y = np.log(g["target"].to_numpy())
        e = np.maximum(np.log(g[f"{prefix}_lo"]) - y, y - np.log(g[f"{prefix}_hi"]))
        n = len(e)
        level = min(1.0, (1 - alpha) * (n + 1) / n)
        q[int(h)] = float(np.quantile(e, level))
    return q


def apply_conformal(df: pd.DataFrame, prefix: str, q: dict[int, float]) -> None:
    adj = df["horizon_days"].map(q).to_numpy()
    df[f"{prefix}_lo"] = np.exp(np.log(df[f"{prefix}_lo"]) - adj)
    df[f"{prefix}_hi"] = np.exp(np.log(df[f"{prefix}_hi"]) + adj)


# ------------------------------------------------------------------ baseline
def baseline_point(df: pd.DataFrame, fallback: pd.Series) -> np.ndarray:
    p = df["comp_price_median_90d"].fillna(df["dong_price_median_90d"]).fillna(df["gu_price_median_90d"])
    key = list(zip(df["gu"].astype(str), df["housing_category"].astype(str)))
    fb = pd.Series([fallback.get(k, np.nan) for k in key], index=df.index)
    return p.fillna(fb).clip(lower=1).to_numpy()


def run_baseline(df: pd.DataFrame) -> None:
    tr = df["split"] == "train"
    fallback = df.loc[tr].groupby([df["gu"].astype(str), df["housing_category"].astype(str)])["target"].median()
    df["base_point"] = baseline_point(df, fallback)
    r = np.log(df["target"]) - np.log(df["base_point"])
    qs = r[tr].groupby(df.loc[tr, "horizon_days"]).quantile(list(mc.QUANTILES)).unstack()
    for name, q in zip(("lo", "mid", "hi"), mc.QUANTILES):
        df[f"base_{name}"] = df["base_point"] * np.exp(df["horizon_days"].map(qs[q]))


# ------------------------------------------------------------------ model
def anchor_fallback_table(df: pd.DataFrame, mask) -> dict[str, float]:
    """구×주택대분류별 target median ('gu|category' -> 값). anchor가 없을 때의 마지막 fallback."""
    fb = df.loc[mask].groupby([df["gu"].astype(str), df["housing_category"].astype(str)])["target"].median()
    return {f"{g}|{c}": float(v) for (g, c), v in fb.items()}


def apply_anchor(df: pd.DataFrame, mode: str, feats: list[str], fallback: dict[str, float]) -> list[str]:
    """target 표현 방식에 맞춰 log_anchor 컬럼과 최종 feature 목록을 만든다 (학습·서빙 공용).

    anchor 모드: anchor = as-of 시점 구 90일 median (없으면 동 median → fallback 표의 구×주택대분류 median).
    가격 수준 feature는 anchor 대비 비율(rel_*)로 바꿔 넣는다.
    """
    if mode == "absolute":
        df["log_anchor"] = 0.0
        return list(feats)
    key = df["gu"].astype(str) + "|" + df["housing_category"].astype(str)
    anchor = df[mc.ANCHOR_FEATURE].fillna(df["dong_price_median_90d"])
    anchor = anchor.fillna(key.map(fallback)).clip(lower=1)
    df["log_anchor"] = np.log(anchor.astype(float))
    out = []
    for f in feats:
        if f in mc.RELATIVE_PRICE_FEATURES:
            df[f"rel_{f}"] = df[f] / anchor
            out.append(f"rel_{f}")
        elif f != mc.ANCHOR_FEATURE:
            out.append(f)
    return out


def train_quantile_models(df: pd.DataFrame, feats: list[str]) -> tuple[dict, dict]:
    """target = log(가격) - log_anchor (absolute 모드에서는 log_anchor=0)."""
    tr, va = df["split"] == "train", df["split"] == "valid"
    y = np.log(df["target"]) - df["log_anchor"]
    X_tr, y_tr = df.loc[tr, feats], y[tr]
    X_va, y_va = df.loc[va, feats], y[va]
    models, best = {}, {}
    for q in mc.QUANTILES:
        m = lgb.LGBMRegressor(objective="quantile", alpha=q, **mc.LGBM_PARAMS)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="quantile",
              categorical_feature=mc.CATEGORICAL_FEATURES,
              callbacks=[lgb.early_stopping(mc.EARLY_STOPPING_ROUNDS, verbose=False)])
        models[q], best[q] = m, int(m.best_iteration_)
        log.info("    q%.0f: best_iteration=%d", q * 100, best[q])
    return models, best


def predict_quantiles(models: dict, X: pd.DataFrame, log_anchor) -> np.ndarray:
    la = np.asarray(log_anchor)
    P = np.column_stack([np.exp(models[q].predict(X) + la) for q in mc.QUANTILES])
    return np.sort(P, axis=1)    # 분위수 교차 방지 (lo <= mid <= hi)


# ------------------------------------------------------------------ main
def run_kind(kind: str, df: pd.DataFrame, mode: str = mc.TARGET_MODE) -> dict:
    log.info("[2] %s (target_mode=%s): rows=%s, split=%s", kind, mode, f"{len(df):,}",
             df["split"].value_counts().to_dict())
    df = df[df["split"].notna()].copy()
    cats = prepare_categories(df)
    fallback = anchor_fallback_table(df, df["split"] == "train")
    feats = apply_anchor(df, mode, feature_columns(kind), fallback)

    t0 = time.time()
    run_baseline(df)
    models, best = train_quantile_models(df, feats)
    P = predict_quantiles(models, df[feats], df["log_anchor"])
    df["model_lo"], df["model_mid"], df["model_hi"] = P[:, 0], P[:, 1], P[:, 2]
    log.info("    trained in %.0fs", time.time() - t0)

    calib = df[df["split"] == "calib"]
    calib_raw = metrics_table(calib, None, "model").iloc[0].to_dict()
    calib_raw["median_log_bias"] = round(float(np.median(np.log(calib["target"] / calib["model_mid"]))), 4)
    log.info("    calib(2025H2, 보정 전) metrics: %s", calib_raw)
    conf = {p: fit_conformal(calib, p) for p in ("base", "model")}
    raw_test = df[df["split"] == "test"].copy()
    for p in ("base", "model"):
        apply_conformal(df, p, conf[p])
    test = df[df["split"] == "test"]

    out_dir = mc.MODEL_REPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for p, name in (("base", "baseline"), ("model", "lgbm_quantile")):
        overall = metrics_table(test, None, p).assign(method=name, horizon_days="all")
        by_h = metrics_table(test, ["horizon_days"], p).assign(method=name)
        rows += [overall, by_h]
    by_h_all = pd.concat(rows, ignore_index=True)
    by_h_all.to_csv(out_dir / f"{kind}_metrics_by_horizon.csv", index=False, encoding="utf-8-sig")
    by_gu = pd.concat([metrics_table(test, ["gu", "housing_category"], p).assign(method=n)
                       for p, n in (("base", "baseline"), ("model", "lgbm_quantile"))], ignore_index=True)
    by_gu.to_csv(out_dir / f"{kind}_metrics_by_gu.csv", index=False, encoding="utf-8-sig")
    raw_cov = metrics_table(raw_test, None, "model")["coverage_80"].item()

    imp = pd.DataFrame({"feature": feats,
                        "gain": models[Q_MID].booster_.feature_importance("gain")}).sort_values("gain", ascending=False)
    imp["gain_share"] = (imp["gain"] / imp["gain"].sum()).round(4)
    imp.to_csv(out_dir / f"{kind}_feature_importance.csv", index=False, encoding="utf-8-sig")

    mc.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for q, m in models.items():
        m.booster_.save_model(str(mc.MODEL_DIR / f"{kind}_q{int(q * 100)}.txt"))
    meta = {
        "kind": kind, "target_mode": mode, "anchor_fallback": fallback, "anchor_feature": mc.ANCHOR_FEATURE, "features": feats, "categorical_levels": cats, "quantiles": mc.QUANTILES,
        "best_iteration": {str(k): v for k, v in best.items()},
        "conformal_log_adjustment_by_horizon": conf["model"],
        "horizons": mc.HORIZONS, "reporting_lag_days": mc.REPORTING_LAG_DAYS, "splits": mc.SPLITS,
        "note": "학습 데이터: train(2021-2024). 서비스 배포 전에는 전체 기간으로 재학습 필요.",
    }
    (mc.MODEL_DIR / f"{kind}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    test[["tx_id", "contract_date", "horizon_days", "gu", "dong", "housing_category", "area_bin", "target",
          "base_lo", "base_mid", "base_hi", "model_lo", "model_mid", "model_hi"]].to_parquet(
        out_dir / f"{kind}_test_predictions.parquet", index=False)
    return {"by_h": by_h_all, "by_gu": by_gu, "imp": imp, "raw_cov": raw_cov, "conf": conf["model"],
            "calib_raw": calib_raw}


def train_final(kind: str, df: pd.DataFrame) -> None:
    """서비스용 최종 모델: 평가 run에서 정한 반복 수·conformal 보정량을 그대로 쓰고 전체 기간으로 재학습.

    - 데이터: 2021-01 ~ 2026-09 (모든 split)
    - 반복 수: 평가 run best_iteration × 1.1 (학습 데이터가 늘어난 만큼 약간 증가)
    - 80% 범위 보정량: 평가 run의 calib(2025 H2) 값 재사용 (전체 학습 후에는 hold-out이 없으므로)
    """
    eval_meta = json.loads((mc.MODEL_DIR / f"{kind}_meta.json").read_text(encoding="utf-8"))
    df = df[df["split"].notna()].copy()
    cats = prepare_categories(df)
    fallback = anchor_fallback_table(df, df["contract_date"] >= df["contract_date"].max() - pd.Timedelta(days=365))
    feats = apply_anchor(df, eval_meta["target_mode"], feature_columns(kind), fallback)
    assert feats == eval_meta["features"], "평가 모델과 feature 목록이 다릅니다"
    y = np.log(df["target"]) - df["log_anchor"]
    out_dir = mc.MODEL_DIR / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    for q in mc.QUANTILES:
        n = int(eval_meta["best_iteration"][str(q)] * 1.1)
        params = {**mc.LGBM_PARAMS, "n_estimators": n}
        m = lgb.LGBMRegressor(objective="quantile", alpha=q, **params)
        m.fit(df[feats], y, categorical_feature=mc.CATEGORICAL_FEATURES)
        m.booster_.save_model(str(out_dir / f"{kind}_q{int(q * 100)}.txt"))
        log.info("  [final %s] q%.0f: %d trees, rows=%s", kind, q * 100, n, f"{len(df):,}")
    meta = {**eval_meta, "categorical_levels": cats, "anchor_fallback": fallback,
            "trained_on": [str(df["contract_date"].min().date()), str(df["contract_date"].max().date())],
            "note": "서비스용 최종 모델 (전체 기간 학습). conformal 보정량은 평가 run(calib 2025 H2) 값."}
    (out_dir / f"{kind}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def print_report(kind: str, r: dict) -> str:
    label = {"jeonse": "전세 (보증금, 만원)", "monthly": "월세 (월세금, 만원)"}[kind]
    L = [f"\n================ {label} — test 2026-01 ~ 09 ================"]
    cols = ["method", "horizon_days", "n", "MAPE_%", "MdAPE_%", "MAE", "coverage_80", "rel_width_median"]
    t = r["by_h"][cols].copy()
    t["horizon_days"] = t["horizon_days"].astype(str)
    L.append(t.to_string(index=False))
    L.append(f"\n보정 전 모델 80% 범위 적중률: {r['raw_cov']:.3f}  → conformal 보정량(log) by h: "
             + ", ".join(f"{h}:{v:+.3f}" for h, v in r["conf"].items()))
    g = r["by_gu"]
    piv = g.pivot_table(index=["gu", "housing_category"], columns="method", values=["MdAPE_%", "coverage_80"])
    L.append("\n구별 (모든 horizon 합산):\n" + piv.round(3).to_string())
    L.append("\nFeature importance top 10 (q50, gain):\n" + r["imp"].head(10)[["feature", "gain_share"]].to_string(index=False))
    return "\n".join(L)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--kinds", default="jeonse,monthly")
    p.add_argument("--final", action="store_true", help="평가 run 이후 전체 기간으로 서비스용 모델 학습")
    p.add_argument("--target-mode", default=mc.TARGET_MODE, choices=["absolute", "anchor"])
    args = p.parse_args(argv)
    setup_logging(mc.MODEL_REPORT_DIR / "train_log.txt")
    data = load_or_build(args.rebuild)
    if args.final:
        for kind in args.kinds.split(","):
            train_final(kind, data[kind])
        return
    texts = []
    for kind in args.kinds.split(","):
        r = run_kind(kind, data[kind], args.target_mode)
        texts.append(print_report(kind, r))
    text = "\n".join(texts)
    print(text)
    (mc.MODEL_REPORT_DIR / "model_summary.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
