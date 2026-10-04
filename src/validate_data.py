"""전처리 결과 검증과 리포트용 요약."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config
from .report import Report

log = logging.getLogger("preprocess")


def validate_clean(df: pd.DataFrame, report: Report) -> None:
    """§38 필수 assert + 추가 검증 건수 기록."""
    log.info("[Phase 7] Validate")
    assert df["contract_date"].min() >= pd.Timestamp(config.START_DATE)
    assert df["contract_date"].max() <= pd.Timestamp(config.END_DATE)
    if config.DROP_INVALID_GU:
        assert set(df["gu"].dropna().unique()).issubset(set(config.VALID_GU))
    else:
        outside = set(df["gu"].dropna().unique()) - set(config.VALID_GU)
        if outside:
            log.warning("  VALID_GU 외 지역 존재: %s", outside)
    assert (df["area_m2"].dropna() > 0).all()
    assert (df["deposit_10k"].dropna() >= 0).all()
    assert (df["monthly_rent_10k"].dropna() >= 0).all()
    assert df["tx_id"].is_unique
    assert df["contract_date"].is_monotonic_increasing
    report.check("validate", "required assertions (§38)", 0, "passed")

    report.check("validate", "전세 & monthly_rent > 0", df["anomaly_jeonse_with_rent"].sum(), "flagged")
    report.check("validate", "월세 & monthly_rent == 0", df["anomaly_monthly_zero_rent"].sum(), "flagged")
    report.check("validate", "built_year > contract_year", df["built_after_contract"].sum(), "flagged")
    report.check("validate", "duplicate candidates", df["is_duplicate_candidate"].sum(), "flagged")


def source_distribution_check(df: pd.DataFrame, report: Report) -> pd.DataFrame:
    """gu × housing_category별 전세 비율과 보증금 상한을 비교해 편향된(필터링된) 다운로드를 탐지.

    예) 금액 조건을 건 채 내려받으면 보증금 최대값이 특정 값에서 잘리고 전세가 거의 사라진다.
    """
    g = df.groupby(["gu", "housing_category"])
    t = pd.DataFrame({
        "n": g.size(),
        "jeonse_share": g["rent_type"].apply(lambda s: round(float((s == "전세").mean()), 4)),
        "deposit_max": g["deposit_10k"].max(),
        "deposit_p99": g["deposit_10k"].quantile(.99),
    })
    cat_p99 = df.groupby("housing_category")["deposit_10k"].quantile(.99)
    t["category_deposit_p99"] = [cat_p99[c] for _, c in t.index]
    t["suspect_low_jeonse"] = t["jeonse_share"] < config.MIN_JEONSE_SHARE
    t["suspect_truncated_deposit"] = t["deposit_max"] < t["category_deposit_p99"] * config.MAX_DEPOSIT_RATIO
    for (gu, cat), r in t.iterrows():
        if r["suspect_low_jeonse"] or r["suspect_truncated_deposit"]:
            report.check("validate", f"biased source suspected: {gu}|{cat}", r["n"],
                         "see MODEL_EXCLUDE_GU",
                         f"jeonse_share={r['jeonse_share']:.3f}, deposit_max={r['deposit_max']:.0f}, "
                         f"category p99={r['category_deposit_p99']:.0f}", level="WARNING")
    report.set("source_distribution", {f"{gu}|{cat}": {k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v))
                                                       for k, v in r.items()} for (gu, cat), r in t.iterrows()})
    return t


def _group_mask(df: pd.DataFrame, row: pd.Series, cols) -> np.ndarray:
    m = np.ones(len(df), dtype=bool)
    for c in cols:
        v = row[c]
        m &= (df[c].isna().to_numpy() if pd.isna(v) else (df[c] == v).fillna(False).to_numpy())
    return m


def bruteforce_leakage_check(df: pd.DataFrame, report: Report, n: int = 400, seed: int = 42) -> int:
    """무작위 거래 n건에 대해 historical feature를 단순 필터링(brute force)으로 다시 계산해 비교.

    brute force 정의: 같은 그룹 & t-W <= date < t 인 거래만 사용. 결과가 다르면 mismatch.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
    dates = df["contract_date"].to_numpy()
    price = df["price_10k"].to_numpy()
    mismatches = []
    specs = [
        ("dong_price_median_90d", "dong_tx_count_90d", config.DONG_GROUP, 90, config.MIN_ROLLING_COUNT),
        ("dong_price_median_30d", "dong_tx_count_30d", config.DONG_GROUP, 30, config.MIN_ROLLING_COUNT),
        ("comp_price_median_90d", "comp_tx_count_90d", config.COMP_GROUP, 90, config.MIN_ROLLING_COUNT),
        ("property_price_median_365d", "property_tx_count_365d", config.PROPERTY_GROUP,
         config.PROPERTY_HISTORY_DAYS, 1),
    ]
    for i in idx:
        row = df.iloc[i]
        t = dates[i]
        for med_col, cnt_col, grp, w, min_n in specs:
            g = _group_mask(df, row, grp)
            win = g & (dates >= t - np.timedelta64(w, "D")) & (dates < t)
            vals = price[win]
            exp_cnt = len(vals)
            exp_med = float(np.median(vals)) if exp_cnt >= min_n else np.nan
            got_med, got_cnt = row[med_col], row[cnt_col]
            if got_cnt != exp_cnt or not (np.isclose(got_med, exp_med) or (pd.isna(got_med) and pd.isna(exp_med))):
                mismatches.append((int(row["tx_id"]), med_col, got_med, exp_med, got_cnt, exp_cnt))
        # 직전 거래 가격은 반드시 엄격히 이전 날짜
        g = _group_mask(df, row, config.PROPERTY_GROUP) & (dates < t)
        if g.any():
            last_date = dates[g].max()
            cands = price[g & (dates == last_date)]
            if not (row["property_last_date"] == last_date and row["property_last_price"] in cands):
                mismatches.append((int(row["tx_id"]), "property_last_price", row["property_last_price"],
                                   list(cands), row["property_last_date"], last_date))
        elif pd.notna(row["property_last_price"]):
            mismatches.append((int(row["tx_id"]), "property_last_price", row["property_last_price"], None, None, None))
        if pd.notna(row["property_last_date"]) and row["property_last_date"] >= row["contract_date"]:
            mismatches.append((int(row["tx_id"]), "property_last_date>=t", None, None, None, None))

    report.check("validate", f"brute-force leakage re-check ({len(idx)} random rows)", len(mismatches),
                 "must be 0", str(mismatches[:5]) if mismatches else "all rolling/property features match",
                 level="WARNING")
    report.set("bruteforce_leakage_mismatches", len(mismatches))
    return len(mismatches)


def rolling_trace_example(df: pd.DataFrame) -> dict:
    """§49-G: 실제 거래 1건의 dong 90일 feature가 어떤 과거 거래로 계산됐는지 추적."""
    cand = df[(df["contract_year"] == 2024) & (df["dong_tx_count_90d"] >= 10)
              & (df["housing_category"] == "연립다세대") & (df["rent_type"] == "전세")]
    # 같은 날 같은 그룹에 다른 거래가 있는 행을 고르면 동일 날짜 제외 여부까지 보여줄 수 있다
    same_day = cand.groupby(config.DONG_GROUP + ["contract_date"])["tx_id"].transform("size")
    row = (cand[same_day >= 2] if (same_day >= 2).any() else cand).iloc[len(cand) // 3 if len(cand) else 0]
    t = row["contract_date"]
    g = _group_mask(df, row, config.DONG_GROUP)
    dates = df["contract_date"]
    win = df[g & (dates >= t - pd.Timedelta(days=90)) & (dates < t)]
    same = df[g & (dates == t)]
    future = df[g & (dates > t)]
    return {
        "tx_id": int(row["tx_id"]),
        "group": " | ".join(str(row[c]) for c in config.DONG_GROUP),
        "transaction_date": str(t.date()),
        "transaction_price_10k(target, not used)": float(row["price_10k"]),
        "window": f"[{(t - pd.Timedelta(days=90)).date()}, {t.date()})",
        "earliest_past_tx_in_window": str(win["contract_date"].min().date()),
        "latest_past_tx_in_window": str(win["contract_date"].max().date()),
        "n_past_tx_in_window (recomputed)": int(len(win)),
        "dong_tx_count_90d (feature)": int(row["dong_tx_count_90d"]),
        "median_recomputed": float(win["price_10k"].median()),
        "dong_price_median_90d (feature)": float(row["dong_price_median_90d"]),
        "same_day_tx_excluded": int(len(same)),
        "future_tx_in_group_excluded": int(len(future)),
        "median_if_same_day_were_included": float(pd.concat([win, same])["price_10k"].median()),
    }


def missing_rate(df: pd.DataFrame) -> pd.Series:
    return (df.isna().mean() * 100).round(2).sort_values(ascending=False)


def monthly_market_summary(df: pd.DataFrame) -> pd.DataFrame:
    """gu/dong/연월/rent_type/housing_category별 거래 수 및 가격 요약 (EDA용)."""
    keys = ["gu", "dong", "contract_year", "contract_month", "rent_type", "housing_category"]
    g = df.groupby(keys, dropna=False)
    out = g.size().rename("transaction_count").to_frame()
    dep = g["deposit_10k"]
    out["median_deposit"] = dep.median()
    out["mean_deposit"] = dep.mean().round(1)
    out["std_deposit"] = dep.std().round(1)
    out["q25_deposit"] = dep.quantile(.25)
    out["q75_deposit"] = dep.quantile(.75)
    rent = g["monthly_rent_10k"]
    out["median_monthly_rent"] = rent.median()
    out["mean_monthly_rent"] = rent.mean().round(2)
    out["std_monthly_rent"] = rent.std().round(2)
    out["new_contract_count"] = g["contract_type_clean"].apply(lambda s: int((s == "new").sum()))
    out = out.reset_index().rename(columns={"contract_year": "year", "contract_month": "month"})
    # 월세 행에서는 월세 통계, 전세 행에서는 보증금 통계가 의미 있음
    out.loc[out["rent_type"] == "전세", ["median_monthly_rent", "mean_monthly_rent", "std_monthly_rent"]] = np.nan
    out["is_partial_year"] = (out["year"] == config.PARTIAL_YEAR).astype("int8")
    return out
