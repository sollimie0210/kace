"""미래 시점 예측용 학습 데이터 생성 (direct multi-horizon).

계약일이 t인 신규 거래 하나를 horizon h마다 한 행씩 복제한다.
- 서비스 기준일(as-of): a = t - h
- 실제로 쓸 수 있는 데이터: a - REPORTING_LAG_DAYS 이전 거래 (신고 지연 반영)
- 따라서 모든 historical feature는 offset = h + lag 로 계산한다 → [t-offset-W, t-offset)

즉 "오늘(a) 알 수 있는 정보만으로 h일 뒤 계약 가격을 맞히는" 문제를 그대로 학습한다.
미래 시점의 rolling feature를 지어낼 필요가 없고, h가 클수록 불확실성이 커지는 것도
데이터에서 직접 학습된다.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config
from . import model_config as mc
from .feature_engineering import past_window_stats, previous_transaction

log = logging.getLogger("preprocess")

UNIT_FEATURES = [
    "gu", "dong", "housing_category", "housing_type", "area_bin",
    "area_m2", "floor", "is_basement", "is_ground_floor",
    "built_year", "building_age", "building_name_is_address",
    "contract_month", "month_sin", "month_cos",
    # 위치 (src/location_features.py, 지오코딩 결과가 있을 때)
    "lat", "lon", "station_dist_m", "n_stations_1km", "univ_dist_m",
]
MONTHLY_ONLY_UNIT_FEATURES = ["deposit_10k", "deposit_per_m2"]

HISTORY_FEATURES = [
    "dong_tx_count_30d", "dong_tx_count_90d", "dong_tx_count_180d",
    "dong_price_median_30d", "dong_price_median_90d", "dong_price_median_180d",
    "dong_price_mean_90d", "dong_price_std_90d",
    "gu_tx_count_90d", "gu_price_median_90d",
    "comp_price_median_30d", "comp_price_median_90d", "comp_price_median_180d", "comp_tx_count_90d",
    "property_last_price", "property_days_since_last_transaction",
    "property_tx_count_365d", "property_price_median_365d",
    "dong_price_trend_3m", "comp_price_trend_3m", "gu_price_trend_3m", "gu_price_trend_12m",
    "has_property_history", "has_comp_90d_history", "has_dong_90d_history",
]
MONTHLY_ONLY_HISTORY_FEATURES = ["dong_deposit_median_90d", "comp_deposit_median_90d", "deposit_vs_dong_median"]


def feature_columns(kind: str) -> list[str]:
    feats = ["horizon_days"] + UNIT_FEATURES + HISTORY_FEATURES
    if kind == "monthly":
        feats += MONTHLY_ONLY_UNIT_FEATURES + MONTHLY_ONLY_HISTORY_FEATURES
    return feats


def _ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.where(b > 0) - 1


def history_features_at_offset(hist: pd.DataFrame, offset: int) -> pd.DataFrame:
    """hist 전체 행에 대해 [t-offset-W, t-offset) 기준 historical feature 계산."""
    out = pd.DataFrame(index=hist.index)
    P = "price_10k"
    for w in (30, 90, 180):
        stats = ("count", "median", "mean", "std") if w == 90 else ("count", "median")
        r = past_window_stats(hist, config.DONG_GROUP, P, w, stats, offset_days=offset)
        out[f"dong_tx_count_{w}d"] = r["count"]
        out[f"dong_price_median_{w}d"] = r["median"]
        if w == 90:
            out["dong_price_mean_90d"], out["dong_price_std_90d"] = r["mean"], r["std"]
    r = past_window_stats(hist, config.GU_GROUP, P, 90, ("count", "median"), offset_days=offset)
    out["gu_tx_count_90d"], out["gu_price_median_90d"] = r["count"], r["median"]
    for w in (30, 90, 180):
        r = past_window_stats(hist, config.COMP_GROUP, P, w, ("count", "median"), offset_days=offset)
        out[f"comp_price_median_{w}d"] = r["median"]
        if w == 90:
            out["comp_tx_count_90d"] = r["count"]

    out["dong_deposit_median_90d"] = past_window_stats(
        hist, config.DONG_GROUP, "deposit_10k", 90, ("median",), offset_days=offset)["median"]
    out["comp_deposit_median_90d"] = past_window_stats(
        hist, config.COMP_GROUP, "deposit_10k", 90, ("median",), offset_days=offset)["median"]

    prev = previous_transaction(hist, config.PROPERTY_GROUP, P, offset_days=offset)
    asof = hist["contract_date"] - pd.Timedelta(days=offset)
    out["property_last_price"] = prev["prev_value"]
    out["property_days_since_last_transaction"] = (asof - prev["prev_date"]).dt.days.astype("float64")
    r = past_window_stats(hist, config.PROPERTY_GROUP, P, config.PROPERTY_HISTORY_DAYS,
                          ("count", "median"), offset_days=offset, min_count=1)
    out["property_tx_count_365d"], out["property_price_median_365d"] = r["count"], r["median"]

    # 추세: 직전 90일 median vs 그 이전 구간 median
    for name, grp, cur in [("dong", config.DONG_GROUP, "dong_price_median_90d"),
                           ("comp", config.COMP_GROUP, "comp_price_median_90d"),
                           ("gu", config.GU_GROUP, "gu_price_median_90d")]:
        prev90 = past_window_stats(hist, grp, P, 90, ("median",), offset_days=offset + 90)["median"]
        out[f"{name}_price_trend_3m"] = _ratio(out[cur], prev90)
    prev365 = past_window_stats(hist, config.GU_GROUP, P, 90, ("median",), offset_days=offset + 365)["median"]
    out["gu_price_trend_12m"] = _ratio(out["gu_price_median_90d"], prev365)

    out["has_property_history"] = out["property_last_price"].notna().astype("int8")
    out["has_comp_90d_history"] = out["comp_price_median_90d"].notna().astype("int8")
    out["has_dong_90d_history"] = out["dong_price_median_90d"].notna().astype("int8")
    return out


def assign_model_split(dates: pd.Series) -> pd.Series:
    out = pd.Series(None, index=dates.index, dtype="object")
    for name, (s, e) in mc.SPLITS.items():
        out[(dates >= s) & (dates <= e)] = name
    return out


def target_mask(hist: pd.DataFrame, kind: str) -> pd.Series:
    spec = mc.RENT_TYPES[kind]
    m = (hist["rent_type"] == spec["rent_type"]) & (hist["contract_type_clean"] == "new")
    m &= hist["housing_category"].isin(config.MODEL_HOUSING_CATEGORIES)
    if config.MODEL_EXCLUDE_GU:
        m &= ~hist["gu"].isin(config.MODEL_EXCLUDE_GU)
    if kind == "jeonse":
        m &= hist["deposit_10k"] > 0
    else:
        m &= (hist["monthly_rent_10k"] > 0) & (hist["deposit_10k"] >= 0)
    return m


def build_horizon_datasets(hist: pd.DataFrame, kinds=("jeonse", "monthly"),
                           horizons=mc.HORIZONS) -> dict[str, pd.DataFrame]:
    """hist: all_transactions (모든 거래 = feature 계산용 과거 정보).
    반환: {kind: target 거래 × horizon 행}. historical feature는 horizon마다 한 번만 계산해 공유한다.
    """
    hist = hist.sort_values(["contract_date", "tx_id"], kind="stable").reset_index(drop=True)
    masks = {k: target_mask(hist, k).to_numpy() for k in kinds}
    any_mask = np.logical_or.reduce(list(masks.values()))
    parts = {k: [] for k in kinds}
    for h in horizons:
        offset = h + mc.REPORTING_LAG_DAYS
        log.info("  horizon=%3dd (feature offset %dd)", h, offset)
        feats_all = history_features_at_offset(hist, offset)
        for k in kinds:
            target_col = mc.RENT_TYPES[k]["target_col"]
            unit_cols = UNIT_FEATURES + (MONTHLY_ONLY_UNIT_FEATURES if k == "monthly" else [])
            m = masks[k]
            part = pd.concat([hist.loc[m, ["tx_id", "contract_date", "property_id"] + unit_cols],
                              feats_all.loc[m], hist.loc[m, [target_col]].rename(columns={target_col: "target"})],
                             axis=1)
            part.insert(2, "horizon_days", h)
            part.insert(3, "asof_date", part["contract_date"] - pd.Timedelta(days=h))
            parts[k].append(part)
        del feats_all
    out = {}
    for k in kinds:
        df = pd.concat(parts[k], ignore_index=True)
        if k == "monthly":
            df["deposit_vs_dong_median"] = _ratio(df["deposit_10k"], df["dong_deposit_median_90d"])
        else:
            df = df.drop(columns=["dong_deposit_median_90d", "comp_deposit_median_90d"])
        df["split"] = assign_model_split(df["contract_date"])
        out[k] = df
    assert any_mask.any()
    return out
