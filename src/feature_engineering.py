"""기본 파생변수 + 과거 거래 기반(leakage-free) historical feature + 모델 데이터셋 생성.

Leakage 방지 핵심 (§21)
------------------------
거래일 t의 historical feature는 같은 그룹에서 **t 이전 날짜(date < t)** 의 거래만 사용한다.
같은 날짜의 다른 거래도 사용하지 않는다.

구현 방식은 `past_window_stats` 참고: 각 행마다 '질의(query) 행'을 가격 NaN으로 만들어
실제 거래들과 함께 시간순 정렬하되, 같은 시각의 실제 거래보다 **앞에** 둔다. 그 다음
time-based rolling(closed="left")을 적용하면 질의 행의 window는 정확히 [q - W, q) 가 된다.
- 질의 행 자신은 closed="left"로 제외되고 값도 NaN이다.
- 같은 날짜의 실제 거래는 질의 행 뒤에 있으므로 포함될 수 없다.
- rolling은 뒤쪽 행을 보지 않으므로 미래 거래는 구조적으로 포함될 수 없다.
질의 시각을 q = t - offset 으로 옮기면 [t - offset - W, t - offset) 구간 통계(추세 feature의
'이전 90일')도 같은 코드로 계산된다.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config
from .report import Report

log = logging.getLogger("preprocess")


# =========================================================== basic features
def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df["contract_date"]
    df["contract_year"] = d.dt.year.astype("int16")
    df["contract_month"] = d.dt.month.astype("int8")
    df["contract_quarter"] = d.dt.quarter.astype("int8")
    df["contract_day_of_month"] = d.dt.day.astype("int8")
    df["month_sin"] = np.sin(2 * np.pi * df["contract_month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["contract_month"] / 12)
    df["contract_year_month"] = d.dt.strftime("%Y-%m")          # 외부 월별 데이터 merge key
    df["is_partial_year"] = (df["contract_year"] == config.PARTIAL_YEAR).astype("int8")
    return df


def add_building_age(df: pd.DataFrame, report: Report) -> pd.DataFrame:
    by = df["built_year"]
    report.check("features", "built_year missing", by.isna().sum(), "keep NaN")
    suspicious = by < 1900
    df["built_year_suspicious"] = suspicious.astype("int8")
    report.check("features", "built_year < 1900 (suspicious)", suspicious.sum(), "flag", level="WARNING")
    age = df["contract_year"] - by
    future_built = by > df["contract_year"]
    df["built_after_contract"] = future_built.astype("int8")
    report.check("features", "built_year > contract_year", future_built.sum(),
                 "building_age=NaN, built_year kept (선임대 가능성)", level="WARNING")
    df["building_age"] = age.mask(future_built | suspicious)
    return df


def add_floor_features(df: pd.DataFrame, report: Report) -> pd.DataFrame:
    f = df["floor"]
    df["is_basement"] = (f < 0).astype("int8")
    df["is_ground_floor"] = (f == 1).astype("int8")
    report.check("features", "floor < 0 (basement)", (f < 0).sum(), "keep, is_basement=1")
    report.check("features", "floor == 0", (f == 0).sum(), "keep", level="WARNING")
    report.check("features", "floor missing", f.isna().sum(), "keep NaN")
    hi = f.quantile(0.999)
    report.check("features", "floor > p99.9", (f > hi).sum(), "keep",
                 f"p99.9={hi:.0f}, max={f.max():.0f}")
    return df


def add_area_features(df: pd.DataFrame) -> pd.DataFrame:
    df["area_pyeong"] = df["area_m2"] / config.PYEONG_M2
    df["area_bin"] = pd.cut(df["area_m2"], bins=config.AREA_BINS, labels=config.AREA_BIN_LABELS,
                            right=False).astype("str")
    return df


def add_address_keys(df: pd.DataFrame) -> pd.DataFrame:
    """location/parcel/property 식별자와 geocoding용 주소.

    건물명이 '(461-0)'처럼 주소를 반복하는 경우가 많아 건물명 단독으로 식별하지 않고
    gu|dong|jibun|building_name 조합을 property_id로 사용한다.
    """
    gu, dong = df["gu"].fillna(""), df["dong"].fillna("")
    jibun, bname = df["jibun"].fillna(""), df["building_name"].fillna("")
    df["location_id"] = gu + "|" + dong
    df["parcel_id"] = gu + "|" + dong + "|" + jibun
    df["property_id"] = gu + "|" + dong + "|" + jibun + "|" + bname
    base = df["sido"].fillna("") + " " + gu
    df["full_jibun_address"] = (base + " " + dong + " " + jibun).str.strip().where(df["jibun"].notna())
    df["full_road_address"] = (base + " " + df["road_name"]).where(df["road_name"].notna())
    df["building_name_is_address"] = df["building_name"].str.fullmatch(r"\(\d+(-\d+)?\)").fillna(False).astype("int8")
    return df


def add_contract_period(df: pd.DataFrame, report: Report) -> pd.DataFrame:
    """'202201~202401' → lease_start_ym=202201, lease_end_ym=202401, lease_duration_months=24."""
    m = df["contract_period"].str.extract(r"^(\d{6})\s*~\s*(\d{6})$")
    start = pd.to_datetime(m[0], format="%Y%m", errors="coerce")
    end = pd.to_datetime(m[1], format="%Y%m", errors="coerce")
    df["lease_start_ym"] = start.dt.strftime("%Y-%m")
    df["lease_end_ym"] = end.dt.strftime("%Y-%m")
    dur = (end.dt.year - start.dt.year) * 12 + (end.dt.month - start.dt.month)
    df["lease_duration_months"] = dur.astype("float64")
    failed = df["contract_period"].notna() & dur.isna()
    report.check("features", "contract_period missing", df["contract_period"].isna().sum(), "keep NaN")
    report.check("features", "contract_period parse failure", failed.sum(), "set NaN", level="WARNING")
    report.check("features", "lease_duration_months < 0", (dur < 0).sum(), "keep", level="WARNING")
    return df


def add_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    """rent_type별 가격(price_10k)과 EDA/갱신 분석 전용 파생값.

    price_10k 는 현재 거래의 target 이므로 **절대 모델 feature로 쓰지 않는다**
    (historical feature 계산의 입력으로만 사용).
    """
    df["price_10k"] = np.select(
        [df["rent_type"] == rt for rt in config.PRICE_COLUMN_BY_RENT_TYPE],
        [df[c] for c in config.PRICE_COLUMN_BY_RENT_TYPE.values()], default=np.nan)
    # 월세 모델에서는 feature로 사용 가능(보증금만으로 계산), 전세 모델에서는 target 파생값 → 금지
    df["deposit_per_m2"] = df["deposit_10k"] / df["area_m2"]
    df["monthly_rent_per_m2_eda_only"] = df["monthly_rent_10k"] / df["area_m2"]

    # 갱신 계약 분석 전용 (기본 prediction dataset에서는 제외)
    renewal = df["contract_type_clean"] == "renewal"
    pd_, pr = df["previous_deposit_10k"], df["previous_monthly_rent_10k"]
    df["deposit_change_from_previous"] = (df["deposit_10k"] - pd_).where(renewal)
    df["monthly_rent_change_from_previous"] = (df["monthly_rent_10k"] - pr).where(renewal)
    df["deposit_change_rate_from_previous"] = (df["deposit_10k"] / pd_.where(pd_ > 0) - 1).where(renewal)
    df["monthly_rent_change_rate_from_previous"] = (df["monthly_rent_10k"] / pr.where(pr > 0) - 1).where(renewal)
    return df


def add_basic_features(df: pd.DataFrame, report: Report) -> pd.DataFrame:
    log.info("[Phase 4] Basic features")
    df = df.copy()
    df = add_date_features(df)
    df = add_building_age(df, report)
    df = add_floor_features(df, report)
    df = add_area_features(df)
    df = add_address_keys(df)
    df = add_contract_period(df, report)
    df = add_price_columns(df)
    return df


# ====================================================== historical features
def _group_ids(df: pd.DataFrame, group_cols: list[str]) -> np.ndarray:
    return df.groupby(group_cols, dropna=False, sort=True).ngroup().to_numpy()


def past_window_stats(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str | None,
    window_days: int,
    stats=("count", "median"),
    offset_days: int = 0,
    min_count: int = config.MIN_ROLLING_COUNT,
    date_col: str = "contract_date",
) -> pd.DataFrame:
    """각 행 i에 대해 같은 그룹의 과거 거래 j 중
        t_i - offset - window <= t_j < t_i - offset
    을 만족하는 거래들의 통계를 계산한다 (t = 날짜, 일 단위).

    - offset_days=0 이면 [t-W, t): 같은 날짜 거래는 제외.
    - 'count'는 거래 수(없으면 0). median/mean/std는 거래 수가 min_count 미만이면 NaN.
    - value_col=None이면 count만 의미가 있다.

    반환: df.index에 정렬된 DataFrame, 컬럼 = stats.
    """
    gid = _group_ids(df, group_cols)
    t = df[date_col].to_numpy()
    values = df[value_col].to_numpy(dtype="float64") if value_col else np.ones(len(df))

    data = pd.DataFrame({"gid": gid, "t": t, "v": values, "order": 1})
    q_t = t - np.timedelta64(offset_days, "D")
    queries = pd.DataFrame({"gid": gid, "t": q_t}).drop_duplicates()
    queries["v"] = np.nan
    queries["order"] = 0          # 같은 시각이면 질의 행이 실제 거래보다 먼저 오도록
    comb = (pd.concat([queries, data], ignore_index=True)
            .sort_values(["gid", "t", "order"], kind="stable").reset_index(drop=True))

    roll = comb.set_index("t").groupby("gid", sort=True)["v"].rolling(
        f"{window_days}D", closed="left", min_periods=0)
    out = {}
    cnt_s = roll.count()
    # groupby-rolling 결과는 (gid, t) 순서로 나온다. comb가 같은 순서로 정렬돼 있어야 위치 기반 매핑이 맞다.
    assert np.array_equal(cnt_s.index.get_level_values(0).to_numpy(), comb["gid"].to_numpy())
    assert np.array_equal(cnt_s.index.get_level_values(1).to_numpy(), comb["t"].to_numpy())
    cnt = cnt_s.to_numpy()
    for s in stats:
        if s == "count":
            out[s] = cnt
        else:
            vals = getattr(roll, s)().to_numpy()
            out[s] = np.where(cnt >= min_count, vals, np.nan)
    res = pd.DataFrame(out)
    res["gid"], res["t"], res["order"] = comb["gid"].to_numpy(), comb["t"].to_numpy(), comb["order"].to_numpy()
    res = res.loc[res["order"] == 0].drop(columns="order")

    left = pd.DataFrame({"gid": gid, "t": q_t}, index=df.index)
    merged = left.merge(res, on=["gid", "t"], how="left", validate="many_to_one")
    merged.index = df.index
    return merged[list(stats)]


def previous_transaction(df: pd.DataFrame, group_cols: list[str], value_col: str,
                         date_col: str = "contract_date", offset_days: int = 0) -> pd.DataFrame:
    """같은 그룹에서 (현재 날짜 - offset_days)보다 **엄격히 이전** 날짜의 가장 최근 거래(가격, 날짜).

    merge_asof(allow_exact_matches=False)를 사용하므로 기준일과 같은 날짜 거래는 제외된다.
    직전 날짜에 거래가 여러 건이면 tx_id가 가장 큰(원본 정렬상 마지막) 거래를 사용한다.
    """
    gid = _group_ids(df, group_cols)
    q_t = df[date_col].to_numpy() - np.timedelta64(offset_days, "D")
    left = pd.DataFrame({"gid": gid, "t": q_t, "_row": np.arange(len(df))})
    right = pd.DataFrame({"gid": gid, "t": df[date_col].to_numpy(),
                          "prev_value": df[value_col].to_numpy(dtype="float64"),
                          "prev_date": df[date_col].to_numpy(), "_tx": df["tx_id"].to_numpy()})
    left = left.sort_values("t", kind="stable")
    # 가격이 없는 행(서빙 시 붙이는 가상 거래)은 '직전 거래'가 될 수 없다
    right = right[right["prev_value"].notna()].sort_values(["t", "_tx"], kind="stable")
    m = pd.merge_asof(left, right.drop(columns="_tx"), on="t", by="gid",
                      allow_exact_matches=False, direction="backward")
    m = m.sort_values("_row")
    return pd.DataFrame({"prev_value": m["prev_value"].to_numpy(),
                         "prev_date": m["prev_date"].to_numpy()}, index=df.index)


def _trend(recent: pd.Series, previous: pd.Series) -> pd.Series:
    prev = previous.where(previous > 0)
    return recent / prev - 1


def add_coverage_gap_flag(df: pd.DataFrame, report: Report, lookback_days: int) -> pd.DataFrame:
    """수집 누락 월(해당 gu×housing_category의 거래가 0건인 달)을 찾아 보고하고,
    각 행의 과거 lookback 구간에 누락 월이 포함되면 history_window_has_data_gap=1.

    누락 월이 섞인 window의 count/median은 '시장이 조용했다'가 아니라 '데이터가 없다'는 뜻이므로
    모델 단계에서 이 flag로 구분할 수 있게 한다. (데이터 시작 이전 구간은 누락으로 보지 않는다.)
    """
    months = pd.period_range(config.START_DATE, config.END_DATE, freq="M")
    m_idx = {p: i for i, p in enumerate(months)}
    grp = config.COVERAGE_GROUP
    ym = df["contract_date"].dt.to_period("M")
    present = df.assign(_m=ym.map(m_idx)).groupby(grp)["_m"].unique()

    gaps, cum = {}, {}
    for key, ms in present.items():
        missing = np.ones(len(months), dtype=int)
        missing[list(ms)] = 0
        cum[key] = np.concatenate([[0], np.cumsum(missing)])
        miss_months = [str(months[i]) for i in np.flatnonzero(missing)]
        if miss_months:
            gaps["|".join(key)] = miss_months
    for g in config.VALID_GU:
        for cat in config.HOUSING_CATEGORIES:
            if (g, cat) not in present.index:
                gaps[f"{g}|{cat}"] = "NO DATA AT ALL"
    report.set("coverage_gaps", gaps)
    for k, v in gaps.items():
        log.warning("  데이터 공백: %s -> %s", k, v if isinstance(v, str) else f"{len(v)}개월 ({v[0]} ~ {v[-1]})")

    start_m = (df["contract_date"] - pd.Timedelta(days=lookback_days)).dt.to_period("M")
    end_m = (df["contract_date"] - pd.Timedelta(days=1)).dt.to_period("M")
    s_i = start_m.map(lambda p: m_idx.get(p, 0) if p >= months[0] else 0).to_numpy()
    e_i = end_m.map(lambda p: m_idx.get(p, -1)).to_numpy()
    keys = list(df[grp].itertuples(index=False, name=None))
    flag = np.zeros(len(df), dtype="int8")
    for i, (k, s, e) in enumerate(zip(keys, s_i, e_i)):
        if e >= s and e >= 0:
            c = cum[k]
            flag[i] = int(c[e + 1] - c[s] > 0)
    df["history_window_has_data_gap"] = flag
    report.check("features", f"rows whose {lookback_days}d history overlaps a data gap", flag.sum(),
                 "flag history_window_has_data_gap", level="WARNING")
    return df


def add_historical_features(df: pd.DataFrame, report: Report) -> pd.DataFrame:
    """Phase 5. df는 contract_date 오름차순 정렬되어 있어야 한다 (assert)."""
    log.info("[Phase 5] Historical features (leakage-free, window = [t-W, t))")
    df = df.sort_values(["contract_date", "tx_id"], kind="stable").reset_index(drop=True)
    assert df["contract_date"].is_monotonic_increasing
    wins, min_n = config.ROLLING_WINDOWS, config.MIN_ROLLING_COUNT

    # --- 지역(동) 거래량 / 가격 (§22, §23)
    for w in wins:
        stats = ("count", "median", "mean", "std") if w == 90 else ("count", "median")
        r = past_window_stats(df, config.DONG_GROUP, "price_10k", w, stats)
        df[f"dong_tx_count_{w}d"] = r["count"].astype("int32")
        df[f"dong_price_median_{w}d"] = r["median"]
        if w == 90:
            df["dong_price_mean_90d"] = r["mean"]
            df["dong_price_std_90d"] = r["std"]
    df["gu_tx_count_90d"] = past_window_stats(df, config.GU_GROUP, None, 90, ("count",))["count"].astype("int32")

    # --- 비교 가능 주택 (§24)
    for w in wins:
        stats = ("count", "median")
        r = past_window_stats(df, config.COMP_GROUP, "price_10k", w, stats)
        df[f"comp_price_median_{w}d"] = r["median"]
        if w == 90:
            df["comp_tx_count_90d"] = r["count"].astype("int32")

    # --- 월세용: 과거 보증금 수준 (보증금-월세 trade-off 보정용)
    df["dong_deposit_median_90d"] = past_window_stats(df, config.DONG_GROUP, "deposit_10k", 90, ("median",))["median"]
    df["comp_deposit_median_90d"] = past_window_stats(df, config.COMP_GROUP, "deposit_10k", 90, ("median",))["median"]

    # --- 동일 건물 (§25)
    prev = previous_transaction(df, config.PROPERTY_GROUP, "price_10k")
    df["property_last_price"] = prev["prev_value"]
    df["property_last_date"] = prev["prev_date"]
    df["property_days_since_last_transaction"] = (df["contract_date"] - prev["prev_date"]).dt.days.astype("float64")
    r = past_window_stats(df, config.PROPERTY_GROUP, "price_10k", config.PROPERTY_HISTORY_DAYS,
                          ("count", "median"), min_count=1)
    df["property_tx_count_365d"] = r["count"].astype("int32")
    df["property_price_median_365d"] = r["median"]

    # --- 추세 (§26): 최근 90일 median / 그 이전 90일 median - 1
    tw = config.TREND_WINDOW_DAYS
    for name, grp in [("dong", config.DONG_GROUP), ("comp", config.COMP_GROUP)]:
        recent = df[f"{name}_price_median_{tw}d"]
        previous = past_window_stats(df, grp, "price_10k", tw, ("median",), offset_days=tw)["median"]
        df[f"{name}_price_trend_3m"] = _trend(recent, previous)

    # --- missing indicators (§30)
    df["has_property_history"] = df["property_last_price"].notna().astype("int8")
    df["has_comp_90d_history"] = df["comp_price_median_90d"].notna().astype("int8")
    df["has_dong_90d_history"] = df["dong_price_median_90d"].notna().astype("int8")

    df = add_coverage_gap_flag(df, report, max(wins))
    for c in ["dong_price_median_90d", "comp_price_median_90d", "property_last_price", "dong_price_trend_3m"]:
        rule = "no earlier tx of same property" if c.startswith("property") else f"< {min_n} past tx"
        report.check("features", f"{c} NaN ({rule})",
                     df[c].isna().sum(), "keep NaN")
    return df


# ================================================================ outliers
def add_outlier_flags(df: pd.DataFrame, report: Report) -> pd.DataFrame:
    """log 가격 분포의 그룹별 IQR fence 밖이면 후보로 flag (삭제하지 않음).

    주의: 전체 기간 분포를 사용하므로 **모델 feature로 쓰면 안 된다** (EDA/필터 검토용).
    그룹 표본이 OUTLIER_MIN_GROUP_SIZE 미만이면 (gu, rent_type, housing_category)로 fallback.
    """
    k = config.OUTLIER_IQR_K

    def flag(value: pd.Series, mask: pd.Series) -> pd.Series:
        lv = np.log1p(value.where(mask & (value >= 0)))
        out = pd.Series(False, index=df.index)
        fine, coarse = config.OUTLIER_GROUP, ["gu", "rent_type", "housing_category"]
        fence = {}
        for grp in (coarse, fine):   # fine이 충분하면 덮어씀
            g = lv.groupby([df[c] for c in grp], dropna=False)
            q1, q3, n = g.transform(lambda s: s.quantile(.25)), g.transform(lambda s: s.quantile(.75)), g.transform("count")
            ok = n >= config.OUTLIER_MIN_GROUP_SIZE
            iqr = q3 - q1
            lo, hi = q1 - k * iqr, q3 + k * iqr
            if not fence:
                fence = {"lo": lo, "hi": hi}
            else:
                fence["lo"], fence["hi"] = lo.where(ok, fence["lo"]), hi.where(ok, fence["hi"])
        out = (lv < fence["lo"]) | (lv > fence["hi"])
        return out.fillna(False).astype("int8")

    is_j, is_m = df["rent_type"] == "전세", df["rent_type"] == "월세"
    df["deposit_outlier_candidate"] = flag(df["deposit_10k"], is_j | is_m)
    df["monthly_rent_outlier_candidate"] = flag(df["monthly_rent_10k"], is_m & (df["monthly_rent_10k"] > 0))
    df["is_price_outlier_candidate"] = np.where(
        is_j, df["deposit_outlier_candidate"], df["monthly_rent_outlier_candidate"]).astype("int8")
    for c in ["deposit_outlier_candidate", "monthly_rent_outlier_candidate", "is_price_outlier_candidate"]:
        report.check("features", c, df[c].sum(), "flag only (not removed)",
                     f"log-price IQR x{k} within {'/'.join(config.OUTLIER_GROUP)}")
    for rt, col in config.PRICE_COLUMN_BY_RENT_TYPE.items():
        q = df.loc[df["rent_type"] == rt, col].quantile([.001, .01, .5, .99, .999]).round(1).to_dict()
        report.check("features", f"{rt} {col} quantiles", int((df["rent_type"] == rt).sum()), "info", str(q))
    return df


# =========================================================== model datasets
COMMON_FEATURES = [
    "contract_year", "contract_month", "contract_quarter", "month_sin", "month_cos",
    "gu", "dong", "property_id", "housing_category", "housing_type",
    "area_m2", "area_pyeong", "area_bin", "floor", "is_basement", "is_ground_floor",
    "built_year", "building_age", "building_name_is_address",
    "dong_tx_count_30d", "dong_tx_count_90d", "dong_tx_count_180d", "gu_tx_count_90d",
    "dong_price_median_30d", "dong_price_median_90d", "dong_price_median_180d",
    "dong_price_mean_90d", "dong_price_std_90d",
    "comp_price_median_30d", "comp_price_median_90d", "comp_price_median_180d", "comp_tx_count_90d",
    "property_last_price", "property_days_since_last_transaction",
    "property_tx_count_365d", "property_price_median_365d",
    "dong_price_trend_3m", "comp_price_trend_3m",
    "has_property_history", "has_comp_90d_history", "has_dong_90d_history",
    "history_window_has_data_gap",
]
JEONSE_FEATURES = COMMON_FEATURES
MONTHLY_FEATURES = COMMON_FEATURES + [
    "deposit_10k", "deposit_per_m2", "dong_deposit_median_90d", "comp_deposit_median_90d",
]
# 모델 데이터에 함께 저장하지만 feature가 아닌 식별/merge/split 컬럼
ID_COLUMNS = ["tx_id", "contract_date", "contract_year_month", "sido", "full_jibun_address",
              "full_road_address", "contract_type_clean", "is_partial_year", "split"]

# 절대 feature로 들어가면 안 되는 컬럼 (§17, §27, §28)
FORBIDDEN_COMMON = {
    "price_10k", "monthly_rent_per_m2_eda_only", "renewal_request_used", "renewal_right_used",
    "previous_deposit_10k", "previous_monthly_rent_10k",
    "deposit_change_from_previous", "monthly_rent_change_from_previous",
    "deposit_change_rate_from_previous", "monthly_rent_change_rate_from_previous",
    "deposit_outlier_candidate", "monthly_rent_outlier_candidate", "is_price_outlier_candidate",
}
FORBIDDEN = {
    "jeonse": FORBIDDEN_COMMON | {"deposit_10k", "deposit_per_m2", "monthly_rent_10k",
                                  "dong_deposit_median_90d", "comp_deposit_median_90d"},
    "monthly": FORBIDDEN_COMMON | {"monthly_rent_10k"},
}


def assign_split(dates: pd.Series) -> pd.Series:
    out = pd.Series(pd.NA, index=dates.index, dtype="string")
    for name, (s, e) in config.SPLIT_BOUNDARIES.items():
        out[(dates >= s) & (dates <= e)] = name
    return out


def _contract_filter(df: pd.DataFrame, new_only: bool) -> pd.Series:
    if not new_only:
        return pd.Series(True, index=df.index)
    ok = df["contract_type_clean"] == "new"
    if config.INCLUDE_UNKNOWN_CONTRACT_TYPE:
        ok |= df["contract_type_clean"] == "unknown"
    return ok


def build_model_datasets(df: pd.DataFrame, report: Report, new_only: bool = config.NEW_CONTRACT_ONLY,
                         start_date: str | None = config.MODEL_DATA_START_DATE,
                         exclude_gu: list[str] = config.MODEL_EXCLUDE_GU):
    log.info("[Phase 6] Model datasets (NEW_CONTRACT_ONLY=%s, start=%s, categories=%s, exclude_gu=%s)",
             new_only, start_date, config.MODEL_HOUSING_CATEGORIES, exclude_gu)
    df = df.copy()
    df["split"] = assign_split(df["contract_date"])
    base = df["housing_category"].isin(config.MODEL_HOUSING_CATEGORIES) & _contract_filter(df, new_only)
    if start_date:
        base &= df["contract_date"] >= pd.Timestamp(start_date)
    if exclude_gu:
        excluded = df["gu"].isin(exclude_gu)
        report.check("model_data", f"rows excluded by MODEL_EXCLUDE_GU={exclude_gu}", excluded.sum(),
                     "excluded from model data (kept in all_transactions)", level="WARNING")
        base &= ~excluded
    report.set("model_filters", {"new_only": new_only, "include_unknown_contract_type": config.INCLUDE_UNKNOWN_CONTRACT_TYPE,
                                 "start_date": start_date, "exclude_gu": exclude_gu,
                                 "housing_categories": config.MODEL_HOUSING_CATEGORIES})

    specs = {
        "jeonse": (JEONSE_FEATURES, "deposit_10k", "target_deposit_10k",
                   (df["rent_type"] == "전세") & (df["deposit_10k"] > 0)),
        "monthly": (MONTHLY_FEATURES, "monthly_rent_10k", "target_monthly_rent_10k",
                    (df["rent_type"] == "월세") & (df["monthly_rent_10k"] > 0) & (df["deposit_10k"] >= 0)),
    }
    out = {}
    for name, (feats, tcol, tname, rule) in specs.items():
        leaked = set(feats) & FORBIDDEN[name]
        assert not leaked, f"{name}: leakage feature 포함 {leaked}"
        assert tcol not in feats, f"{name}: target이 feature에 포함"
        mask = base & rule
        d = df.loc[mask, ID_COLUMNS + feats + [tcol]].rename(columns={tcol: tname})
        d = d.sort_values(["contract_date", "tx_id"]).reset_index(drop=True)
        out[name] = d
        report.check("model_data", f"{name} rows", len(d), "saved",
                     f"rule excluded={int((base & ~rule & (df['rent_type'] == ('전세' if name == 'jeonse' else '월세'))).sum())}, "
                     f"features={len(feats)}")
        report.set(f"model_{name}", {
            "rows": len(d), "n_features": len(feats), "features": feats, "target": tname,
            "date_min": str(d["contract_date"].min().date()), "date_max": str(d["contract_date"].max().date()),
            "split_counts": d["split"].value_counts().to_dict(),
        })
    return out
