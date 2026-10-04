"""전처리 핵심 로직 테스트. 특히 rolling feature의 미래/동시점 정보 누수 여부를 검증한다.

    python -m pytest -q
"""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.clean_data import normalize_strings, parse_numeric
from src.feature_engineering import (FORBIDDEN, JEONSE_FEATURES, MONTHLY_FEATURES,
                                     past_window_stats, previous_transaction)
from src.load_data import HeaderNotFoundError, read_molit_csv
from src.report import Report


def _frame(rows, group="A"):
    df = pd.DataFrame(rows, columns=["contract_date", "price"])
    df["contract_date"] = pd.to_datetime(df["contract_date"])
    df["g"] = group
    df["tx_id"] = np.arange(len(df))
    return df


# ------------------------------------------------------------------ §39
def test_leakage_spec_example():
    """2024-01-10 행의 과거 median에는 2024-01-01(10)만 들어가야 한다. 1000은 절대 영향 X."""
    df = _frame([("2024-01-01", 10), ("2024-01-10", 20), ("2024-01-20", 1000)])
    r = past_window_stats(df, ["g"], "price", 90, ("count", "median"), min_count=1)
    assert np.isnan(r.loc[0, "median"]) and r.loc[0, "count"] == 0
    assert r.loc[1, "median"] == 10 and r.loc[1, "count"] == 1
    assert r.loc[2, "median"] == 15 and r.loc[2, "count"] == 2   # median(10, 20)


def test_future_values_do_not_change_past_features():
    df = _frame([("2024-01-01", 10), ("2024-01-10", 20), ("2024-01-20", 1000)])
    a = past_window_stats(df, ["g"], "price", 90, ("count", "median", "mean"), min_count=1)
    df2 = df.copy()
    df2.loc[2, "price"] = -99999
    b = past_window_stats(df2, ["g"], "price", 90, ("count", "median", "mean"), min_count=1)
    pd.testing.assert_frame_equal(a.iloc[:2], b.iloc[:2])


def test_same_day_transactions_excluded():
    df = _frame([("2024-01-01", 10), ("2024-01-05", 50), ("2024-01-05", 70), ("2024-01-05", 90)])
    r = past_window_stats(df, ["g"], "price", 30, ("count", "median"), min_count=1)
    # 1/5의 세 거래 모두 1/1 거래만 본다 (서로를 보지 않음)
    assert (r.loc[1:, "count"] == 1).all()
    assert (r.loc[1:, "median"] == 10).all()


def test_window_boundaries_and_min_count():
    df = _frame([("2024-01-01", 1), ("2024-01-02", 2), ("2024-01-03", 3), ("2024-01-31", 4), ("2024-02-01", 5)])
    r = past_window_stats(df, ["g"], "price", 30, ("count", "median"), min_count=3)
    # 2024-01-31: window [01-01, 01-31) → 1,2,3 포함
    assert r.loc[3, "count"] == 3 and r.loc[3, "median"] == 2
    # 2024-02-01: window [01-02, 02-01) → 2,3,4 (01-01은 밖)
    assert r.loc[4, "count"] == 3 and r.loc[4, "median"] == 3
    # 거래 수 < min_count → NaN
    assert np.isnan(r.loc[2, "median"]) and r.loc[2, "count"] == 2


def test_groups_are_isolated():
    a = _frame([("2024-01-01", 10), ("2024-01-10", 20)], group="A")
    b = _frame([("2024-01-05", 999)], group="B")
    df = pd.concat([a, b], ignore_index=True).sort_values("contract_date").reset_index(drop=True)
    df["tx_id"] = np.arange(len(df))
    r = past_window_stats(df, ["g"], "price", 90, ("median",), min_count=1)
    assert r.loc[df["contract_date"] == "2024-01-10", "median"].item() == 10


def test_offset_window_for_trend():
    """offset=90: [t-180, t-90) 구간만 사용."""
    df = _frame([("2024-01-01", 10), ("2024-03-01", 20), ("2024-05-01", 30), ("2024-07-01", 40)])
    r = past_window_stats(df, ["g"], "price", 90, ("count", "median"), offset_days=90, min_count=1)
    # 2024-07-01: [2024-01-03, 2024-04-02) → 03-01(20)만
    assert r.loc[3, "count"] == 1 and r.loc[3, "median"] == 20


def test_matches_bruteforce_on_random_data():
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame({
        "contract_date": pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 400, n), unit="D"),
        "g": rng.choice(["A", "B", "C"], n),
        "price": rng.integers(1, 1000, n).astype(float),
    }).sort_values("contract_date").reset_index(drop=True)
    df["tx_id"] = np.arange(n)
    r = past_window_stats(df, ["g"], "price", 60, ("count", "median", "mean"), min_count=2)
    t = df["contract_date"].to_numpy()
    for i in rng.choice(n, 80, replace=False):
        m = (df["g"] == df.loc[i, "g"]).to_numpy() & (t < t[i]) & (t >= t[i] - np.timedelta64(60, "D"))
        vals = df.loc[m, "price"]
        assert r.loc[i, "count"] == len(vals)
        if len(vals) >= 2:
            assert np.isclose(r.loc[i, "median"], vals.median())
            assert np.isclose(r.loc[i, "mean"], vals.mean())
        else:
            assert np.isnan(r.loc[i, "median"])


def test_previous_transaction_strictly_before():
    df = _frame([("2024-01-01", 10), ("2024-01-05", 20), ("2024-01-05", 30), ("2024-02-01", 40)])
    r = previous_transaction(df, ["g"], "price")
    assert np.isnan(r.loc[0, "prev_value"])
    # 같은 날짜(1/5) 거래끼리는 서로의 '직전 거래'가 될 수 없다
    assert r.loc[1, "prev_value"] == 10 and r.loc[2, "prev_value"] == 10
    assert r.loc[3, "prev_value"] in (20, 30) and r.loc[3, "prev_date"] == pd.Timestamp("2024-01-05")


# ------------------------------------------------------------------ loading / cleaning
HEADER = ('"NO","시군구","번지","본번","부번","건물명","전월세구분","전용면적(㎡)","계약년월","계약일",'
          '"보증금(만원)","월세금(만원)","층","건축년도","도로명","계약기간","계약구분","갱신요구권 사용",'
          '"종전계약 보증금(만원)","종전계약 월세(만원)","주택유형"')
ROW = ('"1","서울특별시 서대문구 북가좌동","295-9","0295","0009","케이하우스","전세","59.9300","202112","31",'
       '"21,525","0","-1","2014","증가로23길 46","-","-","-","","","연립다세대"')


@pytest.mark.parametrize("n_preamble", [0, 3, 15, 22])
def test_csv_header_detection(tmp_path, n_preamble):
    lines = [f'"□ 안내문 {i}"' for i in range(n_preamble)] + [HEADER, ROW]
    p = tmp_path / "x.csv"
    p.write_text("\n".join(lines), encoding="cp949")
    df, info = read_molit_csv(p)
    assert info.header_line_index == n_preamble and info.encoding == "cp949"
    assert len(df) == 1 and df.loc[0, "본번"] == "0295"


def test_csv_without_header_raises(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text('"□ 안내문"\n"a","b"\n', encoding="cp949")
    with pytest.raises(HeaderNotFoundError):
        read_molit_csv(p)


def test_numeric_and_missing_tokens():
    df = pd.DataFrame({c: ["x", "x", "x"] for c in config.COLUMN_MAP.values()})
    df["deposit_10k"] = ["21,525", "-", " "]
    df["bonbun"] = ["0295", "0009", "-"]
    for c in ["area_m2", "monthly_rent_10k", "floor", "built_year", "previous_deposit_10k",
              "previous_monthly_rent_10k"]:
        df[c] = ["1", "-1", ""]
    rep = Report()
    out = parse_numeric(normalize_strings(df, rep), rep)
    assert out.loc[0, "deposit_10k"] == 21525
    assert out["deposit_10k"].isna().sum() == 2            # '-'와 공백은 0이 아니라 NaN
    assert out.loc[0, "bonbun"] == "0295"                  # leading zero 보존
    assert pd.isna(out.loc[2, "bonbun"])
    assert out.loc[1, "floor"] == -1                       # 지하층 유지


# ------------------------------------------------------------------ target leakage lists
def test_model_feature_lists_have_no_forbidden_columns():
    assert not set(JEONSE_FEATURES) & FORBIDDEN["jeonse"]
    assert not set(MONTHLY_FEATURES) & FORBIDDEN["monthly"]
    assert "deposit_10k" in MONTHLY_FEATURES                 # 월세 모델은 보증금을 반드시 사용
    assert "deposit_per_m2" not in JEONSE_FEATURES
    assert "monthly_rent_per_m2_eda_only" not in MONTHLY_FEATURES


# ------------------------------------------------------------------ horizon (미래 예측) feature
def test_horizon_features_respect_asof_and_reporting_lag():
    """h=60, lag=30 → 계약일 t의 feature는 t-90일 이전 거래만 사용해야 한다."""
    from src.horizon_dataset import history_features_at_offset
    rows = [("2024-01-01", 100), ("2024-02-01", 200), ("2024-03-01", 300),
            ("2024-04-15", 9999), ("2024-05-01", 9999), ("2024-05-20", 500)]
    df = pd.DataFrame(rows, columns=["contract_date", "price_10k"])
    df["contract_date"] = pd.to_datetime(df["contract_date"])
    for c, v in dict(gu="g", dong="d", rent_type="전세", housing_category="연립다세대",
                     area_bin="20-25", property_id="p").items():
        df[c] = v
    df["deposit_10k"] = df["price_10k"]
    df["tx_id"] = np.arange(len(df))
    f = history_features_at_offset(df, offset=60 + 30)
    # 2024-05-20 - 90일 = 2024-02-20 → [2023-11-22, 2024-02-20) 의 거래: 100, 200 만
    last = f.iloc[-1]
    assert last["dong_tx_count_90d"] == 2
    assert last["property_last_price"] == 200
    assert last["property_days_since_last_transaction"] == (pd.Timestamp("2024-02-20") - pd.Timestamp("2024-02-01")).days
    assert last["property_price_median_365d"] == 150     # 9999는 절대 포함되면 안 됨


def test_serving_synthetic_rows_match_training_features():
    """서빙 시 가격 NaN 가상 거래를 붙여 계산한 feature == 학습 때 실제 거래의 feature (학습-서빙 불일치 방지)."""
    from src.horizon_dataset import HISTORY_FEATURES, history_features_at_offset
    rng = np.random.default_rng(1)
    n = 400
    df = pd.DataFrame({
        "contract_date": pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 300, n), unit="D"),
        "gu": "g", "dong": rng.choice(["d1", "d2"], n), "rent_type": "월세", "housing_category": "연립다세대",
        "area_bin": rng.choice(["15-20", "20-25"], n), "property_id": rng.choice([f"p{i}" for i in range(15)], n),
        "price_10k": rng.integers(30, 90, n).astype(float), "deposit_10k": rng.integers(500, 3000, n).astype(float),
    }).sort_values("contract_date").reset_index(drop=True)
    df["tx_id"] = np.arange(n)
    off = 60 + 30
    ref = history_features_at_offset(df, off)
    pick = rng.choice(n, 50, replace=False)
    syn = df.iloc[pick].copy()
    syn["price_10k"], syn["deposit_10k"], syn["tx_id"] = np.nan, np.nan, syn["tx_id"] + 10**9
    got = history_features_at_offset(pd.concat([df, syn], ignore_index=True), off).iloc[n:].reset_index(drop=True)
    exp = ref.iloc[pick].reset_index(drop=True)
    for c in HISTORY_FEATURES:
        assert np.allclose(got[c].astype(float), exp[c].astype(float), equal_nan=True), c
