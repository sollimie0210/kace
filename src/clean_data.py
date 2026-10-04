"""컬럼 rename, 문자열/숫자/날짜 정규화, 행정구역 파싱, 중복 제거, 오류 행 제거.

원칙
- 명백한 오류(면적<=0, 음수 가격, 잘못된 날짜)만 제거하고 나머지는 flag로 남긴다.
- '-' 같은 결측 토큰을 0으로 바꾸지 않는다.
- 제거/flag 건수는 모두 Report에 기록한다.
"""
from __future__ import annotations

import logging
import unicodedata

import numpy as np
import pandas as pd

from . import config
from .report import Report

log = logging.getLogger("preprocess")

META_COLUMNS = ["source_file", "source_sheet", "source_table_id", "source_cond_gu",
                "source_cond_period", "housing_category"]


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=config.COLUMN_MAP)


def normalize_strings(df: pd.DataFrame, report: Report) -> pd.DataFrame:
    """모든 문자열 컬럼: NFKC 정규화, strip, 결측 토큰('', '-', 'nan' ...) → NA.

    본번/부번 등은 leading zero를 보존하기 위해 문자열로 유지한다.
    """
    df = df.copy()
    raw_cols = [c for c in config.COLUMN_MAP.values() if c in df.columns]
    n_missing_tokens = 0
    for col in raw_cols:
        s = df[col].astype("str")
        s = s.map(lambda x: unicodedata.normalize("NFKC", x) if isinstance(x, str) else x).astype("str")
        s = s.str.strip()
        is_missing = s.isin(config.MISSING_TOKENS)
        n_missing_tokens += int(is_missing.sum())
        df[col] = s.mask(is_missing)
    report.check("clean", "missing tokens ('', '-', ' ', 'nan') -> NA", n_missing_tokens, "set NA")
    return df


def parse_numeric(df: pd.DataFrame, report: Report) -> pd.DataFrame:
    """콤마 제거 후 숫자 변환. 원래 값이 있었는데 변환에 실패한 건수를 기록."""
    df = df.copy()
    failures = {}
    for col in config.NUMERIC_COLUMNS:
        raw = df[col]
        num = pd.to_numeric(raw.str.replace(",", "", regex=False), errors="coerce")
        failed = raw.notna() & num.isna()
        failures[col] = int(failed.sum())
        if failed.any():
            log.warning("  %s 숫자 변환 실패 %d건, 예: %s", col, failed.sum(),
                        raw[failed].value_counts().head(5).to_dict())
        df[col] = num.astype("float64")
    report.set("numeric_parse_failures", failures)
    report.check("clean", "numeric parse failures (total)", sum(failures.values()), "set NaN",
                 ", ".join(f"{k}={v}" for k, v in failures.items() if v))
    return df


def parse_dates(df: pd.DataFrame, report: Report) -> pd.DataFrame:
    """contract_ym(YYYYMM) + contract_day(DD) → contract_date. 범위 밖/파싱 불가는 제거."""
    df = df.copy()
    ymd = df["contract_ym"].fillna("") + df["contract_day"].fillna("").str.zfill(2)
    df["contract_date"] = pd.to_datetime(ymd, format="%Y%m%d", errors="coerce")
    unparsable = df["contract_date"].isna()
    before = df["contract_date"] < pd.Timestamp(config.START_DATE)
    after = df["contract_date"] > pd.Timestamp(config.END_DATE)
    report.check("clean", "invalid date: unparsable", unparsable.sum(), "remove", level="WARNING")
    report.check("clean", f"invalid date: before {config.START_DATE}", before.sum(), "remove", level="WARNING")
    report.check("clean", f"invalid date: after {config.END_DATE}", after.sum(), "remove", level="WARNING")
    invalid = unparsable | before | after
    report.set("n_invalid_dates", int(invalid.sum()))
    return df.loc[~invalid].copy()


def parse_location(df: pd.DataFrame, report: Report) -> pd.DataFrame:
    """'서울특별시 서대문구 북가좌동' → sido / gu / dong."""
    df = df.copy()
    parts = df["sigungu"].str.split(r"\s+", n=2, expand=True, regex=True)
    parts = parts.reindex(columns=[0, 1, 2])
    df["sido"], df["gu"], df["dong"] = parts[0], parts[1], parts[2]
    failed = df[["sido", "gu", "dong"]].isna().any(axis=1)
    report.check("clean", "sigungu split failure (sido/gu/dong)", failed.sum(), "keep, NA parts",
                 str(df.loc[failed, "sigungu"].value_counts().head(5).to_dict()) if failed.any() else "",
                 level="WARNING")

    # 검색조건의 시군구(CSV에만 있음)와 실제 데이터의 구가 다른지 확인
    has_cond = df["source_cond_gu"].notna()
    mismatch = has_cond & (df["source_cond_gu"] != df["gu"])
    report.check("clean", "search-condition gu != data gu", mismatch.sum(), "keep (data wins)", level="WARNING")

    invalid_gu = ~df["gu"].isin(config.VALID_GU)
    if invalid_gu.any():
        detail = str(df.loc[invalid_gu, "gu"].value_counts(dropna=False).to_dict())
        action = "remove" if config.DROP_INVALID_GU else "keep (DROP_INVALID_GU=False)"
        report.check("clean", "gu outside VALID_GU", invalid_gu.sum(), action, detail, level="WARNING")
        if config.DROP_INVALID_GU:
            df = df.loc[~invalid_gu].copy()
    else:
        report.check("clean", "gu outside VALID_GU", 0, "none")
    return df


DUP_CANDIDATE_KEYS = ["contract_date", "gu", "dong", "jibun", "building_name", "area_m2",
                      "floor", "rent_type", "deposit_10k", "monthly_rent_10k"]
EXACT_DUP_KEYS = [c for c in config.COLUMN_MAP.values() if c != "row_no"] + ["housing_category"]


def remove_duplicates(df: pd.DataFrame, report: Report) -> pd.DataFrame:
    """2단계 중복 처리.

    1) 완전 중복: row_no를 제외한 모든 원본 컬럼이 동일한 행.
       단, 같은 파일(표) 안에서 완전히 같은 행이 여러 번 나오는 것은 실제로 서로 다른
       호실의 거래일 수 있다(공개자료에 호수가 없음). 따라서 '같은 표 안에서 k번째 등장'
       순번(dup_rank)을 키에 포함해, 다른 파일에서 같은 거래가 반복된 경우(파일 중복
       다운로드/기간 겹침)만 제거한다.
    2) 거래 중복 후보: DUP_CANDIDATE_KEYS가 같은 행 → 제거하지 않고 flag + report.
    """
    df = df.copy()
    n0 = len(df)
    keys = df[EXACT_DUP_KEYS].astype("string").fillna("<NA>")
    df["_dup_rank"] = keys.assign(_t=df["source_table_id"]).groupby(
        list(keys.columns) + ["_t"], dropna=False).cumcount()
    keys["_dup_rank"] = df["_dup_rank"]
    exact_dup = keys.duplicated(keep="first")
    within_table_identical = int(keys.drop(columns="_dup_rank").assign(_t=df["source_table_id"]).duplicated().sum())
    report.check("clean", "exact duplicates across files", exact_dup.sum(), "remove", level="WARNING")
    report.check("clean", "identical rows within the same file", within_table_identical,
                 "keep (likely different units), flagged as dup candidate")
    df = df.loc[~exact_dup].drop(columns="_dup_rank")
    report.set("n_exact_duplicates_removed", int(n0 - len(df)))

    cand = df[DUP_CANDIDATE_KEYS].astype("string").fillna("<NA>").duplicated(keep=False)
    df["is_duplicate_candidate"] = cand.astype("int8")
    report.check("clean", "transaction duplicate candidates (key match)", cand.sum(), "flag only",
                 "keys=" + ",".join(DUP_CANDIDATE_KEYS))
    report.set("n_duplicate_candidates", int(cand.sum()))
    return df


def apply_validity_rules(df: pd.DataFrame, report: Report) -> pd.DataFrame:
    """절대 규칙 위반 행만 제거하고, 전세/월세 규칙 위반은 flag로 남긴다."""
    df = df.copy()
    bad_area = df["area_m2"].isna() | (df["area_m2"] <= 0)
    bad_dep = df["deposit_10k"] < 0
    bad_rent = df["monthly_rent_10k"] < 0
    report.check("clean", "area_m2 <= 0 or missing", bad_area.sum(), "remove", level="WARNING")
    report.check("clean", "deposit_10k < 0", bad_dep.sum(), "remove", level="WARNING")
    report.check("clean", "monthly_rent_10k < 0", bad_rent.sum(), "remove", level="WARNING")
    report.set("n_invalid_area", int(bad_area.sum()))
    report.set("n_invalid_price", int((bad_dep | bad_rent).sum()))
    df = df.loc[~(bad_area | bad_dep | bad_rent)].copy()

    rt = df["rent_type"]
    report.check("clean", "rent_type distribution", len(df), "info",
                 str(rt.value_counts(dropna=False).to_dict()))
    unknown_rt = ~rt.isin(["전세", "월세"])
    report.check("clean", "rent_type not in {전세, 월세}", unknown_rt.sum(), "keep, excluded from models",
                 level="WARNING")

    anomalies = {
        "anomaly_jeonse_with_rent": (rt == "전세") & (df["monthly_rent_10k"] > 0),
        "anomaly_jeonse_zero_deposit": (rt == "전세") & ~(df["deposit_10k"] > 0),
        "anomaly_monthly_zero_rent": (rt == "월세") & ~(df["monthly_rent_10k"] > 0),
    }
    for name, mask in anomalies.items():
        df[name] = mask.astype("int8")
        report.check("clean", name, mask.sum(), "flag (excluded by model filter)", level="WARNING")
    return df


def clean_contract_fields(df: pd.DataFrame, report: Report) -> pd.DataFrame:
    """계약구분/갱신요구권 정규화."""
    df = df.copy()
    ct = df["contract_type"]
    df["contract_type_clean"] = np.select(
        [ct == "신규", ct == "갱신"], ["new", "renewal"], default="unknown")
    unexpected = ct.notna() & ~ct.isin(["신규", "갱신"])
    report.check("clean", "contract_type unexpected values", unexpected.sum(), "-> unknown",
                 str(ct[unexpected].value_counts().head().to_dict()), level="WARNING")

    by_year = (df.assign(y=df["contract_date"].dt.year)
               .groupby("y")["contract_type_clean"].apply(lambda s: round(float((s == "unknown").mean()), 4)))
    report.set("contract_type_missing_rate_by_year", {int(k): v for k, v in by_year.items()})
    log.info("  contract_type 결측(unknown) 비율 by year: %s", by_year.to_dict())

    # '사용' → 1, 그 외('-'/결측: 미사용 또는 해당없음) → 0
    rr = df["renewal_request_used"]
    unexpected_rr = rr.notna() & (rr != "사용")
    report.check("clean", "renewal_request_used unexpected values", unexpected_rr.sum(), "-> 0",
                 str(rr[unexpected_rr].value_counts().head().to_dict()), level="WARNING")
    df["renewal_right_used"] = (rr == "사용").astype("int8")

    # 종전계약 정보는 갱신 계약에만 있어야 한다
    prev_on_non_renewal = df["previous_deposit_10k"].notna() & (df["contract_type_clean"] != "renewal")
    report.check("clean", "previous-contract info on non-renewal rows", prev_on_non_renewal.sum(),
                 "keep", level="WARNING")
    return df


def clean(df_raw: pd.DataFrame, report: Report) -> pd.DataFrame:
    """Phase 3 전체. 입력은 원본 한글 컬럼 + 메타 컬럼."""
    log.info("[Phase 3] Clean")
    df = rename_columns(df_raw)
    df = normalize_strings(df, report)
    df = parse_numeric(df, report)
    df = parse_dates(df, report)
    df = parse_location(df, report)
    df = remove_duplicates(df, report)
    df = apply_validity_rules(df, report)
    df = clean_contract_fields(df, report)
    df["row_no"] = pd.to_numeric(df["row_no"], errors="coerce").astype("Int64")
    df = df.sort_values(["contract_date", "source_table_id", "row_no"], kind="stable").reset_index(drop=True)
    # 이후 외부 데이터/모델 데이터 간 join을 위한 안정적인 거래 id
    df.insert(0, "tx_id", np.arange(len(df), dtype="int64"))
    return df
