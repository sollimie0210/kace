"""전월세 실거래가 전처리 파이프라인 진입점.

    python -m src.preprocess
    python -m src.preprocess --raw-dir data/raw --new-only true --model-start 2022-01-01
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from . import config
from .clean_data import clean
from .feature_engineering import add_basic_features, add_historical_features, add_outlier_flags, build_model_datasets
from .load_data import load_all, tables_to_frame
from .location_features import add_location_features
from .report import Report, setup_logging
from .validate_data import (bruteforce_leakage_check, missing_rate, monthly_market_summary,
                            rolling_trace_example, source_distribution_check, validate_clean)


def _bool(s: str) -> bool:
    return str(s).lower() in {"1", "true", "yes", "y"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="전월세 실거래가 전처리")
    p.add_argument("--raw-dir", type=Path, default=config.RAW_DATA_DIR)
    p.add_argument("--new-only", type=_bool, default=config.NEW_CONTRACT_ONLY)
    p.add_argument("--model-start", default=config.MODEL_DATA_START_DATE)
    p.add_argument("--exclude-gu", default=",".join(config.MODEL_EXCLUDE_GU),
                   help="모델 데이터에서 제외할 구 (콤마 구분, 빈 문자열이면 전체 포함)")
    p.add_argument("--leakage-samples", type=int, default=400)
    p.add_argument("--skip-csv", action="store_true", help="대용량 CSV 산출물 생략")
    return p.parse_args(argv)


def _rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(config.PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)


def main(argv=None):
    args = parse_args(argv)
    for d in (config.INTERIM_DATA_DIR, config.PROCESSED_DATA_DIR, config.REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)
    log = setup_logging(config.REPORT_DIR / "preprocessing_log.txt")
    report = Report()
    t0 = time.time()

    # ---------------------------------------------------------- Phase 1-2
    log.info("[Phase 1-2] Inspect & Load (%s)", _rel(args.raw_dir))
    raw, infos = load_all(args.raw_dir)
    tables = tables_to_frame(infos)
    tables.to_csv(config.REPORT_DIR / "input_files.csv", index=False, encoding="utf-8-sig")
    n_files = tables["source_file"].nunique()
    report.set("n_raw_files", int(n_files))
    report.set("n_raw_tables", int(len(tables)))
    report.set("n_failed_tables", int((~tables["ok"]).sum()))
    report.set("n_rows_raw", int(tables["n_rows"].sum()))
    report.set("n_rows_after_concat", int(len(raw)))
    raw.to_csv(config.INTERIM_DATA_DIR / "merged_raw.csv", index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------- Phase 3
    df = clean(raw, report)
    if not args.skip_csv:
        df.to_csv(config.INTERIM_DATA_DIR / "cleaned_all.csv", index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------- Phase 4-5
    df = add_basic_features(df, report)
    df = add_location_features(df, report)
    df = add_historical_features(df, report)
    df = add_outlier_flags(df, report)

    # ---------------------------------------------------------- Phase 7 (clean data)
    validate_clean(df, report)
    source_distribution_check(df, report)
    n_mis = bruteforce_leakage_check(df, report, n=args.leakage_samples)
    if n_mis:
        raise AssertionError(f"leakage brute-force 검증 실패: {n_mis}건 불일치")
    trace = rolling_trace_example(df)
    report.set("rolling_trace_example", trace)

    # ---------------------------------------------------------- Phase 6
    exclude_gu = [g.strip() for g in args.exclude_gu.split(",") if g.strip()]
    models = build_model_datasets(df, report, new_only=args.new_only, start_date=args.model_start,
                                  exclude_gu=exclude_gu)

    # ---------------------------------------------------------- save
    out_all = config.PROCESSED_DATA_DIR / "all_transactions.parquet"
    df.to_parquet(out_all, index=False)
    outputs = [out_all]
    for name, d in models.items():
        p = config.PROCESSED_DATA_DIR / f"{name}_model_data.parquet"
        d.to_parquet(p, index=False)
        outputs.append(p)
        if not args.skip_csv:
            d.to_csv(p.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    monthly_market_summary(df).to_csv(config.REPORT_DIR / "monthly_market_summary.csv",
                                      index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------- summary json
    ct = df["contract_type_clean"].value_counts()
    mr = missing_rate(df[[c for c in config.COLUMN_MAP.values() if c in df.columns]])
    report.set("n_rows_clean", int(len(df)))
    report.set("n_jeonse", int((df["rent_type"] == "전세").sum()))
    report.set("n_monthly", int((df["rent_type"] == "월세").sum()))
    report.set("n_new_contract", int(ct.get("new", 0)))
    report.set("n_renewal_contract", int(ct.get("renewal", 0)))
    report.set("n_unknown_contract_type", int(ct.get("unknown", 0)))
    report.set("date_min", str(df["contract_date"].min().date()))
    report.set("date_max", str(df["contract_date"].max().date()))
    report.set("gu_counts", df["gu"].value_counts().to_dict())
    report.set("gu_housing_counts", {f"{g}|{c}": int(n) for (g, c), n in
                                     df.groupby(["gu", "housing_category"]).size().items()})
    report.set("rent_type_counts", df["rent_type"].value_counts().to_dict())
    report.set("housing_category_counts", df["housing_category"].value_counts().to_dict())
    report.set("missing_rate", mr.to_dict())
    report.set("config", {k: (_rel(v) if isinstance(v, Path) else v) for k, v in vars(config).items()
                          if k.isupper() and k not in {"COLUMN_MAP"}})
    report.set("elapsed_sec", round(time.time() - t0, 1))
    with open(config.REPORT_DIR / "preprocessing_summary.json", "w", encoding="utf-8") as f:
        json.dump(report.summary, f, ensure_ascii=False, indent=2, default=str)
    pd.DataFrame(report.quality).to_csv(config.REPORT_DIR / "data_quality_report.csv",
                                        index=False, encoding="utf-8-sig")

    print_summary(report, tables, df, models, outputs, trace, mr)
    return df, models


def print_summary(report, tables, df, models, outputs, trace, mr):
    s = report.summary
    L = []
    w = L.append
    w("\n================ PREPROCESSING SUMMARY ================\n")
    w(f"Raw files                 : {s['n_raw_files']} ({s['n_raw_tables']} tables/sheets, failed={s['n_failed_tables']})")
    w(f"Raw rows                  : {s['n_rows_raw']:,}")
    w(f"Exact duplicates removed  : {s['n_exact_duplicates_removed']:,}")
    w(f"Invalid rows removed      : {s['n_invalid_dates'] + s['n_invalid_area'] + s['n_invalid_price']:,}")
    w(f"Final clean rows          : {s['n_rows_clean']:,}")
    w(f"\nDate range:\n{s['date_min']} ~ {s['date_max']}")
    w("\nDistricts (연립다세대 / 오피스텔):")
    for g in config.VALID_GU:
        a = s["gu_housing_counts"].get(f"{g}|연립다세대", 0)
        b = s["gu_housing_counts"].get(f"{g}|오피스텔", 0)
        w(f"{g:<8}{s['gu_counts'].get(g, 0):>10,}   ({a:,} / {b:,})")
    w("\nRent type:")
    for k, v in s["rent_type_counts"].items():
        w(f"{k:<10}{v:>10,}")
    w("\nContract type:")
    w(f"{'신규':<10}{s['n_new_contract']:>10,}\n{'갱신':<10}{s['n_renewal_contract']:>10,}\n"
      f"{'Unknown':<10}{s['n_unknown_contract_type']:>10,}")
    if s.get("coverage_gaps"):
        w("\nData coverage gaps (gu|category → missing months):")
        for k, v in s["coverage_gaps"].items():
            w(f"  {k}: {v if isinstance(v, str) else ', '.join(v) if len(v) <= 6 else f'{len(v)} months {v[0]}~{v[-1]}'}")
    biased = [k for k, v in s.get("source_distribution", {}).items()
              if v["suspect_low_jeonse"] or v["suspect_truncated_deposit"]]
    if biased:
        w("\nBiased source suspected (low 전세 share / truncated deposit):")
        for k in biased:
            v = s["source_distribution"][k]
            w(f"  {k}: n={v['n']:.0f}, 전세 share={v['jeonse_share']:.3f}, deposit max={v['deposit_max']:.0f}")
        w(f"  -> model data excludes gu: {s['model_filters']['exclude_gu']}")
    w("\nMissing rate top 10 (raw columns, %):")
    for k, v in mr.head(10).items():
        w(f"  {k:<28}{v:>7.2f}")
    w("\nModel datasets:")
    for name, d in models.items():
        m = s[f"model_{name}"]
        w(f"  {name:<8} rows={m['rows']:>7,}  features={m['n_features']}  "
          f"{m['date_min']} ~ {m['date_max']}  split={m['split_counts']}")
    w("\nRolling trace example (dong 90d):")
    for k, v in trace.items():
        w(f"  {k:<42}{v}")
    w(f"\nBrute-force leakage mismatches: {s['bruteforce_leakage_mismatches']}")
    w("\nOutput:")
    for p in outputs:
        w(_rel(p))
    w("\n=======================================================")
    text = "\n".join(L)
    print(text)
    with open(config.REPORT_DIR / "preprocessing_log.txt", "a", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
