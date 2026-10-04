"""웹앱 배포용 경량 거래 데이터 생성 (예측 엔진이 쓰는 컬럼만).

    python -m src.export_app_data
"""
import pandas as pd

from . import config
from .predictor import APP_COLUMNS, APP_DATA


def main():
    df = pd.read_parquet(config.PROCESSED_DATA_DIR / "all_transactions.parquet", columns=APP_COLUMNS)
    APP_DATA.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(APP_DATA, index=False, compression="zstd")
    print(f"{len(df):,} rows x {len(APP_COLUMNS)} cols -> {APP_DATA} ({APP_DATA.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
