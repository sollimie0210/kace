"""예측 모델(미래 시점 전월세가) 설정값."""
from . import config

MODEL_DIR = config.PROJECT_ROOT / "models"
MODEL_REPORT_DIR = config.REPORT_DIR / "model"
HORIZON_DATA_DIR = config.PROCESSED_DATA_DIR

# 예측 기간: 오늘(as-of) 기준 h일 뒤 계약. 최대 6개월.
HORIZONS = [0, 30, 60, 90, 120, 150, 180]
# 실거래 신고 지연(계약 후 30일 이내 신고). 서비스 시점에 최근 30일 거래는 아직 다 안 들어와 있으므로
# 학습 feature도 '기준일 - 30일' 이전 거래만 사용해 실제 서비스 환경과 맞춘다.
REPORTING_LAG_DAYS = 30

QUANTILES = (0.1, 0.5, 0.9)      # 80% 적정 범위 + 중앙값
TARGET_COVERAGE = 0.8

# 계약일(target 날짜) 기준 시간 분할
SPLITS = {
    "train": ("2021-01-01", "2024-12-31"),
    "valid": ("2025-01-01", "2025-06-30"),   # early stopping
    "calib": ("2025-07-01", "2025-12-31"),   # conformal 보정
    "test": ("2026-01-01", "2026-09-15"),    # 최종 평가 (부분연도)
}

RENT_TYPES = {
    "jeonse": {"rent_type": "전세", "target_col": "deposit_10k"},
    "monthly": {"rent_type": "월세", "target_col": "monthly_rent_10k"},
}

LGBM_PARAMS = dict(
    learning_rate=0.05, num_leaves=63, min_child_samples=50,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    reg_lambda=1.0, n_estimators=6000, verbose=-1, random_state=42,
)
EARLY_STOPPING_ROUNDS = 100

CATEGORICAL_FEATURES = ["gu", "dong", "housing_category", "housing_type", "area_bin"]

# target 표현 방식
#  - "absolute": log(가격)을 직접 예측
#  - "anchor"  : log(가격 / 시장수준)을 예측. 시장수준(anchor) = as-of 시점 구×전월세×주택대분류 90일 median.
#                트리 모델은 학습 범위 밖의 가격 수준으로 외삽하지 못하므로, 시장 전체가 오를 때
#                체계적으로 과소예측한다. '시장 대비 얼마나 비싼 매물인가'를 학습하면 수준 변화에 강하다.
TARGET_MODE = "anchor"
ANCHOR_FEATURE = "gu_price_median_90d"
# anchor 모드에서 anchor 대비 비율(rel_*)로 바꿔 넣는 가격 수준 feature
RELATIVE_PRICE_FEATURES = [
    "dong_price_median_30d", "dong_price_median_90d", "dong_price_median_180d",
    "dong_price_mean_90d", "dong_price_std_90d",
    "comp_price_median_30d", "comp_price_median_90d", "comp_price_median_180d",
    "property_last_price", "property_price_median_365d",
]
