"""전처리 파이프라인 설정값."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"

# ---------------------------------------------------------------- scope
VALID_GU = ["동작구", "노원구", "서대문구", "동대문구", "성북구"]
DROP_INVALID_GU = False          # True면 VALID_GU 외 지역 행을 제거 (기본: WARNING만)
START_DATE = "2021-01-01"
END_DATE = "2026-09-15"          # 2026년은 부분연도
PARTIAL_YEAR = 2026

# 원본 데이터에 존재하는 주택 대분류. 동작구는 오피스텔 원본이 없음(README 참고).
HOUSING_CATEGORIES = ["연립다세대", "오피스텔"]
# 모델 데이터셋에 포함할 대분류. ["연립다세대"]로 바꾸면 빌라 전용 모델 데이터가 된다.
MODEL_HOUSING_CATEGORIES = ["연립다세대", "오피스텔"]

# ---------------------------------------------------------------- loading
ENCODINGS = ["cp949", "euc-kr", "utf-8-sig", "utf-8"]
RAW_FILE_PATTERNS = ["*.csv", "*.xlsx"]

COLUMN_MAP = {
    "NO": "row_no",
    "시군구": "sigungu",
    "번지": "jibun",
    "본번": "bonbun",
    "부번": "bubun",
    "건물명": "building_name",
    "전월세구분": "rent_type",
    "전용면적(㎡)": "area_m2",
    "계약년월": "contract_ym",
    "계약일": "contract_day",
    "보증금(만원)": "deposit_10k",
    "월세금(만원)": "monthly_rent_10k",
    "층": "floor",
    "건축년도": "built_year",
    "도로명": "road_name",
    "계약기간": "contract_period",
    "계약구분": "contract_type",
    "갱신요구권 사용": "renewal_request_used",
    "종전계약 보증금(만원)": "previous_deposit_10k",
    "종전계약 월세(만원)": "previous_monthly_rent_10k",
    "주택유형": "housing_type",
}
RAW_COLUMNS = list(COLUMN_MAP)
# 오피스텔 원본은 '건물명' 대신 '단지명'을 쓰고 '주택유형' 컬럼이 없다.
RAW_COLUMN_ALIASES = {"단지명": "건물명"}
OFFICETEL_RAW_COLUMNS = [c for c in RAW_COLUMNS if c != "주택유형"]

STRING_COLUMNS = ["jibun", "bonbun", "bubun", "building_name", "road_name", "sigungu"]
NUMERIC_COLUMNS = [
    "area_m2", "deposit_10k", "monthly_rent_10k", "floor", "built_year",
    "previous_deposit_10k", "previous_monthly_rent_10k",
]
MISSING_TOKENS = {"", "-", "nan", "NaN", "None", "none", "null"}

# ---------------------------------------------------------------- features
PYEONG_M2 = 3.305785
AREA_BINS = [0, 15, 20, 25, 30, 40, 60, 85, float("inf")]
AREA_BIN_LABELS = ["0-15", "15-20", "20-25", "25-30", "30-40", "40-60", "60-85", "85+"]

ROLLING_WINDOWS = [30, 90, 180]
PROPERTY_HISTORY_DAYS = 365
TREND_WINDOW_DAYS = 90
MIN_ROLLING_COUNT = 3

# rent_type별 가격 기준 컬럼 (rolling 가격 통계의 대상)
PRICE_COLUMN_BY_RENT_TYPE = {"전세": "deposit_10k", "월세": "monthly_rent_10k"}

# 그룹 정의 (§22~24). 서로 다른 시장인 연립다세대/오피스텔 통계가 섞이지 않도록
# dong 그룹에도 housing_category를 포함한다.
# comp 그룹은 원본 housing_type(다세대/연립/연립다세대) 대신 housing_category를 쓴다:
# 서대문구 원본은 전부 '연립다세대'로, 다른 구는 '다세대'/'연립'으로 표기되어
# 원본 라벨이 출처(파일)에 따라 달라지기 때문이다.
DONG_GROUP = ["gu", "dong", "rent_type", "housing_category"]
GU_GROUP = ["gu", "rent_type", "housing_category"]
COMP_GROUP = ["gu", "dong", "rent_type", "housing_category", "area_bin"]
# 데이터 공백 판단 단위 (이 조합에서 한 달 거래가 0건이면 수집 누락으로 간주)
COVERAGE_GROUP = ["gu", "housing_category"]
PROPERTY_GROUP = ["property_id", "rent_type"]

# 이상치 flag (삭제하지 않음)
OUTLIER_GROUP = ["gu", "rent_type", "housing_category", "area_bin"]
OUTLIER_IQR_K = 3.0              # log 가격 기준 IQR 배수
OUTLIER_MIN_GROUP_SIZE = 30

# ---------------------------------------------------------------- model data
NEW_CONTRACT_ONLY = True
# '신규'만 쓸 때 contract_type 결측(2021년 초 등 제도 도입 전)을 포함할지 여부
INCLUDE_UNKNOWN_CONTRACT_TYPE = False
MODEL_DATA_START_DATE = None     # 예: "2022-01-01" (2021년을 rolling warm-up으로 사용)
# 원본이 편향된 구는 all_transactions에는 남기되 모델 데이터에서 제외한다.
# (2026-10-04) 동작구: 금액 필터가 걸린 다운로드였던 원본을 '금액선택: 전체'로 재다운로드해 교체 → 제외 해제.
MODEL_EXCLUDE_GU = []

# 원본 다운로드 편향 탐지 (gu × housing_category 단위)
MIN_JEONSE_SHARE = 0.05          # 전세 비율이 이보다 낮으면 경고
MAX_DEPOSIT_RATIO = 0.25         # 보증금 최대값 < (같은 category 전체 p99 × 비율) 이면 절단(truncation) 의심

SPLIT_BOUNDARIES = {
    "train": ("2021-01-01", "2024-12-31"),
    "validation": ("2025-01-01", "2025-12-31"),
    "test": ("2026-01-01", "2026-09-15"),
}
