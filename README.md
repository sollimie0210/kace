# 우리 학교 근처 전월세, 얼마가 적정할까?

서울 5개 구(동작·노원·서대문·동대문·성북) 대학생을 위한 전월세 적정가 예측 서비스.
학교와 조건(전세/월세, 계약 시점 0~6개월, 방 크기, 보증금)을 고르면 **예상 가격 · 80% 적정 범위 · 지금 계약 vs 나중 ·
보고 있는 매물의 저렴/적정/비쌈 판정**을 보여준다. 국토교통부 실거래가(2021-01 ~ 2026-09-15) 기반.

## 웹앱 실행
```bash
pip install -r requirements.txt
streamlit run app.py
```
배포: Streamlit Community Cloud에서 이 저장소의 `app.py`를 선택 (Python 3.12). 앱에 필요한 데이터
(`data/app/`, `data/external/`)와 모델(`models/final/`)은 저장소에 포함되어 있다.

| 경로 | 내용 |
|---|---|
| `app.py` | Streamlit 웹앱 |
| `src/predictor.py` | 학교 기반 예측 엔진 (CLI: `python -m src.predictor --school 고대 --type 월세 --months 3`) |
| `src/preprocess.py` 외 | 전처리 파이프라인 (아래) |
| `src/train_model.py`, `src/horizon_dataset.py` | 미래 시점 분위수 예측 모델 학습·평가 |
| `src/geocode.py`, `src/location_features.py` | 카카오 로컬 API 지오코딩, 역·학교 거리 |
| `tests/` | leakage·학습/서빙 일치 테스트 |

## 데이터 다시 만들기 (로컬)
원본 CSV/XLSX(국토교통부 실거래가 공개시스템 다운로드)는 용량 때문에 저장소에 없다. `data/raw/`에 넣은 뒤:
```bash
python -m src.preprocess          # 전처리
python -m src.geocode             # 좌표 (.env 에 KAKAO_REST_API_KEY 필요)
python -m src.preprocess          # 좌표 feature 포함 재실행
python -m src.train_model --rebuild && python -m src.train_model --final
python -m src.export_app_data     # 웹앱용 경량 데이터
```

---

# 전처리 파이프라인

전세(target: 보증금)·월세(target: 월세금) 예측 모델 학습 전 단계의 재현 가능한 전처리 + feature engineering.

## 1. 데이터 출처
국토교통부 실거래가 공개시스템(rt.molit.go.kr) 전월세 실거래 다운로드 파일.
- 연립다세대(전월세), 오피스텔(전월세)
- CSV(cp949)와 XLSX가 섞여 있으며, 노원구·성북구는 여러 연도를 수작업으로 합친 XLSX(시트별 연립다세대/오피스텔)

## 2. 대상 지역
서울특별시 동작구, 노원구, 서대문구, 동대문구, 성북구

## 3. 대상 기간
2021-01-01 ~ 2026-09-15 (2026년은 **부분연도**. 연간 합계/평균을 2021~2025년과 직접 비교하지 말 것.
`is_partial_year` 컬럼 제공)

## 4. 원본 파일의 특수 구조
| 형태 | 구조 | 처리 |
|---|---|---|
| MOLIT CSV | 안내문 + 검색조건(`키 : 값`) + 헤더 + 데이터 | 인코딩 cp949→euc-kr→utf-8-sig→utf-8 순서로 시도, `"NO","시군구"` 헤더 행을 탐색 |
| MOLIT XLSX | 같은 구조, 시군구 검색조건 없음, 시트 dimension 메타데이터가 깨져 있음 | `reset_dimensions()` 후 헤더 탐색 |
| 수작업 XLSX | 시트 여러 개, 헤더가 없는 시트도 있음(노원구 오피스텔) | 헤더가 없으면 컬럼 수(20=오피스텔, 21=연립다세대)로 schema 추론 + WARNING |
- 오피스텔 원본은 `건물명` 대신 `단지명`을 쓰고 `주택유형` 컬럼이 없음 → `건물명`으로 통일, `주택유형='오피스텔'`
- 파일명이 아니라 파일 내부 데이터(시군구 컬럼)로 구를 판단. 검색조건의 시군구가 있으면(CSV) 교차 검증.
- 파일별 인코딩/헤더 위치/검색조건/행 수는 `reports/input_files.csv`와 로그에 기록

## 5. 실행 방법
```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements-dev.txt    # Windows (웹앱만 실행할 땐 requirements.txt)
.venv\Scripts\python -m src.preprocess
.venv\Scripts\python -m pytest -q                     # 테스트

# 옵션
python -m src.preprocess --raw-dir data/raw --new-only true --model-start 2022-01-01 --exclude-gu ""
```
원본 파일은 `data/raw/` 아래 어디에 두어도 재귀 탐색된다(`*.csv`, `*.xlsx`). 설정값은 `src/config.py`.

## 6. 출력 파일
| 파일 | 내용 |
|---|---|
| `data/interim/merged_raw.csv` | 원본 한글 컬럼 그대로 병합 + source 메타 컬럼 |
| `data/interim/cleaned_all.csv` | 정제 후(snake_case), feature 생성 전 |
| `data/processed/all_transactions.parquet` | 전체 거래: clean 컬럼 + 주소/날짜/기본 파생 + rolling feature + anomaly/outlier flag |
| `data/processed/jeonse_model_data.{parquet,csv}` | 전세 모델용 (target `target_deposit_10k`) |
| `data/processed/monthly_model_data.{parquet,csv}` | 월세 모델용 (target `target_monthly_rent_10k`) |
| `reports/preprocessing_summary.json` | 건수, 결측률, 공백/편향 탐지, 모델 데이터 요약, rolling 검증 예시 |
| `reports/data_quality_report.csv` | 검사 항목별 건수와 처리 방식 |
| `reports/monthly_market_summary.csv` | 구/동/연월/전월세/주택대분류별 거래수·가격 요약 (EDA용) |
| `reports/input_files.csv` | 입력 파일(시트)별 파싱 결과 |
| `reports/preprocessing_log.txt` | 전체 로그 + 콘솔 summary |

모든 데이터셋에 `tx_id`(안정적인 거래 id)가 있어 all_transactions의 flag/주소 등을 다시 join할 수 있다.

## 7. 주요 파생변수
- 날짜: `contract_date, contract_year, contract_month, contract_quarter, contract_day_of_month, month_sin, month_cos, contract_year_month("YYYY-MM")`
- 위치/식별자: `sido, gu, dong, location_id, parcel_id, property_id(gu|dong|jibun|building_name), full_jibun_address, full_road_address`
- 건물: `area_pyeong, area_bin, building_age, is_basement, is_ground_floor, building_name_is_address`
- 계약: `contract_type_clean(new/renewal/unknown), renewal_right_used, lease_start_ym, lease_end_ym, lease_duration_months`
- Historical (모두 과거 거래만 사용):
  - 동 단위 `(gu, dong, rent_type, housing_category)`: `dong_tx_count_{30,90,180}d, dong_price_median_{30,90,180}d, dong_price_mean_90d, dong_price_std_90d, dong_price_trend_3m`
  - 구 단위: `gu_tx_count_90d`
  - 비교주택 `(gu, dong, rent_type, housing_category, area_bin)`: `comp_price_median_{30,90,180}d, comp_tx_count_90d, comp_price_trend_3m`
  - 동일 건물 `(property_id, rent_type)`: `property_last_price, property_days_since_last_transaction, property_tx_count_365d, property_price_median_365d`
  - 월세용 과거 보증금 수준: `dong_deposit_median_90d, comp_deposit_median_90d`
  - 결측 indicator: `has_property_history, has_comp_90d_history, has_dong_90d_history`
  - 데이터 공백 indicator: `history_window_has_data_gap`
- '가격'은 전세 행은 보증금, 월세 행은 월세금 (`price_10k`, 모델 feature 아님)
- 갱신 분석 전용(모델 데이터 제외): `deposit_change_from_previous` 등 4개
- EDA 전용: `monthly_rent_per_m2_eda_only`. `deposit_per_m2`는 월세 모델에서만 feature

## 8. Data leakage 방지
- 거래일 t의 historical feature는 같은 그룹의 **t 이전 날짜(date < t)** 거래만 사용. **같은 날짜 거래도 제외.**
- 구현(`feature_engineering.past_window_stats`): 행마다 가격이 NaN인 '질의 행'을 만들어 같은 시각의 실제 거래보다 앞에 두고 time-based `rolling(closed="left")`를 적용 → window가 정확히 `[t-W, t)`.
  추세의 '이전 90일'은 질의 시각을 t-90일로 옮겨 `[t-180, t-90)`을 계산.
- 동일 건물 직전 거래: `merge_asof(allow_exact_matches=False)`로 엄격히 이전 날짜만.
- 전체 기간 groupby 평균/중앙값을 행에 붙이는 방식은 사용하지 않음.
- 검증: (1) `tests/`의 synthetic 테스트(§39 예시 포함), (2) 실행 시 무작위 실제 거래 400건을 brute-force로 재계산해 불일치 0건을 assert, (3) summary에 실제 거래 1건의 rolling 추적 결과 출력.
- 이상치 flag(`*_outlier_candidate`)는 전체 기간 분포로 계산하므로 **모델 feature로 쓰지 않는다** (모델 데이터에 미포함, tx_id로 join해 필터 용도로만 사용).

## 9. Target 정의
- 전세: `rent_type=='전세' & deposit_10k>0` → `target_deposit_10k`. 금지 feature: `deposit_10k, deposit_per_m2`, 종전계약 정보
- 월세: `rent_type=='월세' & monthly_rent_10k>0 & deposit_10k>=0` → `target_monthly_rent_10k`. `deposit_10k`는 **필수 feature**. 금지: `monthly_rent_per_m2`, 종전계약 정보
- 금지 목록은 코드(`FORBIDDEN`)로 관리되며 데이터셋 생성 시 assert.

## 10. 신규 계약 필터 정책
- 기본 `NEW_CONTRACT_ONLY=True`: `계약구분=='신규'`만 모델 데이터에 포함 (새 집을 구하는 사용자 관점).
- `계약구분` 결측은 삭제하지 않고 `unknown`으로 보존. 결측률이 2021년 48%, 2022년 19%로 높다(계약구분 공개 이전 거래).
  `INCLUDE_UNKNOWN_CONTRACT_TYPE=True`로 포함 가능.
- Historical feature는 신규/갱신 구분 없이 **모든 과거 거래**로 계산한다(모델 데이터 필터와 별개).
- 시간 기반 split 컬럼: train(2021~2024) / validation(2025) / test(2026-01-01~09-15). Random split 금지.
- `MODEL_DATA_START_DATE="2022-01-01"`로 2021년을 rolling warm-up 구간으로만 쓸 수 있다.

## 11. 알려진 한계 / 데이터 이슈
- 동작구 원본은 처음에 금액 조건이 걸린 채 다운로드되어(보증금 ≤ 3,000만원, 2025년·오피스텔 누락) 2026-10-04에
  '금액선택: 전체'로 재다운로드해 교체함. 같은 문제를 자동으로 잡기 위해 실행 시 gu×주택대분류별 전세 비율·보증금 상한을
  검사(`source_distribution_check`)하고, 월 단위 수집 공백을 탐지(`history_window_has_data_gap`)한다.
  편향이 의심되는 구는 `MODEL_EXCLUDE_GU`로 모델 데이터에서 뺄 수 있다.
- 원본 `주택유형` 라벨이 출처마다 다름(서대문구는 전부 '연립다세대', 다른 구는 '다세대'/'연립'). 그룹핑에는 `housing_category`(연립다세대/오피스텔)를 사용.
- 공개자료에 호수가 없어 같은 건물·면적·층·가격·날짜의 거래는 구분 불가. 같은 파일 안의 완전 동일 행은 서로 다른 호실일 수 있어 삭제하지 않고 `is_duplicate_candidate`로 표시. 다른 파일 사이의 완전 중복만 제거.
- 월세 rolling 가격 통계는 보증금 조건을 반영하지 않은 단순 월세 median. 보증금 수준 feature(`*_deposit_median_90d`)를 함께 제공하며, 환산월세 feature는 `config.PRICE_COLUMN_BY_RENT_TYPE`/`past_window_stats`로 쉽게 추가 가능.
- 2021년 초반 거래는 과거 window가 비어 rolling feature가 NaN (오류 아님).
- 외부 데이터(금리, 부동산원 지수, 역/학교 거리 등)는 아직 미결합. merge key: `contract_year_month, sido, gu, dong, full_jibun_address, full_road_address`.

---

# 미래 시점 전월세가 예측 모델 (v1, 동 단위)

```bash
.venv\Scripts\python -m src.train_model            # horizon 데이터 없으면 생성 후 학습/평가
.venv\Scripts\python -m src.train_model --rebuild  # all_transactions가 바뀌면 horizon 데이터 재생성
```

## 문제 정의
"오늘(as-of) 알 수 있는 정보로 h일 뒤(h = 0~180) 신규 계약 가격의 중앙값과 80% 적정 범위"를 예측.
- 학습 데이터(`src/horizon_dataset.py`): 신규 거래 하나를 h ∈ {0,30,60,90,120,150,180}마다 복제하고,
  모든 historical feature를 **계약일 − h − 30일(신고 지연)** 이전 거래로 계산 → 서비스 환경과 동일한 정보만 사용.
- 모델(`src/train_model.py`): LightGBM quantile regression q10/q50/q90.
  target = log(가격 / 시장수준), 시장수준 = as-of 시점 구×전월세×주택대분류 90일 median (`TARGET_MODE="anchor"`).
  트리 모델은 학습 범위 밖 가격 수준으로 외삽하지 못해 시장 상승기에 과소예측하므로, '시장 대비 상대가격'을 학습.
- 80% 범위 보정: 2025 H2(calib)에서 horizon별 CQR(conformalized quantile regression).
- 분할(계약일 기준): train 2021–2024 / valid 2025 H1(early stopping) / calib 2025 H2 / test 2026-01~09.
- 모델 선택(absolute vs anchor)은 test가 아니라 calib 지표로 결정.

## 성능 (test 2026-01 ~ 09, 모든 horizon)
| | 방법 | MdAPE | MAPE | 80% 범위 적중률 | 범위 폭(중앙값 대비) |
|---|---|---|---|---|---|
| 전세 | baseline (비교주택 90일 median) | 19.4% | 36.7% | 0.80 | 0.92 |
| 전세 | LightGBM (동 단위) | 8.7% | 17.2% | 0.79 | 0.37 |
| 전세 | **LightGBM + 위치** | **8.5%** | 16.7% | 0.78 | 0.36 |
| 월세 | baseline | 26.7% | 90.6% | 0.81 | 1.70 |
| 월세 | LightGBM (동 단위) | 11.2% | 31.4% | 0.81 | 0.54 |
| 월세 | **LightGBM + 위치** | **10.9%** | 30.4% | 0.81 | 0.52 |

horizon별·구별 지표: `reports/model/*_metrics_by_*.csv`, 모델: `models/`, 요약: `reports/model/model_summary.txt`

## 알려진 한계
- 2026년 후반으로 갈수록 실제 가격이 예측 중앙값보다 1~3% 높고 80% 범위 적중률이 0.73~0.78로 떨어지는 달이 있음
  (시장 상승 속도를 다 따라가지 못함). 외부 지표(금리, 부동산원 지수) 추가로 개선 예정.
- horizon이 커져도 오차가 거의 늘지 않음 → 가격 차이의 대부분이 '어떤 매물인가'에서 오고, 6개월 내 시장 변화는
  상대적으로 작다는 뜻. '지금 vs 나중' 판단은 대부분 "차이 작음"이 될 가능성이 높음.
- 서비스 배포 전에는 전체 기간(~2026-09)으로 재학습 필요 (현재 모델은 2024년까지 학습).

## 위치 데이터 (카카오 로컬 API)
```bash
.venv\Scripts\python -m src.geocode    # .env 의 KAKAO_REST_API_KEY 필요, 캐시 기반(새 주소만 호출)
.venv\Scripts\python -m src.preprocess # 좌표·거리 feature가 all_transactions에 붙음
```
- `data/external/address_coords.csv`: 지번주소 13,242개 → 좌표 (지번 13,241 / 도로명 1)
- `data/external/universities.csv`: 대상 5개 구 대학 22개 (서비스의 학교 선택 목록)
- `data/external/subway_stations.csv`: 지하철역 365개 (환승역은 노선별로 1건)
- feature: `lat, lon, station_dist_m, n_stations_1km(1km 내 역·노선 수), univ_dist_m`
- 한계: 신규 개통역은 개통 전 거래에도 '역이 있는 것'으로 계산됨. `.env`는 `.gitignore`에 포함.

## 학교 기반 예측 엔진 (`src/predictor.py`)
```bash
.venv\Scripts\python -m src.train_model --final          # 서비스용 모델: 전체 기간(2021-01~2026-09) 재학습 → models/final/
.venv\Scripts\python -m src.predictor --school 고대 --type 월세 --months 3 --room 원룸 --deposit 1000 --price 60
.venv\Scripts\python -m src.predictor --school 중앙대 --type 전세 --months 6 --room 투룸 --housing 연립다세대 --json
```
입력: 학교(정식명/줄임말: 고대·외대·과기대·이대…), 전세/월세, 몇 개월 뒤 계약(0~6), 방 크기(원룸 <25㎡ / 투룸 25~45㎡ / 쓰리룸+ / 전체),
월세 보증금(미입력 시 근처 최근 1년 중앙값), 매물 가격(선택), 주택 유형(전체/연립다세대/오피스텔).

출력: 예상 가격(중앙값), 적정 범위(80%), 지금 계약 시 예상가와 차이 + 메시지, 매물 판정(저렴/적정/다소 비쌈/비쌈 = 분포 내 백분위
<25 / 25~75 / 75~90 / >90), 근처 최근 실거래 5건, 주의사항.

동작 방식
1. 학교 반경 1km(후보 건물 30개 미만이면 1.5km → 2km) 안에서 최근 1년 해당 전월세 거래가 있고 요청 면적대 호실이 있는 건물 선택
2. 건물마다 대표 호실(해당 면적대 중앙값 면적·층)로 '계약일 = 예정일, 가격 미정'인 가상 거래를 만들어 실제 이력에 붙이고,
   **학습과 같은 함수**로 기준일−30일 시점 feature 계산 → q10/q50/q90 + conformal 보정
   (가상 거래 feature == 학습 feature 임을 테스트로 검증: `test_serving_synthetic_rows_match_training_features`)
3. 건물별 예측분포(log 공간 split-normal)를 최근 2년 거래량 가중으로 섞어 학교 주변 가격 분포 생성 → 분위수/백분위 계산

한계
- 기준일은 데이터 스냅샷(2026-09-15). 실서비스에서는 국토부 데이터를 주기적으로 받아 전처리→재학습해야 함.
- 1회 예측 약 9초(전체 이력에 대해 feature 재계산). 웹에서는 학교×조건 조합 사전 계산 또는 기준일 feature 캐싱 필요.
- '지금 vs 나중' 차이는 대부분 ±1% 이내 → 모델이 6개월 내 시장 변화를 작게 본다는 뜻(평가 결과와 일치).
- 노원구 등 거래가 적은 학교·조건은 후보 건물이 적어 범위가 불안정할 수 있음(결과에 경고 표시).
