# 전월세 실거래가 데이터 전처리 구현 프롬프트

## 0. 역할과 목표

너는 부동산 실거래 데이터 분석 및 머신러닝 파이프라인을 구축하는 시니어 데이터 엔지니어다.

현재 보유한 데이터는 **국토교통부 실거래가 공개시스템에서 내려받은 연립·다세대 전월세 CSV 파일들**이다.  
이 데이터를 이용해 이후 다음 두 종류의 예측 모델을 학습할 예정이다.

1. **전세 모델**
   - target: `보증금(만원)`
2. **월세 모델**
   - target: `월세금(만원)`
   - 월세 모델에서는 `보증금(만원)`을 중요한 입력 feature로 사용한다.

이번 작업의 목적은 **모델 학습 이전 단계의 재현 가능한 데이터 전처리 + feature engineering 파이프라인을 완성하는 것**이다.

단순히 CSV를 합치는 수준이 아니라 다음 조건을 모두 만족해야 한다.

- 원본 CSV의 특수한 헤더 구조를 안전하게 파싱
- 여러 구/여러 연도의 데이터를 자동 탐색하여 병합
- 데이터 타입 정리
- 결측/이상치 처리
- 주소/건물 식별자 생성
- 날짜 feature 생성
- 신규/갱신 계약 구분
- 과거 거래만을 이용한 rolling/lag feature 생성
- 절대로 미래 정보가 섞이지 않도록 **data leakage 방지**
- 전세/월세용 데이터셋을 각각 생성
- 처리 과정과 제거 행 수를 로그로 남김
- 최종 데이터 품질 검증 결과를 출력
- 이후 외부 데이터(금리, 가격지수, 역세권 등)를 merge하기 쉬운 구조로 설계

---

# 1. 데이터 범위

현재 데이터 범위는 다음과 같다.

## 지역
서울특별시의 다음 5개 구:

- 동작구
- 노원구
- 서대문구
- 동대문구
- 성북구

## 기간

- 시작: **2021-01-01**
- 종료: **2026-09-15**

2026년은 9월 15일까지의 **부분연도(partial year)** 데이터이므로, 연간 통계 계산 시 2026년 전체 연도와 동일하게 취급하지 말 것.

## 주택유형

현재 예시 파일은:

- `연립다세대(전월세)`

이며 원본의 `주택유형` 컬럼에도 `연립다세대`가 기록되어 있다.

---

# 2. 예시 원본 CSV 구조

예시 파일:

`연립다세대(전월세)_실거래가_20260915195728.csv`

국토교통부 실거래가 CSV는 일반 CSV와 달리 파일 앞부분에 검색조건/안내문이 포함되어 있다.

예시 구조:

```text
1  "□ 본 서비스에서 제공하는 정보는 ..."
2  "□ 신고정보가 ..."
...
7  "□ 검색조건"
8  "계약일자 : 2021-01-01 ~ 2021-12-31"
9  "실거래구분 : 연립다세대(전월세)"
10 "주소구분 : 지번주소"
11 "시도 : 서울특별시"
12 "시군구 : 서대문구"
...
15 "금액선택 : 전체"
16 "NO","시군구","번지","본번","부번","건물명","전월세구분","전용면적(㎡)","계약년월","계약일","보증금(만원)","월세금(만원)","층","건축년도","도로명","계약기간","계약구분","갱신요구권 사용","종전계약 보증금(만원)","종전계약 월세(만원)","주택유형"
17 데이터 시작
```

주의:

- 파일 인코딩은 **CP949**로 읽는 것을 우선 시도한다.
- 고정적으로 무조건 `skiprows=15`만 가정하지 말고, 가능하면 `"NO","시군구"`로 시작하는 실제 헤더 행을 탐색해서 읽도록 구현한다.
- 단, 헤더 탐색 실패 시 명확한 에러 메시지를 출력한다.
- 파일마다 검색조건 행 수가 달라져도 견딜 수 있게 작성한다.

---

# 3. 원본 컬럼

예시 파일에서 확인된 컬럼은 다음과 같다.

```text
NO
시군구
번지
본번
부번
건물명
전월세구분
전용면적(㎡)
계약년월
계약일
보증금(만원)
월세금(만원)
층
건축년도
도로명
계약기간
계약구분
갱신요구권 사용
종전계약 보증금(만원)
종전계약 월세(만원)
주택유형
```

코드에서는 가능하면 원본 한글 컬럼을 보존한 `raw/clean intermediate` 버전과, 머신러닝용 영문 snake_case 컬럼 버전을 구분하라.

권장 rename:

```python
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
```

---

# 4. 프로젝트 디렉터리 구조

다음과 같이 구성하라.

```text
project/
├─ data/
│  ├─ raw/
│  │  ├─ ...원본 csv들...
│  ├─ interim/
│  │  ├─ merged_raw.csv
│  │  ├─ cleaned_all.csv
│  └─ processed/
│     ├─ all_transactions.parquet
│     ├─ jeonse_model_data.parquet
│     ├─ monthly_model_data.parquet
│     ├─ jeonse_model_data.csv
│     └─ monthly_model_data.csv
├─ reports/
│  ├─ preprocessing_summary.json
│  ├─ data_quality_report.csv
│  └─ preprocessing_log.txt
├─ src/
│  ├─ config.py
│  ├─ load_data.py
│  ├─ clean_data.py
│  ├─ feature_engineering.py
│  ├─ validate_data.py
│  └─ preprocess.py
├─ tests/
│  └─ test_preprocessing.py
├─ requirements.txt
└─ README.md
```

CSV뿐 아니라 Parquet도 함께 저장하라.  
내부 계산과 이후 모델링은 Parquet 사용을 우선 권장한다.

---

# 5. 원본 파일 자동 탐색

`data/raw/` 안에 있는 모든 `.csv`를 자동 탐색한다.

파일명을 신뢰하여 지역/연도를 파싱하는 방식에 의존하지 말고, **파일 내부의 실제 데이터와 검색조건을 우선 신뢰**한다.

각 파일 처리 시 다음 정보를 로그로 남긴다.

```text
파일명
검색조건상의 시군구
검색조건상의 계약기간
읽은 행 수
정상 파싱 여부
```

같은 거래 파일이 중복으로 들어간 경우도 대비한다.

---

# 6. CSV 로딩

## 인코딩

순서대로 시도:

1. `cp949`
2. `euc-kr`
3. `utf-8-sig`
4. `utf-8`

단, 첫 번째 성공 인코딩을 로그로 남긴다.

## 헤더 탐색

파일을 텍스트로 먼저 열어 다음과 같은 행을 탐색한다.

```text
"NO","시군구",...
```

또는 첫 두 필드가 `NO`, `시군구`인 행.

찾은 index를 `header_line_index`로 사용해 `pd.read_csv()`를 수행한다.

---

# 7. 문자열 정규화

모든 object/string 컬럼에 대해:

- 앞뒤 공백 제거
- `""` → missing
- `"-"` → missing
- `" "` → missing
- `"nan"` 문자열 → missing
- 가능하면 Unicode 정규화 적용

다만 `번지`, `본번`, `부번`, `도로명`, `건물명` 값은 의미 있는 문자열이므로 숫자로 강제 변환하지 않는다.

특히:

```text
본번 = "0295"
부번 = "0009"
```

처럼 leading zero가 있으므로 string 유지.

---

# 8. 숫자형 변환

다음 컬럼은 comma를 제거한 후 numeric으로 변환한다.

```text
area_m2
deposit_10k
monthly_rent_10k
floor
built_year
previous_deposit_10k
previous_monthly_rent_10k
```

예:

```text
"21,525" -> 21525
"3,000" -> 3000
```

변환 실패 값은 `NaN`.

단, 변환 실패 건수를 반드시 report에 기록한다.

---

# 9. 날짜 생성

`contract_ym` 예:

```text
202112
```

`contract_day` 예:

```text
31
```

이를 결합해:

```python
contract_date = 2021-12-31
```

형식의 `datetime64`를 생성한다.

추가 feature:

```text
contract_year
contract_month
contract_quarter
contract_day_of_month
```

추가로 계절성 표현용:

```python
month_sin = sin(2*pi*contract_month/12)
month_cos = cos(2*pi*contract_month/12)
```

## 날짜 검증

다음은 제거 또는 별도 invalid 처리:

- 파싱 불가능한 날짜
- 2021-01-01 이전
- 2026-09-15 이후

제거 건수를 기록한다.

---

# 10. 행정구역 파싱

`sigungu` 예:

```text
서울특별시 서대문구 북가좌동
서울특별시 서대문구 연희동
서울특별시 동대문구 제기동
```

다음 컬럼을 파생한다.

```text
sido
gu
dong
```

예:

```text
sido = 서울특별시
gu = 서대문구
dong = 북가좌동
```

분리 실패 케이스를 로그에 남긴다.

최종적으로 `gu`가 다음 5개 중 하나인지 검증:

```text
동작구
노원구
서대문구
동대문구
성북구
```

그 외 지역이 있으면 제거하지 말고 먼저 WARNING을 출력하고 report에 남긴 뒤, 설정값에 따라 제외 가능하게 한다.

---

# 11. 건물 식별자 생성

같은 건물의 과거 거래를 추적하기 위해 deterministic identifier를 만든다.

권장:

```python
property_id = (
    gu + "|" +
    dong + "|" +
    jibun.fillna("") + "|" +
    building_name.fillna("")
)
```

단, 건물명이 `(461-0)`, `(187-16)`처럼 주소를 단순 반복하는 경우도 있으므로 건물명 하나만을 식별자로 사용하면 안 된다.

추가 식별자:

```text
location_id = gu + "|" + dong
parcel_id   = gu + "|" + dong + "|" + jibun
```

향후 geocoding을 위해:

```text
full_jibun_address
full_road_address
```

도 생성한다.

예:

```text
서울특별시 서대문구 북가좌동 295-9
서울특별시 서대문구 증가로23길 46
```

---

# 12. 면적 feature

원본:

```text
area_m2
```

추가:

```text
area_pyeong = area_m2 / 3.305785
```

면적 bin을 생성한다.

추천:

```text
0~15
15~20
20~25
25~30
30~40
40~60
60~85
85+
```

코드에서는 경계값을 config에 분리한다.

예:

```python
AREA_BINS = [0, 15, 20, 25, 30, 40, 60, 85, float("inf")]
```

결과:

```text
area_bin
```

rolling comparable 거래를 계산할 때도 활용한다.

---

# 13. 건물 연식

다음 생성:

```python
building_age = contract_year - built_year
```

검증:

- `building_age < 0` → invalid
- `built_year < 1900` → suspicious
- `built_year > contract_year` → invalid

invalid 값은 원본 행 전체를 무조건 삭제하지 말고 `built_year` 또는 `building_age`만 NaN 처리하는 방식을 우선 사용한다.

관련 건수 report.

---

# 14. 층

연립·다세대 데이터에는 음수층이 있을 수 있다.

예:

```text
floor = -1
```

이는 지하층 의미일 수 있으므로 **이상치로 삭제하지 않는다.**

다음 파생 feature 생성:

```python
is_basement = (floor < 0).astype(int)
is_ground_floor = (floor == 1).astype(int)
```

`floor == 0` 또는 극단값은 빈도를 확인하여 report.

임의 삭제 금지.

---

# 15. 전세/월세 분리

`rent_type` 값 분포를 먼저 출력한다.

정상 예상:

```text
전세
월세
```

## 전세

정상적인 전세의 기본 규칙:

```text
rent_type == "전세"
monthly_rent_10k == 0
deposit_10k > 0
```

단, 데이터에 예외가 있으면 바로 제거하지 말고 anomaly report에 기록.

## 월세

기본 규칙:

```text
rent_type == "월세"
monthly_rent_10k > 0
deposit_10k >= 0
```

월세에서는 보증금을 삭제하지 않는다.

---

# 16. 신규/갱신 계약

`contract_type`은 다음처럼 포함될 수 있다.

```text
신규
갱신
missing
```

정규화 후:

```text
new
renewal
unknown
```

형태의 별도 컬럼 `contract_type_clean` 생성 가능.

## 핵심 원칙

최종 ML 데이터셋은 최소 두 가지 버전을 만들 수 있게 구현한다.

### 기본 권장 버전

**신규 계약만**

이유:

이 서비스의 주 사용자는 앞으로 새 집을 구하려는 사람이므로 신규 계약 가격이 핵심 target이다.

```python
contract_type == "신규"
```

### 전체 버전

필요하면 신규 + 갱신 모두 사용할 수 있도록 config 옵션 제공:

```python
NEW_CONTRACT_ONLY = True
```

`contract_type`이 결측인 과거 데이터는 함부로 모두 삭제하지 않는다.  
연도별 missing 비율을 확인한 뒤 report에 남긴다.

---

# 17. 갱신요구권 / 종전계약 정보

다음 컬럼:

```text
renewal_request_used
previous_deposit_10k
previous_monthly_rent_10k
```

은 **일반 신규 계약 가격 예측 feature로 사용하지 않는다.**

이유:

신규 주택 탐색 시 사용자가 알 수 없는 정보일 수 있고, 갱신 계약의 현재 가격과 지나치게 직접적으로 연결되어 있다.

다만 원본 clean dataset에는 보존한다.

그리고 갱신 분석용 feature를 선택적으로 생성:

```text
deposit_change_from_previous
monthly_rent_change_from_previous
deposit_change_rate_from_previous
monthly_rent_change_rate_from_previous
```

이 feature들은 **갱신 계약 분석 전용**이며 기본 prediction dataset에서는 제외한다.

---

# 18. 계약기간 파싱

`contract_period` 예:

```text
202201~202401
202112~202404
-
```

가능하면 다음으로 파싱:

```text
lease_start_ym
lease_end_ym
lease_duration_months
```

그러나 `contract_period`가 missing인 행이 많을 수 있으므로 필수 feature로 요구하지 않는다.

파싱 실패는 NaN.

---

# 19. 중복 제거

중복 판단을 한 종류로만 하지 말고 두 단계로 수행한다.

## 19-1. 완전 중복

모든 주요 컬럼이 동일한 행 제거.

## 19-2. 거래 중복 후보

다음 조합이 동일한 경우 duplicate candidate로 report:

```text
contract_date
gu
dong
jibun
building_name
area_m2
floor
rent_type
deposit_10k
monthly_rent_10k
```

동일 거래가 파일 중복 다운로드 때문에 반복된 것인지 확인 가능하도록 한다.

기본적으로 완전 동일 row는 제거한다.

---

# 20. 이상치 처리 원칙

부동산 가격 데이터는 극단값이 실제 고가주택일 수 있으므로 단순 IQR로 자동 삭제하지 않는다.

절대 규칙 기반 오류만 우선 제거:

```text
area_m2 <= 0
deposit_10k < 0
monthly_rent_10k < 0
invalid contract date
```

그 외 가격 극단치는:

- quantile
- IQR
- log distribution

을 계산해 `is_price_outlier_candidate` flag만 만들고 원본은 유지한다.

예:

```python
monthly_rent_outlier_candidate
deposit_outlier_candidate
```

지역/주택유형/면적구간별 분포 기반으로 판단하도록 한다.

---

# 21. 핵심: 미래 정보 누수 방지

이 부분은 절대 위반하면 안 된다.

예측 대상 거래가 `2024-03-20`이라면, feature 생성 시 사용할 수 있는 거래는 반드시:

```text
2024-03-19 이전
```

뿐이다.

즉 같은 날짜의 다른 거래도 원칙적으로 미래/동시점 정보 누수 가능성이 있으므로 rolling feature 계산 시 다음 방식 중 하나를 사용한다.

권장:

```python
closed="left"
```

또는 현재 행을 shift(1)한 후 rolling.

절대로 전체 데이터에서 그룹 평균/중앙값을 먼저 계산해서 각 행에 붙이지 않는다.

잘못된 예:

```python
df.groupby("dong")["monthly_rent"].transform("median")
```

이 방식은 미래 거래를 포함하므로 모델 feature로 금지.

---

# 22. Rolling / Historical Feature Engineering

전처리 완료 후 데이터를 반드시 `contract_date` 기준으로 오름차순 정렬한다.

## 22-1. 지역 거래량

각 거래 시점 이전 데이터를 사용해:

```text
dong_tx_count_30d
dong_tx_count_90d
dong_tx_count_180d
```

기준:

```text
gu + dong + rent_type
```

추가 후보:

```text
gu_tx_count_90d
```

---

# 23. 지역 최근 가격 통계

## 전세

과거 거래의 `deposit_10k`

## 월세

과거 거래의 `monthly_rent_10k`

단, 월세는 보증금 조건에 따라서 월세가 달라지므로 단순 월세 median만 쓰는 것에는 한계가 있다.

그래도 baseline feature로 생성하되, 이후 `deposit bin` 또는 환산월세 feature를 추가할 수 있도록 구조를 만들어라.

생성:

```text
dong_price_median_30d
dong_price_median_90d
dong_price_median_180d

dong_price_mean_90d
dong_price_std_90d
```

그룹:

```text
gu + dong + rent_type
```

---

# 24. 비교 가능 주택(Comparable) 가격

지역 전체보다 다음 그룹이 더 중요하다.

```text
gu
dong
rent_type
housing_type
area_bin
```

이 그룹 기준으로 과거:

```text
comp_price_median_30d
comp_price_median_90d
comp_price_median_180d
comp_tx_count_90d
```

를 생성한다.

거래 수가 너무 적을 경우 NaN 허용.

절대로 미래 거래를 fallback 계산에 사용하지 않는다.

---

# 25. 동일 건물 과거 거래

`property_id` 기준으로:

```text
property_last_price
property_days_since_last_transaction
property_tx_count_365d
property_price_median_365d
```

생성.

동일 건물 거래가 없으면 NaN.

`property_last_price`는 현재 행 바로 이전 거래만 사용.

---

# 26. 가격 추세 feature

예:

```text
최근 90일 median
vs
그 이전 90일 median
```

을 비교.

가능하면:

```python
price_trend_3m =
    recent_90d_median / previous_90d_median - 1
```

단 denominator가 0 또는 NaN이면 NaN.

추천:

```text
dong_price_trend_3m
comp_price_trend_3m
```

표본이 너무 적으면 계산하지 말고 NaN.

최소 거래 수 threshold를 config로 둔다.

예:

```python
MIN_ROLLING_COUNT = 3
```

---

# 27. 월세의 보증금 효과

월세 모델에서는 반드시:

```text
deposit_10k
```

를 원본 feature로 남긴다.

추가로 exploratory feature:

```text
deposit_per_m2
monthly_rent_per_m2
```

생성 가능.

단 `monthly_rent_per_m2`는 **target에서 직접 파생된 값이므로 월세 예측 입력 feature로 절대 넣지 않는다.**

마찬가지로 전세의:

```text
deposit_per_m2
```

도 target 파생이므로 전세 예측 feature에 넣지 않는다.

EDA용으로만 저장 가능하며, `*_eda_only`처럼 명확히 구분한다.

---

# 28. Target Leakage 방지 목록

아래 항목은 해당 모델의 feature에서 제외한다.

## 전세 모델

target:

```text
deposit_10k
```

금지:

```text
deposit_per_m2
deposit 기반 현재 거래 파생값
현재 거래의 종전계약 정보를 통한 직접 target 추정값
```

## 월세 모델

target:

```text
monthly_rent_10k
```

금지:

```text
monthly_rent_per_m2
현재 월세에서 계산한 파생값
```

월세 모델의 `deposit_10k`는 사용 가능하며 중요 feature다.

---

# 29. 최종 모델용 데이터셋

## 29-1. 전체 clean dataset

`all_transactions.parquet`

포함:

- 원본 clean 컬럼
- 주소 파싱 컬럼
- 날짜 컬럼
- 기본 파생변수
- rolling feature
- anomaly flag

---

## 29-2. 전세 데이터

`jeonse_model_data.parquet`

필터 기본값:

```python
rent_type == "전세"
deposit_10k > 0
NEW_CONTRACT_ONLY 조건 적용
```

target:

```text
target_deposit_10k
```

예상 feature:

```text
contract_date
contract_year
contract_month
contract_quarter
month_sin
month_cos

gu
dong
property_id

area_m2
area_pyeong
area_bin
floor
is_basement
is_ground_floor

built_year
building_age

housing_type

dong_tx_count_30d
dong_tx_count_90d
dong_tx_count_180d

dong_price_median_30d
dong_price_median_90d
dong_price_median_180d
dong_price_std_90d

comp_price_median_30d
comp_price_median_90d
comp_price_median_180d
comp_tx_count_90d

property_last_price
property_days_since_last_transaction
property_price_median_365d

dong_price_trend_3m
comp_price_trend_3m
```

---

## 29-3. 월세 데이터

`monthly_model_data.parquet`

필터 기본값:

```python
rent_type == "월세"
monthly_rent_10k > 0
deposit_10k >= 0
NEW_CONTRACT_ONLY 조건 적용
```

target:

```text
target_monthly_rent_10k
```

전세 feature + 반드시:

```text
deposit_10k
deposit_per_m2
```

단 `deposit_per_m2`는 deposit에서만 파생되므로 월세 예측에서는 사용 가능하다.

---

# 30. 결측치

결측치는 무작정 0으로 채우지 않는다.

원칙:

## categorical

예:

```text
building_name
contract_type
renewal_request_used
```

필요할 경우:

```text
"UNKNOWN"
```

## numeric

rolling feature NaN은:

```text
해당 시점 이전에 충분한 거래가 없었다
```

라는 의미이므로 중요한 정보다.

0으로 강제 대체하지 말고 NaN 유지.

필요하면 missing indicator 생성:

```text
has_property_history
has_comp_90d_history
has_dong_90d_history
```

모델 단계에서 CatBoost/LightGBM의 missing 처리 기능을 사용할 수 있게 한다.

---

# 31. 2021년 초반 데이터의 rolling feature

데이터 시작이 2021-01-01이므로 초반 거래는 과거 180일 데이터가 없을 수 있다.

이를 오류로 취급하지 않는다.

다만 실제 모델 학습 시 충분한 historical context를 확보하려면:

- 원본 데이터는 2021년부터 모두 유지
- 필요하면 2021년 일부를 rolling warm-up 기간으로 사용
- 2022년 이후부터 모델 학습에 사용하는 옵션

을 지원할 수 있도록 한다.

예:

```python
MODEL_DATA_START_DATE = None
# 또는 "2022-01-01"
```

---

# 32. 2026년 부분연도 처리

데이터 종료일은:

```text
2026-09-15
```

따라서:

- 2026년 연평균을 2021~2025년 완전연도와 직접 비교하지 않는다.
- 연간 거래량 feature를 만들 경우 partial-year bias를 주의한다.
- 월/rolling 기반 feature에는 그대로 사용 가능하다.

---

# 33. 데이터 분할을 고려한 설계

이번 단계에서 실제 train/val/test split까지 수행해도 되고 별도 모델링 단계로 넘겨도 된다.

단 향후 split은 **random split 금지**.

권장 예:

```text
Train: 2021-01-01 ~ 2024-12-31
Validation: 2025-01-01 ~ 2025-12-31
Test: 2026-01-01 ~ 2026-09-15
```

단순 baseline에서는 이 시간 분할을 사용할 수 있도록 `split` 컬럼을 추가해도 좋다.

```text
train
validation
test
```

중요:

rolling feature는 split 이전에 생성해도 되지만 각 행의 **과거 거래만** 사용해야 한다.

---

# 34. 외부 데이터 merge를 위한 key 미리 준비

아직 이번 작업에서 외부 데이터는 붙이지 않는다.

하지만 이후 다음 데이터를 추가할 예정이다.

- 한국은행 기준금리
- 전세자금대출/시장금리
- 한국부동산원 전세/월세 가격지수
- 지하철역 거리
- 학교/대학교 거리
- 입주물량
- 인구/가구 데이터

따라서 미리 다음 key를 생성:

```text
contract_year_month
sido
gu
dong
full_jibun_address
full_road_address
```

예:

```text
contract_year_month = "2024-03"
```

---

# 35. 데이터 품질 리포트

`reports/preprocessing_summary.json`에는 최소 다음 내용을 저장한다.

```json
{
  "n_raw_files": 0,
  "n_rows_raw": 0,
  "n_rows_after_concat": 0,
  "n_exact_duplicates_removed": 0,
  "n_invalid_dates": 0,
  "n_invalid_area": 0,
  "n_invalid_price": 0,
  "n_rows_clean": 0,
  "n_jeonse": 0,
  "n_monthly": 0,
  "n_new_contract": 0,
  "n_renewal_contract": 0,
  "n_unknown_contract_type": 0,
  "date_min": "",
  "date_max": "",
  "gu_counts": {},
  "rent_type_counts": {},
  "missing_rate": {}
}
```

---

# 36. 콘솔 summary

전처리 종료 시 사람이 바로 볼 수 있도록 다음과 비슷하게 출력한다.

```text
================ PREPROCESSING SUMMARY ================

Raw files                 : 30
Raw rows                  : 123,456
Exact duplicates removed  : 312
Invalid rows removed      : 41
Final clean rows          : 123,103

Date range:
2021-01-01 ~ 2026-09-15

Districts:
동작구       xx,xxx
노원구       xx,xxx
서대문구     xx,xxx
동대문구     xx,xxx
성북구       xx,xxx

Rent type:
전세         xx,xxx
월세         xx,xxx

Contract type:
신규         xx,xxx
갱신         xx,xxx
Unknown      xx,xxx

Output:
data/processed/all_transactions.parquet
data/processed/jeonse_model_data.parquet
data/processed/monthly_model_data.parquet

=======================================================
```

---

# 37. EDA용 요약 파일

다음 통계도 별도 CSV로 저장하면 좋다.

## 지역/연도/월별 거래 수

```text
gu
dong
year
month
rent_type
transaction_count
```

## 가격 요약

전세:

```text
median_deposit
mean_deposit
std_deposit
q25_deposit
q75_deposit
```

월세:

```text
median_monthly_rent
mean_monthly_rent
std_monthly_rent
median_deposit
```

파일 예:

```text
reports/monthly_market_summary.csv
```

---

# 38. 반드시 작성할 validation

전처리 후 assert 또는 validation report로 확인:

```python
assert df["contract_date"].min() >= "2021-01-01"
assert df["contract_date"].max() <= "2026-09-15"

assert set(df["gu"].dropna().unique()).issubset(
    {"동작구", "노원구", "서대문구", "동대문구", "성북구"}
)

assert (df["area_m2"].dropna() > 0).all()
assert (df["deposit_10k"].dropna() >= 0).all()
assert (df["monthly_rent_10k"].dropna() >= 0).all()
```

추가 검증:

- 전세인데 월세금 > 0인 행 수
- 월세인데 월세금 == 0인 행 수
- built_year > contract_year
- 동일 거래 duplicate 후보
- rolling feature에 현재/미래 거래가 들어가지 않는지 unit test

---

# 39. Leakage unit test

특히 rolling feature가 중요하므로 테스트를 반드시 작성한다.

작은 synthetic dataset:

```text
2024-01-01 price=10
2024-01-10 price=20
2024-01-20 price=1000
```

2024-01-10 행의 historical median 계산에는:

```text
2024-01-01 price=10
```

만 들어가야 한다.

2024-01-20의 값 1000은 절대로 1월 10일 feature에 영향을 주면 안 된다.

이를 자동 테스트로 검증한다.

---

# 40. 성능/구현 요구사항

데이터가 수십만 행이 될 가능성을 고려한다.

가능하면:

- pandas vectorized operation
- groupby + rolling
- merge_asof
- time-based rolling

을 사용한다.

행 단위 Python `for` loop는 가능한 피한다.

단, 잘못된 고속 구현보다 정확한 leakage-free 계산을 우선한다.

rolling 계산이 복잡하면 함수별로 분리하고 docstring을 상세히 작성한다.

---

# 41. config로 분리할 값

`src/config.py`에 최소:

```python
RAW_DATA_DIR
INTERIM_DATA_DIR
PROCESSED_DATA_DIR
REPORT_DIR

VALID_GU
START_DATE
END_DATE

AREA_BINS
ROLLING_WINDOWS = [30, 90, 180]
PROPERTY_HISTORY_DAYS = 365
MIN_ROLLING_COUNT = 3

NEW_CONTRACT_ONLY = True
MODEL_DATA_START_DATE = None
```

---

# 42. 실행 방법

최종적으로 루트에서 다음 한 줄로 실행되게 한다.

```bash
python -m src.preprocess
```

옵션 지원도 가능:

```bash
python -m src.preprocess --raw-dir data/raw --new-only true
```

---

# 43. requirements.txt

최소:

```text
pandas
numpy
pyarrow
```

테스트:

```text
pytest
```

시각화가 필요하면 optional:

```text
matplotlib
```

이번 전처리 단계에서 scipy/plotly/streamlit은 필수가 아니다.

---

# 44. README

README에 다음을 명시:

1. 데이터 출처
2. 대상 지역
3. 대상 기간
4. 원본 CSV의 특수 헤더 구조
5. 실행 방법
6. 출력 파일
7. 주요 파생변수
8. data leakage 방지 방식
9. 전세/월세 target 정의
10. 신규 계약 필터 정책
11. 알려진 한계

---

# 45. 하지 말아야 할 것

다음은 절대 하지 않는다.

### 1.
모든 기간의 지역 평균 가격을 계산해 현재 행에 붙이지 말 것.

```python
# 금지
df.groupby("dong")["price"].transform("mean")
```

모델용 feature라면 미래 정보가 포함된다.

### 2.
전세와 월세 가격을 하나의 target으로 섞지 말 것.

### 3.
월세 모델에서 보증금을 버리지 말 것.

### 4.
전세 target인 현재 보증금에서 계산한 `deposit_per_m2`를 전세 입력 feature로 넣지 말 것.

### 5.
월세 target에서 계산한 `monthly_rent_per_m2`를 월세 입력 feature로 넣지 말 것.

### 6.
`-` 값을 무조건 숫자 0으로 바꾸지 말 것.

### 7.
지하층 `floor=-1`을 오류라고 삭제하지 말 것.

### 8.
가격의 상위 1%를 자동으로 이상치라 판단해서 삭제하지 말 것.

### 9.
2026년을 완전한 1년 데이터처럼 연간 비교하지 말 것.

### 10.
random train-test split을 기본 방법으로 사용하지 말 것.

---

# 46. 구현 순서

아래 순서대로 작업한다.

## Phase 1 — Inspect

- `data/raw/` 파일 목록 출력
- 각 CSV의 encoding / header line / rows / columns 확인
- 각 파일 검색조건 확인
- 컬럼 schema 일치 여부 검사

## Phase 2 — Load & Merge

- CSV 읽기
- source filename 기록
- source search condition 기록
- 모든 데이터 concat
- raw merged 저장

## Phase 3 — Clean

- 컬럼 rename
- string normalization
- numeric parsing
- date parsing
- location parsing
- duplicate removal
- invalid row flag/remove
- clean dataset 저장

## Phase 4 — Basic Features

- date features
- building age
- floor flags
- area bins
- address keys
- property identifiers
- contract period parser

## Phase 5 — Historical Features

- 반드시 시간 정렬
- transaction count
- dong rolling price
- comparable rolling price
- property history
- price trends
- leakage 검증

## Phase 6 — Model Datasets

- 전세 분리
- 월세 분리
- 신규계약 옵션 적용
- target rename
- leakage feature 제거
- parquet/csv 저장

## Phase 7 — Validate & Report

- schema validation
- 기간 검증
- 구 검증
- 이상치/결측 요약
- leakage unit test
- summary 출력

---

# 47. 최종 산출물

작업 완료 시 다음을 제공한다.

```text
src/config.py
src/load_data.py
src/clean_data.py
src/feature_engineering.py
src/validate_data.py
src/preprocess.py

tests/test_preprocessing.py

data/interim/merged_raw.csv
data/interim/cleaned_all.csv

data/processed/all_transactions.parquet
data/processed/jeonse_model_data.parquet
data/processed/monthly_model_data.parquet

reports/preprocessing_summary.json
reports/data_quality_report.csv
reports/monthly_market_summary.csv
reports/preprocessing_log.txt

requirements.txt
README.md
```

---

# 48. 작업 중 의사결정 원칙

불명확한 데이터가 발견되면 임의로 조용히 처리하지 말고:

1. 실제 값 분포를 먼저 출력한다.
2. 어떤 문제가 있는지 설명한다.
3. 가장 보수적인 처리방법을 적용한다.
4. 해당 처리 건수를 report에 남긴다.

특히 다음 컬럼은 실제 데이터 분포 확인 후 처리한다.

```text
계약구분
갱신요구권 사용
계약기간
건물명
층
건축년도
종전계약 보증금
종전계약 월세
```

---

# 49. 첫 실행 시 나에게 보여줄 결과

전체 구현이 끝난 뒤 무조건 다음 결과를 요약해서 보여줘.

### A. 입력 파일

- 총 파일 개수
- 각 파일별 구/기간/행 수

### B. 통합 데이터

- 총 행 수
- 기간
- 5개 구별 행 수
- 전세/월세 건수

### C. 결측률 Top 10

컬럼별 missing percentage

### D. 계약구분 분포

```text
신규
갱신
결측
```

### E. 이상 데이터

- invalid date
- area <= 0
- negative deposit
- negative monthly rent
- built_year > contract_year
- 전세인데 월세 > 0
- 월세인데 월세 == 0
- duplicate

### F. 최종 모델 데이터

```text
전세 rows
월세 rows
feature 개수
각 데이터셋 기간
```

### G. rolling feature 검증

실제 거래 한 건을 골라:

```text
해당 거래 날짜
feature 계산에 들어간 가장 최근 과거 거래 날짜
계산된 90일 median
해당 기간 과거 거래 수
```

를 보여줘.

이를 통해 미래 정보가 들어가지 않았음을 확인할 수 있어야 한다.

---

# 50. 가장 중요한 목표

이번 전처리의 성공 기준은 단순히 CSV가 에러 없이 만들어지는 것이 아니다.

다음 세 조건을 반드시 만족해야 한다.

1. **모든 feature가 실제 예측 시점에 이용 가능한 정보로만 구성될 것**
2. **과거 거래 기반 feature에 현재/미래 거래가 절대로 섞이지 않을 것**
3. **이후 금리·한국부동산원 지수·지하철/학교 거리 등의 외부 데이터를 쉽게 merge할 수 있을 것**

코드를 작성하기 전에 먼저 예시 CSV 한 개를 실제로 읽고 schema와 값 분포를 확인한 다음 구현을 시작하라.
