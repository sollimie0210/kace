"""학교 기반 전월세 예측 엔진 (웹사이트 백엔드의 핵심 로직).

    python -m src.predictor --school 고려대 --type 월세 --months 3 --room 원룸 --deposit 1000 --price 60

흐름
1) 학교 반경(기본 1km, 표본 부족 시 1.5km → 2km) 안에서 최근 1년 거래가 있는 건물을 후보로 선택
2) 후보 건물마다 사용자 조건(방 크기, 보증금)을 넣은 '가상 신규 계약'을 만들고,
   학습과 **동일한 코드**(history_features_at_offset)로 기준일 시점 feature를 계산해 q10/q50/q90 예측
   (학습-서빙 불일치 방지)
3) 건물별 예측 분포를 거래량 가중으로 합쳐 '이 학교 근처에서 계약하면 형성될 가격 분포'를 만든다
4) 예상 가격(중앙값) / 적정 범위(80%) / 매물 가격 판정 / 지금 vs 계약 예정 시점 비교

기준일(as-of)은 데이터 스냅샷 날짜(config.END_DATE). 학습과 같게, 기준일 30일 이전 거래까지만 사용한다.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config
from . import model_config as mc
from .geocode import UNIVERSITY_FILE
from .horizon_dataset import MONTHLY_ONLY_UNIT_FEATURES, UNIT_FEATURES, history_features_at_offset
from .location_features import haversine_matrix
from .train_model import apply_anchor

FINAL_DIR = mc.MODEL_DIR / "final"
# 웹앱 배포용 경량 거래 데이터 (python -m src.export_app_data). 없으면 전체 all_transactions 사용.
APP_DATA = config.PROJECT_ROOT / "data" / "app" / "transactions.parquet"
APP_COLUMNS = [
    "tx_id", "contract_date", "gu", "dong", "rent_type", "housing_category", "housing_type", "area_bin",
    "property_id", "price_10k", "deposit_10k", "monthly_rent_10k", "contract_type_clean", "area_m2", "floor",
    "built_year", "building_name_is_address", "building_name", "full_road_address",
    "lat", "lon", "station_dist_m", "n_stations_1km", "univ_dist_m",
]
KIND_BY_TYPE = {"전세": "jeonse", "월세": "monthly"}

# 방 크기 선택지 (전용면적 ㎡)
ROOM_TYPES = {
    "원룸": (0, 25),
    "투룸": (25, 45),
    "쓰리룸+": (45, 200),
    "전체": (0, 1000),
}
RADII_M = [1000, 1500, 2000]          # 후보 건물이 부족하면 순서대로 확대
MIN_BUILDINGS = 30
CANDIDATE_LOOKBACK_DAYS = 365
N_SAMPLES = 20000
# 학생들이 실제로 쓰는 줄임말 → 학교 검색어(universities.csv의 query)
SCHOOL_ALIASES = {
    "중대": "중앙대학교", "숭실대": "숭실대학교", "총신대": "총신대학교",
    "광운대": "광운대학교", "과기대": "서울과학기술대학교", "서울과기대": "서울과학기술대학교", "서울과기": "서울과학기술대학교",
    "삼육대": "삼육대학교", "서울여대": "서울여자대학교", "인덕대": "인덕대학교",
    "연대": "연세대학교", "연세대": "연세대학교", "이대": "이화여자대학교", "이화여대": "이화여자대학교",
    "명지대": "명지대학교", "추계예대": "추계예술대학교", "경기대": "경기대학교",
    "경희대": "경희대학교", "외대": "한국외국어대학교", "한국외대": "한국외국어대학교",
    "시립대": "서울시립대학교", "서울시립대": "서울시립대학교",
    "고대": "고려대학교", "고려대": "고려대학교", "성신여대": "성신여자대학교", "국민대": "국민대학교",
    "한성대": "한성대학교", "서경대": "서경대학교", "동덕여대": "동덕여자대학교", "중앙대": "중앙대학교",
}
Z90 = 1.2815515655446004              # 표준정규 90% 분위수


@dataclass
class Prediction:
    school: str
    rent_type: str
    room: str
    asof_date: str
    contract_date: str
    horizon_days: int
    radius_m: int
    n_buildings: int
    deposit_10k: float | None
    expected: float
    range_low: float
    range_high: float
    now_expected: float
    change_vs_now_pct: float
    timing_message: str
    price_check: dict | None
    recent_examples: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    school_lat: float | None = None
    school_lon: float | None = None
    buildings: list = field(default_factory=list)       # 지도용: 후보 건물별 예측
    samples: list = field(default_factory=list)         # 차트용: 가격 분포 표본(2,000개)

    def to_dict(self, include_samples: bool = False):
        d = self.__dict__.copy()
        if not include_samples:
            d.pop("samples")
        return d


class RentPredictor:
    def __init__(self):
        path = APP_DATA if APP_DATA.exists() else config.PROCESSED_DATA_DIR / "all_transactions.parquet"
        self.hist = pd.read_parquet(path, columns=APP_COLUMNS)
        self.hist = self.hist.sort_values(["contract_date", "tx_id"], kind="stable").reset_index(drop=True)
        self.asof = pd.Timestamp(config.END_DATE)
        self.schools = pd.read_csv(UNIVERSITY_FILE)
        self.models, self.meta = {}, {}
        for kind in ("jeonse", "monthly"):
            self.meta[kind] = json.loads((FINAL_DIR / f"{kind}_meta.json").read_text(encoding="utf-8"))
            self.models[kind] = {q: lgb.Booster(model_file=str(FINAL_DIR / f"{kind}_q{int(q * 100)}.txt"))
                                 for q in mc.QUANTILES}

    # -------------------------------------------------------------- helpers
    def find_school(self, name: str) -> pd.Series:
        """정식 이름, 캠퍼스명 포함 이름, 줄임말(고대/외대/과기대 ...) 모두 허용."""
        s = self.schools
        key = name.replace(" ", "")
        key = SCHOOL_ALIASES.get(key, key)
        norm = lambda col: s[col].str.replace(" ", "")
        hit = s[norm("query").str.startswith(key) | norm("school_name").str.startswith(key)]
        if hit.empty:
            hit = s[norm("query").str.contains(key, regex=False) | norm("school_name").str.contains(key, regex=False)]
        if hit.empty:
            raise ValueError(f"학교를 찾을 수 없습니다: {name}. 지원 학교: {', '.join(s['school_name'])}")
        return hit.iloc[0]

    def _candidates(self, school: pd.Series, rent_type: str, area_rng, categories) -> tuple[pd.DataFrame, int]:
        """학교 반경 내, 최근 1년 해당 전월세 거래가 있고 요청 면적대 호실이 있는 건물들의 대표 호실."""
        h = self.hist
        recent_cut = self.asof - pd.Timedelta(days=CANDIDATE_LOOKBACK_DAYS)
        base = h[(h["rent_type"] == rent_type) & h["housing_category"].isin(categories)
                 & h["area_m2"].between(area_rng[0], area_rng[1], inclusive="left")]
        d = haversine_matrix(base["lat"].to_numpy(), base["lon"].to_numpy(),
                             np.array([school["lat"]]), np.array([school["lon"]]))[:, 0]
        base = base.assign(dist_m=d)
        for r in RADII_M:
            near = base[base["dist_m"] <= r]
            active = near.loc[near["contract_date"] >= recent_cut, "property_id"].unique()
            if len(active) >= MIN_BUILDINGS or r == RADII_M[-1]:
                break
        near = near[near["property_id"].isin(active)]
        # 건물 대표 호실: 해당 면적대 거래의 중앙값 면적/층, 최신 거래의 건물 속성
        last = near.sort_values("contract_date").groupby("property_id").tail(1).set_index("property_id")
        agg = near.groupby("property_id").agg(
            area_m2=("area_m2", "median"), floor=("floor", "median"),
            weight=("contract_date", lambda s: int((s >= self.asof - pd.Timedelta(days=730)).sum())),
            dist_m=("dist_m", "first"))
        keep = ["gu", "dong", "housing_category", "housing_type", "built_year", "building_name_is_address",
                "lat", "lon", "station_dist_m", "n_stations_1km", "univ_dist_m", "building_name", "full_road_address"]
        cand = agg.join(last[keep]).reset_index()
        cand["floor"] = cand["floor"].round()
        cand["weight"] = cand["weight"].clip(lower=1)
        return cand, r

    def _scenario_rows(self, cand: pd.DataFrame, rent_type: str, contract_date: pd.Timestamp,
                       deposit: float | None) -> pd.DataFrame:
        """후보 건물마다 '계약일 = contract_date'인 가상 거래 행 (가격 NaN)."""
        s = cand.copy()
        s["rent_type"] = rent_type
        s["contract_date"] = contract_date
        s["contract_type_clean"] = "new"
        s["price_10k"] = np.nan
        s["deposit_10k"] = np.nan if deposit is None else float(deposit)
        s["area_bin"] = pd.cut(s["area_m2"], bins=config.AREA_BINS, labels=config.AREA_BIN_LABELS,
                               right=False).astype(str)
        s["is_basement"] = (s["floor"] < 0).astype("int8")
        s["is_ground_floor"] = (s["floor"] == 1).astype("int8")
        s["contract_month"] = contract_date.month
        s["month_sin"] = np.sin(2 * np.pi * contract_date.month / 12)
        s["month_cos"] = np.cos(2 * np.pi * contract_date.month / 12)
        s["building_age"] = (contract_date.year - s["built_year"]).where(lambda a: a >= 0)
        s["deposit_per_m2"] = s["deposit_10k"] / s["area_m2"]
        return s

    def _predict_rows(self, kind: str, scen: pd.DataFrame, horizon: int) -> np.ndarray:
        """가상 거래를 실제 거래 이력에 붙여 학습과 같은 방식으로 feature 계산 → q10/q50/q90."""
        n_hist = len(self.hist)
        syn = scen.copy()
        syn["tx_id"] = np.arange(n_hist, n_hist + len(syn)) + 10**9
        combined = pd.concat([self.hist, syn[[c for c in syn.columns if c in self.hist.columns or c == "tx_id"]]],
                             ignore_index=True)
        # 가상 행은 가격이 NaN이므로 다른 행의 통계에 영향을 주지 않고, 학습 때와 같은 offset을 쓴다
        feats = history_features_at_offset(combined, horizon + mc.REPORTING_LAG_DAYS).iloc[n_hist:]
        X = pd.concat([scen.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
        X = X.loc[:, ~X.columns.duplicated(keep="last")]
        X["horizon_days"] = min(horizon, max(mc.HORIZONS))
        if kind == "monthly":
            X["deposit_vs_dong_median"] = X["deposit_10k"] / X["dong_deposit_median_90d"].where(lambda v: v > 0) - 1
        meta = self.meta[kind]
        feats_used = apply_anchor(X, meta["target_mode"], [f for f in self._base_features(kind)], meta["anchor_fallback"])
        assert feats_used == meta["features"]
        for c, levels in meta["categorical_levels"].items():
            X[c] = pd.Categorical(X[c].astype("object"), categories=levels)
        P = np.column_stack([np.exp(self.models[kind][q].predict(X[feats_used]) + X["log_anchor"].to_numpy())
                             for q in mc.QUANTILES])
        P = np.sort(P, axis=1)
        adj = meta["conformal_log_adjustment_by_horizon"]
        a = adj[str(min(mc.HORIZONS, key=lambda h: abs(h - horizon)))]
        P[:, 0] *= np.exp(-a)
        P[:, 2] *= np.exp(a)
        return P

    @staticmethod
    def _base_features(kind):
        from .horizon_dataset import feature_columns
        return feature_columns(kind)

    @staticmethod
    def _pool(P: np.ndarray, w: np.ndarray, seed: int = 0) -> np.ndarray:
        """건물별 (q10,q50,q90)을 log 공간 split-normal로 보고 거래량 가중 혼합분포에서 표본 추출."""
        rng = np.random.default_rng(seed)
        n = np.maximum(1, np.round(N_SAMPLES * w / w.sum()).astype(int))
        lo, mid, hi = np.log(P[:, 0]), np.log(P[:, 1]), np.log(P[:, 2])
        sl, sh = (mid - lo) / Z90, (hi - mid) / Z90
        out = []
        for i in range(len(P)):
            z = rng.standard_normal(n[i])
            out.append(np.exp(mid[i] + np.where(z < 0, z * sl[i], z * sh[i])))
        return np.concatenate(out)

    # -------------------------------------------------------------- main API
    def predict(self, school: str, rent_type: str, months_ahead: int = 0, room: str = "전체",
                deposit: float | None = None, listing_price: float | None = None,
                categories=("연립다세대", "오피스텔")) -> Prediction:
        if rent_type not in KIND_BY_TYPE:
            raise ValueError("rent_type은 '전세' 또는 '월세'")
        kind = KIND_BY_TYPE[rent_type]
        sch = self.find_school(school)
        area_rng = ROOM_TYPES[room]
        notes = []

        contract_date = self.asof + pd.DateOffset(months=months_ahead)
        horizon = int((contract_date - self.asof).days)
        if horizon > max(mc.HORIZONS) + 7:      # 6개월(=182~184일)은 정상 범위로 취급
            notes.append(f"예측 기간이 학습 범위(최대 {max(mc.HORIZONS)}일)를 넘어 {max(mc.HORIZONS)}일로 계산했습니다.")

        cand, radius = self._candidates(sch, rent_type, area_rng, list(categories))
        if cand.empty:
            raise ValueError("조건에 맞는 근처 거래가 없습니다. 방 크기나 주택 유형 조건을 넓혀 주세요.")
        if len(cand) < MIN_BUILDINGS:
            notes.append(f"반경 {radius}m 안의 후보 건물이 {len(cand)}개로 적어 범위가 불안정할 수 있습니다.")

        if kind == "monthly" and deposit is None:
            recent = self.hist[(self.hist["rent_type"] == "월세")
                               & self.hist["property_id"].isin(cand["property_id"])
                               & (self.hist["contract_date"] >= self.asof - pd.Timedelta(days=365))]
            deposit = float(recent["deposit_10k"].median())
            notes.append(f"보증금을 입력하지 않아 근처 최근 1년 월세 보증금 중앙값 {deposit:,.0f}만원으로 계산했습니다.")

        w = cand["weight"].to_numpy(dtype=float)
        P_t = self._predict_rows(kind, self._scenario_rows(cand, rent_type, contract_date, deposit), horizon)
        P_0 = self._predict_rows(kind, self._scenario_rows(cand, rent_type, self.asof, deposit), 0)
        dist_t, dist_0 = self._pool(P_t, w), self._pool(P_0, w)
        q10, q50, q90 = np.quantile(dist_t, [0.1, 0.5, 0.9])
        now50 = float(np.quantile(dist_0, 0.5))
        change = (q50 / now50 - 1) * 100

        half_band = (q90 - q10) / 2 / q50 * 100
        if abs(change) < 2 or abs(change) < half_band / 4:
            timing = (f"지금 계약해도, {months_ahead}개월 뒤에 계약해도 예상 차이는 {change:+.1f}%로 작아요. "
                      "시점보다 매물 선택이 가격에 훨씬 큰 영향을 줍니다.")
        elif change > 0:
            timing = f"{months_ahead}개월 뒤에는 지금보다 약 {change:+.1f}% 오를 것으로 예상돼요. 서두르는 편이 유리할 수 있어요."
        else:
            timing = f"{months_ahead}개월 뒤에는 지금보다 약 {change:+.1f}% 낮아질 것으로 예상돼요. 급하지 않다면 기다려볼 만해요."

        check = None
        if listing_price is not None:
            pct = float((dist_t < listing_price).mean() * 100)
            if pct < 25:
                verdict = "저렴"
            elif pct <= 75:
                verdict = "적정"
            elif pct <= 90:
                verdict = "다소 비쌈"
            else:
                verdict = "비쌈"
            check = {"listing_price": listing_price, "percentile": round(pct, 1), "verdict": verdict,
                     "message": f"입력한 {listing_price:,.0f}만원은 근처 예상 가격 분포에서 하위 {pct:.0f}% 위치 → {verdict}"}

        ex = self.hist[(self.hist["rent_type"] == rent_type) & self.hist["property_id"].isin(cand["property_id"])
                       & self.hist["area_m2"].between(area_rng[0], area_rng[1], inclusive="left")]
        if kind == "monthly" and deposit and deposit > 0:
            # 월세 사례는 보증금이 비슷한 거래만 (반전세 등과 섞이면 비교가 안 됨)
            similar = ex[ex["deposit_10k"].between(deposit * 0.5, deposit * 2)]
            ex = similar if len(similar) >= 3 else ex
        ex = ex.merge(cand[["property_id", "dist_m"]], on="property_id").sort_values("contract_date").tail(5)
        examples = [{"date": str(r.contract_date.date()), "building": r.building_name, "dong": r.dong,
                     "area_m2": r.area_m2, "floor": r.floor, "deposit_10k": r.deposit_10k,
                     "monthly_rent_10k": r.monthly_rent_10k, "distance_m": int(r.dist_m)}
                    for r in ex[::-1].itertuples()]

        return Prediction(
            school=sch["school_name"], rent_type=rent_type, room=room,
            asof_date=str(self.asof.date()), contract_date=str(contract_date.date()), horizon_days=horizon,
            radius_m=radius, n_buildings=len(cand), deposit_10k=deposit if kind == "monthly" else None,
            expected=round(float(q50), 1), range_low=round(float(q10), 1), range_high=round(float(q90), 1),
            now_expected=round(now50, 1), change_vs_now_pct=round(float(change), 2),
            timing_message=timing, price_check=check, recent_examples=examples, notes=notes,
            school_lat=float(sch["lat"]), school_lon=float(sch["lon"]),
            buildings=[{"building": r.building_name, "dong": r.dong, "lat": r.lat, "lon": r.lon,
                        "distance_m": int(r.dist_m), "area_m2": round(r.area_m2, 1),
                        "expected": round(float(m), 1), "low": round(float(lo), 1), "high": round(float(hi), 1)}
                       for r, (lo, m, hi) in zip(cand.itertuples(), P_t)],
            samples=np.random.default_rng(1).choice(dist_t, size=min(2000, len(dist_t)), replace=False).round(1).tolist(),
        )


def format_prediction(p: Prediction) -> str:
    unit = "만원"
    what = "전세 보증금" if p.rent_type == "전세" else f"월세 (보증금 {p.deposit_10k:,.0f}만원 기준)"
    L = [f"\n[{p.school}] {p.room} {what}",
         f"  기준 데이터: ~{p.asof_date} | 계약 예정: {p.contract_date} | 반경 {p.radius_m}m 건물 {p.n_buildings}개",
         f"  예상 가격 : {p.expected:,.0f}{unit}",
         f"  적정 범위 : {p.range_low:,.0f} ~ {p.range_high:,.0f}{unit} (80%)",
         f"  지금 계약 시 예상: {p.now_expected:,.0f}{unit} ({p.change_vs_now_pct:+.1f}%)",
         f"  → {p.timing_message}"]
    if p.price_check:
        L.append(f"  매물 판정: {p.price_check['message']}")
    if p.recent_examples:
        L.append("  근처 최근 실거래:")
        for e in p.recent_examples:
            price = (f"{e['deposit_10k']:,.0f}" if p.rent_type == "전세"
                     else f"{e['deposit_10k']:,.0f}/{e['monthly_rent_10k']:,.0f}")
            L.append(f"    {e['date']}  {e['building']} ({e['dong']}, {e['area_m2']:.1f}㎡, {e['floor']:.0f}층, {e['distance_m']}m)  {price}")
    for n in p.notes:
        L.append(f"  ※ {n}")
    return "\n".join(L)


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--school", required=True)
    ap.add_argument("--type", required=True, choices=list(KIND_BY_TYPE))
    ap.add_argument("--months", type=int, default=0, help="몇 개월 뒤 계약 (0~6)")
    ap.add_argument("--room", default="전체", choices=list(ROOM_TYPES))
    ap.add_argument("--deposit", type=float, help="월세 보증금(만원)")
    ap.add_argument("--price", type=float, help="보고 있는 매물 가격(만원): 전세=보증금, 월세=월세")
    ap.add_argument("--housing", default="전체", choices=["전체", "연립다세대", "오피스텔"])
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    cats = ("연립다세대", "오피스텔") if a.housing == "전체" else (a.housing,)
    p = RentPredictor().predict(a.school, a.type, a.months, a.room, a.deposit, a.price, categories=cats)
    print(json.dumps(p.to_dict(), ensure_ascii=False, indent=2) if a.json else format_prediction(p))


if __name__ == "__main__":
    main()
