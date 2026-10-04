"""카카오 로컬 API로 주소/학교/지하철역 좌표를 수집한다 (캐시 기반, 재실행 시 새 주소만 호출).

    python -m src.geocode

산출물 (data/external/)
- address_coords.csv : full_jibun_address → lat, lon, geo_precision(jibun/road/dong), 사용한 질의
- universities.csv   : 대상 5개 구 대학 좌표 (서비스의 '학교 선택' 목록)
- subway_stations.csv: 대상 지역 지하철역 좌표

API 키는 프로젝트 루트 .env 의 KAKAO_REST_API_KEY 에서 읽는다 (코드/로그에 출력하지 않음).
"""
from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from . import config
from .report import setup_logging

log = logging.getLogger("preprocess")

EXTERNAL_DIR = config.PROJECT_ROOT / "data" / "external"
ADDRESS_CACHE = EXTERNAL_DIR / "address_coords.csv"
UNIVERSITY_FILE = EXTERNAL_DIR / "universities.csv"
STATION_FILE = EXTERNAL_DIR / "subway_stations.csv"

API = "https://dapi.kakao.com/v2/local/search/{}.json"
MAX_WORKERS = 6
MIN_INTERVAL_SEC = 0.04      # 전체 호출 속도 상한 (~25 req/s)

# 대상 5개 구에 캠퍼스가 있는 대학 (검색어, 기대하는 구)
UNIVERSITIES = [
    ("중앙대학교 서울캠퍼스", "동작구"), ("숭실대학교", "동작구"), ("총신대학교", "동작구"),
    ("광운대학교", "노원구"), ("서울과학기술대학교", "노원구"), ("삼육대학교", "노원구"),
    ("서울여자대학교", "노원구"), ("인덕대학교", "노원구"),
    ("연세대학교 신촌캠퍼스", "서대문구"), ("이화여자대학교", "서대문구"),
    ("명지대학교 인문캠퍼스", "서대문구"), ("추계예술대학교", "서대문구"), ("경기대학교 서울캠퍼스", "서대문구"),
    ("경희대학교 서울캠퍼스", "동대문구"), ("한국외국어대학교 서울캠퍼스", "동대문구"), ("서울시립대학교", "동대문구"),
    ("고려대학교 서울캠퍼스", "성북구"), ("성신여자대학교 돈암수정캠퍼스", "성북구"), ("국민대학교", "성북구"),
    ("한성대학교", "성북구"), ("서경대학교", "성북구"), ("동덕여자대학교", "성북구"),
]


# ------------------------------------------------------------------ client
def load_api_key() -> str:
    env = config.PROJECT_ROOT / ".env"
    for line in env.read_text(encoding="utf-8-sig").splitlines():
        if line.strip().startswith("KAKAO_REST_API_KEY"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError(".env 에 KAKAO_REST_API_KEY 가 없습니다")


class KakaoLocal:
    def __init__(self, key: str):
        self.key = key
        self._lock = threading.Lock()
        self._last = 0.0
        self.n_calls = 0

    def _throttle(self):
        with self._lock:
            wait = self._last + MIN_INTERVAL_SEC - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            self.n_calls += 1

    def get(self, kind: str, **params) -> dict:
        url = API.format(kind) + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"Authorization": "KakaoAK " + self.key})
        for attempt in range(4):
            self._throttle()
            try:
                with urllib.request.urlopen(req, timeout=10) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503) and attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Kakao API HTTP {e.code}: {e.read().decode()[:200]}") from None
            except (urllib.error.URLError, TimeoutError):
                if attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                raise
        return {"documents": []}

    def address(self, query: str):
        docs = self.get("address", query=query, size=1).get("documents", [])
        return (float(docs[0]["y"]), float(docs[0]["x"])) if docs else None


# ------------------------------------------------------------------ addresses
def unique_addresses(hist: pd.DataFrame) -> pd.DataFrame:
    """지오코딩 대상: 지번주소별 대표 도로명주소와 동 주소(실패 시 fallback용)."""
    g = hist.groupby("full_jibun_address", dropna=True)
    return pd.DataFrame({
        "road": g["full_road_address"].agg(lambda s: s.dropna().mode().iloc[0] if s.notna().any() else None),
        "dong_addr": g["location_id"].first().str.replace("|", " ", regex=False).radd("서울특별시 "),
    }).reset_index()


def geocode_addresses(client: KakaoLocal, hist: pd.DataFrame) -> pd.DataFrame:
    todo = unique_addresses(hist)
    cache = pd.read_csv(ADDRESS_CACHE) if ADDRESS_CACHE.exists() else pd.DataFrame(
        columns=["full_jibun_address", "lat", "lon", "geo_precision", "query"])
    todo = todo[~todo["full_jibun_address"].isin(cache["full_jibun_address"])]
    log.info("주소 지오코딩: 전체 %d, 캐시 %d, 신규 호출 대상 %d",
             len(todo) + len(cache), len(cache), len(todo))
    if todo.empty:
        return cache

    dong_cache: dict[str, tuple | None] = {}

    def one(row):
        for q, prec in ((row.full_jibun_address, "jibun"), (row.road, "road")):
            if q:
                hit = client.address(q)
                if hit:
                    return row.full_jibun_address, hit[0], hit[1], prec, q
        if row.dong_addr not in dong_cache:
            dong_cache[row.dong_addr] = client.address(row.dong_addr)
        hit = dong_cache[row.dong_addr]
        return (row.full_jibun_address, *(hit or (np.nan, np.nan)), "dong" if hit else "failed", row.dong_addr)

    results, t0 = [], time.time()
    rows = list(todo.itertuples(index=False))
    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        for i, r in enumerate(ex.map(one, rows), 1):
            results.append(r)
            if i % 1000 == 0 or i == len(rows):
                log.info("  %d / %d (%.0fs)", i, len(rows), time.time() - t0)
                part = pd.DataFrame(results, columns=cache.columns)
                pd.concat([cache, part]).to_csv(ADDRESS_CACHE, index=False, encoding="utf-8-sig")
    out = pd.read_csv(ADDRESS_CACHE)
    log.info("  precision: %s", out["geo_precision"].value_counts().to_dict())
    return out


# ------------------------------------------------------------------ universities
def geocode_universities(client: KakaoLocal) -> pd.DataFrame:
    rows = []
    for query, gu in UNIVERSITIES:
        docs = client.get("keyword", query=query, category_group_code="SC4", size=15)["documents"]
        docs = [d for d in docs if gu in d["address_name"] and ("대학교" in d["category_name"] or "전문대학" in d["category_name"])
                and not any(w in d["place_name"] for w in ("병원", "기숙사", "도서관", "학과", "대학원"))]
        if not docs:
            log.warning("  학교 검색 실패: %s (%s)", query, gu)
            continue
        base = query.split()[0]          # '연세대학교 신촌캠퍼스' -> '연세대학교'
        docs.sort(key=lambda d: (not d["place_name"].startswith(base), len(d["place_name"])))
        d = docs[0]
        # 이름이 검색어와 다르면(예: 인덕대학교 → '인덕학원', 같은 캠퍼스) 표시 이름은 검색어를 사용
        name = d["place_name"] if d["place_name"].startswith(base) else query
        rows.append({"school_name": name, "query": query, "gu": gu, "lat": float(d["y"]),
                     "lon": float(d["x"]), "address": d["address_name"], "kakao_place_id": d["id"]})
    df = pd.DataFrame(rows)
    df.to_csv(UNIVERSITY_FILE, index=False, encoding="utf-8-sig")
    log.info("대학 %d개 저장 -> %s", len(df), UNIVERSITY_FILE.name)
    return df


# ------------------------------------------------------------------ subway
def collect_stations(client: KakaoLocal, coords: pd.DataFrame, pad_deg: float = 0.01,
                     step_m: int = 1500, radius_m: int = 1200) -> pd.DataFrame:
    """거래 좌표 범위를 격자로 덮어 카테고리(SW8) 검색. 한 질의는 최대 45건이라 촘촘한 격자를 쓴다."""
    lat0, lat1 = coords["lat"].min() - pad_deg, coords["lat"].max() + pad_deg
    lon0, lon1 = coords["lon"].min() - pad_deg, coords["lon"].max() + pad_deg
    dlat = step_m / 111_000
    dlon = step_m / (111_000 * np.cos(np.radians((lat0 + lat1) / 2)))
    found = {}
    for la in np.arange(lat0, lat1 + dlat, dlat):
        for lo in np.arange(lon0, lon1 + dlon, dlon):
            for page in (1, 2, 3):
                d = client.get("category", category_group_code="SW8", x=f"{lo:.6f}", y=f"{la:.6f}",
                               radius=radius_m, page=page, size=15)
                for s in d["documents"]:
                    found[s["id"]] = {"station_name": s["place_name"], "lat": float(s["y"]), "lon": float(s["x"]),
                                      "address": s["address_name"], "kakao_place_id": s["id"]}
                if d["meta"]["is_end"]:
                    break
    df = pd.DataFrame(found.values()).sort_values("station_name")
    df.to_csv(STATION_FILE, index=False, encoding="utf-8-sig")
    log.info("지하철역 %d개 저장 -> %s", len(df), STATION_FILE.name)
    return df


def main():
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging(config.REPORT_DIR / "geocode_log.txt")
    client = KakaoLocal(load_api_key())
    hist = pd.read_parquet(config.PROCESSED_DATA_DIR / "all_transactions.parquet",
                           columns=["full_jibun_address", "full_road_address", "location_id"])
    coords = geocode_addresses(client, hist)
    geocode_universities(client)
    collect_stations(client, coords.dropna(subset=["lat"]))
    log.info("총 API 호출 %d회", client.n_calls)


if __name__ == "__main__":
    main()
