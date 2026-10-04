"""좌표·거리 기반 위치 feature (data/external 의 지오코딩 결과 사용).

위치는 시간에 따라 변하지 않는 매물 속성이므로 leakage 문제가 없다.
(단, 신규 개통역은 개통 전 거래에도 '역이 있는 것'으로 계산된다 — README 한계 참고)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .geocode import ADDRESS_CACHE, STATION_FILE, UNIVERSITY_FILE
from .report import Report

log = logging.getLogger("preprocess")

EARTH_R = 6_371_000.0
LOCATION_FEATURES = ["lat", "lon", "station_dist_m", "n_stations_1km", "univ_dist_m"]


def haversine_matrix(lat1, lon1, lat2, lon2) -> np.ndarray:
    """(n,) 점들과 (m,) 점들 사이 거리(m) 행렬 (n, m)."""
    p1, p2 = np.radians(lat1)[:, None], np.radians(lat2)[None, :]
    dp = p2 - p1
    dl = np.radians(lon2)[None, :] - np.radians(lon1)[:, None]
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def available() -> bool:
    return ADDRESS_CACHE.exists() and STATION_FILE.exists() and UNIVERSITY_FILE.exists()


def add_location_features(df: pd.DataFrame, report: Report) -> pd.DataFrame:
    if not available():
        log.warning("  지오코딩 결과 없음 → 위치 feature 생략 (python -m src.geocode 먼저 실행)")
        return df
    coords = pd.read_csv(ADDRESS_CACHE)[["full_jibun_address", "lat", "lon", "geo_precision"]]
    df = df.drop(columns=[c for c in LOCATION_FEATURES + ["geo_precision", "nearest_station", "nearest_univ"]
                          if c in df.columns])
    df = df.merge(coords, on="full_jibun_address", how="left", validate="many_to_one")
    report.check("features", "geocode precision", len(df), "info",
                 str(df["geo_precision"].value_counts(dropna=False).to_dict()))
    report.check("features", "rows without coordinates", df["lat"].isna().sum(), "location features NaN",
                 level="WARNING")

    # 같은 주소는 거리 계산을 한 번만
    u = df[["lat", "lon"]].dropna().drop_duplicates()
    st, un = pd.read_csv(STATION_FILE), pd.read_csv(UNIVERSITY_FILE)
    ds = haversine_matrix(u["lat"].to_numpy(), u["lon"].to_numpy(), st["lat"].to_numpy(), st["lon"].to_numpy())
    du = haversine_matrix(u["lat"].to_numpy(), u["lon"].to_numpy(), un["lat"].to_numpy(), un["lon"].to_numpy())
    u = u.assign(
        station_dist_m=ds.min(axis=1).round(0),
        nearest_station=st["station_name"].to_numpy()[ds.argmin(axis=1)],
        n_stations_1km=(ds <= 1000).sum(axis=1),
        univ_dist_m=du.min(axis=1).round(0),
        nearest_univ=un["school_name"].to_numpy()[du.argmin(axis=1)],
    )
    df = df.merge(u, on=["lat", "lon"], how="left")
    log.info("  위치 feature 추가: 역 %d개, 대학 %d개 기준 (역거리 중앙값 %.0fm, 대학거리 중앙값 %.0fm)",
             len(st), len(un), df["station_dist_m"].median(), df["univ_dist_m"].median())
    return df
