"""원본 실거래가 파일(CSV/XLSX) 탐색·파싱·병합.

국토교통부 실거래가 공개시스템 파일은 앞부분에 안내문/검색조건이 붙어 있고,
실제 헤더("NO","시군구",...)의 위치가 포맷(CSV/XLSX)마다 다르다. 이 모듈은
헤더 행을 직접 탐색하고, 검색조건(key : value)을 메타데이터로 보존한다.

지원하는 입력 형태
- MOLIT CSV  : cp949, 안내문 + 검색조건 + 헤더 + 데이터
- MOLIT XLSX : 시트 1개, 안내문 + 검색조건 + 헤더 + 데이터 (dimension 메타데이터가 깨져 있음)
- 수작업 병합 XLSX : 여러 시트, 헤더가 있거나 없음 (헤더 없는 시트는 컬럼 수로 schema 추론)
"""
from __future__ import annotations

import csv
import io
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import config

log = logging.getLogger("preprocess")


class HeaderNotFoundError(ValueError):
    pass


@dataclass
class TableInfo:
    """원본 파일 내 하나의 표(CSV 파일 또는 XLSX 시트)에 대한 파싱 메타데이터."""
    source_file: str
    source_sheet: str | None
    encoding: str | None
    header_line_index: int | None
    header_inferred: bool
    conditions: dict = field(default_factory=dict)
    n_rows: int = 0
    n_repeated_header_rows: int = 0
    housing_category: str | None = None
    ok: bool = True
    error: str | None = None

    @property
    def cond_gu(self):
        return self.conditions.get("시군구")

    @property
    def cond_period(self):
        return self.conditions.get("계약일자")


def discover_raw_files(raw_dir: Path) -> list[Path]:
    files = []
    for pattern in config.RAW_FILE_PATTERNS:
        files.extend(p for p in Path(raw_dir).rglob(pattern) if not p.name.startswith("~$"))
    return sorted(files)


# ------------------------------------------------------------------ helpers
def _is_header_row(values) -> bool:
    vals = [str(v).strip().strip('"') if v is not None else "" for v in values[:2]]
    return len(vals) == 2 and vals[0] == "NO" and vals[1] == "시군구"


def _parse_conditions(lines) -> dict:
    """헤더 이전 행에서 '키 : 값' 형태의 검색조건을 추출."""
    cond = {}
    for line in lines:
        if line is None:
            continue
        text = str(line).strip().strip('"')
        if " : " in text:
            k, v = text.split(" : ", 1)
            cond[k.strip()] = v.strip()
    return cond


def _cell_to_str(v) -> str:
    """XLSX 셀 값을 CSV와 동일한 문자열 표현으로 맞춘다 (1.0 -> '1')."""
    if v is None:
        return ""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _standardize_columns(df: pd.DataFrame, source_hint: str) -> tuple[pd.DataFrame, str]:
    """단지명→건물명 alias 적용, 누락된 주택유형 보완, housing_category 결정."""
    df = df.rename(columns=config.RAW_COLUMN_ALIASES)
    is_officetel = "주택유형" not in df.columns or "오피스텔" in source_hint
    category = "오피스텔" if is_officetel else "연립다세대"
    if "주택유형" not in df.columns:
        df["주택유형"] = "오피스텔"
    missing = [c for c in config.RAW_COLUMNS if c not in df.columns]
    if missing:
        raise HeaderNotFoundError(f"필수 컬럼 누락: {missing}")
    extra = [c for c in df.columns if c not in config.RAW_COLUMNS]
    if extra:
        log.warning("  예상하지 못한 컬럼 무시: %s", extra)
    return df[config.RAW_COLUMNS].copy(), category


# ------------------------------------------------------------------ CSV
def read_molit_csv(path: Path) -> tuple[pd.DataFrame, TableInfo]:
    text, used_enc = None, None
    for enc in config.ENCODINGS:
        try:
            text = path.read_text(encoding=enc)
            used_enc = enc
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise UnicodeError(f"{path.name}: 지원 인코딩 {config.ENCODINGS} 모두 실패")

    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        first = next(csv.reader([line]), [])
        if _is_header_row(first):
            header_idx = i
            break
    if header_idx is None:
        raise HeaderNotFoundError(
            f"{path.name}: '\"NO\",\"시군구\"'로 시작하는 헤더 행을 찾지 못했습니다 "
            f"(encoding={used_enc}, 총 {len(lines)}행)"
        )

    df = pd.read_csv(
        io.StringIO("\n".join(lines[header_idx:])),
        dtype=str, keep_default_na=False, na_filter=False,
    )
    info = TableInfo(
        source_file=str(path.relative_to(config.PROJECT_ROOT) if path.is_relative_to(config.PROJECT_ROOT) else path),
        source_sheet=None, encoding=used_enc, header_line_index=header_idx,
        header_inferred=False, conditions=_parse_conditions(lines[:header_idx]),
    )
    hint = info.conditions.get("실거래구분", "") + path.name
    df, info.housing_category = _standardize_columns(df, hint)
    info.n_rows = len(df)
    return df, info


# ------------------------------------------------------------------ XLSX
def read_xlsx_tables(path: Path) -> list[tuple[pd.DataFrame, TableInfo]]:
    """XLSX의 모든 시트를 읽는다. 시트마다 (DataFrame, TableInfo)를 반환."""
    import openpyxl

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "Workbook contains no default style"
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    out = []
    try:
        for ws in wb.worksheets:
            ws.reset_dimensions()  # MOLIT xlsx는 dimension이 A1로 잘못 기록돼 있음
            rows = [tuple(_cell_to_str(v) for v in r) for r in ws.iter_rows(values_only=True)]
            rows = [r for r in rows if any(c.strip() for c in r)]
            src = str(path.relative_to(config.PROJECT_ROOT) if path.is_relative_to(config.PROJECT_ROOT) else path)
            info = TableInfo(source_file=src, source_sheet=ws.title, encoding="xlsx",
                             header_line_index=None, header_inferred=False)
            try:
                df = _rows_to_frame(rows, info, path)
                hint = info.conditions.get("실거래구분", "") + ws.title + path.name
                df, info.housing_category = _standardize_columns(df, hint)
                info.n_rows = len(df)
                out.append((df, info))
            except HeaderNotFoundError as e:
                info.ok, info.error = False, str(e)
                log.error("  [%s / %s] %s", path.name, ws.title, e)
                out.append((None, info))
    finally:
        wb.close()
    return out


def _rows_to_frame(rows, info: TableInfo, path: Path) -> pd.DataFrame:
    header_idx = next((i for i, r in enumerate(rows) if _is_header_row(r)), None)
    if header_idx is not None:
        info.header_line_index = header_idx
        info.conditions = _parse_conditions(r[0] for r in rows[:header_idx])
        header = [h.strip() for h in rows[header_idx]]
        body = rows[header_idx + 1:]
    else:
        # 헤더 없는 시트: 첫 행이 데이터처럼 생겼으면 컬럼 수로 schema를 추론
        if not rows:
            raise HeaderNotFoundError(f"{path.name}: 빈 시트")
        # 뒤쪽 빈 셀(종전계약 등)도 컬럼이므로 시트 전체의 최대 행 폭을 사용
        ncol = max(len(r) for r in rows)
        looks_like_data = rows[0][0].isdigit() and rows[0][1].startswith(("서울", "경기", "인천"))
        if looks_like_data and ncol == len(config.OFFICETEL_RAW_COLUMNS):
            header = config.OFFICETEL_RAW_COLUMNS
        elif looks_like_data and ncol == len(config.RAW_COLUMNS):
            header = config.RAW_COLUMNS
        else:
            raise HeaderNotFoundError(
                f"{path.name}: 헤더 행('NO','시군구')을 찾지 못했고 컬럼 수({ncol})로 schema를 추론할 수 없습니다"
            )
        info.header_inferred = True
        body = rows
        log.warning("  [%s / %s] 헤더 행 없음 → 컬럼 수 %d로 schema 추론", path.name, info.source_sheet, ncol)

    # 수작업 병합 시트 내부에 반복된 헤더 행 제거
    repeated = [r for r in body if _is_header_row(r)]
    info.n_repeated_header_rows = len(repeated)
    body = [r[:len(header)] + ("",) * max(0, len(header) - len(r)) for r in body if not _is_header_row(r)]
    return pd.DataFrame(body, columns=header, dtype=str)


# ------------------------------------------------------------------ entry
def load_all(raw_dir: Path) -> tuple[pd.DataFrame, list[TableInfo]]:
    """raw_dir 하위 모든 CSV/XLSX를 읽어 하나의 DataFrame으로 병합한다.

    각 행에는 source_file / source_sheet / source_table_id / source_cond_* /
    housing_category 메타 컬럼이 붙는다. 원본 한글 컬럼명은 그대로 유지한다.
    """
    files = discover_raw_files(raw_dir)
    if not files:
        raise FileNotFoundError(f"{raw_dir} 에서 CSV/XLSX 파일을 찾지 못했습니다")
    log.info("원본 파일 %d개 발견 (%s)", len(files), raw_dir)

    frames, infos = [], []
    for path in files:
        try:
            if path.suffix.lower() == ".csv":
                tables = [read_molit_csv(path)]
            else:
                tables = read_xlsx_tables(path)
        except (HeaderNotFoundError, UnicodeError) as e:
            log.error("  [%s] 파싱 실패: %s", path.name, e)
            infos.append(TableInfo(str(path), None, None, None, False, ok=False, error=str(e)))
            continue
        for df, info in tables:
            infos.append(info)
            if df is None:
                continue
            table_id = len(infos) - 1
            df["source_file"] = info.source_file
            df["source_sheet"] = info.source_sheet
            df["source_table_id"] = table_id
            df["source_cond_gu"] = info.cond_gu
            df["source_cond_period"] = info.cond_period
            df["housing_category"] = info.housing_category
            frames.append(df)
            log.info(
                "  %-60s sheet=%-12s enc=%-6s header@%-4s cond_gu=%-6s period=%-25s cat=%-6s rows=%6d%s",
                info.source_file, info.source_sheet, info.encoding,
                "infer" if info.header_inferred else info.header_line_index,
                info.cond_gu, info.cond_period, info.housing_category, info.n_rows,
                f" (반복헤더 {info.n_repeated_header_rows}행 제거)" if info.n_repeated_header_rows else "",
            )
    if not frames:
        raise RuntimeError("정상적으로 파싱된 파일이 없습니다")
    merged = pd.concat(frames, ignore_index=True)
    return merged, infos


def tables_to_frame(infos: list[TableInfo]) -> pd.DataFrame:
    return pd.DataFrame([{
        "source_file": i.source_file, "source_sheet": i.source_sheet, "encoding": i.encoding,
        "header_line_index": i.header_line_index, "header_inferred": i.header_inferred,
        "cond_gu": i.cond_gu, "cond_period": i.cond_period, "housing_category": i.housing_category,
        "n_rows": i.n_rows, "n_repeated_header_rows": i.n_repeated_header_rows,
        "ok": i.ok, "error": i.error,
    } for i in infos])
