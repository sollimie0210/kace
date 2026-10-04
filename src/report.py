"""전처리 과정의 건수/경고를 모으는 report 객체와 로깅 설정."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

log = logging.getLogger("preprocess")


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("preprocess")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


class Report:
    """summary(json용 key-value)와 quality(검사 항목별 건수 표)를 함께 관리."""

    def __init__(self):
        self.summary: dict = {}
        self.quality: list[dict] = []

    def set(self, key, value):
        self.summary[key] = value

    def check(self, stage: str, check: str, n: int, action: str, detail: str = "", level="INFO"):
        """품질 검사 결과 한 줄을 기록하고 로그로 출력."""
        n = int(n)
        self.quality.append({"stage": stage, "check": check, "n_rows": n,
                             "action": action, "detail": detail})
        lvl = logging.WARNING if (level == "WARNING" and n > 0) else logging.INFO
        log.log(lvl, "  [%s] %-45s %8s  -> %s %s", stage, check, f"{n:,}", action,
                f"({detail})" if detail else "")
