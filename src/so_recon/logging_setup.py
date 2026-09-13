"""Logging rules: every record carries run_id; stderr + per-run file; no data rows in logs."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

LOGGER_NAME = "so_recon"
_FORMAT = "%(asctime)s %(levelname)s run=%(run_id)s %(name)s: %(message)s"


class _RunIdFilter(logging.Filter):
    def __init__(self, run_id: str) -> None:
        super().__init__()
        self.run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = self.run_id
        return True


def configure_logging(
    run_id: str, log_file: Path | None = None, level: int = logging.INFO
) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
    for f in list(logger.filters):
        logger.removeFilter(f)
    logger.setLevel(level)
    logger.propagate = False
    formatter = logging.Formatter(_FORMAT)
    run_filter = _RunIdFilter(run_id)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    for h in handlers:
        h.setFormatter(formatter)
        h.addFilter(run_filter)
        logger.addHandler(h)
    return logger
