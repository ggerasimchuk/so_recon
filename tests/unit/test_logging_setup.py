import logging
from pathlib import Path

from so_recon.logging_setup import configure_logging


def test_log_lines_carry_run_id_and_go_to_file(tmp_path: Path) -> None:
    log_file = tmp_path / "run.log"
    logger = configure_logging("20260913T120000Z-smoke-deadbeef", log_file)
    logger.info("hello")
    logging.getLogger("so_recon.child").warning("child message")
    for h in logger.handlers:
        h.flush()
    text = log_file.read_text(encoding="utf-8")
    assert "run=20260913T120000Z-smoke-deadbeef" in text
    assert "hello" in text
    assert "child message" in text


def test_reconfigure_replaces_handlers(tmp_path: Path) -> None:
    configure_logging("run-a", tmp_path / "a.log")
    logger = configure_logging("run-b", tmp_path / "b.log")
    assert len([h for h in logger.handlers if isinstance(h, logging.FileHandler)]) == 1
