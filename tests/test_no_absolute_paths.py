import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = (
    "src",
    "configs",
    "julia",
    "tests",
    "scripts",
    "Makefile",
    "pyproject.toml",
    "README.md",
)
SKIP_SUFFIXES = {".pyc", ".parquet", ".png"}
FORBIDDEN = re.compile(r"(/Users/|/home/|[Cc]:\\Users|/Volumes/)")

SELF = Path(__file__).resolve()


def _files() -> list[Path]:
    out: list[Path] = []
    for name in SCAN_DIRS:
        p = ROOT / name
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            out.extend(f for f in p.rglob("*") if f.is_file() and f.suffix not in SKIP_SUFFIXES)
    return out


def test_no_personal_absolute_paths_in_repo_sources() -> None:
    offenders: list[str] = []
    for f in _files():
        if "__pycache__" in f.parts or f.resolve() == SELF:
            continue  # the test file itself contains the forbidden pattern
        text = f.read_text(encoding="utf-8", errors="ignore")
        for m in FORBIDDEN.finditer(text):
            line_no = text.count("\n", 0, m.start()) + 1
            offenders.append(f"{f.relative_to(ROOT)}:{line_no}")
    assert offenders == [], f"personal absolute paths found: {offenders}"
