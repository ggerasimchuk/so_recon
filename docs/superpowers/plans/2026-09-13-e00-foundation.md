# E00 Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Создать воспроизводимое основание проекта SO-RECON: Python/Julia-окружение с lock-файлами, typed configuration, каталоги данных, manifest исходных файлов, run registry с полной lineage, единый CLI и один детерминированный smoke fixture, который проходит на чистой установке.

**Architecture:** Python-пакет `so_recon` (src-layout, `uv`, pydantic-конфиги) отвечает за конфигурацию, hashing, manifests, run registry и CLI. Julia-окружение `julia/` с зафиксированным `Manifest.toml` держит JutulDarcy и минимальный smoke-скрипт, который Python вызывает через subprocess и читает JSON. Каждый запуск CLI получает `run_id`, каталог `artifacts/runs/<run_id>/` с `run.json`, `resolved_config.json` и логом; ни один модуль не содержит личных абсолютных путей.

**Tech Stack:** Python 3.13 (`uv`, `pydantic>=2`, `pyyaml`, `numpy`, `pyarrow`, `pytest`, `ruff`, `mypy`), Julia 1.12 (`juliaup`, `JutulDarcy 0.3.x`, `Jutul 0.4.x`, `JSON`), Make, git.

**Spec:** `docs/SPEC.md` (v3.0, разделы 7.1, 19, 20), `docs/STAGES.md` (v3.0, раздел «E00»), `docs/DATA_AUDIT.md` (раздел 3 — контракты чтения файлов), `docs/README.md`.

## Global Constraints

Скопировано из спецификации; каждая задача обязана их соблюдать.

- `SPEC.md` версия **3.0**, `STAGES.md` версия **3.0**. Любое расхождение с данными «требует новой версии спецификации, а не скрытой адаптации кода».
- Gate E00 (STAGES): «чистая установка выполняет один детерминированный fixture; версии и hashes сохраняются; код не зависит от личных абсолютных путей».
- Не входит в E00 (STAGES): «чтение всех промысловых данных, построение модели пласта, обучение ML».
- Ключевые выходы E00 (STAGES): `pyproject.toml` и lock-файл Python; `Project.toml` и `Manifest.toml` Julia; каркас `src/so_recon/`, `julia/`, `tests/`, `configs/`, `reports/`; `source_manifest.json`; `environment_report.md`; `reports/stages/E00.md`.
- SPEC 19.1: Python — «ETL, data contracts, … artifact registry, reporting»; Julia/JutulDarcy — «forward simulation, state/restart, adjoint sensitivities»; configuration — «versioned YAML/TOML validated by typed schemas».
- SPEC 19.2: «Модуль не должен читать произвольные raw CSV вне `data`». В E00 raw CSV не читаются содержательно — только хешируются и считаются строки.
- SPEC 19.10: «Resolved config сохраняется рядом с output; environment variables/secrets не меняют scientific parameters».
- SPEC 19.11: «Pickle как единственный долговременный scientific artifact запрещён». Форматы: Parquet, Zarr, NetCDF, JSON/YAML/TOML.
- SPEC 19.12: каждый output хранит «SHA-256 raw inputs; source/canonical schema versions; git commit; environment lock hash; Jutul/Julia version; model checkpoint hash; resolved config hash; parent artifact IDs; creation timestamp; command/run ID».
- SPEC 19.13: при фиксированных seeds воспроизводятся «data transformations bitwise или within declared tolerance».
- SPEC 19.15: «Исходные закрытые данные не включаются в публичный release»; «Logs не должны печатать полный закрытый dataset или credentials».
- SPEC 7.1: «Удаление записи без записи причины запрещено» — в E00 это означает: отсутствующий исходный файл вызывает явную ошибку, а не молчаливый пропуск.
- STAGES §1: отчёт `reports/stages/E00.md` фиксирует «версия SPEC.md и конфигурации; входные артефакты и их hashes; выполненные команды; результаты тестов и численных проверок; созданные артефакты; найденные ограничения; статус этапа; что разрешено передать следующему этапу».
- Статусы этапа: `NOT_RUN`, `IN_PROGRESS`, `PASS`, `PASS_WITH_LIMITATIONS`, `FAIL`.

## Замечания о фактическом состоянии репозитория (на 13.09.2026)

- Репозиторий содержит только `docs/`, `data/`, `.idea/`, `.gitignore` (`/data/`, `/research/`). Кода нет. Ветка `master`.
- Raw CSV лежат в `data/Ромашка_сырые/` (пять файлов). Каталог `data/` целиком в `.gitignore` — закрытые данные не версионируются, это соответствует SPEC 19.15.
- Локально: Python 3.13.2 (pyenv), `uv 0.12.7`, Homebrew, Docker, `git-lfs`. **Julia и `juliaup` не установлены.**
- Актуальные версии на дату плана: Julia stable **1.12.7**, LTS 1.10.12; JutulDarcy **0.3.11** (compat `julia = "1.7"`, `Jutul = "0.4.31"`).
- `STAGES.md` §8 ссылается на `/docs/README_SO_RECON.md`, фактический файл — `docs/README.md`. В коде ничего не ломает; зафиксировать в отчёте E00 как замечание к документации.

## Решения, принятые в плане (не меняют SPEC)

1. **Raw-каталог.** Канонический каталог raw — `data/raw/`. Пять CSV переносятся из `data/Ромашка_сырые/` в `data/raw/` командой `mv` (Task 6). Это локальная операция над неверсионированными файлами; целостность подтверждается SHA-256 в manifest. Дополнительные файлы (`so_data_dossier.md`, PDF) остаются в `data/`.
2. **Что версионируется.** `reports/` (включая `reports/manifests/*.json` и `reports/environment_report.md`) — в git; `artifacts/` и `data/` — нет. Manifest содержит только имена файлов, размеры и хеши, не содержимое.
3. **Julia-окружение.** `julia/Project.toml` + `julia/Manifest.toml` — общее окружение проекта. Пакет `julia/SOReconSimulator/` создаётся в E05 и будет `dev`-подключён в это же окружение. В E00 — только `julia/smoke/smoke_case.jl`.
4. **Smoke fixture** состоит из двух детерминированных частей: (a) синтетические таблицы `wells` и `well_month` из `numpy.random.default_rng(seed)`, сохранённые в Parquet и захешированные по каноническому JSON содержимого; (b) 1D oil–water JutulDarcy-кейс, возвращающий скалярные summaries. Ожидаемые значения замораживаются в `configs/smoke_expected.json` с относительным допуском `1e-6`.
5. **CLI** — stdlib `argparse`, точка входа `so-recon`. Подкоманды: `manifest`, `env-report`, `smoke`.
6. **Логи** — stdlib `logging`; каждая запись содержит `run_id`; пишутся в консоль и в `artifacts/runs/<run_id>/run.log`. Содержимое строк данных не логируется.
7. **CI** не создаётся в E00: JutulDarcy precompile в CI занимает десятки минут, а remote у репозитория нет. Роль «чистой установки» выполняет `make gate` (удаляет `.venv`, пересоздаёт окружение, прогоняет manifest, env-report, smoke и тесты).

## Структура файлов

```text
so_field/
├── pyproject.toml                  # метаданные, зависимости, ruff/mypy/pytest
├── uv.lock                         # lock Python
├── .python-version                 # 3.13
├── Makefile                        # setup / test / lint / smoke / gate
├── README.md                       # как поставить и прогнать gate
├── .gitignore                      # + .venv, artifacts, кеши
├── configs/
│   ├── project.yml                 # единый typed config (paths, sources, smoke, julia)
│   └── smoke_expected.json         # замороженные ожидания smoke (Task 12)
├── src/so_recon/
│   ├── __init__.py                 # __version__, SPEC_VERSION
│   ├── paths.py                    # find_repo_root, ProjectPaths
│   ├── logging_setup.py            # configure_logging(run_id, log_file)
│   ├── cli.py                      # argparse: manifest | env-report | smoke
│   ├── smoke.py                    # оркестрация smoke: fixture → julia → сравнение с expected
│   ├── config/
│   │   ├── __init__.py
│   │   ├── schema.py               # pydantic-модели (StrictModel, ProjectConfig, ...)
│   │   └── load.py                 # load_project_config, resolved_config_dict, config_hash
│   ├── registry/
│   │   ├── __init__.py
│   │   ├── hashing.py              # sha256_file, sha256_bytes, canonical_json, sha256_json
│   │   ├── gitinfo.py              # git_commit, git_is_dirty
│   │   ├── run.py                  # make_run_id, RunRecord, RunContext, environment_lock_hash
│   │   └── source_manifest.py      # hash_and_count, build_source_manifest, write_source_manifest
│   ├── environment/
│   │   ├── __init__.py
│   │   └── report.py               # parse_julia_manifest, collect_environment, render_markdown
│   ├── synthetic/
│   │   ├── __init__.py
│   │   └── fixture.py              # build_smoke_fixture, write_smoke_fixture
│   └── simulator/
│       ├── __init__.py
│       └── julia_bridge.py         # find_julia, JuliaLauncher, SubprocessJuliaLauncher, run_julia_smoke
├── julia/
│   ├── Project.toml                # JutulDarcy, Jutul, JSON + [compat]
│   ├── Manifest.toml               # зафиксированные версии (генерируется Pkg.instantiate)
│   ├── README.md
│   └── smoke/smoke_case.jl         # 1D oil–water кейс → JSON
├── tests/
│   ├── conftest.py                 # фикстура tmp_project
│   ├── test_no_absolute_paths.py
│   ├── unit/
│   │   ├── test_hashing.py
│   │   ├── test_paths.py
│   │   ├── test_config.py
│   │   ├── test_run_registry.py
│   │   ├── test_logging_setup.py
│   │   ├── test_source_manifest.py
│   │   ├── test_fixture.py
│   │   ├── test_julia_bridge.py
│   │   ├── test_environment_report.py
│   │   ├── test_smoke.py
│   │   └── test_cli.py
│   └── integration/
│       └── test_julia_smoke.py     # @pytest.mark.julia
├── reports/
│   ├── environment_report.md
│   ├── manifests/
│   │   ├── source_manifest.json
│   │   └── environment.json
│   └── stages/E00.md
├── artifacts/.gitkeep              # runs/<run_id>/ создаются в runtime (ignored)
└── data/                           # ignored: raw/ interim/ processed/
```

Ответственности: `registry/*` — всё, что относится к lineage (hashes, git, run record, manifest); `config/*` — схема и загрузка; `environment/*` — отчёт об окружении; `synthetic/fixture.py` — данные smoke; `simulator/julia_bridge.py` — единственное место, знающее, как запускать Julia; `smoke.py` — сценарий; `cli.py` — только парсинг аргументов и вызов сценариев.

---

### Task 1: Каркас Python-проекта, зависимости и lock

**Files:**
- Create: `pyproject.toml`, `.python-version`, `Makefile`, `README.md`, `artifacts/.gitkeep`, `reports/.gitkeep`, `reports/manifests/.gitkeep`, `reports/stages/.gitkeep`, `configs/.gitkeep`
- Create: `src/so_recon/__init__.py`, пустые `__init__.py` в `src/so_recon/{config,registry,environment,synthetic,simulator}/`
- Modify: `.gitignore`
- Test: `tests/unit/test_package.py`

**Interfaces:**
- Produces: `so_recon.__version__ == "0.0.1"`, `so_recon.SPEC_VERSION == "3.0"`; команды `uv sync`, `uv run pytest`, `make test`, `make lint`.

- [ ] **Step 1: Создать `pyproject.toml`**

```toml
[project]
name = "so-recon"
version = "0.0.1"
description = "SO-RECON: physically constrained Bayesian reconstruction of current oil saturation"
readme = "README.md"
requires-python = ">=3.13,<3.14"
license = { text = "Proprietary" }
dependencies = [
  "pydantic>=2.11,<3",
  "pyyaml>=6.0,<7",
  "numpy>=2.2,<3",
  "pyarrow>=19",
]

[project.scripts]
so-recon = "so_recon.cli:main"

[dependency-groups]
dev = [
  "pytest>=8.3",
  "ruff>=0.12",
  "mypy>=1.16",
  "types-PyYAML>=6.0",
]

[build-system]
requires = ["hatchling>=1.27"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/so_recon"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
markers = [
  "julia: requires an instantiated Julia environment with JutulDarcy",
]

[tool.ruff]
line-length = 100
target-version = "py313"
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "PTH"]

[tool.mypy]
python_version = "3.13"
strict = true
plugins = ["pydantic.mypy"]
mypy_path = "src"
packages = ["so_recon"]

[[tool.mypy.overrides]]
module = ["pyarrow", "pyarrow.*"]
ignore_missing_imports = true
```

- [ ] **Step 2: Создать `.python-version`, `.gitignore`, `Makefile`, `README.md`, `.gitkeep`**

`.python-version`:
```text
3.13
```

`.gitignore` (полностью заменить содержимое):
```gitignore
# closed field data and private research notes
/data/
/research/

# python
.venv/
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
.ruff_cache/
dist/
build/

# run artifacts (lineage lives in run.json inside; never committed)
/artifacts/*
!/artifacts/.gitkeep

# os / ide
.DS_Store
```

`Makefile`:
```makefile
UV ?= uv
JULIA ?= julia

.PHONY: setup setup-julia test lint smoke manifest env-report gate

setup:
	$(UV) sync --frozen

setup-julia:
	$(JULIA) --project=julia --startup-file=no -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

test:
	$(UV) run pytest -q

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .
	$(UV) run mypy

manifest:
	$(UV) run so-recon manifest

env-report:
	$(UV) run so-recon env-report

smoke:
	$(UV) run so-recon smoke

# E00 gate: clean Python environment, locked Julia environment, manifests, smoke, tests.
gate:
	rm -rf .venv
	$(UV) sync --frozen
	$(MAKE) setup-julia
	$(UV) run so-recon manifest
	$(UV) run so-recon env-report
	$(UV) run so-recon smoke
	$(UV) run pytest -q
	$(UV) run ruff check .
	$(UV) run mypy
```

`README.md`:
```markdown
# SO-RECON

Физически ограниченная байесовская реконструкция текущей нефтенасыщенности.
Нормативные документы: `docs/SPEC.md` (v3.0), `docs/STAGES.md`, `docs/DATA_AUDIT.md`, `docs/RESEARCH.md`.

## Требования

- Python 3.13 и [`uv`](https://docs.astral.sh/uv/)
- Julia 1.12 через [`juliaup`](https://github.com/JuliaLang/juliaup)

## Установка

    make setup          # Python: uv sync --frozen
    make setup-julia    # Julia: Pkg.instantiate() по julia/Manifest.toml

## Данные

Закрытые исходные CSV лежат в `data/raw/` (каталог не версионируется).
Ожидаемые файлы перечислены в `configs/project.yml` → `sources.files`.
`so-recon manifest` записывает их SHA-256 и число строк в `reports/manifests/source_manifest.json`.

## Команды

    uv run so-recon manifest      # manifest исходных файлов
    uv run so-recon env-report    # reports/environment_report.md
    uv run so-recon smoke         # детерминированный fixture + JutulDarcy smoke
    make test                     # pytest
    make lint                     # ruff + mypy
    make gate                     # чистая установка + полный gate E00

Каждый запуск создаёт `artifacts/runs/<run_id>/` с `run.json`, `resolved_config.json`, `run.log`.
```

Создать пустые файлы: `artifacts/.gitkeep`, `reports/.gitkeep`, `reports/manifests/.gitkeep`, `reports/stages/.gitkeep`, `configs/.gitkeep`.

- [ ] **Step 3: Создать пакет и падающий тест**

`src/so_recon/__init__.py`:
```python
"""SO-RECON: physically constrained Bayesian reconstruction of current oil saturation."""

__version__ = "0.0.1"
SPEC_VERSION = "3.0"
```

Пустые `src/so_recon/config/__init__.py`, `src/so_recon/registry/__init__.py`, `src/so_recon/environment/__init__.py`, `src/so_recon/synthetic/__init__.py`, `src/so_recon/simulator/__init__.py` (каждый — одна строка docstring, например `"""Typed configuration."""`).

`tests/unit/__init__.py` и `tests/integration/__init__.py` — пустые. `tests/__init__.py` — пустой.

`tests/unit/test_package.py`:
```python
import so_recon


def test_version_and_spec_version() -> None:
    assert so_recon.__version__ == "0.0.1"
    assert so_recon.SPEC_VERSION == "3.0"
```

- [ ] **Step 4: Установить окружение и запустить тест**

```bash
cd /Users/george/Documents/so_field && uv lock && uv sync && uv run pytest -q
```
Ожидается: `1 passed`; появился `uv.lock` и `.venv/`.

- [ ] **Step 5: Проверить lint и mypy на пустом каркасе**

```bash
cd /Users/george/Documents/so_field && uv run ruff check . && uv run ruff format --check . && uv run mypy
```
Ожидается: `All checks passed!`, `Success: no issues found`.

- [ ] **Step 6: Commit**

```bash
cd /Users/george/Documents/so_field && git add pyproject.toml uv.lock .python-version Makefile README.md .gitignore src tests artifacts/.gitkeep reports configs && git commit -m "feat(e00): scaffold python project with uv lock, make targets and package skeleton

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Hashing и канонический JSON

**Files:**
- Create: `src/so_recon/registry/hashing.py`
- Test: `tests/unit/test_hashing.py`

**Interfaces:**
- Produces:
  - `sha256_bytes(data: bytes) -> str` — hex-строка 64 символа.
  - `sha256_file(path: Path, chunk_size: int = 1 << 20) -> str` — потоковый SHA-256.
  - `canonical_json(obj: object) -> str` — `sort_keys=True`, `separators=(",", ":")`, `ensure_ascii=False`, `allow_nan=False`.
  - `sha256_json(obj: object) -> str` — `sha256_bytes(canonical_json(obj).encode("utf-8"))`.

- [ ] **Step 1: Написать падающий тест**

`tests/unit/test_hashing.py`:
```python
import hashlib
from pathlib import Path

import pytest

from so_recon.registry.hashing import canonical_json, sha256_bytes, sha256_file, sha256_json


def test_sha256_bytes_matches_hashlib() -> None:
    assert sha256_bytes(b"abc") == hashlib.sha256(b"abc").hexdigest()


def test_sha256_file_streams_large_file(tmp_path: Path) -> None:
    p = tmp_path / "big.bin"
    payload = b"x" * (3 * 1024 * 1024 + 17)
    p.write_bytes(payload)
    assert sha256_file(p, chunk_size=1024) == hashlib.sha256(payload).hexdigest()


def test_canonical_json_is_key_order_independent() -> None:
    a = canonical_json({"b": 1, "a": [1, 2, {"z": None, "y": "ё"}]})
    b = canonical_json({"a": [1, 2, {"y": "ё", "z": None}], "b": 1})
    assert a == b
    assert a == '{"a":[1,2,{"y":"ё","z":null}],"b":1}'


def test_canonical_json_rejects_nan() -> None:
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})


def test_sha256_json_is_stable() -> None:
    assert sha256_json({"k": 1}) == sha256_bytes(b'{"k":1}')
```

- [ ] **Step 2: Убедиться, что тест падает**

```bash
cd /Users/george/Documents/so_field && uv run pytest tests/unit/test_hashing.py -q
```
Ожидается: `ModuleNotFoundError: No module named 'so_recon.registry.hashing'`.

- [ ] **Step 3: Реализовать `hashing.py`**

```python
"""Content hashing helpers shared by manifests, run records and fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(obj: object) -> str:
    """Deterministic JSON: sorted keys, no whitespace, UTF-8 text, NaN/Inf forbidden."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_json(obj: object) -> str:
    return sha256_bytes(canonical_json(obj).encode("utf-8"))
```

- [ ] **Step 4: Прогнать тесты**

```bash
cd /Users/george/Documents/so_field && uv run pytest tests/unit/test_hashing.py -q
```
Ожидается: `5 passed`.

- [ ] **Step 5: Commit**

```bash
cd /Users/george/Documents/so_field && git add src/so_recon/registry/hashing.py tests/unit/test_hashing.py && git commit -m "feat(e00): add sha256 and canonical json hashing helpers

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Корень репозитория и каталоги проекта

**Files:**
- Create: `src/so_recon/paths.py`
- Test: `tests/unit/test_paths.py`, `tests/test_no_absolute_paths.py`

**Interfaces:**
- Produces:
  - `class RepoRootNotFoundError(RuntimeError)`
  - `find_repo_root(start: Path | None = None) -> Path` — порядок: env `SO_RECON_ROOT` → подъём от `start` (по умолчанию `Path.cwd()`) → подъём от `__file__`. Маркер корня: есть `pyproject.toml` и каталог `src/so_recon`.
  - `@dataclass(frozen=True) class ProjectPaths` с полями `root, raw, interim, processed, artifacts, reports, configs, julia` (все `Path`, абсолютные) и свойствами `runs = artifacts/"runs"`, `manifests = reports/"manifests"`, `stages = reports/"stages"`; методы `relative(path: Path) -> str` (posix относительно root), `ensure_dirs() -> None` (создаёт raw/interim/processed/runs/manifests/stages), classmethod `ProjectPaths.default(root: Path) -> ProjectPaths` (стандартные подкаталоги `data/raw`, `data/interim`, `data/processed`, `artifacts`, `reports`, `configs`, `julia`).
- Примечание: конфигурируемые подкаталоги подключаются в Task 4 через `ProjectPaths.from_config`.

- [ ] **Step 1: Написать падающие тесты**

`tests/unit/test_paths.py`:
```python
from pathlib import Path

import pytest

from so_recon.paths import ProjectPaths, RepoRootNotFoundError, find_repo_root


def _make_root(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    (tmp_path / "src" / "so_recon").mkdir(parents=True)
    return tmp_path


def test_find_repo_root_walks_up_from_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SO_RECON_ROOT", raising=False)
    root = _make_root(tmp_path)
    nested = root / "a" / "b"
    nested.mkdir(parents=True)
    assert find_repo_root(nested) == root.resolve()


def test_find_repo_root_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _make_root(tmp_path)
    monkeypatch.setenv("SO_RECON_ROOT", str(root))
    assert find_repo_root(Path("/")) == root.resolve()


def test_find_repo_root_raises_without_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SO_RECON_ROOT", raising=False)
    with pytest.raises(RepoRootNotFoundError):
        find_repo_root(tmp_path)


def test_default_paths_and_relative(tmp_path: Path) -> None:
    paths = ProjectPaths.default(tmp_path)
    assert paths.raw == tmp_path / "data" / "raw"
    assert paths.runs == tmp_path / "artifacts" / "runs"
    assert paths.manifests == tmp_path / "reports" / "manifests"
    assert paths.stages == tmp_path / "reports" / "stages"
    assert paths.relative(paths.raw / "mer.csv") == "data/raw/mer.csv"


def test_ensure_dirs_creates_runtime_dirs(tmp_path: Path) -> None:
    paths = ProjectPaths.default(tmp_path)
    paths.ensure_dirs()
    for p in (paths.raw, paths.interim, paths.processed, paths.runs, paths.manifests, paths.stages):
        assert p.is_dir()
```

`tests/test_no_absolute_paths.py` — gate «код не зависит от личных абсолютных путей»:
```python
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = ("src", "configs", "julia", "tests", "Makefile", "pyproject.toml", "README.md")
SKIP_SUFFIXES = {".pyc", ".parquet", ".png"}
FORBIDDEN = re.compile(r"(/Users/|/home/|C:\\Users|/Volumes/)")


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
```

- [ ] **Step 2: Убедиться, что тесты падают**

```bash
cd /Users/george/Documents/so_field && uv run pytest tests/unit/test_paths.py tests/test_no_absolute_paths.py -q
```
Ожидается: `ModuleNotFoundError: so_recon.paths` для первого файла; второй проходит (нарушений пока нет).

- [ ] **Step 3: Реализовать `paths.py`**

```python
"""Repository root discovery and canonical project directories."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT_ENV_VAR = "SO_RECON_ROOT"


class RepoRootNotFoundError(RuntimeError):
    """Raised when no directory with pyproject.toml + src/so_recon is found."""


def _is_root(p: Path) -> bool:
    return (p / "pyproject.toml").is_file() and (p / "src" / "so_recon").is_dir()


def _walk_up(start: Path) -> Path | None:
    start = start.resolve()
    for candidate in (start, *start.parents):
        if _is_root(candidate):
            return candidate
    return None


def find_repo_root(start: Path | None = None) -> Path:
    env = os.environ.get(ROOT_ENV_VAR)
    if env:
        return Path(env).resolve()
    for origin in (start or Path.cwd(), Path(__file__)):
        found = _walk_up(origin)
        if found is not None:
            return found
    raise RepoRootNotFoundError(
        f"cannot locate repository root (pyproject.toml + src/so_recon); set {ROOT_ENV_VAR}"
    )


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    raw: Path
    interim: Path
    processed: Path
    artifacts: Path
    reports: Path
    configs: Path
    julia: Path

    @classmethod
    def default(cls, root: Path) -> ProjectPaths:
        root = root.resolve()
        return cls(
            root=root,
            raw=root / "data" / "raw",
            interim=root / "data" / "interim",
            processed=root / "data" / "processed",
            artifacts=root / "artifacts",
            reports=root / "reports",
            configs=root / "configs",
            julia=root / "julia",
        )

    @property
    def runs(self) -> Path:
        return self.artifacts / "runs"

    @property
    def manifests(self) -> Path:
        return self.reports / "manifests"

    @property
    def stages(self) -> Path:
        return self.reports / "stages"

    def relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.root).as_posix()

    def ensure_dirs(self) -> None:
        for p in (self.raw, self.interim, self.processed, self.runs, self.manifests, self.stages):
            p.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 4: Прогнать тесты**

```bash
cd /Users/george/Documents/so_field && uv run pytest tests/unit/test_paths.py tests/test_no_absolute_paths.py -q
```
Ожидается: `6 passed`.

- [ ] **Step 5: Commit**

```bash
cd /Users/george/Documents/so_field && git add src/so_recon/paths.py tests/unit/test_paths.py tests/test_no_absolute_paths.py && git commit -m "feat(e00): add repo root discovery, project paths and absolute-path guard test

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Typed configuration (pydantic) и `configs/project.yml`

**Files:**
- Create: `src/so_recon/config/schema.py`, `src/so_recon/config/load.py`, `configs/project.yml`
- Modify: `src/so_recon/paths.py` (добавить `ProjectPaths.from_config`)
- Test: `tests/unit/test_config.py`

**Interfaces:**
- Consumes: `sha256_json` (Task 2), `ProjectPaths` (Task 3).
- Produces (`so_recon.config.schema`):
  - `class StrictModel(BaseModel)` — `extra="forbid"`, `frozen=True`.
  - `class PathsConfig(StrictModel)`: `raw: str = "data/raw"`, `interim: str = "data/interim"`, `processed: str = "data/processed"`, `artifacts: str = "artifacts"`, `reports: str = "reports"`, `configs: str = "configs"`, `julia: str = "julia"`; валидатор: значения относительные.
  - `class SourceFileSpec(StrictModel)`: `name: str`, `path: str`, `encoding: Literal["utf-8", "utf-8-sig", "cp1251"]`, `delimiter: Literal[",", ";"]`, `decimal: Literal[".", ","]`, `required: bool = True`.
  - `class SourcesConfig(StrictModel)`: `files: list[SourceFileSpec]`.
  - `class SmokeFixtureConfig(StrictModel)`: `seed: int = 20260913`, `n_wells: int = 4`, `n_months: int = 12`, `nx: int = 20`, `n_steps: int = 12`, `rel_tol: float = 1e-6`.
  - `class JuliaConfig(StrictModel)`: `project: str = "julia"`, `smoke_script: str = "julia/smoke/smoke_case.jl"`, `timeout_s: int = 1800`.
  - `class ProjectConfig(StrictModel)`: `spec_version: Literal["3.0"]`, `config_version: str`, `project_name: str = "SO-RECON"`, `paths: PathsConfig`, `sources: SourcesConfig`, `smoke: SmokeFixtureConfig`, `julia: JuliaConfig`.
- Produces (`so_recon.config.load`):
  - `class ConfigError(ValueError)`
  - `load_project_config(path: Path) -> ProjectConfig`
  - `resolved_config_dict(cfg: ProjectConfig) -> dict[str, Any]` — `cfg.model_dump(mode="json")`.
  - `config_hash(cfg: ProjectConfig) -> str` — `sha256_json(resolved_config_dict(cfg))`.
- Produces (`so_recon.paths`): `ProjectPaths.from_config(root: Path, cfg: PathsConfig) -> ProjectPaths`.

- [ ] **Step 1: Написать падающие тесты**

`tests/unit/test_config.py`:
```python
from pathlib import Path

import pytest
from pydantic import ValidationError

from so_recon.config.load import ConfigError, config_hash, load_project_config, resolved_config_dict
from so_recon.config.schema import PathsConfig, ProjectConfig, SourceFileSpec, SourcesConfig
from so_recon.paths import ProjectPaths

MINIMAL_YAML = """
spec_version: "3.0"
config_version: "test.1"
paths: {}
sources:
  files:
    - {name: coords, path: data/raw/coords.csv, encoding: utf-8, delimiter: ",", decimal: "."}
smoke: {}
julia: {}
"""


def test_load_minimal_config(tmp_path: Path) -> None:
    p = tmp_path / "project.yml"
    p.write_text(MINIMAL_YAML, encoding="utf-8")
    cfg = load_project_config(p)
    assert cfg.spec_version == "3.0"
    assert cfg.paths.raw == "data/raw"
    assert cfg.sources.files[0].encoding == "utf-8"
    assert cfg.smoke.seed == 20260913
    assert cfg.julia.smoke_script == "julia/smoke/smoke_case.jl"


def test_unknown_field_is_rejected(tmp_path: Path) -> None:
    p = tmp_path / "project.yml"
    p.write_text(MINIMAL_YAML + "unexpected: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_project_config(p)


def test_wrong_spec_version_is_rejected(tmp_path: Path) -> None:
    p = tmp_path / "project.yml"
    p.write_text(MINIMAL_YAML.replace('"3.0"', '"2.9"'), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_project_config(p)


def test_absolute_path_in_paths_config_is_rejected() -> None:
    with pytest.raises(ValidationError):
        PathsConfig(raw="/tmp/raw")


def test_absolute_source_path_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SourceFileSpec(name="x", path="/abs/x.csv", encoding="utf-8", delimiter=",", decimal=".")


def test_config_hash_is_order_independent_and_stable(tmp_path: Path) -> None:
    p = tmp_path / "project.yml"
    p.write_text(MINIMAL_YAML, encoding="utf-8")
    cfg = load_project_config(p)
    assert config_hash(cfg) == config_hash(load_project_config(p))
    assert resolved_config_dict(cfg)["smoke"]["seed"] == 20260913


def test_project_paths_from_config(tmp_path: Path) -> None:
    cfg = ProjectConfig(
        spec_version="3.0",
        config_version="t",
        paths=PathsConfig(raw="custom/raw"),
        sources=SourcesConfig(files=[]),
    )
    paths = ProjectPaths.from_config(tmp_path, cfg.paths)
    assert paths.raw == tmp_path / "custom" / "raw"
    assert paths.reports == tmp_path / "reports"


def test_repo_config_file_is_valid() -> None:
    repo_cfg = Path(__file__).resolve().parents[2] / "configs" / "project.yml"
    cfg = load_project_config(repo_cfg)
    names = [f.name for f in cfg.sources.files]
    assert names == ["coords", "gis", "mer", "perf", "plastoper"]
```

- [ ] **Step 2: Убедиться, что тесты падают**

```bash
cd /Users/george/Documents/so_field && uv run pytest tests/unit/test_config.py -q
```
Ожидается: `ModuleNotFoundError: so_recon.config.load`.

- [ ] **Step 3: Реализовать `schema.py`**

```python
"""Typed configuration schema (SPEC 19.1: versioned YAML validated by typed schemas)."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _must_be_relative(value: str) -> str:
    if PurePosixPath(value).is_absolute() or value.startswith("~"):
        raise ValueError(f"path must be relative to repository root, got {value!r}")
    return value


class PathsConfig(StrictModel):
    raw: str = "data/raw"
    interim: str = "data/interim"
    processed: str = "data/processed"
    artifacts: str = "artifacts"
    reports: str = "reports"
    configs: str = "configs"
    julia: str = "julia"

    @field_validator("raw", "interim", "processed", "artifacts", "reports", "configs", "julia")
    @classmethod
    def _relative(cls, v: str) -> str:
        return _must_be_relative(v)


class SourceFileSpec(StrictModel):
    name: str
    path: str
    encoding: Literal["utf-8", "utf-8-sig", "cp1251"]
    delimiter: Literal[",", ";"]
    decimal: Literal[".", ","]
    required: bool = True

    @field_validator("path")
    @classmethod
    def _relative(cls, v: str) -> str:
        return _must_be_relative(v)


class SourcesConfig(StrictModel):
    files: list[SourceFileSpec]


class SmokeFixtureConfig(StrictModel):
    seed: int = 20260913
    n_wells: int = Field(default=4, ge=2)
    n_months: int = Field(default=12, ge=1)
    nx: int = Field(default=20, ge=3)
    n_steps: int = Field(default=12, ge=1)
    rel_tol: float = Field(default=1e-6, gt=0)


class JuliaConfig(StrictModel):
    project: str = "julia"
    smoke_script: str = "julia/smoke/smoke_case.jl"
    timeout_s: int = Field(default=1800, ge=1)

    @field_validator("project", "smoke_script")
    @classmethod
    def _relative(cls, v: str) -> str:
        return _must_be_relative(v)


class ProjectConfig(StrictModel):
    spec_version: Literal["3.0"]
    config_version: str
    project_name: str = "SO-RECON"
    paths: PathsConfig = PathsConfig()
    sources: SourcesConfig
    smoke: SmokeFixtureConfig = SmokeFixtureConfig()
    julia: JuliaConfig = JuliaConfig()
```

- [ ] **Step 4: Реализовать `load.py`**

```python
"""Loading and hashing of the project configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from so_recon.config.schema import ProjectConfig
from so_recon.registry.hashing import sha256_json


class ConfigError(ValueError):
    """Configuration file is missing, malformed or violates the schema."""


def load_project_config(path: Path) -> ProjectConfig:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"config root must be a mapping: {path}")
    try:
        return ProjectConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid config {path}:\n{exc}") from exc


def resolved_config_dict(cfg: ProjectConfig) -> dict[str, Any]:
    return cfg.model_dump(mode="json")


def config_hash(cfg: ProjectConfig) -> str:
    return sha256_json(resolved_config_dict(cfg))
```

- [ ] **Step 5: Добавить `ProjectPaths.from_config` в `paths.py`**

В `paths.py` добавить импорт под `TYPE_CHECKING` и classmethod:
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from so_recon.config.schema import PathsConfig
```
и внутри `ProjectPaths`:
```python
    @classmethod
    def from_config(cls, root: Path, cfg: PathsConfig) -> ProjectPaths:
        root = root.resolve()
        return cls(
            root=root,
            raw=root / cfg.raw,
            interim=root / cfg.interim,
            processed=root / cfg.processed,
            artifacts=root / cfg.artifacts,
            reports=root / cfg.reports,
            configs=root / cfg.configs,
            julia=root / cfg.julia,
        )
```

- [ ] **Step 6: Создать `configs/project.yml`** (значения encoding/delimiter/decimal взяты из `DATA_AUDIT.md` §3 и проверены по заголовкам файлов)

```yaml
# SO-RECON project configuration. Validated by so_recon.config.schema.ProjectConfig.
# All paths are relative to the repository root. No personal absolute paths.
spec_version: "3.0"
config_version: "E00.1"
project_name: SO-RECON

paths:
  raw: data/raw
  interim: data/interim
  processed: data/processed
  artifacts: artifacts
  reports: reports
  configs: configs
  julia: julia

sources:
  files:
    - name: coords
      path: data/raw/coords.csv
      encoding: utf-8
      delimiter: ","
      decimal: "."
    - name: gis
      path: data/raw/gis.csv
      encoding: cp1251
      delimiter: ";"
      decimal: ","
    - name: mer
      path: data/raw/mer.csv
      encoding: utf-8-sig
      delimiter: ";"
      decimal: ","
    - name: perf
      path: data/raw/perf.csv
      encoding: cp1251
      delimiter: ";"
      decimal: ","
    - name: plastoper
      path: data/raw/plastoper.csv
      encoding: utf-8-sig
      delimiter: ";"
      decimal: "."

smoke:
  seed: 20260913
  n_wells: 4
  n_months: 12
  nx: 20
  n_steps: 12
  rel_tol: 1.0e-6

julia:
  project: julia
  smoke_script: julia/smoke/smoke_case.jl
  timeout_s: 1800
```
Удалить `configs/.gitkeep`.

- [ ] **Step 7: Прогнать тесты, lint, mypy**

```bash
cd /Users/george/Documents/so_field && uv run pytest tests/unit/test_config.py tests/unit/test_paths.py -q && uv run ruff check . && uv run mypy
```
Ожидается: `13 passed`, без ошибок ruff/mypy.

- [ ] **Step 8: Commit**

```bash
cd /Users/george/Documents/so_field && git add src/so_recon/config src/so_recon/paths.py configs tests/unit/test_config.py && git commit -m "feat(e00): add typed project configuration schema, loader and configs/project.yml

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Run registry (run ID, run record, lineage) и логирование

**Files:**
- Create: `src/so_recon/registry/gitinfo.py`, `src/so_recon/registry/run.py`, `src/so_recon/logging_setup.py`
- Test: `tests/unit/test_run_registry.py`, `tests/unit/test_logging_setup.py`

**Interfaces:**
- Consumes: `sha256_file`, `sha256_bytes`, `sha256_json`, `canonical_json` (Task 2); `ProjectPaths` (Task 3); `ProjectConfig`, `config_hash`, `resolved_config_dict`, `StrictModel` (Task 4).
- Produces (`so_recon.registry.gitinfo`):
  - `git_commit(root: Path) -> str | None` — полный SHA HEAD или `None`, если не git-репозиторий / git недоступен.
  - `git_is_dirty(root: Path) -> bool | None` — `git status --porcelain` не пуст.
- Produces (`so_recon.registry.run`):
  - `RunStatus = Literal["RUNNING", "PASS", "FAIL"]`
  - `make_run_id(command: str, config_hash: str, git_commit: str | None, now: datetime) -> str` — формат `YYYYmmddTHHMMSSZ-<command>-<8 hex>`. Если каталог с таким `run_id` уже существует (повторный запуск в ту же секунду), `RunContext.start` добавляет суффикс `-01`, `-02`, … `-99`; после 99 попыток — `FileExistsError`.
  - `environment_lock_hash(paths: ProjectPaths) -> str` — `sha256_json({"uv.lock": h, "julia/Manifest.toml": h})`, где отсутствующий файл даёт строку `"missing"`.
  - `class RunRecord(StrictModel)` — поля: `run_id: str`, `command: str`, `argv: list[str]`, `created_at: str` (ISO-8601 UTC), `finished_at: str | None = None`, `git_commit: str | None`, `git_dirty: bool | None`, `spec_version: str`, `config_version: str`, `resolved_config_hash: str`, `environment_lock_hash: str`, `python_version: str`, `julia_version: str | None = None`, `jutuldarcy_version: str | None = None`, `model_checkpoint_hash: str | None = None`, `schema_versions: dict[str, str]`, `raw_input_hashes: dict[str, str]`, `parent_run_ids: list[str]`, `status: RunStatus`, `outputs: dict[str, str]`, `notes: list[str]`.
  - `class RunContext` с атрибутами `run_id: str`, `run_dir: Path`, `record: RunRecord`; `RunContext.start(*, command, argv, cfg, paths, parent_run_ids=(), raw_input_hashes=None, schema_versions=None, now=None) -> RunContext` (создаёт `run_dir`, пишет `resolved_config.json` и `run.json`); `update(**fields) -> None`; `finish(status, outputs, notes=()) -> None`; `write() -> None`.
- Produces (`so_recon.logging_setup`): `configure_logging(run_id: str, log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger` — логгер `so_recon`, формат `%(asctime)s %(levelname)s run=%(run_id)s %(name)s: %(message)s`; обработчики предыдущих запусков снимаются.

- [ ] **Step 1: Написать падающие тесты**

`tests/unit/test_run_registry.py`:
```python
import json
import re
from datetime import UTC, datetime
from pathlib import Path

from so_recon.config.schema import ProjectConfig, SourcesConfig
from so_recon.paths import ProjectPaths
from so_recon.registry.run import RunContext, environment_lock_hash, make_run_id


def _cfg() -> ProjectConfig:
    return ProjectConfig(spec_version="3.0", config_version="t.1", sources=SourcesConfig(files=[]))


def test_make_run_id_format_and_determinism() -> None:
    now = datetime(2026, 9, 13, 12, 0, 0, tzinfo=UTC)
    a = make_run_id("smoke", "c" * 64, "abc123", now)
    b = make_run_id("smoke", "c" * 64, "abc123", now)
    assert a == b
    assert re.fullmatch(r"20260913T120000Z-smoke-[0-9a-f]{8}", a)
    assert make_run_id("smoke", "d" * 64, "abc123", now) != a
    assert make_run_id("smoke", "c" * 64, None, now) != a


def test_environment_lock_hash_marks_missing_files(tmp_path: Path) -> None:
    paths = ProjectPaths.default(tmp_path)
    h_missing = environment_lock_hash(paths)
    (tmp_path / "uv.lock").write_text("lock")
    h_with_lock = environment_lock_hash(paths)
    assert h_missing != h_with_lock
    assert h_with_lock == environment_lock_hash(paths)


def test_run_context_writes_lineage_files(tmp_path: Path) -> None:
    paths = ProjectPaths.default(tmp_path)
    cfg = _cfg()
    now = datetime(2026, 9, 13, 12, 0, 0, tzinfo=UTC)
    ctx = RunContext.start(
        command="smoke",
        argv=["so-recon", "smoke"],
        cfg=cfg,
        paths=paths,
        raw_input_hashes={"coords": "a" * 64},
        now=now,
    )
    assert ctx.run_dir == paths.runs / ctx.run_id
    record = json.loads((ctx.run_dir / "run.json").read_text())
    assert record["status"] == "RUNNING"
    assert record["spec_version"] == "3.0"
    assert record["resolved_config_hash"]
    assert record["raw_input_hashes"] == {"coords": "a" * 64}
    assert record["created_at"] == "2026-09-13T12:00:00+00:00"
    resolved = json.loads((ctx.run_dir / "resolved_config.json").read_text())
    assert resolved["config_version"] == "t.1"

    ctx.update(julia_version="1.12.7")
    ctx.finish("PASS", outputs={"fixture": "artifacts/runs/x/fixture"}, notes=["ok"])
    record = json.loads((ctx.run_dir / "run.json").read_text())
    assert record["status"] == "PASS"
    assert record["julia_version"] == "1.12.7"
    assert record["outputs"] == {"fixture": "artifacts/runs/x/fixture"}
    assert record["finished_at"] is not None


def test_run_context_suffixes_same_second_runs(tmp_path: Path) -> None:
    paths = ProjectPaths.default(tmp_path)
    now = datetime(2026, 9, 13, 12, 0, 0, tzinfo=UTC)
    first = RunContext.start(command="x", argv=[], cfg=_cfg(), paths=paths, now=now)
    second = RunContext.start(command="x", argv=[], cfg=_cfg(), paths=paths, now=now)
    third = RunContext.start(command="x", argv=[], cfg=_cfg(), paths=paths, now=now)
    assert second.run_id == f"{first.run_id}-01"
    assert third.run_id == f"{first.run_id}-02"
    assert second.record.run_id == second.run_id
    assert (paths.runs / second.run_id / "run.json").is_file()
```

`tests/unit/test_logging_setup.py`:
```python
import logging
from pathlib import Path

from so_recon.logging_setup import configure_logging


def test_log_lines_carry_run_id_and_go_to_file(tmp_path: Path) -> None:
    log_file = tmp_path / "run.log"
    logger = configure_logging("20260913T120000Z-smoke-deadbeef", log_file)
    logger.info("hello")
    child = logging.getLogger("so_recon.child")
    child.warning("child message")
    for h in logger.handlers:
        h.flush()
    text = log_file.read_text()
    assert "run=20260913T120000Z-smoke-deadbeef" in text
    assert "hello" in text
    assert "child message" in text


def test_reconfigure_replaces_handlers(tmp_path: Path) -> None:
    configure_logging("run-a", tmp_path / "a.log")
    logger = configure_logging("run-b", tmp_path / "b.log")
    assert len([h for h in logger.handlers if isinstance(h, logging.FileHandler)]) == 1
```

- [ ] **Step 2: Убедиться, что тесты падают**

```bash
cd /Users/george/Documents/so_field && uv run pytest tests/unit/test_run_registry.py tests/unit/test_logging_setup.py -q
```
Ожидается: `ModuleNotFoundError` для `so_recon.registry.run` и `so_recon.logging_setup`.

- [ ] **Step 3: Реализовать `gitinfo.py`**

```python
"""Git provenance helpers (SPEC 19.12: git commit in every output)."""

from __future__ import annotations

import subprocess
from pathlib import Path


def _git(root: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=False, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout


def git_commit(root: Path) -> str | None:
    out = _git(root, "rev-parse", "HEAD")
    return out.strip() if out else None


def git_is_dirty(root: Path) -> bool | None:
    out = _git(root, "status", "--porcelain")
    if out is None:
        return None
    return bool(out.strip())
```

- [ ] **Step 4: Реализовать `run.py`**

```python
"""Run identity and lineage record (SPEC 19.12, 19.13, 20.9)."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from so_recon import SPEC_VERSION
from so_recon.config.load import config_hash, resolved_config_dict
from so_recon.config.schema import ProjectConfig, StrictModel
from so_recon.paths import ProjectPaths
from so_recon.registry.gitinfo import git_commit, git_is_dirty
from so_recon.registry.hashing import canonical_json, sha256_bytes, sha256_file, sha256_json

RunStatus = Literal["RUNNING", "PASS", "FAIL"]

LOCK_FILES = ("uv.lock", "julia/Manifest.toml")


def make_run_id(command: str, config_hash: str, git_commit: str | None, now: datetime) -> str:
    stamp = now.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    short = sha256_bytes(f"{command}|{config_hash}|{git_commit or 'nogit'}".encode())[:8]
    return f"{stamp}-{command}-{short}"


def environment_lock_hash(paths: ProjectPaths) -> str:
    parts: dict[str, str] = {}
    for rel in LOCK_FILES:
        p = paths.root / rel
        parts[rel] = sha256_file(p) if p.is_file() else "missing"
    return sha256_json(parts)


def _claim_run_dir(runs_root: Path, base_id: str, max_suffix: int = 99) -> tuple[str, Path]:
    """Create a unique run directory; same-second reruns get -01, -02, ... suffixes."""
    for i in range(max_suffix + 1):
        run_id = base_id if i == 0 else f"{base_id}-{i:02d}"
        run_dir = runs_root / run_id
        try:
            run_dir.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            continue
        return run_id, run_dir
    raise FileExistsError(f"more than {max_suffix} runs with id {base_id} in one second")


class RunRecord(StrictModel):
    run_id: str
    command: str
    argv: list[str]
    created_at: str
    finished_at: str | None = None
    git_commit: str | None
    git_dirty: bool | None
    spec_version: str
    config_version: str
    resolved_config_hash: str
    environment_lock_hash: str
    python_version: str
    julia_version: str | None = None
    jutuldarcy_version: str | None = None
    model_checkpoint_hash: str | None = None
    schema_versions: dict[str, str]
    raw_input_hashes: dict[str, str]
    parent_run_ids: list[str]
    status: RunStatus
    outputs: dict[str, str]
    notes: list[str]


@dataclass
class RunContext:
    run_id: str
    run_dir: Path
    record: RunRecord

    @classmethod
    def start(
        cls,
        *,
        command: str,
        argv: list[str],
        cfg: ProjectConfig,
        paths: ProjectPaths,
        parent_run_ids: tuple[str, ...] = (),
        raw_input_hashes: dict[str, str] | None = None,
        schema_versions: dict[str, str] | None = None,
        now: datetime | None = None,
    ) -> RunContext:
        now = now or datetime.now(UTC)
        cfg_hash = config_hash(cfg)
        commit = git_commit(paths.root)
        base_id = make_run_id(command, cfg_hash, commit, now)
        paths.runs.mkdir(parents=True, exist_ok=True)
        run_id, run_dir = _claim_run_dir(paths.runs, base_id)
        (run_dir / "resolved_config.json").write_text(
            canonical_json(resolved_config_dict(cfg)), encoding="utf-8"
        )
        record = RunRecord(
            run_id=run_id,
            command=command,
            argv=list(argv),
            created_at=now.isoformat(),
            git_commit=commit,
            git_dirty=git_is_dirty(paths.root),
            spec_version=SPEC_VERSION,
            config_version=cfg.config_version,
            resolved_config_hash=cfg_hash,
            environment_lock_hash=environment_lock_hash(paths),
            python_version=platform.python_version(),
            schema_versions=dict(schema_versions or {}),
            raw_input_hashes=dict(raw_input_hashes or {}),
            parent_run_ids=list(parent_run_ids),
            status="RUNNING",
            outputs={},
            notes=[],
        )
        ctx = cls(run_id=run_id, run_dir=run_dir, record=record)
        ctx.write()
        return ctx

    def write(self) -> None:
        (self.run_dir / "run.json").write_text(
            self.record.model_dump_json(indent=2), encoding="utf-8"
        )

    def update(self, **fields: Any) -> None:
        self.record = self.record.model_copy(update=fields)
        self.write()

    def finish(
        self, status: RunStatus, outputs: dict[str, str], notes: tuple[str, ...] | list[str] = ()
    ) -> None:
        self.update(
            status=status,
            outputs=dict(outputs),
            notes=[*self.record.notes, *notes],
            finished_at=datetime.now(UTC).isoformat(),
        )
```

> `model_copy(update=...)` не валидирует поля; это допустимо, потому что все обновления идут из кода, а не из внешних данных. Тип `Any` в `update` осознан.

- [ ] **Step 5: Реализовать `logging_setup.py`**

```python
"""Logging rules: every record carries run_id; console + per-run file; no data rows in logs."""

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
```

> Фильтр стоит на обработчиках, а не на логгере, поэтому записи дочерних логгеров (`so_recon.child`) тоже получают `run_id`.

- [ ] **Step 6: Прогнать тесты, lint, mypy**

```bash
cd /Users/george/Documents/so_field && uv run pytest tests/unit/test_run_registry.py tests/unit/test_logging_setup.py -q && uv run ruff check . && uv run mypy
```
Ожидается: `6 passed`, без ошибок.

- [ ] **Step 7: Commit**

```bash
cd /Users/george/Documents/so_field && git add src/so_recon/registry/gitinfo.py src/so_recon/registry/run.py src/so_recon/logging_setup.py tests/unit/test_run_registry.py tests/unit/test_logging_setup.py && git commit -m "feat(e00): add run registry with lineage record, run ids and run-scoped logging

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Source manifest исходных CSV

**Files:**
- Create: `src/so_recon/registry/source_manifest.py`
- Test: `tests/unit/test_source_manifest.py`
- Runtime (не в git): перенос `data/Ромашка_сырые/*.csv` → `data/raw/`

**Interfaces:**
- Consumes: `SourcesConfig`, `SourceFileSpec`, `StrictModel` (Task 4); `ProjectPaths` (Task 3); `sha256_json` (Task 2).
- Produces:
  - `class MissingSourceError(FileNotFoundError)` — атрибут `missing: list[str]`.
  - `hash_and_count(path: Path, chunk_size: int = 1 << 20) -> tuple[str, int, int]` — `(sha256, size_bytes, line_count)` за один проход; `line_count` = число `\n` плюс 1, если файл не пуст и не заканчивается `\n`.
  - `class SourceEntry(StrictModel)`: `name, path, sha256, size_bytes, line_count, encoding, delimiter, decimal, required`.
  - `class SourceManifest(StrictModel)`: `manifest_version: Literal["1"]`, `created_at: str`, `git_commit: str | None`, `spec_version: str`, `config_version: str`, `sources: list[SourceEntry]`; метод `hashes() -> dict[str, str]` (`name → sha256`).
  - `build_source_manifest(sources: SourcesConfig, paths: ProjectPaths, *, config_version: str, git_commit: str | None, now: datetime) -> SourceManifest` — все отсутствующие `required` файлы собираются и бросается один `MissingSourceError`; отсутствующие `required=False` пропускаются с записью в `logging`.
  - `write_source_manifest(manifest: SourceManifest, path: Path) -> None` — JSON с `indent=2`, `ensure_ascii=False`, сортированные ключи.
  - `load_source_manifest(path: Path) -> SourceManifest`.

- [ ] **Step 1: Написать падающие тесты**

`tests/unit/test_source_manifest.py`:
```python
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from so_recon.config.schema import SourceFileSpec, SourcesConfig
from so_recon.paths import ProjectPaths
from so_recon.registry.source_manifest import (
    MissingSourceError,
    build_source_manifest,
    hash_and_count,
    load_source_manifest,
    write_source_manifest,
)

NOW = datetime(2026, 9, 13, 12, 0, tzinfo=UTC)


def _spec(name: str, required: bool = True) -> SourceFileSpec:
    return SourceFileSpec(
        name=name, path=f"data/raw/{name}.csv", encoding="utf-8", delimiter=";", decimal=",",
        required=required,
    )


def test_hash_and_count(tmp_path: Path) -> None:
    p = tmp_path / "f.csv"
    p.write_bytes(b"a;b\r\n1;2\r\n3;4")
    sha, size, lines = hash_and_count(p, chunk_size=2)
    assert sha == hashlib.sha256(b"a;b\r\n1;2\r\n3;4").hexdigest()
    assert size == 13
    assert lines == 3
    p.write_bytes(b"")
    assert hash_and_count(p) == (hashlib.sha256(b"").hexdigest(), 0, 0)


def test_build_manifest_records_every_file(tmp_path: Path) -> None:
    paths = ProjectPaths.default(tmp_path)
    paths.ensure_dirs()
    (paths.raw / "mer.csv").write_bytes(b"\xef\xbb\xbfh1;h2\n1;2\n")
    (paths.raw / "gis.csv").write_bytes(b"x\n")
    sources = SourcesConfig(files=[_spec("mer"), _spec("gis")])
    m = build_source_manifest(sources, paths, config_version="t", git_commit="abc", now=NOW)
    assert [s.name for s in m.sources] == ["mer", "gis"]
    assert m.sources[0].path == "data/raw/mer.csv"
    assert m.sources[0].line_count == 2
    assert m.hashes()["gis"] == hashlib.sha256(b"x\n").hexdigest()
    assert m.created_at == "2026-09-13T12:00:00+00:00"


def test_missing_required_files_raise_with_full_list(tmp_path: Path) -> None:
    paths = ProjectPaths.default(tmp_path)
    paths.ensure_dirs()
    sources = SourcesConfig(files=[_spec("mer"), _spec("gis"), _spec("opt", required=False)])
    with pytest.raises(MissingSourceError) as exc:
        build_source_manifest(sources, paths, config_version="t", git_commit=None, now=NOW)
    assert exc.value.missing == ["data/raw/mer.csv", "data/raw/gis.csv"]


def test_optional_missing_file_is_skipped(tmp_path: Path) -> None:
    paths = ProjectPaths.default(tmp_path)
    paths.ensure_dirs()
    (paths.raw / "mer.csv").write_bytes(b"1\n")
    sources = SourcesConfig(files=[_spec("mer"), _spec("opt", required=False)])
    m = build_source_manifest(sources, paths, config_version="t", git_commit=None, now=NOW)
    assert [s.name for s in m.sources] == ["mer"]


def test_write_and_load_roundtrip(tmp_path: Path) -> None:
    paths = ProjectPaths.default(tmp_path)
    paths.ensure_dirs()
    (paths.raw / "mer.csv").write_bytes(b"1\n")
    m = build_source_manifest(
        SourcesConfig(files=[_spec("mer")]), paths, config_version="t", git_commit=None, now=NOW
    )
    out = tmp_path / "source_manifest.json"
    write_source_manifest(m, out)
    text = out.read_text(encoding="utf-8")
    assert json.loads(text)["manifest_version"] == "1"
    assert load_source_manifest(out) == m
```

- [ ] **Step 2: Убедиться, что тесты падают**

```bash
cd /Users/george/Documents/so_field && uv run pytest tests/unit/test_source_manifest.py -q
```
Ожидается: `ModuleNotFoundError: so_recon.registry.source_manifest`.

- [ ] **Step 3: Реализовать `source_manifest.py`**

```python
"""Manifest of raw source files: SHA-256, size and line counts (SPEC 7.1, 19.12)."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

from so_recon import SPEC_VERSION
from so_recon.config.schema import SourcesConfig, StrictModel
from so_recon.paths import ProjectPaths

log = logging.getLogger(__name__)


class MissingSourceError(FileNotFoundError):
    def __init__(self, missing: list[str]) -> None:
        super().__init__(f"required source files are missing: {missing}")
        self.missing = missing


def hash_and_count(path: Path, chunk_size: int = 1 << 20) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    size = 0
    newlines = 0
    last = b""
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
            size += len(chunk)
            newlines += chunk.count(b"\n")
            last = chunk[-1:]
    lines = newlines + (1 if size > 0 and last != b"\n" else 0)
    return digest.hexdigest(), size, lines


class SourceEntry(StrictModel):
    name: str
    path: str
    sha256: str
    size_bytes: int
    line_count: int
    encoding: str
    delimiter: str
    decimal: str
    required: bool


class SourceManifest(StrictModel):
    manifest_version: Literal["1"] = "1"
    created_at: str
    git_commit: str | None
    spec_version: str
    config_version: str
    sources: list[SourceEntry]

    def hashes(self) -> dict[str, str]:
        return {s.name: s.sha256 for s in self.sources}


def build_source_manifest(
    sources: SourcesConfig,
    paths: ProjectPaths,
    *,
    config_version: str,
    git_commit: str | None,
    now: datetime,
) -> SourceManifest:
    entries: list[SourceEntry] = []
    missing: list[str] = []
    for spec in sources.files:
        full = paths.root / spec.path
        if not full.is_file():
            if spec.required:
                missing.append(spec.path)
            else:
                log.warning("optional source %s not found at %s; skipped", spec.name, spec.path)
            continue
        sha, size, lines = hash_and_count(full)
        log.info("hashed %s: %d bytes, %d lines", spec.name, size, lines)
        entries.append(
            SourceEntry(
                name=spec.name,
                path=spec.path,
                sha256=sha,
                size_bytes=size,
                line_count=lines,
                encoding=spec.encoding,
                delimiter=spec.delimiter,
                decimal=spec.decimal,
                required=spec.required,
            )
        )
    if missing:
        raise MissingSourceError(missing)
    return SourceManifest(
        created_at=now.isoformat(),
        git_commit=git_commit,
        spec_version=SPEC_VERSION,
        config_version=config_version,
        sources=entries,
    )


def write_source_manifest(manifest: SourceManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.model_dump(mode="json")
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_source_manifest(path: Path) -> SourceManifest:
    return SourceManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
```

- [ ] **Step 4: Прогнать тесты, lint, mypy**

```bash
cd /Users/george/Documents/so_field && uv run pytest tests/unit/test_source_manifest.py -q && uv run ruff check . && uv run mypy
```
Ожидается: `5 passed`.

- [ ] **Step 5: Перенести raw CSV в канонический каталог (локально, вне git)**

Перед переносом зафиксировать хеши для контроля:
```bash
cd /Users/george/Documents/so_field && shasum -a 256 data/Ромашка_сырые/*.csv > /tmp/so_field_raw_before.sha256 && cat /tmp/so_field_raw_before.sha256
```
Перенос:
```bash
cd /Users/george/Documents/so_field && mkdir -p data/raw && mv data/Ромашка_сырые/coords.csv data/Ромашка_сырые/gis.csv data/Ромашка_сырые/mer.csv data/Ромашка_сырые/perf.csv data/Ромашка_сырые/plastoper.csv data/raw/ && ls -la data/raw
```
Проверка, что хеши не изменились (сравнить по столбцу хеша):
```bash
cd /Users/george/Documents/so_field && shasum -a 256 data/raw/*.csv | awk '{print $1}' | sort > /tmp/after.txt && awk '{print $1}' /tmp/so_field_raw_before.sha256 | sort > /tmp/before.txt && diff /tmp/before.txt /tmp/after.txt && echo "HASHES IDENTICAL"
```
Ожидается: `HASHES IDENTICAL`. Ожидаемые значения для двух малых файлов известны заранее: `coords.csv` → `3298fda637a5f36d2ecbd5ec014e037a95bbd9631973db488c6dcbc1ade78996`, `plastoper.csv` → `129c32e7deb51a06ede288be21b1d878a36fb0f7edf9ce52b591b445674f934b`.

- [ ] **Step 6: Commit**

```bash
cd /Users/george/Documents/so_field && git add src/so_recon/registry/source_manifest.py tests/unit/test_source_manifest.py && git commit -m "feat(e00): add raw source manifest with sha256, size and line counts

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Детерминированный synthetic fixture (Python-часть)

**Files:**
- Create: `src/so_recon/synthetic/fixture.py`
- Test: `tests/unit/test_fixture.py`

**Interfaces:**
- Consumes: `SmokeFixtureConfig` (Task 4), `sha256_json` (Task 2).
- Produces:
  - `@dataclass(frozen=True) class SmokeFixture`: `wells: pa.Table`, `well_month: pa.Table`, `content_hash: str`, `seed: int`.
  - `build_smoke_fixture(cfg: SmokeFixtureConfig) -> SmokeFixture` — детерминирован по `cfg.seed`; `wells` имеет колонки `well_id: str`, `role: str` (`injector`/`producer`), `x_m: float`, `y_m: float`; `well_month` — `well_id: str`, `month: str` (`YYYY-MM-01`), `liquid_m3: float`, `injection_m3: float`, все объёмы округлены до 6 знаков и неотрицательны; injector имеет `liquid_m3 = 0`, producer — `injection_m3 = 0`.
  - `write_smoke_fixture(fixture: SmokeFixture, out_dir: Path) -> dict[str, Path]` — пишет `wells.parquet`, `well_month.parquet`, `fixture_meta.json` (`seed`, `content_hash`, `n_rows`); возвращает пути по именам.
  - `fixture_content_hash(wells: pa.Table, well_month: pa.Table) -> str` — `sha256_json({"wells": wells.to_pylist(), "well_month": well_month.to_pylist()})`.

- [ ] **Step 1: Написать падающие тесты**

`tests/unit/test_fixture.py`:
```python
import json
from pathlib import Path

import pyarrow.parquet as pq

from so_recon.config.schema import SmokeFixtureConfig
from so_recon.synthetic.fixture import build_smoke_fixture, fixture_content_hash, write_smoke_fixture


def test_fixture_is_deterministic_for_same_seed() -> None:
    a = build_smoke_fixture(SmokeFixtureConfig(seed=7, n_wells=4, n_months=3))
    b = build_smoke_fixture(SmokeFixtureConfig(seed=7, n_wells=4, n_months=3))
    assert a.content_hash == b.content_hash
    assert a.wells.to_pylist() == b.wells.to_pylist()


def test_fixture_changes_with_seed() -> None:
    a = build_smoke_fixture(SmokeFixtureConfig(seed=7))
    b = build_smoke_fixture(SmokeFixtureConfig(seed=8))
    assert a.content_hash != b.content_hash


def test_fixture_shape_and_invariants() -> None:
    fx = build_smoke_fixture(SmokeFixtureConfig(seed=1, n_wells=5, n_months=4))
    assert fx.wells.num_rows == 5
    assert fx.well_month.num_rows == 20
    assert fx.wells.column_names == ["well_id", "role", "x_m", "y_m"]
    assert fx.well_month.column_names == ["well_id", "month", "liquid_m3", "injection_m3"]
    roles = dict(zip(fx.wells["well_id"].to_pylist(), fx.wells["role"].to_pylist(), strict=True))
    for row in fx.well_month.to_pylist():
        assert row["liquid_m3"] >= 0 and row["injection_m3"] >= 0
        if roles[row["well_id"]] == "injector":
            assert row["liquid_m3"] == 0.0
        else:
            assert row["injection_m3"] == 0.0
    months = sorted(set(fx.well_month["month"].to_pylist()))
    assert months == ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"]


def test_write_fixture_roundtrips_and_records_hash(tmp_path: Path) -> None:
    fx = build_smoke_fixture(SmokeFixtureConfig(seed=3, n_wells=2, n_months=2))
    files = write_smoke_fixture(fx, tmp_path / "fixture")
    wells = pq.read_table(files["wells"])
    well_month = pq.read_table(files["well_month"])
    assert fixture_content_hash(wells, well_month) == fx.content_hash
    meta = json.loads(files["meta"].read_text())
    assert meta["content_hash"] == fx.content_hash
    assert meta["seed"] == 3
```

- [ ] **Step 2: Убедиться, что тесты падают**

```bash
cd /Users/george/Documents/so_field && uv run pytest tests/unit/test_fixture.py -q
```
Ожидается: `ModuleNotFoundError: so_recon.synthetic.fixture`.

- [ ] **Step 3: Реализовать `fixture.py`**

```python
"""Minimal deterministic synthetic fixture for the E00 smoke run.

Not a physical model: two tiny tables (wells, well_month) generated from a seeded PCG64
stream, hashed by canonical JSON content so the hash is independent of Parquet encoding.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from so_recon.config.schema import SmokeFixtureConfig
from so_recon.registry.hashing import sha256_json

_START_YEAR = 2020
_START_MONTH = 1


def _month_label(index: int) -> str:
    total = (_START_YEAR * 12 + (_START_MONTH - 1)) + index
    return f"{total // 12:04d}-{total % 12 + 1:02d}-01"


def fixture_content_hash(wells: pa.Table, well_month: pa.Table) -> str:
    return sha256_json({"wells": wells.to_pylist(), "well_month": well_month.to_pylist()})


@dataclass(frozen=True)
class SmokeFixture:
    wells: pa.Table
    well_month: pa.Table
    content_hash: str
    seed: int


def build_smoke_fixture(cfg: SmokeFixtureConfig) -> SmokeFixture:
    rng = np.random.default_rng(cfg.seed)
    well_ids = [f"W{i + 1:03d}" for i in range(cfg.n_wells)]
    roles = ["injector" if i % 2 == 0 else "producer" for i in range(cfg.n_wells)]
    xs = np.round(rng.uniform(0.0, 1000.0, size=cfg.n_wells), 3)
    ys = np.round(rng.uniform(0.0, 1000.0, size=cfg.n_wells), 3)
    wells = pa.table(
        {
            "well_id": pa.array(well_ids, pa.string()),
            "role": pa.array(roles, pa.string()),
            "x_m": pa.array(xs.tolist(), pa.float64()),
            "y_m": pa.array(ys.tolist(), pa.float64()),
        }
    )

    wm_ids: list[str] = []
    wm_month: list[str] = []
    wm_liq: list[float] = []
    wm_inj: list[float] = []
    for wid, role in zip(well_ids, roles, strict=True):
        volumes = np.round(rng.uniform(10.0, 100.0, size=cfg.n_months), 6)
        for m in range(cfg.n_months):
            wm_ids.append(wid)
            wm_month.append(_month_label(m))
            v = float(volumes[m])
            wm_liq.append(0.0 if role == "injector" else v)
            wm_inj.append(v if role == "injector" else 0.0)
    well_month = pa.table(
        {
            "well_id": pa.array(wm_ids, pa.string()),
            "month": pa.array(wm_month, pa.string()),
            "liquid_m3": pa.array(wm_liq, pa.float64()),
            "injection_m3": pa.array(wm_inj, pa.float64()),
        }
    )
    return SmokeFixture(
        wells=wells,
        well_month=well_month,
        content_hash=fixture_content_hash(wells, well_month),
        seed=cfg.seed,
    )


def write_smoke_fixture(fixture: SmokeFixture, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    wells_path = out_dir / "wells.parquet"
    wm_path = out_dir / "well_month.parquet"
    meta_path = out_dir / "fixture_meta.json"
    pq.write_table(fixture.wells, wells_path)
    pq.write_table(fixture.well_month, wm_path)
    meta = {
        "seed": fixture.seed,
        "content_hash": fixture.content_hash,
        "n_rows": {"wells": fixture.wells.num_rows, "well_month": fixture.well_month.num_rows},
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"wells": wells_path, "well_month": wm_path, "meta": meta_path}
```

- [ ] **Step 4: Прогнать тесты, lint, mypy**

```bash
cd /Users/george/Documents/so_field && uv run pytest tests/unit/test_fixture.py -q && uv run ruff check . && uv run mypy
```
Ожидается: `4 passed`. Если mypy жалуется на `pa.Table` в dataclass — override для `pyarrow` уже задан в `pyproject.toml` (`ignore_missing_imports`), тип станет `Any`; это допустимо.

- [ ] **Step 5: Commit**

```bash
cd /Users/george/Documents/so_field && git add src/so_recon/synthetic/fixture.py tests/unit/test_fixture.py && git commit -m "feat(e00): add deterministic synthetic smoke fixture with content hash

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Julia-окружение с JutulDarcy и smoke-скрипт

**Files:**
- Create: `julia/Project.toml`, `julia/Manifest.toml` (генерируется), `julia/README.md`, `julia/smoke/smoke_case.jl`

**Interfaces:**
- Produces: команда
  `julia --project=julia --startup-file=no julia/smoke/smoke_case.jl --out <file.json> --nx 20 --nsteps 12`
  пишет JSON с ключами: `status` (`"ok"`), `julia_version`, `jutuldarcy_version`, `jutul_version`, `nx`, `n_steps`, `cumulative_oil_m3` (>0), `cumulative_water_injected_m3` (>0), `mean_so_final` (в (0,1) и меньше 0.8), `wall_time_s`. Код возврата 0. При ошибке — JSON `{"status":"error","message":...}` и код возврата 1.

- [ ] **Step 1: Установить Julia 1.12 через juliaup (если `julia` не найден)**

```bash
brew install juliaup && juliaup add 1.12 && juliaup default 1.12 && julia --version
```
Ожидается: `julia version 1.12.x`. Если `brew` недоступен, альтернатива из официальной документации juliaup: `curl -fsSL https://install.julialang.org | sh -s -- --yes --default-channel 1.12`, затем перезапустить shell.

- [ ] **Step 2: Создать `julia/Project.toml`**

```toml
name = "SOReconEnv"
uuid = "0f9b3f1e-7c4a-4a7e-9a3d-5e00e00e0001"
authors = ["SO-RECON"]
version = "0.0.1"

[deps]
JSON = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
Jutul = "2b460a1a-8a2b-45b2-b125-b5c536396eb9"
JutulDarcy = "82210473-ab04-4dce-b31b-11573c4f8e0a"
Pkg = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[compat]
JSON = "0.21, 1"
Jutul = "0.4"
JutulDarcy = "0.3"
julia = "1.12"
```

> UUID зависимостей проверяются командой `Pkg.add` на следующем шаге: если UUID расходятся с реестром, `Pkg.add` их перепишет. Поэтому фактический путь: создать `Project.toml` только с `[compat]` и добавить пакеты через `Pkg.add`, что само проставит UUID.

Практический порядок:
```bash
cd /Users/george/Documents/so_field && mkdir -p julia/smoke && printf 'name = "SOReconEnv"\nuuid = "0f9b3f1e-7c4a-4a7e-9a3d-5e00e00e0001"\nversion = "0.0.1"\n\n[deps]\n\n[compat]\nJSON = "0.21, 1"\nJutul = "0.4"\nJutulDarcy = "0.3"\njulia = "1.12"\n' > julia/Project.toml && julia --project=julia --startup-file=no -e 'using Pkg; Pkg.add(["JutulDarcy", "Jutul", "JSON"]); Pkg.precompile(); Pkg.status()'
```
Ожидается (порядка 5–15 минут на первый precompile): `Pkg.status()` показывает `JutulDarcy v0.3.x`, `Jutul v0.4.x`, `JSON`. Появился `julia/Manifest.toml`.

- [ ] **Step 3: Написать `julia/smoke/smoke_case.jl`**

```julia
# E00 smoke: tiny 1D oil–water JutulDarcy case. Not a verification test (that is E05);
# it proves the locked Julia environment runs and produces deterministic scalar summaries.
using JutulDarcy, Jutul, JSON

function parse_cli(args::Vector{String})
    opts = Dict{String,String}("nx" => "20", "nsteps" => "12")
    i = 1
    while i <= length(args)
        key = args[i]
        startswith(key, "--") || error("unexpected argument $(key)")
        i + 1 <= length(args) || error("missing value for $(key)")
        opts[key[3:end]] = args[i + 1]
        i += 2
    end
    haskey(opts, "out") || error("--out <file.json> is required")
    return opts
end

function run_smoke(nx::Int, nsteps::Int)
    Darcy, bar, kg, meter, day = si_units(:darcy, :bar, :kilogram, :meter, :day)
    g = CartesianMesh((nx, 1, 1), (nx * 50.0, 50.0, 10.0) .* meter)
    domain = reservoir_domain(g, permeability = 0.1Darcy, porosity = 0.25)
    injector = setup_vertical_well(domain, 1, 1, name = :Injector)
    producer = setup_vertical_well(domain, nx, 1, name = :Producer)

    rhoWS = 1000.0kg / meter^3
    rhoOS = 850.0kg / meter^3
    sys = ImmiscibleSystem((AqueousPhase(), LiquidPhase()), reference_densities = [rhoWS, rhoOS])
    model = setup_reservoir_model(domain, sys, wells = [injector, producer])
    parameters = setup_parameters(model)
    state0 = setup_reservoir_state(model, Pressure = 150bar, Saturations = [0.2, 0.8])

    dt = fill(30.0day, nsteps)
    pv = pore_volume(model, parameters)
    inj_rate = 0.5 * sum(pv) / sum(dt)          # inject half a pore volume over the run
    i_ctrl = InjectorControl(TotalRateTarget(inj_rate), [1.0, 0.0], density = rhoWS)
    p_ctrl = ProducerControl(BottomHolePressureTarget(100bar))
    forces = setup_reservoir_forces(model, control = Dict(:Injector => i_ctrl, :Producer => p_ctrl))

    wd, states, t = simulate_reservoir(state0, model, dt,
        parameters = parameters, forces = forces, info_level = -1)

    orat = wd[:Producer, :orat]                  # surface oil rate, negative for production
    wrat_inj = wd[:Injector, :wrat]              # surface water rate, positive for injection
    cum_oil = -sum(orat .* dt)
    cum_winj = sum(wrat_inj .* dt)
    so_final = states[end][:Saturations][2, :]
    mean_so = sum(so_final .* pv) / sum(pv)

    return Dict(
        "status" => "ok",
        "julia_version" => string(VERSION),
        "jutuldarcy_version" => string(pkgversion(JutulDarcy)),
        "jutul_version" => string(pkgversion(Jutul)),
        "nx" => nx,
        "n_steps" => nsteps,
        "cumulative_oil_m3" => cum_oil,
        "cumulative_water_injected_m3" => cum_winj,
        "mean_so_final" => mean_so,
    )
end

function main(args::Vector{String})
    opts = parse_cli(args)
    out = opts["out"]
    t0 = time()
    result = try
        r = run_smoke(parse(Int, opts["nx"]), parse(Int, opts["nsteps"]))
        r["wall_time_s"] = time() - t0
        r
    catch err
        Dict("status" => "error", "message" => sprint(showerror, err), "wall_time_s" => time() - t0)
    end
    mkpath(dirname(abspath(out)))
    open(out, "w") do io
        write(io, JSON.json(result))   # JSON.json exists in both JSON.jl 0.21 and 1.x
    end
    return result["status"] == "ok" ? 0 : 1
end

exit(main(ARGS))
```

- [ ] **Step 4: Запустить скрипт вручную и проверить инварианты**

```bash
cd /Users/george/Documents/so_field && julia --project=julia --startup-file=no julia/smoke/smoke_case.jl --out /tmp/so_smoke.json --nx 20 --nsteps 12; echo "exit=$?"; cat /tmp/so_smoke.json
```
Ожидается: `exit=0`; в JSON `status: "ok"`, `cumulative_oil_m3 > 0`, `cumulative_water_injected_m3 > 0`, `0 < mean_so_final < 0.8`. Если API-вызов не найден (например, изменилось имя `setup_vertical_well`, `pore_volume` или индексация `wd[:Producer, :orat]`), свериться с документацией установленной версии: `julia --project=julia -e 'using JutulDarcy; println(@doc setup_vertical_well)'`, поправить только имя вызова, не меняя физики кейса. Зафиксировать любую такую правку в `julia/README.md`.

- [ ] **Step 5: Проверить детерминизм двумя запусками**

```bash
cd /Users/george/Documents/so_field && julia --project=julia --startup-file=no julia/smoke/smoke_case.jl --out /tmp/so_smoke_2.json && python3 -c "
import json
a=json.load(open('/tmp/so_smoke.json')); b=json.load(open('/tmp/so_smoke_2.json'))
for k in ('cumulative_oil_m3','cumulative_water_injected_m3','mean_so_final'):
    rel=abs(a[k]-b[k])/max(1.0,abs(a[k])); print(k, a[k], b[k], rel); assert rel <= 1e-6, k
print('DETERMINISTIC')"
```
Ожидается: `DETERMINISTIC`.

- [ ] **Step 6: Написать `julia/README.md`**

```markdown
# Julia environment (SO-RECON)

`Project.toml` + `Manifest.toml` — зафиксированное окружение проекта. Единственный
operational forward backend — JutulDarcy (SPEC 10.1). Пакет `SOReconSimulator/`
появится в E05 и будет подключён в это окружение через `Pkg.develop`.

Установка (Julia 1.12 через juliaup):

    julia --project=julia --startup-file=no -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

Smoke-кейс E00 (не верификация физики — она в E05):

    julia --project=julia --startup-file=no julia/smoke/smoke_case.jl --out artifacts/tmp/smoke.json

Обновление зависимостей выполняется только осознанно (`Pkg.update`) с коммитом нового
`Manifest.toml` и новой строкой в `reports/environment_report.md`.
```

- [ ] **Step 7: Убедиться, что `Manifest.toml` не содержит личных путей и закоммитить**

```bash
cd /Users/george/Documents/so_field && ! grep -n "/Users/" julia/Manifest.toml && uv run pytest tests/test_no_absolute_paths.py -q && git add julia && git commit -m "feat(e00): add locked Julia environment with JutulDarcy and 1D oil-water smoke case

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: Python-мост к Julia

**Files:**
- Create: `src/so_recon/simulator/julia_bridge.py`
- Test: `tests/unit/test_julia_bridge.py`, `tests/integration/test_julia_smoke.py`

**Interfaces:**
- Consumes: `StrictModel` (Task 4), `ProjectPaths` (Task 3), `JuliaConfig` (Task 4), скрипт Task 8.
- Produces:
  - `class JuliaNotFoundError(RuntimeError)`, `class JuliaRunError(RuntimeError)`.
  - `find_julia(explicit: str | None = None) -> Path` — порядок: `explicit` → env `SO_RECON_JULIA` → `shutil.which("julia")` → `Path.home()/".juliaup"/"bin"/"julia"`; иначе `JuliaNotFoundError`.
  - `class JuliaLauncher(Protocol)`: `def launch(self, script: Path, args: list[str], out_path: Path) -> None`.
  - `class SubprocessJuliaLauncher`: `__init__(self, julia_exe: Path, project: Path, timeout_s: int)`; `launch(...)` выполняет `[julia_exe, f"--project={project}", "--startup-file=no", str(script), *args, "--out", str(out_path)]`, при ненулевом коде возврата бросает `JuliaRunError` с хвостом stderr (последние 4000 символов).
  - `class JuliaSmokeResult(StrictModel)`: `status: Literal["ok"]`, `julia_version: str`, `jutuldarcy_version: str`, `jutul_version: str`, `nx: int`, `n_steps: int`, `cumulative_oil_m3: float`, `cumulative_water_injected_m3: float`, `mean_so_final: float`, `wall_time_s: float`.
  - `run_julia_smoke(launcher: JuliaLauncher, script: Path, out_path: Path, *, nx: int, n_steps: int) -> JuliaSmokeResult` — вызывает `launch`, читает JSON; если `status != "ok"` — `JuliaRunError(message)`.
  - `default_launcher(paths: ProjectPaths, cfg: JuliaConfig, julia_exe: str | None = None) -> SubprocessJuliaLauncher`.

- [ ] **Step 1: Написать падающие тесты**

`tests/unit/test_julia_bridge.py`:
```python
import json
from pathlib import Path

import pytest

from so_recon.simulator.julia_bridge import (
    JuliaNotFoundError,
    JuliaRunError,
    SubprocessJuliaLauncher,
    find_julia,
    run_julia_smoke,
)

OK_PAYLOAD = {
    "status": "ok",
    "julia_version": "1.12.7",
    "jutuldarcy_version": "0.3.11",
    "jutul_version": "0.4.40",
    "nx": 20,
    "n_steps": 12,
    "cumulative_oil_m3": 1234.5,
    "cumulative_water_injected_m3": 2000.0,
    "mean_so_final": 0.55,
    "wall_time_s": 3.2,
}


class FakeLauncher:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls: list[tuple[Path, list[str], Path]] = []

    def launch(self, script: Path, args: list[str], out_path: Path) -> None:
        self.calls.append((script, args, out_path))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.payload))


def test_run_julia_smoke_parses_result(tmp_path: Path) -> None:
    launcher = FakeLauncher(OK_PAYLOAD)
    res = run_julia_smoke(launcher, tmp_path / "s.jl", tmp_path / "out.json", nx=20, n_steps=12)
    assert res.cumulative_oil_m3 == 1234.5
    assert launcher.calls[0][1] == ["--nx", "20", "--nsteps", "12"]


def test_run_julia_smoke_raises_on_error_status(tmp_path: Path) -> None:
    launcher = FakeLauncher({"status": "error", "message": "boom", "wall_time_s": 0.1})
    with pytest.raises(JuliaRunError, match="boom"):
        run_julia_smoke(launcher, tmp_path / "s.jl", tmp_path / "out.json", nx=20, n_steps=12)


def test_find_julia_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exe = tmp_path / "julia"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    monkeypatch.setenv("SO_RECON_JULIA", str(exe))
    assert find_julia() == exe


def test_find_julia_raises_when_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SO_RECON_JULIA", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(JuliaNotFoundError):
        find_julia()


def test_subprocess_launcher_builds_command_and_reports_failure(tmp_path: Path) -> None:
    fake = tmp_path / "fake_julia.sh"
    fake.write_text("#!/bin/sh\necho \"$@\" > \"$(dirname \"$0\")/argv.txt\"\necho fatal >&2\nexit 3\n")
    fake.chmod(0o755)
    launcher = SubprocessJuliaLauncher(fake, tmp_path / "proj", timeout_s=30)
    with pytest.raises(JuliaRunError, match="fatal"):
        launcher.launch(tmp_path / "s.jl", ["--nx", "5"], tmp_path / "o.json")
    argv = (tmp_path / "argv.txt").read_text().split()
    assert argv[0] == f"--project={tmp_path / 'proj'}"
    assert argv[1] == "--startup-file=no"
    assert argv[-2:] == ["--out", str(tmp_path / "o.json")]
```

`tests/integration/test_julia_smoke.py`:
```python
from pathlib import Path

import pytest

from so_recon.config.load import load_project_config
from so_recon.paths import ProjectPaths
from so_recon.simulator.julia_bridge import JuliaNotFoundError, default_launcher, run_julia_smoke

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.julia
def test_real_julia_smoke_case_runs(tmp_path: Path) -> None:
    cfg = load_project_config(ROOT / "configs" / "project.yml")
    paths = ProjectPaths.from_config(ROOT, cfg.paths)
    if not (paths.julia / "Manifest.toml").is_file():
        pytest.skip("julia/Manifest.toml missing; run make setup-julia")
    try:
        launcher = default_launcher(paths, cfg.julia)
    except JuliaNotFoundError:
        pytest.skip("julia executable not found")
    res = run_julia_smoke(
        launcher, paths.root / cfg.julia.smoke_script, tmp_path / "smoke.json",
        nx=cfg.smoke.nx, n_steps=cfg.smoke.n_steps,
    )
    assert res.status == "ok"
    assert res.cumulative_oil_m3 > 0
    assert res.cumulative_water_injected_m3 > 0
    assert 0.0 < res.mean_so_final < 0.8
    assert res.jutuldarcy_version.startswith("0.3.")
```

- [ ] **Step 2: Убедиться, что тесты падают**

```bash
cd /Users/george/Documents/so_field && uv run pytest tests/unit/test_julia_bridge.py -q
```
Ожидается: `ModuleNotFoundError: so_recon.simulator.julia_bridge`.

- [ ] **Step 3: Реализовать `julia_bridge.py`**

```python
"""The only module that knows how to launch Julia. E05 builds the simulator adapter on top."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Literal, Protocol

from so_recon.config.schema import JuliaConfig, StrictModel
from so_recon.paths import ProjectPaths

log = logging.getLogger(__name__)

JULIA_ENV_VAR = "SO_RECON_JULIA"


class JuliaNotFoundError(RuntimeError):
    pass


class JuliaRunError(RuntimeError):
    pass


def find_julia(explicit: str | None = None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get(JULIA_ENV_VAR)
    if env:
        candidates.append(Path(env))
    which = shutil.which("julia")
    if which:
        candidates.append(Path(which))
    candidates.append(Path.home() / ".juliaup" / "bin" / "julia")
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return c
    raise JuliaNotFoundError(
        f"julia executable not found; install via juliaup or set {JULIA_ENV_VAR}"
    )


class JuliaLauncher(Protocol):
    def launch(self, script: Path, args: list[str], out_path: Path) -> None: ...


class SubprocessJuliaLauncher:
    def __init__(self, julia_exe: Path, project: Path, timeout_s: int) -> None:
        self.julia_exe = julia_exe
        self.project = project
        self.timeout_s = timeout_s

    def launch(self, script: Path, args: list[str], out_path: Path) -> None:
        cmd = [
            str(self.julia_exe),
            f"--project={self.project}",
            "--startup-file=no",
            str(script),
            *args,
            "--out",
            str(out_path),
        ]
        log.info("launching julia: %s", " ".join(cmd[:4]))
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, check=False, timeout=self.timeout_s
            )
        except subprocess.TimeoutExpired as exc:
            raise JuliaRunError(f"julia timed out after {self.timeout_s}s") from exc
        if proc.returncode != 0:
            raise JuliaRunError(
                f"julia exited with {proc.returncode}; stderr tail:\n{proc.stderr[-4000:]}"
            )


class JuliaSmokeResult(StrictModel):
    status: Literal["ok"]
    julia_version: str
    jutuldarcy_version: str
    jutul_version: str
    nx: int
    n_steps: int
    cumulative_oil_m3: float
    cumulative_water_injected_m3: float
    mean_so_final: float
    wall_time_s: float


def run_julia_smoke(
    launcher: JuliaLauncher, script: Path, out_path: Path, *, nx: int, n_steps: int
) -> JuliaSmokeResult:
    launcher.launch(script, ["--nx", str(nx), "--nsteps", str(n_steps)], out_path)
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    if payload.get("status") != "ok":
        raise JuliaRunError(f"julia smoke failed: {payload.get('message', payload)}")
    return JuliaSmokeResult.model_validate(payload)


def default_launcher(
    paths: ProjectPaths, cfg: JuliaConfig, julia_exe: str | None = None
) -> SubprocessJuliaLauncher:
    return SubprocessJuliaLauncher(
        julia_exe=find_julia(julia_exe),
        project=paths.root / cfg.project,
        timeout_s=cfg.timeout_s,
    )
```

- [ ] **Step 4: Прогнать unit-тесты, затем integration-тест с реальной Julia**

```bash
cd /Users/george/Documents/so_field && uv run pytest tests/unit/test_julia_bridge.py -q && uv run ruff check . && uv run mypy
```
Ожидается: `5 passed`.

```bash
cd /Users/george/Documents/so_field && uv run pytest tests/integration/test_julia_smoke.py -q -m julia
```
Ожидается: `1 passed` (не `skipped`; если skipped — Julia не найдена, вернуться к Task 8 Step 1).

- [ ] **Step 5: Commit**

```bash
cd /Users/george/Documents/so_field && git add src/so_recon/simulator/julia_bridge.py tests/unit/test_julia_bridge.py tests/integration/test_julia_smoke.py && git commit -m "feat(e00): add python-julia bridge with injectable launcher and smoke result model

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 10: Environment report

**Files:**
- Create: `src/so_recon/environment/report.py`
- Test: `tests/unit/test_environment_report.py`

**Interfaces:**
- Consumes: `StrictModel` (Task 4), `ProjectPaths` (Task 3), `sha256_file` (Task 2), `environment_lock_hash` (Task 5), `git_commit`, `git_is_dirty` (Task 5), `find_julia`, `JuliaNotFoundError` (Task 9).
- Produces:
  - `parse_julia_manifest(path: Path) -> tuple[str | None, dict[str, str]]` — `(julia_version, {package: version})` из `Manifest.toml` (формат v2: `data["julia_version"]`, `data["deps"][name][0]["version"]`); включаются пакеты `JutulDarcy`, `Jutul`, `JSON` при наличии.
  - `class EnvironmentReport(StrictModel)`: `created_at: str`, `os: str`, `arch: str`, `python_version: str`, `uv_version: str | None`, `uv_lock_sha256: str | None`, `julia_executable_version: str | None`, `julia_manifest_version: str | None`, `julia_manifest_sha256: str | None`, `julia_packages: dict[str, str]`, `git_commit: str | None`, `git_dirty: bool | None`, `environment_lock_hash: str`.
  - `VersionProbe = Callable[[list[str]], str | None]` — выполняет команду и возвращает stdout или `None`.
  - `subprocess_probe(cmd: list[str]) -> str | None`.
  - `collect_environment(paths: ProjectPaths, *, now: datetime, probe: VersionProbe = subprocess_probe) -> EnvironmentReport`.
  - `render_markdown(report: EnvironmentReport) -> str`.
  - `write_environment_report(report: EnvironmentReport, md_path: Path, json_path: Path) -> None`.

- [ ] **Step 1: Написать падающие тесты**

`tests/unit/test_environment_report.py`:
```python
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from so_recon.environment.report import (
    collect_environment,
    parse_julia_manifest,
    render_markdown,
    write_environment_report,
)
from so_recon.paths import ProjectPaths

MANIFEST = """
julia_version = "1.12.7"
manifest_format = "2.0"
project_hash = "abc"

[[deps.JSON]]
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.1.2"

[[deps.Jutul]]
uuid = "2b460a1a-8a2b-45b2-b125-b5c536396eb9"
version = "0.4.40"

[[deps.JutulDarcy]]
uuid = "82210473-ab04-4dce-b31b-11573c4f8e0a"
version = "0.3.11"

[[deps.Other]]
version = "9.9.9"
"""
NOW = datetime(2026, 9, 13, 12, 0, tzinfo=UTC)


def test_parse_julia_manifest(tmp_path: Path) -> None:
    p = tmp_path / "Manifest.toml"
    p.write_text(MANIFEST)
    julia_version, pkgs = parse_julia_manifest(p)
    assert julia_version == "1.12.7"
    assert pkgs == {"JSON": "1.1.2", "Jutul": "0.4.40", "JutulDarcy": "0.3.11"}


def _probe(cmd: list[str]) -> str | None:
    if cmd[0] == "uv":
        return "uv 0.12.7 (Homebrew)\n"
    if cmd[0].endswith("julia"):
        return "julia version 1.12.7\n"
    return None


def test_collect_environment_with_and_without_lock_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SO_RECON_JULIA", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    paths = ProjectPaths.default(tmp_path)
    rep = collect_environment(paths, now=NOW, probe=_probe)
    assert rep.uv_lock_sha256 is None
    assert rep.julia_manifest_version is None
    assert rep.julia_executable_version is None
    assert rep.uv_version == "0.12.7"

    (tmp_path / "uv.lock").write_text("lock")
    (tmp_path / "julia").mkdir()
    (tmp_path / "julia" / "Manifest.toml").write_text(MANIFEST)
    rep2 = collect_environment(paths, now=NOW, probe=_probe)
    assert rep2.uv_lock_sha256 is not None
    assert rep2.julia_manifest_version == "1.12.7"
    assert rep2.julia_packages["JutulDarcy"] == "0.3.11"
    assert rep2.environment_lock_hash != rep.environment_lock_hash


def test_render_and_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SO_RECON_JULIA", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    paths = ProjectPaths.default(tmp_path)
    rep = collect_environment(paths, now=NOW, probe=_probe)
    md = render_markdown(rep)
    assert "# Environment report" in md
    assert "environment_lock_hash" in md
    md_path = tmp_path / "reports" / "environment_report.md"
    json_path = tmp_path / "reports" / "manifests" / "environment.json"
    write_environment_report(rep, md_path, json_path)
    assert md_path.read_text() == md
    assert json.loads(json_path.read_text())["python_version"] == rep.python_version
```

- [ ] **Step 2: Убедиться, что тесты падают**

```bash
cd /Users/george/Documents/so_field && uv run pytest tests/unit/test_environment_report.py -q
```
Ожидается: `ModuleNotFoundError: so_recon.environment.report`.

- [ ] **Step 3: Реализовать `report.py`**

```python
"""Environment report: versions and lock hashes (SPEC 19.12; STAGES E00 output)."""

from __future__ import annotations

import json
import platform
import subprocess
import tomllib
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from so_recon.config.schema import StrictModel
from so_recon.paths import ProjectPaths
from so_recon.registry.gitinfo import git_commit, git_is_dirty
from so_recon.registry.hashing import sha256_file
from so_recon.registry.run import environment_lock_hash
from so_recon.simulator.julia_bridge import JuliaNotFoundError, find_julia

TRACKED_JULIA_PACKAGES = ("JutulDarcy", "Jutul", "JSON")

VersionProbe = Callable[[list[str]], str | None]


def subprocess_probe(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout if out.returncode == 0 else None


def parse_julia_manifest(path: Path) -> tuple[str | None, dict[str, str]]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    julia_version = data.get("julia_version")
    deps = data.get("deps", {})
    packages: dict[str, str] = {}
    for name in TRACKED_JULIA_PACKAGES:
        entries = deps.get(name)
        if entries and isinstance(entries, list) and "version" in entries[0]:
            packages[name] = str(entries[0]["version"])
    return (str(julia_version) if julia_version else None), packages


class EnvironmentReport(StrictModel):
    created_at: str
    os: str
    arch: str
    python_version: str
    uv_version: str | None
    uv_lock_sha256: str | None
    julia_executable_version: str | None
    julia_manifest_version: str | None
    julia_manifest_sha256: str | None
    julia_packages: dict[str, str]
    git_commit: str | None
    git_dirty: bool | None
    environment_lock_hash: str


def _second_token(text: str | None) -> str | None:
    if not text:
        return None
    parts = text.split()
    return parts[1] if len(parts) >= 2 else None


def _julia_exe_version(probe: VersionProbe) -> str | None:
    try:
        exe = find_julia()
    except JuliaNotFoundError:
        return None
    out = probe([str(exe), "--version"])
    # "julia version 1.12.7"
    if not out:
        return None
    parts = out.split()
    return parts[2] if len(parts) >= 3 else None


def collect_environment(
    paths: ProjectPaths, *, now: datetime, probe: VersionProbe = subprocess_probe
) -> EnvironmentReport:
    uv_lock = paths.root / "uv.lock"
    manifest = paths.julia / "Manifest.toml"
    julia_manifest_version: str | None = None
    julia_packages: dict[str, str] = {}
    if manifest.is_file():
        julia_manifest_version, julia_packages = parse_julia_manifest(manifest)
    return EnvironmentReport(
        created_at=now.isoformat(),
        os=f"{platform.system()} {platform.release()}",
        arch=platform.machine(),
        python_version=platform.python_version(),
        uv_version=_second_token(probe(["uv", "--version"])),
        uv_lock_sha256=sha256_file(uv_lock) if uv_lock.is_file() else None,
        julia_executable_version=_julia_exe_version(probe),
        julia_manifest_version=julia_manifest_version,
        julia_manifest_sha256=sha256_file(manifest) if manifest.is_file() else None,
        julia_packages=julia_packages,
        git_commit=git_commit(paths.root),
        git_dirty=git_is_dirty(paths.root),
        environment_lock_hash=environment_lock_hash(paths),
    )


def render_markdown(report: EnvironmentReport) -> str:
    rows = [
        ("created_at", report.created_at),
        ("os", report.os),
        ("arch", report.arch),
        ("python_version", report.python_version),
        ("uv_version", report.uv_version),
        ("uv_lock_sha256", report.uv_lock_sha256),
        ("julia_executable_version", report.julia_executable_version),
        ("julia_manifest_version", report.julia_manifest_version),
        ("julia_manifest_sha256", report.julia_manifest_sha256),
        ("git_commit", report.git_commit),
        ("git_dirty", report.git_dirty),
        ("environment_lock_hash", report.environment_lock_hash),
    ]
    lines = ["# Environment report", "", "| key | value |", "|---|---|"]
    lines += [f"| `{k}` | `{v}` |" for k, v in rows]
    lines += ["", "## Julia packages", "", "| package | version |", "|---|---|"]
    lines += [f"| `{k}` | `{v}` |" for k, v in sorted(report.julia_packages.items())]
    return "\n".join(lines) + "\n"


def write_environment_report(report: EnvironmentReport, md_path: Path, json_path: Path) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_markdown(report), encoding="utf-8")
    json_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
```

- [ ] **Step 4: Прогнать тесты, lint, mypy**

```bash
cd /Users/george/Documents/so_field && uv run pytest tests/unit/test_environment_report.py -q && uv run ruff check . && uv run mypy
```
Ожидается: `3 passed`.

- [ ] **Step 5: Commit**

```bash
cd /Users/george/Documents/so_field && git add src/so_recon/environment/report.py tests/unit/test_environment_report.py && git commit -m "feat(e00): add environment report with python/julia versions and lock hashes

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 11: Сценарий smoke и CLI `so-recon`

**Files:**
- Create: `src/so_recon/smoke.py`, `src/so_recon/cli.py`, `tests/conftest.py`
- Test: `tests/unit/test_smoke.py`, `tests/unit/test_cli.py`

**Interfaces:**
- Consumes: всё из Tasks 2–10.
- Produces (`so_recon.smoke`):
  - `class SmokeExpectation(StrictModel)`: `frozen_at: str`, `fixture_content_hash: str`, `nx: int`, `n_steps: int`, `cumulative_oil_m3: float`, `cumulative_water_injected_m3: float`, `mean_so_final: float`, `julia_version: str`, `jutuldarcy_version: str`.
  - `EXPECTED_FILENAME = "smoke_expected.json"` (лежит в `paths.configs`).
  - `compare_with_expected(expected: SmokeExpectation, fixture_hash: str, result: JuliaSmokeResult, rel_tol: float) -> list[str]` — список расхождений (пустой = совпало). Хеш и целые сравниваются точно; float — `abs(a-b) <= rel_tol*max(1,|b|)`; расхождение версий Julia/JutulDarcy — **не ошибка**, а заметка вида `note: julia_version 1.12.7 -> 1.12.8` (добавляется в `notes`, не в mismatches).
  - `run_smoke(*, cfg: ProjectConfig, paths: ProjectPaths, argv: list[str], launcher: JuliaLauncher, freeze_expected: bool = False) -> RunContext` — шаги: `RunContext.start(command="smoke")` → `configure_logging` → fixture в `run_dir/fixture/` → Julia в `run_dir/julia_smoke.json` → если `freeze_expected`: записать `configs/smoke_expected.json` и `PASS`; иначе, если файл ожиданий отсутствует — `FAIL` с заметкой `expected file missing; run with --freeze-expected`; иначе сравнить → `PASS`/`FAIL`. Любое исключение Julia/IO → `FAIL` с текстом в `notes`, исключение пробрасывается дальше после записи `run.json`.
- Produces (`so_recon.cli`):
  - `main(argv: Sequence[str] | None = None, *, launcher_factory: Callable[[ProjectPaths, JuliaConfig], JuliaLauncher] | None = None) -> int`.
  - Общие флаги: `--config PATH` (по умолчанию `<root>/configs/project.yml`), `--root PATH` (по умолчанию `find_repo_root()`).
  - `so-recon manifest` → `reports/manifests/source_manifest.json`, run record `command="manifest"`, `raw_input_hashes` заполнены; код 0/1.
  - `so-recon env-report` → `reports/environment_report.md` + `reports/manifests/environment.json`, run record `command="env-report"`.
  - `so-recon smoke [--freeze-expected] [--julia PATH]` → код 0 при `PASS`, 1 при `FAIL`; печатает `run_dir` и статус в stdout.

- [ ] **Step 1: Написать `tests/conftest.py` с фикстурой временного проекта**

```python
from collections.abc import Callable
from pathlib import Path

import pytest

PROJECT_YAML = """
spec_version: "3.0"
config_version: "test.1"
paths: {}
sources:
  files:
    - {name: a, path: data/raw/a.csv, encoding: utf-8, delimiter: ",", decimal: "."}
    - {name: b, path: data/raw/b.csv, encoding: cp1251, delimiter: ";", decimal: ","}
smoke: {seed: 11, n_wells: 2, n_months: 2, nx: 5, n_steps: 2, rel_tol: 1.0e-6}
julia: {}
"""


@pytest.fixture
def tmp_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A minimal repository layout: marker files, config, two fake raw sources."""
    monkeypatch.delenv("SO_RECON_ROOT", raising=False)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    (tmp_path / "src" / "so_recon").mkdir(parents=True)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "project.yml").write_text(PROJECT_YAML, encoding="utf-8")
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "raw" / "a.csv").write_bytes(b"x,y\n1,2\n")
    (tmp_path / "data" / "raw" / "b.csv").write_bytes(b"x;y\r\n1;2\r\n")
    (tmp_path / "julia" / "smoke").mkdir(parents=True)
    (tmp_path / "julia" / "smoke" / "smoke_case.jl").write_text("# fake\n")
    (tmp_path / "uv.lock").write_text("lock\n")
    return tmp_path


@pytest.fixture
def fake_launcher_factory() -> Callable[[dict[str, object]], object]:
    import json

    class FakeLauncher:
        def __init__(self, payload: dict[str, object]) -> None:
            self.payload = payload
            self.calls = 0

        def launch(self, script: Path, args: list[str], out_path: Path) -> None:
            self.calls += 1
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(self.payload))

    return FakeLauncher
```

- [ ] **Step 2: Написать падающие тесты для `smoke.py`**

Тесты вызывают `run_smoke` несколько раз подряд с одинаковым конфигом; уникальность `run_id` обеспечивает суффиксация из Task 5.

`tests/unit/test_smoke.py`:
```python
import json
from pathlib import Path
from typing import Any

import pytest

from so_recon.config.load import load_project_config
from so_recon.paths import ProjectPaths
from so_recon.simulator.julia_bridge import JuliaRunError, JuliaSmokeResult
from so_recon.smoke import EXPECTED_FILENAME, SmokeExpectation, compare_with_expected, run_smoke

OK = {
    "status": "ok", "julia_version": "1.12.7", "jutuldarcy_version": "0.3.11",
    "jutul_version": "0.4.40", "nx": 5, "n_steps": 2, "cumulative_oil_m3": 100.0,
    "cumulative_water_injected_m3": 150.0, "mean_so_final": 0.6, "wall_time_s": 1.0,
}


def _setup(tmp_project: Path) -> tuple[Any, ProjectPaths]:
    cfg = load_project_config(tmp_project / "configs" / "project.yml")
    return cfg, ProjectPaths.from_config(tmp_project, cfg.paths)


def test_compare_with_expected_tolerances() -> None:
    exp = SmokeExpectation(
        frozen_at="t", fixture_content_hash="h", nx=5, n_steps=2, cumulative_oil_m3=100.0,
        cumulative_water_injected_m3=150.0, mean_so_final=0.6, julia_version="1.12.7",
        jutuldarcy_version="0.3.11",
    )
    res = JuliaSmokeResult.model_validate({**OK, "cumulative_oil_m3": 100.0 + 5e-5})
    assert compare_with_expected(exp, "h", res, rel_tol=1e-6) == []
    res_bad = JuliaSmokeResult.model_validate({**OK, "cumulative_oil_m3": 101.0})
    mism = compare_with_expected(exp, "h", res_bad, rel_tol=1e-6)
    assert any("cumulative_oil_m3" in m for m in mism)
    assert any("fixture_content_hash" in m for m in compare_with_expected(exp, "x", res, 1e-6))


def test_run_smoke_fails_without_expected_then_freezes_then_passes(
    tmp_project: Path, fake_launcher_factory: Any
) -> None:
    cfg, paths = _setup(tmp_project)
    launcher = fake_launcher_factory(OK)

    ctx = run_smoke(cfg=cfg, paths=paths, argv=["smoke"], launcher=launcher)
    assert ctx.record.status == "FAIL"
    assert any("freeze-expected" in n for n in ctx.record.notes)

    ctx2 = run_smoke(cfg=cfg, paths=paths, argv=["smoke"], launcher=launcher, freeze_expected=True)
    assert ctx2.record.status == "PASS"
    expected_path = paths.configs / EXPECTED_FILENAME
    assert expected_path.is_file()
    frozen = json.loads(expected_path.read_text())
    assert frozen["nx"] == 5 and frozen["cumulative_oil_m3"] == 100.0

    ctx3 = run_smoke(cfg=cfg, paths=paths, argv=["smoke"], launcher=launcher)
    assert ctx3.record.status == "PASS"
    assert ctx3.record.julia_version == "1.12.7"
    assert ctx3.record.jutuldarcy_version == "0.3.11"
    assert (ctx3.run_dir / "fixture" / "wells.parquet").is_file()
    assert (ctx3.run_dir / "julia_smoke.json").is_file()
    assert (ctx3.run_dir / "run.log").is_file()
    assert ctx3.record.outputs["julia_smoke"].endswith("julia_smoke.json")


def test_run_smoke_detects_drift(tmp_project: Path, fake_launcher_factory: Any) -> None:
    cfg, paths = _setup(tmp_project)
    run_smoke(cfg=cfg, paths=paths, argv=["smoke"], launcher=fake_launcher_factory(OK),
              freeze_expected=True)
    drifted = fake_launcher_factory({**OK, "mean_so_final": 0.61})
    ctx = run_smoke(cfg=cfg, paths=paths, argv=["smoke"], launcher=drifted)
    assert ctx.record.status == "FAIL"
    assert any("mean_so_final" in n for n in ctx.record.notes)


def test_run_smoke_records_failure_and_reraises(tmp_project: Path, fake_launcher_factory: Any) -> None:
    cfg, paths = _setup(tmp_project)
    broken = fake_launcher_factory({"status": "error", "message": "solver blew up", "wall_time_s": 0})
    with pytest.raises(JuliaRunError):
        run_smoke(cfg=cfg, paths=paths, argv=["smoke"], launcher=broken)
    runs = sorted(paths.runs.iterdir())
    record = json.loads((runs[-1] / "run.json").read_text())
    assert record["status"] == "FAIL"
    assert any("solver blew up" in n for n in record["notes"])
```

- [ ] **Step 3: Написать падающие тесты для CLI**

`tests/unit/test_cli.py`:
```python
import json
from pathlib import Path
from typing import Any

from so_recon.cli import main

OK = {
    "status": "ok", "julia_version": "1.12.7", "jutuldarcy_version": "0.3.11",
    "jutul_version": "0.4.40", "nx": 5, "n_steps": 2, "cumulative_oil_m3": 100.0,
    "cumulative_water_injected_m3": 150.0, "mean_so_final": 0.6, "wall_time_s": 1.0,
}


def test_manifest_command_writes_manifest_and_run_record(tmp_project: Path) -> None:
    code = main(["--root", str(tmp_project), "manifest"])
    assert code == 0
    manifest = json.loads((tmp_project / "reports" / "manifests" / "source_manifest.json").read_text())
    assert [s["name"] for s in manifest["sources"]] == ["a", "b"]
    runs = list((tmp_project / "artifacts" / "runs").iterdir())
    assert len(runs) == 1
    record = json.loads((runs[0] / "run.json").read_text())
    assert record["command"] == "manifest"
    assert record["status"] == "PASS"
    assert set(record["raw_input_hashes"]) == {"a", "b"}


def test_manifest_command_fails_on_missing_source(tmp_project: Path) -> None:
    (tmp_project / "data" / "raw" / "b.csv").unlink()
    assert main(["--root", str(tmp_project), "manifest"]) == 1
    runs = list((tmp_project / "artifacts" / "runs").iterdir())
    record = json.loads((runs[0] / "run.json").read_text())
    assert record["status"] == "FAIL"
    assert any("data/raw/b.csv" in n for n in record["notes"])


def test_env_report_command(tmp_project: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("PATH", str(tmp_project))
    monkeypatch.setenv("HOME", str(tmp_project))
    assert main(["--root", str(tmp_project), "env-report"]) == 0
    assert (tmp_project / "reports" / "environment_report.md").is_file()
    env = json.loads((tmp_project / "reports" / "manifests" / "environment.json").read_text())
    assert env["uv_lock_sha256"] is not None


def test_smoke_command_freeze_then_pass(tmp_project: Path, fake_launcher_factory: Any, capsys: Any) -> None:
    launcher = fake_launcher_factory(OK)
    factory = lambda paths, cfg: launcher  # noqa: E731
    assert main(["--root", str(tmp_project), "smoke", "--freeze-expected"], launcher_factory=factory) == 0
    assert main(["--root", str(tmp_project), "smoke"], launcher_factory=factory) == 0
    out = capsys.readouterr().out
    assert "status=PASS" in out
    assert "artifacts/runs/" in out


def test_smoke_command_returns_1_on_fail(tmp_project: Path, fake_launcher_factory: Any) -> None:
    launcher = fake_launcher_factory(OK)
    assert main(["--root", str(tmp_project), "smoke"], launcher_factory=lambda p, c: launcher) == 1


def test_unknown_command_returns_2(tmp_project: Path) -> None:
    import pytest

    with pytest.raises(SystemExit) as exc:
        main(["--root", str(tmp_project), "nope"])
    assert exc.value.code == 2
```

- [ ] **Step 4: Убедиться, что тесты падают**

```bash
cd /Users/george/Documents/so_field && uv run pytest tests/unit/test_smoke.py tests/unit/test_cli.py -q
```
Ожидается: `ModuleNotFoundError` для `so_recon.smoke` и `so_recon.cli`.

- [ ] **Step 5: Реализовать `smoke.py`**

```python
"""E00 smoke scenario: deterministic fixture + JutulDarcy smoke case + comparison with frozen expectations."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from so_recon.config.schema import ProjectConfig, StrictModel
from so_recon.logging_setup import configure_logging
from so_recon.paths import ProjectPaths
from so_recon.registry.run import RunContext
from so_recon.simulator.julia_bridge import JuliaLauncher, JuliaSmokeResult, run_julia_smoke
from so_recon.synthetic.fixture import build_smoke_fixture, write_smoke_fixture

EXPECTED_FILENAME = "smoke_expected.json"
SCHEMA_VERSIONS = {"smoke_fixture": "1", "julia_smoke_result": "1", "run_record": "1"}


class SmokeExpectation(StrictModel):
    frozen_at: str
    fixture_content_hash: str
    nx: int
    n_steps: int
    cumulative_oil_m3: float
    cumulative_water_injected_m3: float
    mean_so_final: float
    julia_version: str
    jutuldarcy_version: str


def _close(a: float, b: float, rel_tol: float) -> bool:
    return abs(a - b) <= rel_tol * max(1.0, abs(b))


def compare_with_expected(
    expected: SmokeExpectation, fixture_hash: str, result: JuliaSmokeResult, rel_tol: float
) -> list[str]:
    mismatches: list[str] = []
    if fixture_hash != expected.fixture_content_hash:
        mismatches.append(
            f"fixture_content_hash: got {fixture_hash}, expected {expected.fixture_content_hash}"
        )
    for name in ("nx", "n_steps"):
        got, exp = getattr(result, name), getattr(expected, name)
        if got != exp:
            mismatches.append(f"{name}: got {got}, expected {exp}")
    for name in ("cumulative_oil_m3", "cumulative_water_injected_m3", "mean_so_final"):
        got, exp = getattr(result, name), getattr(expected, name)
        if not _close(got, exp, rel_tol):
            mismatches.append(f"{name}: got {got!r}, expected {exp!r} (rel_tol={rel_tol})")
    return mismatches


def version_notes(expected: SmokeExpectation, result: JuliaSmokeResult) -> list[str]:
    notes: list[str] = []
    if expected.julia_version != result.julia_version:
        notes.append(f"note: julia_version {expected.julia_version} -> {result.julia_version}")
    if expected.jutuldarcy_version != result.jutuldarcy_version:
        notes.append(
            f"note: jutuldarcy_version {expected.jutuldarcy_version} -> {result.jutuldarcy_version}"
        )
    return notes


def _write_expected(path: Path, expectation: SmokeExpectation) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(expectation.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_smoke(
    *,
    cfg: ProjectConfig,
    paths: ProjectPaths,
    argv: list[str],
    launcher: JuliaLauncher,
    freeze_expected: bool = False,
) -> RunContext:
    paths.ensure_dirs()
    ctx = RunContext.start(
        command="smoke", argv=argv, cfg=cfg, paths=paths, schema_versions=SCHEMA_VERSIONS
    )
    log = configure_logging(ctx.run_id, ctx.run_dir / "run.log")
    outputs: dict[str, str] = {"run_log": paths.relative(ctx.run_dir / "run.log")}
    try:
        fixture = build_smoke_fixture(cfg.smoke)
        write_smoke_fixture(fixture, ctx.run_dir / "fixture")
        outputs["fixture_dir"] = paths.relative(ctx.run_dir / "fixture")
        log.info("fixture built: seed=%d hash=%s", fixture.seed, fixture.content_hash)

        julia_out = ctx.run_dir / "julia_smoke.json"
        result = run_julia_smoke(
            launcher, paths.root / cfg.julia.smoke_script, julia_out,
            nx=cfg.smoke.nx, n_steps=cfg.smoke.n_steps,
        )
        outputs["julia_smoke"] = paths.relative(julia_out)
        ctx.update(julia_version=result.julia_version, jutuldarcy_version=result.jutuldarcy_version)
        log.info(
            "julia smoke ok: oil=%.6g winj=%.6g mean_so=%.6f (%.1fs)",
            result.cumulative_oil_m3, result.cumulative_water_injected_m3,
            result.mean_so_final, result.wall_time_s,
        )

        expected_path = paths.configs / EXPECTED_FILENAME
        if freeze_expected:
            expectation = SmokeExpectation(
                frozen_at=datetime.now(UTC).isoformat(),
                fixture_content_hash=fixture.content_hash,
                nx=result.nx,
                n_steps=result.n_steps,
                cumulative_oil_m3=result.cumulative_oil_m3,
                cumulative_water_injected_m3=result.cumulative_water_injected_m3,
                mean_so_final=result.mean_so_final,
                julia_version=result.julia_version,
                jutuldarcy_version=result.jutuldarcy_version,
            )
            _write_expected(expected_path, expectation)
            outputs["smoke_expected"] = paths.relative(expected_path)
            ctx.finish("PASS", outputs, notes=[f"expected frozen to {paths.relative(expected_path)}"])
            return ctx

        if not expected_path.is_file():
            ctx.finish(
                "FAIL", outputs,
                notes=[f"expected file missing: {paths.relative(expected_path)}; "
                       "run with --freeze-expected"],
            )
            return ctx

        expected = SmokeExpectation.model_validate(
            json.loads(expected_path.read_text(encoding="utf-8"))
        )
        mismatches = compare_with_expected(expected, fixture.content_hash, result, cfg.smoke.rel_tol)
        notes = version_notes(expected, result)
        if mismatches:
            for m in mismatches:
                log.error("smoke mismatch: %s", m)
            ctx.finish("FAIL", outputs, notes=[*notes, *mismatches])
        else:
            ctx.finish("PASS", outputs, notes=notes)
        return ctx
    except Exception as exc:
        log.exception("smoke failed")
        ctx.finish("FAIL", outputs, notes=[f"exception: {type(exc).__name__}: {exc}"])
        raise
    finally:
        logging.shutdown()
```

- [ ] **Step 6: Реализовать `cli.py`**

```python
"""so-recon command line: manifest | env-report | smoke."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from so_recon.config.load import ConfigError, load_project_config
from so_recon.config.schema import JuliaConfig, ProjectConfig
from so_recon.environment.report import collect_environment, write_environment_report
from so_recon.logging_setup import configure_logging
from so_recon.paths import ProjectPaths, find_repo_root
from so_recon.registry.run import RunContext
from so_recon.registry.source_manifest import (
    MissingSourceError,
    build_source_manifest,
    write_source_manifest,
)
from so_recon.simulator.julia_bridge import (
    JuliaLauncher,
    JuliaNotFoundError,
    JuliaRunError,
    default_launcher,
)
from so_recon.smoke import run_smoke

LauncherFactory = Callable[[ProjectPaths, JuliaConfig], JuliaLauncher]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="so-recon", description="SO-RECON project CLI")
    p.add_argument("--root", type=Path, default=None, help="repository root (default: auto)")
    p.add_argument("--config", type=Path, default=None, help="path to project.yml")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("manifest", help="hash raw sources into reports/manifests/source_manifest.json")
    sub.add_parser("env-report", help="write reports/environment_report.md")
    smoke = sub.add_parser("smoke", help="run deterministic fixture + JutulDarcy smoke case")
    smoke.add_argument("--freeze-expected", action="store_true",
                       help="write configs/smoke_expected.json from this run")
    smoke.add_argument("--julia", default=None, help="path to julia executable")
    return p


def _load(args: argparse.Namespace) -> tuple[ProjectConfig, ProjectPaths]:
    root = args.root.resolve() if args.root else find_repo_root()
    config_path = args.config or (root / "configs" / "project.yml")
    cfg = load_project_config(config_path)
    return cfg, ProjectPaths.from_config(root, cfg.paths)


def _cmd_manifest(cfg: ProjectConfig, paths: ProjectPaths, argv: list[str]) -> int:
    paths.ensure_dirs()
    ctx = RunContext.start(command="manifest", argv=argv, cfg=cfg, paths=paths)
    log = configure_logging(ctx.run_id, ctx.run_dir / "run.log")
    try:
        manifest = build_source_manifest(
            cfg.sources, paths, config_version=cfg.config_version,
            git_commit=ctx.record.git_commit, now=datetime.now(UTC),
        )
    except MissingSourceError as exc:
        log.error("missing required sources: %s", exc.missing)
        ctx.finish("FAIL", {}, notes=[f"missing: {m}" for m in exc.missing])
        return 1
    out = paths.manifests / "source_manifest.json"
    write_source_manifest(manifest, out)
    ctx.update(raw_input_hashes=manifest.hashes())
    ctx.finish("PASS", {"source_manifest": paths.relative(out)})
    print(f"run_id={ctx.run_id} status=PASS manifest={paths.relative(out)}")
    return 0


def _cmd_env_report(cfg: ProjectConfig, paths: ProjectPaths, argv: list[str]) -> int:
    paths.ensure_dirs()
    ctx = RunContext.start(command="env-report", argv=argv, cfg=cfg, paths=paths)
    configure_logging(ctx.run_id, ctx.run_dir / "run.log")
    report = collect_environment(paths, now=datetime.now(UTC))
    md = paths.reports / "environment_report.md"
    js = paths.manifests / "environment.json"
    write_environment_report(report, md, js)
    ctx.update(
        julia_version=report.julia_manifest_version,
        jutuldarcy_version=report.julia_packages.get("JutulDarcy"),
    )
    ctx.finish("PASS", {"environment_report": paths.relative(md), "environment_json": paths.relative(js)})
    print(f"run_id={ctx.run_id} status=PASS report={paths.relative(md)}")
    return 0


def _cmd_smoke(
    cfg: ProjectConfig, paths: ProjectPaths, argv: list[str], args: argparse.Namespace,
    launcher_factory: LauncherFactory | None,
) -> int:
    if launcher_factory is not None:
        launcher: JuliaLauncher = launcher_factory(paths, cfg.julia)
    else:
        launcher = default_launcher(paths, cfg.julia, args.julia)
    ctx = run_smoke(cfg=cfg, paths=paths, argv=argv, launcher=launcher,
                    freeze_expected=args.freeze_expected)
    print(f"run_id={ctx.run_id} status={ctx.record.status} run_dir={paths.relative(ctx.run_dir)}")
    for note in ctx.record.notes:
        print(f"  {note}")
    return 0 if ctx.record.status == "PASS" else 1


def main(argv: Sequence[str] | None = None, *, launcher_factory: LauncherFactory | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(raw_argv)
    try:
        cfg, paths = _load(args)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 1
    full_argv = ["so-recon", *raw_argv]
    try:
        if args.command == "manifest":
            return _cmd_manifest(cfg, paths, full_argv)
        if args.command == "env-report":
            return _cmd_env_report(cfg, paths, full_argv)
        return _cmd_smoke(cfg, paths, full_argv, args, launcher_factory)
    except (JuliaNotFoundError, JuliaRunError) as exc:
        # run.json already carries status=FAIL and the note (see run_smoke)
        print(f"julia error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 7: Прогнать все unit-тесты, lint, mypy**

```bash
cd /Users/george/Documents/so_field && uv run pytest -q -m "not julia" && uv run ruff check . && uv run ruff format --check . && uv run mypy
```
Ожидается: все тесты проходят (около 45), ruff и mypy без ошибок. При замечаниях `ruff format` — выполнить `uv run ruff format .` и перепроверить.

- [ ] **Step 8: Commit**

```bash
cd /Users/george/Documents/so_field && git add src/so_recon/smoke.py src/so_recon/cli.py tests/conftest.py tests/unit/test_smoke.py tests/unit/test_cli.py && git commit -m "feat(e00): add so-recon CLI with manifest, env-report and smoke commands

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 12: Прогон gate, заморозка ожиданий, отчёт `reports/stages/E00.md`

**Files:**
- Create: `configs/smoke_expected.json` (генерируется), `reports/manifests/source_manifest.json`, `reports/manifests/environment.json`, `reports/environment_report.md` (генерируются), `reports/stages/E00.md`
- Modify: `README.md` (при необходимости уточнить команды)

**Interfaces:**
- Consumes: CLI (Task 11), `make gate` (Task 1).
- Produces: статус этапа E00 и разрешённые входы для E01/E05.

- [ ] **Step 1: Заморозить ожидания smoke на реальной Julia**

```bash
cd /Users/george/Documents/so_field && uv run so-recon smoke --freeze-expected && cat configs/smoke_expected.json
```
Ожидается: `status=PASS`, файл содержит `fixture_content_hash`, `cumulative_oil_m3`, `mean_so_final`, `julia_version`, `jutuldarcy_version`.

- [ ] **Step 2: Повторный прогон без заморозки должен пройти**

```bash
cd /Users/george/Documents/so_field && uv run so-recon smoke && ls artifacts/runs | tail -3
```
Ожидается: `status=PASS`, без заметок о расхождениях.

- [ ] **Step 3: Сформировать source manifest и environment report на реальных данных**

```bash
cd /Users/george/Documents/so_field && uv run so-recon manifest && uv run so-recon env-report && cat reports/manifests/source_manifest.json && cat reports/environment_report.md
```
Проверить вручную по `DATA_AUDIT.md` §2: `line_count` для `coords` = 4 241, `gis` = 107 176, `mer` = 1 388 595, `perf` = 33 236, `plastoper` = 3 572 (данные + заголовок; `wc -l` показывал именно эти значения). Если число отличается — **не править код**: зафиксировать расхождение в E00.md как замечание к `DATA_AUDIT.md`.

- [ ] **Step 4: Полный gate на чистой установке**

```bash
cd /Users/george/Documents/so_field && make gate 2>&1 | tail -40
```
Ожидается: `uv sync --frozen` без изменения `uv.lock`; `Pkg.instantiate()` без изменения `Manifest.toml`; три команды CLI со `status=PASS`; pytest — все пройдены, включая `tests/integration/test_julia_smoke.py` (не skipped); ruff и mypy чисты. Проверить, что lock-файлы не изменились:
```bash
cd /Users/george/Documents/so_field && git status --short uv.lock julia/Manifest.toml && echo "(empty above = locks unchanged)"
```

- [ ] **Step 5: Написать `reports/stages/E00.md`**

Заполнить фактическими значениями из `artifacts/runs/<run_id>/run.json`, `reports/manifests/source_manifest.json`, `reports/environment_report.md`, `configs/smoke_expected.json`. Шаблон:

```markdown
# E00 — Основание проекта и воспроизводимое окружение

**Статус:** PASS | PASS_WITH_LIMITATIONS  (выбрать по результату Step 4)
**Дата:** <YYYY-MM-DD>
**SPEC.md:** 3.0 · **STAGES.md:** 3.0 · **config_version:** E00.1
**Git commit отчёта:** <sha из `git rev-parse HEAD` после финального коммита кода; сам отчёт коммитится следом>

## 1. Входные артефакты и hashes

Источник: `reports/manifests/source_manifest.json` (run_id `<...>`).

| файл | sha256 | байт | строк |
|---|---|---:|---:|
| data/raw/coords.csv | <sha> | <n> | <n> |
| data/raw/gis.csv | <sha> | <n> | <n> |
| data/raw/mer.csv | <sha> | <n> | <n> |
| data/raw/perf.csv | <sha> | <n> | <n> |
| data/raw/plastoper.csv | <sha> | <n> | <n> |

Сверка с `DATA_AUDIT.md` §2: <совпадает / перечислить расхождения>.

## 2. Окружение

Источник: `reports/environment_report.md`.

- Python <ver>, uv <ver>, `uv.lock` sha256 `<...>`
- Julia <ver>, `julia/Manifest.toml` sha256 `<...>`, JutulDarcy <ver>, Jutul <ver>
- environment_lock_hash `<...>`
- ОС/арх: <...>

## 3. Выполненные команды

    make gate
    uv run so-recon smoke --freeze-expected   # однократно, run_id <...>
    uv run so-recon smoke                     # run_id <...>, PASS
    uv run so-recon manifest                  # run_id <...>, PASS
    uv run so-recon env-report                # run_id <...>, PASS
    uv run pytest -q                          # <N> passed

## 4. Результаты проверок

- Smoke fixture: content_hash `<...>` (seed 20260913, 4 скважины × 12 месяцев).
- JutulDarcy smoke (1D, nx=20, 12 шагов по 30 сут): cumulative_oil_m3 = <...>, cumulative_water_injected_m3 = <...>, mean_so_final = <...>; повторный прогон в допуске rel_tol = 1e-6.
- Тесты: <N> passed, 0 failed; integration-тест Julia выполнен (не skipped).
- Lint/типизация: ruff clean, mypy strict clean.
- Guard абсолютных путей: `tests/test_no_absolute_paths.py` PASS.
- Lock-файлы после `make gate` не изменились.

## 5. Созданные артефакты

- `pyproject.toml`, `uv.lock`, `.python-version`, `Makefile`, `README.md`
- `julia/Project.toml`, `julia/Manifest.toml`, `julia/smoke/smoke_case.jl`
- `configs/project.yml`, `configs/smoke_expected.json`
- `src/so_recon/` (paths, config, registry, environment, synthetic, simulator, smoke, cli)
- `reports/manifests/source_manifest.json`, `reports/manifests/environment.json`, `reports/environment_report.md`
- `artifacts/runs/<run_id>/` (локально, не в git)

## 6. Ограничения

- CI не настроен (нет remote; precompile JutulDarcy в CI дорог). Роль чистой установки выполняет `make gate`.
- Julia-smoke — проверка работоспособности окружения, не верификация физики (E05).
- Raw CSV перенесены в `data/raw/` локально; хеши совпали с исходными.
- Расхождения документации: `STAGES.md` §8 ссылается на `docs/README_SO_RECON.md`, фактический файл `docs/README.md`. <другие найденные>

## 7. Что передаётся следующим этапам

- E01: `ProjectPaths`, `ProjectConfig`/`SourceFileSpec` (encoding/delimiter/decimal для пяти файлов), `RunContext`, `source_manifest.json` как эталон входов, `hash_and_count`.
- E05: `julia/` окружение с зафиксированным JutulDarcy, `SubprocessJuliaLauncher`/`JuliaLauncher` как точка расширения, шаблон JSON-обмена.
- Все этапы: правило «один запуск = один `run_id` + `run.json` + `resolved_config.json` + `run.log`».
```

- [ ] **Step 6: Финальные проверки и коммит отчёта и сгенерированных файлов**

```bash
cd /Users/george/Documents/so_field && uv run pytest -q && uv run ruff check . && uv run mypy && git add configs/smoke_expected.json reports/manifests/source_manifest.json reports/manifests/environment.json reports/environment_report.md reports/stages/E00.md README.md && git commit -m "docs(e00): freeze smoke expectations, add source/environment manifests and stage report

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

Затем проставить в `reports/stages/E00.md` фактический commit hash кода (если он указан как предыдущий коммит — ничего менять не нужно) и убедиться, что `git status` чист, кроме `.idea/workspace.xml`.

---

## Self-Review

**Spec coverage (STAGES E00 «Основная работа» → задачи):**
- структура Python/Julia-проекта и typed configuration → Tasks 1, 4, 8;
- lock-файлы Python, Julia, JutulDarcy → Tasks 1 (`uv.lock`), 8 (`Manifest.toml`, compat 0.3);
- каталоги raw/interim/processed/artifacts/reports → Task 3 (`ProjectPaths`), Task 4 (`PathsConfig`), Task 6 (перенос raw);
- manifest исходных файлов, configs и environment → Task 6 (source), Task 5 (`resolved_config.json`, `run.json`), Task 10 (environment);
- минимальный synthetic fixture и smoke test → Tasks 7, 8, 9, 11;
- единый CLI/run ID и правила логирования → Tasks 5, 11.
**Ключевые выходы:** `pyproject.toml`+`uv.lock` (T1), `Project.toml`+`Manifest.toml` (T8), каркас каталогов (T1, T8), `source_manifest.json` (T6/T12), `environment_report.md` (T10/T12), `reports/stages/E00.md` (T12).
**Gate:** чистая установка → `make gate` (T1, T12); детерминированный fixture → T7 + T8 Step 5 + `smoke_expected.json` (T12); версии и hashes → `run.json`, manifests (T5, T6, T10); нет абсолютных путей → `tests/test_no_absolute_paths.py` (T3) и валидаторы относительных путей в конфиге (T4).
**SPEC 19.12 поля lineage в `RunRecord`:** raw hashes ✓, schema versions ✓, git commit ✓, environment lock hash ✓, Julia/Jutul версия ✓, checkpoint hash (поле, `None`) ✓, resolved config hash ✓, parent IDs ✓, timestamp ✓, command/run ID ✓.

**Placeholder scan:** значения `<...>` присутствуют только в шаблоне отчёта Task 12 Step 5 и обозначают фактические результаты прогона; все кодовые шаги содержат полный код.

**Type consistency:** `ProjectPaths.from_config(root, cfg.paths)` используется одинаково в T9, T11; `RunContext.start(command=, argv=, cfg=, paths=, ...)` — T5 сигнатура, T11 вызовы; `run_julia_smoke(launcher, script, out_path, *, nx, n_steps)` — T9 определение, T9/T11 вызовы; `JuliaSmokeResult` поля совпадают с JSON-ключами `smoke_case.jl` (T8) и `SmokeExpectation` (T11); `environment_lock_hash(paths)` — T5 определение, T10 использование; `hash_and_count` возвращает `(sha, size, lines)` — T6 тесты и код согласованы.
