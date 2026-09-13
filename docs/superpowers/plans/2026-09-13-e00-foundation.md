# E00 Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Revision 2 (13.09.2026).** Учтены 12 замечаний к ревизии 1. Сводка изменений — раздел «Изменения ревизии 2».

**Goal:** Создать воспроизводимое основание проекта SO-RECON: Python/Julia-окружение с lock-файлами, typed configuration, каталоги данных, manifest исходных файлов, run registry с полной lineage и immutable-артефактами, единый CLI и один детерминированный end-to-end smoke, который проходит на чистой установке.

**Architecture:** Python-пакет `so_recon` (src-layout, `uv`, pydantic-конфиги) отвечает за конфигурацию, hashing, атомарную запись, artifact registry, manifests, run registry и CLI. Julia-окружение `julia/` с зафиксированным `Manifest.toml` держит JutulDarcy и smoke-скрипт. Smoke — сквозной: Python пишет `case.json`, Julia читает его, возвращает `input_sha256` прочитанного файла, Python сверяет его со своим хешем. Каждый запуск CLI получает `run_id`, каталог `artifacts/runs/<run_id>/` с `run.json`, `resolved_config.json` и логом; ни один модуль не содержит личных абсолютных путей.

**Tech Stack:** Python 3.13.2 (`uv 0.12.x`, `pydantic>=2`, `pyyaml`, `numpy`, `pyarrow`, `pytest`, `ruff`, `mypy`), Julia — точная project-pinned версия (`julia/.julia-version`), `JutulDarcy 0.3.x`, `Jutul 0.4.x`, `JSON`, `SHA`; Make + bash-скрипты, git.

**Spec:** `docs/SPEC.md` (v3.0, разделы 7.1, 19, 20), `docs/STAGES.md` (v3.0, раздел «E00»), `docs/DATA_AUDIT.md` (раздел 3 — контракты чтения файлов), `docs/README.md`.

## Изменения ревизии 2

1. `plastoper.csv` — `decimal: ","` (проверено по байтам: `269,7600098`). Во всех manifest-записях разделены `physical_line_count` и `data_rows`; для `plastoper.csv` это `3573` и `3572` (файл не заканчивается переводом строки).
2. Разрушительный `mv` исходных CSV удалён. Исходники остаются неизменяемыми на исходном пути `data/Ромашка_сырые/`; E00 их только читает и хеширует. `ProjectPaths.ensure_dirs()` больше не создаёт каталог raw.
3. Smoke стал сквозным: Python формирует `case.json`, Julia его читает и возвращает `input_sha256`; Python сверяет хеш.
4. Введён `ArtifactRef` (`sha256`, `schema_version`, `producer_run_id`, `parent_artifact_ids`); повторная запись артефакта с другим содержимым запрещена.
5. Любое исключение в любой CLI-команде приводит к записи `run.json` со `status: FAIL` — включая отсутствие Julia, обнаруженное до запуска симуляции.
6. Gate переписан как bash-скрипт с `set -euo pipefail` и `tee`; добавлен `ruff format --check`; добавлена отдельная цель `gate-clean` с временным `JULIA_DEPOT_PATH`.
7. Версии Python и Julia зафиксированы точно. Julia описывается как **project-pinned**, а не как «актуальная stable».
8. Запрещены `..`, Windows absolute/UNC пути и выход за пределы repository root через symlink.
9. Несовпадение версий Julia/Jutul/JutulDarcy с lock — **FAIL**, а не заметка.
10. Детерминированные manifests отделены от run-timestamps: повторный gate не создаёт diff в git.
11. Из всех команд плана удалены личные абсолютные пути; команды выполняются из корня репозитория.
12. `run.json`, `resolved_config.json` и все manifest-файлы пишутся атомарно.

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

**Инварианты ревизии 2 (обязательны для каждой задачи):**

- **I1 — Immutability источников.** Ни один код E00 не пишет, не перемещает и не удаляет файлы внутри каталога источников. Источники открываются только в режиме `"rb"`.
- **I2 — Immutability результатов.** Артефакт, записанный под конкретным путём, не перезаписывается другим содержимым. Повторная запись с идентичным содержимым допустима и идемпотентна. Инвариант распространяется на **результаты** (`ArtifactRef`), но не на `run.json`: это журнал запуска, который по определению переходит `RUNNING → PASS|FAIL` и потому переписывается атомарно (инвариант I3). Публикуемые копии в `reports/` тоже не immutable: их изменение означает реальное изменение данных и фиксируется историей git.
- **I3 — Атомарность.** Каждый JSON/Markdown-артефакт пишется через временный файл в том же каталоге + `os.replace`.
- **I4 — Containment путей.** Любой путь из конфигурации обязан быть относительным, без `..`, без Windows drive/UNC, и после `resolve()` обязан лежать внутри repository root.
- **I5 — Детерминизм коммитируемых файлов.** Файлы в `reports/` и `configs/`, попадающие в git, не содержат timestamps, `run_id` и git-состояния. Всё это живёт в `artifacts/runs/<run_id>/` (не версионируется).
- **I6 — FAIL всегда записывается.** Любое исключение после того, как стал известен repository root, приводит к `run.json` со `status: FAIL` и текстом исключения в `notes`.
- **I7 — Никаких личных абсолютных путей** ни в коде, ни в конфигах, ни в командах.

## Фактическое состояние репозитория (на 13.09.2026)

- Репозиторий содержит `docs/`, `data/`, `.idea/`, `.gitignore` (`/data/`, `/research/`). Кода нет. Работа ведётся в ветке `e00-foundation`.
- Raw CSV лежат в `data/Ромашка_сырые/` (пять файлов) и **остаются там**. Каталог `data/` целиком в `.gitignore` — закрытые данные не версионируются (SPEC 19.15).
- Проверено локально: Python 3.13.2 (pyenv), `uv 0.12.7`, Homebrew, GNU Make **3.81**, `/bin/bash` **3.2.57**. **Julia и `juliaup` не установлены.**
- GNU Make 3.81 **не поддерживает `.SHELLFLAGS`** — заданные там `-o pipefail` молча игнорируются. Поэтому gate реализуется отдельным bash-скриптом, а `Makefile` только вызывает его.
- Целевая версия Julia — **project-pinned**, фиксируется в `julia/.julia-version` точным значением той версии, которая фактически установлена и по которой разрешён `Manifest.toml`. План не утверждает, какая версия является актуальной upstream stable.
- `STAGES.md` §8 ссылается на `/docs/README_SO_RECON.md`, фактический файл — `docs/README.md`. Зафиксировать в отчёте E00 как замечание к документации.

### Проверенные контракты исходных файлов

Проверено по байтам (`xxd`, `wc -l`, `iconv`) 13.09.2026:

| файл | кодировка | разделитель | decimal | physical_line_count | data_rows | последний байт |
|---|---|---|---|---:|---:|---|
| `coords.csv` | utf-8 | `,` | `.` | 4241 | 4240 | `\n` |
| `gis.csv` | cp1251 | `;` | `,` | 107176 | 107175 | `\n` |
| `mer.csv` | utf-8-sig | `;` | `,` | 1388595 | 1388594 | `\n` |
| `perf.csv` | cp1251 | `;` | `,` | 33236 | 33235 | `\n` |
| `plastoper.csv` | utf-8-sig | `;` | `,` | 3573 | 3572 | `;` (нет перевода строки) |

`physical_line_count` — число физических строк: количество `\n` плюс 1, если файл непуст и не заканчивается `\n`. `data_rows = physical_line_count - header_lines`, `header_lines = 1` для всех пяти файлов. Это **физический** счёт, а не число разобранных записей: E00 не парсит CSV, поэтому переводы строк внутри закавыченных полей не учитываются. Значения `data_rows` совпадают с колонкой «Строк» в `DATA_AUDIT.md` §2 для всех пяти файлов.

## Решения, принятые в плане (не меняют SPEC)

1. **Каталог источников.** Источники остаются на исходном пути `data/Ромашка_сырые/`. `PathsConfig.raw` указывает на него. Каталоги `data/interim/` и `data/processed/` создаются для последующих этапов. Никакого `mv`, `cp` или переименования исходников E00 не выполняет (инвариант I1).
2. **Что версионируется.** `reports/` (включая `reports/manifests/*.json` и `reports/environment_report.md`) и `configs/` — в git; `artifacts/` и `data/` — нет. Коммитируемые файлы детерминированы (инвариант I5): manifest содержит только имена, размеры, хеши и счётчики строк, без timestamps и `run_id`.
3. **Julia-окружение.** `julia/Project.toml` + `julia/Manifest.toml` + `julia/.julia-version` — общее окружение проекта. Пакет `julia/SOReconSimulator/` создаётся в E05. В E00 — только `julia/smoke/smoke_case.jl`.
4. **Smoke fixture** сквозной и состоит из трёх детерминированных частей: (a) синтетические таблицы `wells` и `well_month` из `numpy.random.default_rng(seed)` в Parquet, хешируемые по каноническому JSON содержимого; (b) `case.json` — канонический JSON, выводимый из fixture и конфигурации, это единственный вход Julia; (c) 1D oil–water JutulDarcy-кейс, читающий `case.json`, возвращающий `input_sha256` и скалярные summaries. Ожидаемые значения замораживаются в `configs/smoke_expected.json` с относительным допуском `rel_tol` из конфигурации.
5. **CLI** — stdlib `argparse`, точка входа `so-recon`. Подкоманды: `manifest`, `env-report`, `smoke`.
6. **Логи** — stdlib `logging`; каждая запись содержит `run_id`; пишутся в stderr и в `artifacts/runs/<run_id>/run.log`. Содержимое строк данных не логируется (SPEC 19.15).
7. **CI** не создаётся в E00: JutulDarcy precompile в CI занимает десятки минут, а remote у репозитория нет. Роль «чистой установки» выполняют `make gate` (чистое Python-окружение) и `make gate-clean` (дополнительно чистый Julia depot).

## Структура файлов

```text
so_field/
├── pyproject.toml                  # метаданные, зависимости, ruff/mypy/pytest
├── uv.lock                         # lock Python
├── .python-version                 # 3.13.2 (точная)
├── Makefile                        # тонкая обёртка над scripts/*.sh
├── scripts/
│   ├── gate.sh                     # set -euo pipefail + tee; полный gate E00
│   └── gate_clean.sh               # gate + временный JULIA_DEPOT_PATH
├── README.md
├── .gitignore
├── configs/
│   ├── project.yml                 # единый typed config
│   └── smoke_expected.json         # замороженные ожидания smoke (Task 13)
├── src/so_recon/
│   ├── __init__.py                 # __version__, SPEC_VERSION
│   ├── paths.py                    # find_repo_root, ProjectPaths, containment путей
│   ├── logging_setup.py            # configure_logging(run_id, log_file)
│   ├── cli.py                      # argparse: manifest | env-report | smoke
│   ├── smoke.py                    # оркестрация smoke
│   ├── config/
│   │   ├── schema.py               # pydantic-модели
│   │   └── load.py                 # load_project_config, resolved_config_dict, config_hash
│   ├── registry/
│   │   ├── hashing.py              # sha256_file, sha256_bytes, canonical_json, sha256_json
│   │   ├── atomic.py               # write_bytes_atomic, write_text_atomic, write_json_atomic
│   │   ├── artifact.py             # ArtifactRef, write_artifact, register_artifact
│   │   ├── gitinfo.py              # git_commit, git_is_dirty
│   │   ├── run.py                  # make_run_id, RunRecord, RunContext, environment_lock_hash
│   │   └── source_manifest.py      # hash_and_count, build_source_manifest, стамп запуска
│   ├── environment/
│   │   └── report.py               # parse_julia_manifest, locked_versions, collect_environment
│   ├── synthetic/
│   │   └── fixture.py              # build_smoke_fixture, build_smoke_case, write_*
│   └── simulator/
│       └── julia_bridge.py         # find_julia, JuliaLauncher, run_julia_smoke
├── julia/
│   ├── Project.toml                # JutulDarcy, Jutul, JSON, SHA + [compat]
│   ├── Manifest.toml               # зафиксированные версии
│   ├── .julia-version              # точная project-pinned версия Julia
│   ├── README.md
│   └── smoke/smoke_case.jl         # читает case.json → JSON с input_sha256
├── tests/
│   ├── conftest.py
│   ├── test_no_absolute_paths.py
│   ├── unit/                       # см. задачи
│   └── integration/test_julia_smoke.py   # @pytest.mark.julia
├── reports/
│   ├── environment_report.md       # детерминированный
│   ├── manifests/{source_manifest.json,environment.json}   # детерминированные
│   └── stages/E00.md
├── artifacts/.gitkeep              # runs/<run_id>/, gate/ — ignored
└── data/
    ├── Ромашка_сырые/              # НЕИЗМЕНЯЕМЫЕ источники (ignored)
    ├── interim/                    # ignored
    └── processed/                  # ignored
```

Ответственности: `registry/hashing.py` — хеши; `registry/atomic.py` — атомарная запись; `registry/artifact.py` — immutable artifact refs; `registry/run.py` — run record и lineage; `registry/source_manifest.py` — manifest источников; `config/*` — схема и загрузка; `environment/*` — отчёт об окружении и версии lock; `synthetic/fixture.py` — данные smoke и `case.json`; `simulator/julia_bridge.py` — единственное место, знающее, как запускать Julia; `smoke.py` — сценарий; `cli.py` — парсинг аргументов, run-контекст и гарантия FAIL-записи.

**Порядок задач и зависимости:**

```text
T1 scaffold
 └ T2 hashing + atomic
    ├ T3 paths (+ containment)
    │  └ T4 config schema + project.yml
    │     ├ T5 artifact registry
    │     │  └ T6 run registry + logging
    │     │     ├ T7 source manifest
    │     │     ├ T8 fixture + case.json
    │     │     │  └ T9 julia env + smoke_case.jl
    │     │     │     └ T10 julia bridge
    │     │     ├ T11 environment report (+ locked versions)
    │     │     └ T12 smoke scenario + CLI   (нужны T7, T10, T11)
    │     │        └ T13 gate + freeze + E00.md
```

---
### Task 1: Каркас Python-проекта, зависимости, lock и gate-скрипты

**Files:**
- Create: `pyproject.toml`, `.python-version`, `Makefile`, `scripts/gate.sh`, `scripts/gate_clean.sh`, `README.md`, `artifacts/.gitkeep`, `reports/.gitkeep`, `reports/manifests/.gitkeep`, `reports/stages/.gitkeep`, `configs/.gitkeep`
- Create: `src/so_recon/__init__.py`, `__init__.py` в `src/so_recon/{config,registry,environment,synthetic,simulator}/`, `tests/__init__.py`, `tests/unit/__init__.py`, `tests/integration/__init__.py`
- Modify: `.gitignore`
- Test: `tests/unit/test_package.py`

**Interfaces:**
- Produces: `so_recon.__version__ == "0.0.1"`, `so_recon.SPEC_VERSION == "3.0"`; цели `make setup|test|lint|format|smoke|manifest|env-report|gate|gate-clean`.

**Замечания ревизии 2, реализуемые здесь:** №6 (gate: pipefail/tee, `ruff format --check`, отдельный `gate-clean` с временным `JULIA_DEPOT_PATH`), №7 (точная версия Python), №11 (нет личных абсолютных путей).

> Все команды выполняются **из корня репозитория**. В плане не указываются абсолютные пути.

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
# ruff formats python code fences inside Markdown. docs/ holds the normative spec and this
# implementation plan: their fenced blocks are quoted text, not project source, and must not
# be rewritten by `ruff format`. Without this, `make lint` and `make gate` fail on the plan.
extend-exclude = ["docs"]

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

- [ ] **Step 2: Создать `.python-version` и `.gitignore`**

`.python-version` — **точная** версия интерпретатора, на которой создаётся `.venv` (замечание №7):
```text
3.13.2
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

# run artifacts and gate logs (lineage lives in run.json inside; never committed)
/artifacts/*
!/artifacts/.gitkeep

# sdd workspace
/.superpowers/

# os / ide
.DS_Store
/.idea/
```

- [ ] **Step 3: Создать `scripts/gate.sh`**

GNU Make 3.81 (установленная версия) молча игнорирует `.SHELLFLAGS`, поэтому `pipefail` обязан жить в скрипте, а не в `Makefile`.

```bash
#!/usr/bin/env bash
# E00 gate: clean Python environment, locked Julia environment, manifests, smoke, tests.
# pipefail lives here because GNU Make 3.81 silently ignores .SHELLFLAGS.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

UV="${UV:-uv}"
JULIA_BIN="${JULIA:-julia}"

LOG_DIR="artifacts/gate"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/gate-$(date -u +%Y%m%dT%H%M%SZ).log"

# Files that must be byte-identical before and after a gate run (invariant I5).
DETERMINISTIC_PATHS=(
  uv.lock
  julia/Manifest.toml
  julia/.julia-version
  configs/smoke_expected.json
  reports/manifests
  reports/environment_report.md
)

check_deterministic_outputs() {
  local dirty
  # Untracked files (??) are ignored: on a first run the artifacts do not exist yet.
  dirty="$(git status --porcelain -- "${DETERMINISTIC_PATHS[@]}" | grep -v '^??' || true)"
  if [ -n "$dirty" ]; then
    echo "gate FAIL: tracked deterministic artifacts changed during the gate run:" >&2
    echo "$dirty" >&2
    return 1
  fi
  echo "deterministic artifacts unchanged (no git pollution)"
}

main() {
  echo "== gate start $(date -u +%Y-%m-%dT%H:%M:%SZ) =="
  echo "-- 1/8 clean python environment --"
  rm -rf .venv
  "$UV" sync --frozen

  echo "-- 2/8 julia environment from lock --"
  "$JULIA_BIN" --project=julia --startup-file=no -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

  echo "-- 3/8 source manifest --"
  "$UV" run so-recon manifest

  echo "-- 4/8 environment report --"
  "$UV" run so-recon env-report

  echo "-- 5/8 end-to-end smoke --"
  "$UV" run so-recon smoke

  echo "-- 6/8 tests --"
  "$UV" run pytest -q

  echo "-- 7/8 lint, format and types --"
  "$UV" run ruff check .
  "$UV" run ruff format --check .
  "$UV" run mypy

  echo "-- 8/8 determinism of committed artifacts --"
  check_deterministic_outputs

  echo "== gate PASS $(date -u +%Y-%m-%dT%H:%M:%SZ) =="
}

main 2>&1 | tee "$LOG_FILE"
```

> `set -o pipefail` обязателен: без него `main | tee` вернул бы код `tee`, и падение gate осталось бы незамеченным. Лог пишется в `artifacts/gate/` — каталог не версионируется.

- [ ] **Step 4: Создать `scripts/gate_clean.sh`**

```bash
#!/usr/bin/env bash
# Same gate, but with a throwaway Julia depot: proves julia/Manifest.toml instantiates
# from scratch. Slow (full JutulDarcy download + precompile), so it is a separate target.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

TMP_DEPOT="$(mktemp -d "${TMPDIR:-/tmp}/so-recon-depot.XXXXXX")"
cleanup() { rm -rf "$TMP_DEPOT"; }
trap cleanup EXIT

export JULIA_DEPOT_PATH="$TMP_DEPOT"
echo "using throwaway JULIA_DEPOT_PATH=$JULIA_DEPOT_PATH"

# No exec: the EXIT trap must still run to remove the temporary depot.
"$ROOT/scripts/gate.sh"
```

Сделать оба скрипта исполняемыми: `chmod +x scripts/gate.sh scripts/gate_clean.sh`.

- [ ] **Step 5: Создать `Makefile`**

`Makefile` — тонкая обёртка; вся логика с `pipefail` в скриптах.

```makefile
UV ?= uv
JULIA ?= julia
export UV
export JULIA

.PHONY: setup setup-julia test lint format smoke manifest env-report gate gate-clean

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

format:
	$(UV) run ruff format .

manifest:
	$(UV) run so-recon manifest

env-report:
	$(UV) run so-recon env-report

smoke:
	$(UV) run so-recon smoke

# Full E00 gate. Logic lives in the script: GNU Make 3.81 ignores .SHELLFLAGS.
gate:
	./scripts/gate.sh

# Gate with a throwaway Julia depot (clean install of the locked Julia environment).
gate-clean:
	./scripts/gate_clean.sh
```

- [ ] **Step 6: Создать `README.md`**

```markdown
# SO-RECON

Физически ограниченная байесовская реконструкция текущей нефтенасыщенности.
Нормативные документы: `docs/SPEC.md` (v3.0), `docs/STAGES.md`, `docs/DATA_AUDIT.md`, `docs/RESEARCH.md`.

## Требования

- Python 3.13.2 (точная версия зафиксирована в `.python-version`) и [`uv`](https://docs.astral.sh/uv/)
- Julia — точная project-pinned версия из `julia/.julia-version`, ставится через
  [`juliaup`](https://github.com/JuliaLang/juliaup). Это версия, зафиксированная проектом,
  а не утверждение о текущем upstream stable-релизе.

## Установка

    make setup          # Python: uv sync --frozen
    make setup-julia    # Julia: Pkg.instantiate() по julia/Manifest.toml

## Данные

Закрытые исходные CSV лежат в `data/Ромашка_сырые/` и **не перемещаются и не изменяются**:
E00 открывает их только на чтение. Каталог `data/` не версионируется.
Перечень файлов и контракты чтения — в `configs/project.yml` → `sources.files`.
`so-recon manifest` записывает SHA-256, размер, `physical_line_count` и `data_rows`
в `reports/manifests/source_manifest.json`.

## Команды

    uv run so-recon manifest      # manifest исходных файлов
    uv run so-recon env-report    # reports/environment_report.md
    uv run so-recon smoke         # end-to-end: fixture -> case.json -> JutulDarcy
    make test                     # pytest
    make lint                     # ruff check + ruff format --check + mypy
    make gate                     # чистое Python-окружение + полный gate E00
    make gate-clean               # то же + чистый временный Julia depot (долго)

Каждый запуск создаёт `artifacts/runs/<run_id>/` с `run.json`, `resolved_config.json`,
`run.log` и артефактами запуска. Коммитируемые файлы в `reports/` и `configs/`
детерминированы: повторный gate не создаёт diff.
```

- [ ] **Step 7: Создать пакет и падающий тест**

`src/so_recon/__init__.py`:
```python
"""SO-RECON: physically constrained Bayesian reconstruction of current oil saturation."""

__version__ = "0.0.1"
SPEC_VERSION = "3.0"
```

Подпакеты — одна строка docstring каждый: `config/__init__.py` → `"""Typed configuration."""`, `registry/__init__.py` → `"""Hashing, atomic writes, artifacts, run records and manifests."""`, `environment/__init__.py` → `"""Environment report."""`, `synthetic/__init__.py` → `"""Deterministic synthetic fixtures."""`, `simulator/__init__.py` → `"""Julia/JutulDarcy integration."""`.

`tests/__init__.py`, `tests/unit/__init__.py`, `tests/integration/__init__.py` — пустые.

`tests/unit/test_package.py`:
```python
import so_recon


def test_version_and_spec_version() -> None:
    assert so_recon.__version__ == "0.0.1"
    assert so_recon.SPEC_VERSION == "3.0"
```

Создать пустые `artifacts/.gitkeep`, `reports/.gitkeep`, `reports/manifests/.gitkeep`, `reports/stages/.gitkeep`, `configs/.gitkeep`.

- [ ] **Step 8: Установить окружение и запустить тест**

```bash
uv lock && uv sync && uv run pytest -q
```
Ожидается: `1 passed`; появились `uv.lock` и `.venv/`. Проверить, что интерпретатор — ровно 3.13.2:
```bash
uv run python -c "import platform; print(platform.python_version())"
```
Ожидается: `3.13.2`. Если версия иная — исправить `.python-version` на фактически доступную точную версию 3.13.x и зафиксировать это в отчёте E00.

- [ ] **Step 9: Проверить lint, format и mypy**

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy
```
Ожидается: `All checks passed!`, `N files already formatted`, `Success: no issues found`.

- [ ] **Step 10: Проверить, что gate-скрипты синтаксически корректны**

```bash
bash -n scripts/gate.sh && bash -n scripts/gate_clean.sh && test -x scripts/gate.sh && test -x scripts/gate_clean.sh && echo "gate scripts ok"
```
Ожидается: `gate scripts ok`. Сам `make gate` на этом этапе ещё не запускается (нет Julia и CLI).

- [ ] **Step 11: Commit**

```bash
git add pyproject.toml uv.lock .python-version Makefile scripts README.md .gitignore src tests artifacts/.gitkeep reports configs && git commit -m "feat(e00): scaffold python project with uv lock, gate scripts and package skeleton

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Hashing, канонический JSON и атомарная запись

**Files:**
- Create: `src/so_recon/registry/hashing.py`, `src/so_recon/registry/atomic.py`
- Test: `tests/unit/test_hashing.py`, `tests/unit/test_atomic.py`

**Interfaces:**
- Produces (`so_recon.registry.hashing`):
  - `sha256_bytes(data: bytes) -> str` — hex-строка 64 символа.
  - `sha256_file(path: Path, chunk_size: int = 1 << 20) -> str` — потоковый SHA-256, файл открывается только на чтение.
  - `canonical_json(obj: object) -> str` — `sort_keys=True`, `separators=(",", ":")`, `ensure_ascii=False`, `allow_nan=False`.
  - `sha256_json(obj: object) -> str` — `sha256_bytes(canonical_json(obj).encode("utf-8"))`.
- Produces (`so_recon.registry.atomic`):
  - `write_bytes_atomic(path: Path, data: bytes) -> None`
  - `write_text_atomic(path: Path, text: str) -> None`
  - `write_json_atomic(path: Path, obj: object, *, indent: int = 2) -> bytes` — детерминированный дамп (`sort_keys=True`, `ensure_ascii=False`, `allow_nan=False`) плюс завершающий `\n`; возвращает записанные байты, чтобы вызывающий мог их захешировать без повторного чтения.

**Замечания ревизии 2, реализуемые здесь:** №12 (атомарная запись).

- [ ] **Step 1: Написать падающие тесты**

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


def test_canonical_json_rejects_infinity() -> None:
    with pytest.raises(ValueError):
        canonical_json({"x": float("inf")})


def test_sha256_file_handles_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "empty.bin"
    p.write_bytes(b"")
    assert sha256_file(p) == hashlib.sha256(b"").hexdigest()


def test_sha256_json_is_stable() -> None:
    assert sha256_json({"k": 1}) == sha256_bytes(b'{"k":1}')
```

`tests/unit/test_atomic.py`:
```python
import json
from pathlib import Path

import pytest

from so_recon.registry import atomic as atomic_module
from so_recon.registry.atomic import write_bytes_atomic, write_json_atomic, write_text_atomic


def test_write_bytes_atomic_creates_parents(tmp_path: Path) -> None:
    p = tmp_path / "a" / "b" / "f.bin"
    write_bytes_atomic(p, b"payload")
    assert p.read_bytes() == b"payload"


def test_write_bytes_atomic_replaces_existing(tmp_path: Path) -> None:
    p = tmp_path / "f.bin"
    write_bytes_atomic(p, b"one")
    write_bytes_atomic(p, b"two")
    assert p.read_bytes() == b"two"


def test_write_atomic_leaves_no_temporary_files(tmp_path: Path) -> None:
    write_text_atomic(tmp_path / "f.txt", "hello")
    assert sorted(q.name for q in tmp_path.iterdir()) == ["f.txt"]


def test_serialisation_failure_keeps_old_content(tmp_path: Path) -> None:
    """An unserialisable object must fail before the filesystem is touched at all.

    This does NOT exercise the rollback branch — see the replace-failure test for that.
    """
    p = tmp_path / "f.json"
    write_json_atomic(p, {"ok": 1})
    with pytest.raises(ValueError):
        write_json_atomic(p, {"bad": float("inf")})
    assert json.loads(p.read_text(encoding="utf-8")) == {"ok": 1}
    assert sorted(q.name for q in tmp_path.iterdir()) == ["f.json"]


def test_cleanup_removes_the_temp_file_when_the_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the rollback branch directly.

    A serialisation failure raises before any file is touched, so it never reaches the
    cleanup path. Without this test a broken `tmp.unlink` would ship green.
    """
    target = tmp_path / "f.bin"
    write_bytes_atomic(target, b"original")

    def boom(src: object, dst: object) -> None:
        raise OSError("replace failed")

    # Path.replace delegates to os.replace, resolving the attribute at call time.
    monkeypatch.setattr(atomic_module.os, "replace", boom)
    with pytest.raises(OSError, match="replace failed"):
        write_bytes_atomic(target, b"replacement")
    assert target.read_bytes() == b"original", "a failed write must not damage the old artifact"
    assert sorted(q.name for q in tmp_path.iterdir()) == ["f.bin"], "temp file was left behind"


def test_write_json_atomic_is_deterministic_and_returns_bytes(tmp_path: Path) -> None:
    p = tmp_path / "f.json"
    first = write_json_atomic(p, {"b": 1, "a": "ё"})
    second = write_json_atomic(tmp_path / "g.json", {"a": "ё", "b": 1})
    assert first == second
    assert first == p.read_bytes()
    assert first.decode("utf-8") == '{\n  "a": "ё",\n  "b": 1\n}\n'
```

- [ ] **Step 2: Убедиться, что тесты падают**

```bash
uv run pytest tests/unit/test_hashing.py tests/unit/test_atomic.py -q
```
Ожидается: `ModuleNotFoundError` для `so_recon.registry.hashing` и `so_recon.registry.atomic`.

- [ ] **Step 3: Реализовать `hashing.py`**

```python
"""Content hashing helpers shared by manifests, artifacts, run records and fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Stream a file's SHA-256. Read-only: never mutates the source (invariant I1)."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(obj: object) -> str:
    """Deterministic JSON: sorted keys, no whitespace, UTF-8 text, NaN/Inf forbidden."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def sha256_json(obj: object) -> str:
    return sha256_bytes(canonical_json(obj).encode("utf-8"))
```

- [ ] **Step 4: Реализовать `atomic.py`**

```python
"""Atomic writes: a reader never observes a half-written artifact (invariant I3).

Every scientific artifact goes through here. The temporary file is created in the
destination directory so that os.replace stays within one filesystem and is atomic.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def _fsync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        # Path.replace, not os.replace: the ruff PTH ruleset this project selects forbids
        # os.replace (PTH105), and Path.replace delegates to it anyway, looking the attribute
        # up on the os module at call time — so monkeypatching os.replace still intercepts it.
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _fsync_dir(path.parent)


def write_text_atomic(path: Path, text: str) -> None:
    write_bytes_atomic(path, text.encode("utf-8"))


def write_json_atomic(path: Path, obj: object, *, indent: int = 2) -> bytes:
    """Deterministic pretty JSON. Serialisation happens before any file is touched,
    so an unserialisable object leaves the previous content intact."""
    payload = (
        json.dumps(obj, indent=indent, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")
    write_bytes_atomic(path, payload)
    return payload
```

- [ ] **Step 5: Прогнать тесты, lint, format, mypy**

```bash
uv run pytest tests/unit/test_hashing.py tests/unit/test_atomic.py -q && uv run ruff check . && uv run ruff format --check . && uv run mypy
```
Ожидается: `13 passed`, без ошибок.

- [ ] **Step 6: Commit**

```bash
git add src/so_recon/registry/hashing.py src/so_recon/registry/atomic.py tests/unit/test_hashing.py tests/unit/test_atomic.py && git commit -m "feat(e00): add sha256/canonical-json hashing and atomic artifact writes

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Корень репозитория, containment путей и каталоги проекта

**Files:**
- Create: `src/so_recon/paths.py`
- Test: `tests/unit/test_paths.py`, `tests/test_no_absolute_paths.py`

**Interfaces:**
- Produces:
  - `class RepoRootNotFoundError(RuntimeError)`, `class PathEscapeError(ValueError)`.
  - `validate_relative_path(value: str) -> str` — единственный валидатор формы пути. Отвергает: пустую строку и строку с обрамляющими пробелами; `~`-префикс; любой backslash (а значит и Windows-разделители, и UNC `\\server\share`); Windows drive-absolute (`C:/`, `C:\`); POSIX-absolute (`/x`) и `//server/share`; любой сегмент `..` или `.`; NUL-байт.
  - `resolve_within_root(root: Path, relative: str) -> Path` — валидирует форму, затем `(root/relative).resolve()` и требует, чтобы результат лежал внутри `root.resolve()`; иначе `PathEscapeError`. Так как `resolve()` раскрывает symlink'и, выход за пределы репозитория через symlink запрещён.
  - `find_repo_root(start: Path | None = None) -> Path` — порядок: env `SO_RECON_ROOT` → подъём от `start` (по умолчанию `Path.cwd()`) → подъём от `__file__`. Маркер корня: есть `pyproject.toml` и каталог `src/so_recon`.
  - `@dataclass(frozen=True) class ProjectPaths` с полями `root, raw, interim, processed, artifacts, reports, configs, julia` (все абсолютные `Path`), свойствами `runs = artifacts/"runs"`, `manifests = reports/"manifests"`, `stages = reports/"stages"`; методы `relative(path) -> str`, `resolve(relative: str) -> Path`, `ensure_dirs() -> None`; classmethod `default(root)`.
  - `ProjectPaths.ensure_dirs()` создаёт **только** `interim`, `processed`, `runs`, `manifests`, `stages`. Каталог `raw` не создаётся: источники обязаны существовать заранее, их отсутствие — явная ошибка (SPEC 7.1, инвариант I1).
- Примечание: `ProjectPaths.from_config` **здесь не реализуется**. Он принимает `PathsConfig`, которого до Task 4 не существует, а mypy в strict-режиме не может разрешить такую аннотацию даже под `TYPE_CHECKING` — Task 3 обязан оставлять `mypy` зелёным. Метод и его тест добавляются в Task 4 одним коммитом вместе с `PathsConfig`.

**Замечания ревизии 2, реализуемые здесь:** №8 (`..`, Windows absolute/UNC, symlink-escape), №2 (`ensure_dirs` не трогает каталог источников), №11.

- [ ] **Step 1: Написать падающие тесты**

`tests/unit/test_paths.py`:
```python
from pathlib import Path

import pytest

from so_recon.paths import (
    PathEscapeError,
    ProjectPaths,
    RepoRootNotFoundError,
    find_repo_root,
    resolve_within_root,
    validate_relative_path,
)


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


@pytest.mark.parametrize(
    "bad",
    [
        "",
        " data/raw",
        "data/raw ",
        "~/data",
        "/abs/data",
        "//server/share",
        "\\\\server\\share",
        "C:/data",
        "C:\\data",
        "c:/data",
        "data\\raw",
        "../outside",
        "data/../../outside",
        "data/./raw",
        "..",
        "data/raw\x00",
    ],
)
def test_validate_relative_path_rejects(bad: str) -> None:
    with pytest.raises(ValueError):
        validate_relative_path(bad)


@pytest.mark.parametrize("good", ["data/raw", "configs", "julia/smoke/smoke_case.jl", "data/Ромашка_сырые"])
def test_validate_relative_path_accepts(good: str) -> None:
    assert validate_relative_path(good) == good


def test_resolve_within_root_returns_absolute_path(tmp_path: Path) -> None:
    assert resolve_within_root(tmp_path, "data/raw") == (tmp_path.resolve() / "data" / "raw")


def test_resolve_within_root_rejects_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    root = tmp_path / "repo"
    root.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(PathEscapeError):
        resolve_within_root(root, "escape/secrets.csv")


def test_resolve_within_root_allows_symlink_inside_root(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "real").mkdir(parents=True)
    (root / "link").symlink_to(root / "real", target_is_directory=True)
    assert resolve_within_root(root, "link/f.csv") == (root.resolve() / "real" / "f.csv")


def test_default_paths_and_relative(tmp_path: Path) -> None:
    paths = ProjectPaths.default(tmp_path)
    assert paths.raw == tmp_path / "data" / "raw"
    assert paths.runs == tmp_path / "artifacts" / "runs"
    assert paths.manifests == tmp_path / "reports" / "manifests"
    assert paths.stages == tmp_path / "reports" / "stages"
    assert paths.relative(paths.raw / "mer.csv") == "data/raw/mer.csv"


def test_relative_rejects_path_outside_root(tmp_path: Path) -> None:
    paths = ProjectPaths.default(tmp_path / "repo")
    with pytest.raises(PathEscapeError):
        paths.relative(tmp_path / "elsewhere" / "f.csv")


def test_ensure_dirs_creates_runtime_dirs_but_not_sources(tmp_path: Path) -> None:
    paths = ProjectPaths.default(tmp_path)
    paths.ensure_dirs()
    for p in (paths.interim, paths.processed, paths.runs, paths.manifests, paths.stages):
        assert p.is_dir()
    assert not paths.raw.exists(), "source directory must never be created by the code"
```

`tests/test_no_absolute_paths.py` — gate «код не зависит от личных абсолютных путей»:
```python
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
```

- [ ] **Step 2: Убедиться, что тесты падают**

```bash
uv run pytest tests/unit/test_paths.py tests/test_no_absolute_paths.py -q
```
Ожидается: `ModuleNotFoundError: so_recon.paths` для первого файла; второй проходит.

- [ ] **Step 3: Реализовать `paths.py`**

```python
"""Repository root discovery, path containment and canonical project directories."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

ROOT_ENV_VAR = "SO_RECON_ROOT"

# "C:/x", "C:\x" and "\\server\share" must never be accepted from configuration.
_WINDOWS_ABSOLUTE = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\)")


class RepoRootNotFoundError(RuntimeError):
    """Raised when no directory with pyproject.toml + src/so_recon is found."""


class PathEscapeError(ValueError):
    """Raised when a path would resolve outside the repository root."""


def validate_relative_path(value: str) -> str:
    """Accept only a plain relative POSIX path that stays inside the repository.

    Rejects absolute POSIX paths, Windows drive-absolute and UNC paths, '~' expansion,
    backslash separators, '.' and '..' segments, and NUL bytes (invariant I4).
    """
    if not value or value != value.strip():
        raise ValueError(f"path must be a non-empty trimmed string, got {value!r}")
    if "\x00" in value:
        raise ValueError(f"path must not contain NUL bytes, got {value!r}")
    if value.startswith("~"):
        raise ValueError(f"path must not use '~' expansion, got {value!r}")
    if "\\" in value:
        raise ValueError(f"path must use '/' separators and must not be a UNC path, got {value!r}")
    if _WINDOWS_ABSOLUTE.match(value):
        raise ValueError(f"path must not be a Windows absolute/UNC path, got {value!r}")
    pure = PurePosixPath(value)
    if pure.is_absolute():
        raise ValueError(f"path must be relative to repository root, got {value!r}")
    # Split the raw string rather than using pure.parts: PurePosixPath silently
    # collapses single "." segments while parsing, so a check on pure.parts would
    # never see them and "data/./raw" would wrongly be accepted.
    if any(segment in ("..", ".") for segment in value.split("/")):
        raise ValueError(f"path must not contain '.' or '..' segments, got {value!r}")
    return value


def resolve_within_root(root: Path, relative: str) -> Path:
    """Validate the form, then resolve and prove the result stays inside the root.

    resolve() expands symlinks, so a symlink pointing outside the repository is rejected.
    """
    validate_relative_path(relative)
    root_resolved = root.resolve()
    candidate = (root_resolved / relative).resolve()
    if candidate != root_resolved and not candidate.is_relative_to(root_resolved):
        raise PathEscapeError(f"{relative!r} resolves to {candidate}, outside {root_resolved}")
    return candidate


def _is_root(p: Path) -> bool:
    return (p / "pyproject.toml").is_file() and (p / "src" / "so_recon").is_dir()


def _walk_up(start: Path) -> Path | None:
    start = start.resolve()
    for candidate in (start, *start.parents):
        if _is_root(candidate):
            return candidate
    return None


def find_repo_root(start: Path | None = None) -> Path:
    """Locate the repository root: env override, then an ancestor walk.

    When `start` is given explicitly it is authoritative: only its own ancestry is
    searched. When `start` is omitted the lookup walks up from the current working
    directory and, failing that, from this module's own location. Falling back to the
    module's own path despite a caller-supplied `start` would make this function find
    THIS repository from anywhere — which silently defeats containment and makes the
    "no marker" case untestable.
    """
    env = os.environ.get(ROOT_ENV_VAR)
    if env:
        return Path(env).resolve()
    origins = (start,) if start is not None else (Path.cwd(), Path(__file__))
    for origin in origins:
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

    def resolve(self, relative: str) -> Path:
        return resolve_within_root(self.root, relative)

    def relative(self, path: Path) -> str:
        resolved = path.resolve()
        root = self.root.resolve()
        if resolved != root and not resolved.is_relative_to(root):
            raise PathEscapeError(f"{path} is outside the repository root {root}")
        return resolved.relative_to(root).as_posix()

    def ensure_dirs(self) -> None:
        """Create runtime directories only. The source directory is never created:
        a missing source must surface as an explicit error (SPEC 7.1, invariant I1)."""
        for p in (self.interim, self.processed, self.runs, self.manifests, self.stages):
            p.mkdir(parents=True, exist_ok=True)
```

> В `paths.py` нет импорта из `so_recon.config`: `ProjectPaths.from_config` добавляется в Task 4. Это сохраняет `mypy --strict` зелёным на каждом коммите и заодно исключает цикл `paths ↔ config` (в Task 4 импорт `PathsConfig` идёт только под `TYPE_CHECKING`, потому что `config.schema` импортирует `paths` в рантайме).

- [ ] **Step 4: Прогнать тесты, lint, format, mypy**

```bash
uv run pytest tests/unit/test_paths.py tests/test_no_absolute_paths.py -q && uv run ruff check . && uv run ruff format --check . && uv run mypy
```
Ожидается: все тесты проходят (параметризованные reject/accept + остальные), без ошибок ruff/mypy.

- [ ] **Step 5: Commit**

```bash
git add src/so_recon/paths.py tests/unit/test_paths.py tests/test_no_absolute_paths.py && git commit -m "feat(e00): add repo root discovery, path containment guards and project paths

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---
### Task 4: Typed configuration (pydantic) и `configs/project.yml`

**Files:**
- Create: `src/so_recon/config/schema.py`, `src/so_recon/config/load.py`, `configs/project.yml`
- Modify: `src/so_recon/paths.py` (добавить `ProjectPaths.from_config`)
- Test: `tests/unit/test_config.py`
- Delete: `configs/.gitkeep`

**Interfaces:**
- Consumes: `sha256_json` (Task 2); `validate_relative_path`, `ProjectPaths` (Task 3).
- Produces (`so_recon.config.schema`):
  - `class StrictModel(BaseModel)` — `extra="forbid"`, `frozen=True`.
  - `class PathsConfig(StrictModel)`: `raw: str = "data/raw"`, `interim: str = "data/interim"`, `processed: str = "data/processed"`, `artifacts: str = "artifacts"`, `reports: str = "reports"`, `configs: str = "configs"`, `julia: str = "julia"`; каждое поле проходит `validate_relative_path`.
  - `class SourceFileSpec(StrictModel)`: `name: str`, `path: str`, `encoding: Literal["utf-8", "utf-8-sig", "cp1251"]`, `delimiter: Literal[",", ";"]`, `decimal: Literal[".", ","]`, `header_lines: int = 1` (`ge=0`), `required: bool = True`; `path` проходит `validate_relative_path`.
  - `class SourcesConfig(StrictModel)`: `files: list[SourceFileSpec]`; валидатор запрещает повторяющиеся `name`.
  - `class SmokeFixtureConfig(StrictModel)`: `seed: int = 20260913`, `n_wells: int = 4` (`ge=2`), `n_months: int = 12` (`ge=1`), `nx: int = 20` (`ge=3`), `n_steps: int = 12` (`ge=1`), `rel_tol: float = 1e-6` (`gt=0`).
  - `class JuliaConfig(StrictModel)`: `project: str = "julia"`, `smoke_script: str = "julia/smoke/smoke_case.jl"`, `timeout_s: int = 1800` (`ge=1`); строковые пути проходят `validate_relative_path`.
  - `class ProjectConfig(StrictModel)`: `spec_version: Literal["3.0"]`, `config_version: str`, `project_name: str = "SO-RECON"`, `paths: PathsConfig`, `sources: SourcesConfig`, `smoke: SmokeFixtureConfig`, `julia: JuliaConfig`.
- Produces (`so_recon.config.load`): `class ConfigError(ValueError)`, `load_project_config(path) -> ProjectConfig`, `resolved_config_dict(cfg) -> dict[str, Any]`, `config_hash(cfg) -> str`.
- Produces (`so_recon.paths`): `ProjectPaths.from_config(root: Path, cfg: PathsConfig) -> ProjectPaths` — переносится сюда из Task 3, потому что раньше `PathsConfig` не существует и `mypy --strict` не может разрешить аннотацию.

**Замечания ревизии 2, реализуемые здесь:** №1 (`plastoper` `decimal: ","`, `header_lines`), №2 (источники на исходном пути), №8 (жёсткая валидация формы пути).

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


def _spec(**over: object) -> SourceFileSpec:
    base: dict[str, object] = {
        "name": "x", "path": "data/raw/x.csv", "encoding": "utf-8",
        "delimiter": ";", "decimal": ",",
    }
    base.update(over)
    return SourceFileSpec(**base)  # type: ignore[arg-type]


def test_load_minimal_config(tmp_path: Path) -> None:
    p = tmp_path / "project.yml"
    p.write_text(MINIMAL_YAML, encoding="utf-8")
    cfg = load_project_config(p)
    assert cfg.spec_version == "3.0"
    assert cfg.paths.raw == "data/raw"
    assert cfg.sources.files[0].encoding == "utf-8"
    assert cfg.sources.files[0].header_lines == 1
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


def test_missing_config_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_project_config(tmp_path / "absent.yml")


@pytest.mark.parametrize(
    "bad", ["/tmp/raw", "../raw", "~/raw", "C:/raw", "C:\\raw", "\\\\server\\share", "//server/share", "data\\raw"]
)
def test_dangerous_paths_config_values_are_rejected(bad: str) -> None:
    with pytest.raises(ValidationError):
        PathsConfig(raw=bad)


@pytest.mark.parametrize("bad", ["/abs/x.csv", "../x.csv", "data/../../x.csv", "C:\\x.csv"])
def test_dangerous_source_paths_are_rejected(bad: str) -> None:
    with pytest.raises(ValidationError):
        _spec(path=bad)


def test_duplicate_source_names_are_rejected() -> None:
    with pytest.raises(ValidationError):
        SourcesConfig(files=[_spec(name="a"), _spec(name="a", path="data/raw/y.csv")])


def test_negative_header_lines_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _spec(header_lines=-1)


def test_config_hash_is_stable(tmp_path: Path) -> None:
    p = tmp_path / "project.yml"
    p.write_text(MINIMAL_YAML, encoding="utf-8")
    cfg = load_project_config(p)
    assert config_hash(cfg) == config_hash(load_project_config(p))
    assert resolved_config_dict(cfg)["smoke"]["seed"] == 20260913


def test_config_hash_changes_with_content(tmp_path: Path) -> None:
    p = tmp_path / "project.yml"
    p.write_text(MINIMAL_YAML, encoding="utf-8")
    q = tmp_path / "other.yml"
    q.write_text(MINIMAL_YAML.replace("test.1", "test.2"), encoding="utf-8")
    assert config_hash(load_project_config(p)) != config_hash(load_project_config(q))


def test_project_paths_from_config(tmp_path: Path) -> None:
    cfg = ProjectConfig(
        spec_version="3.0",
        config_version="t",
        paths=PathsConfig(raw="custom/raw"),
        sources=SourcesConfig(files=[]),
    )
    paths = ProjectPaths.from_config(tmp_path, cfg.paths)
    assert paths.raw == tmp_path.resolve() / "custom" / "raw"
    assert paths.reports == tmp_path.resolve() / "reports"


def test_repo_config_file_is_valid_and_matches_audited_contracts() -> None:
    repo_cfg = Path(__file__).resolve().parents[2] / "configs" / "project.yml"
    cfg = load_project_config(repo_cfg)
    by_name = {f.name: f for f in cfg.sources.files}
    assert list(by_name) == ["coords", "gis", "mer", "perf", "plastoper"]
    # Contracts verified byte-wise against the raw files (DATA_AUDIT.md section 3).
    assert (by_name["coords"].encoding, by_name["coords"].delimiter) == ("utf-8", ",")
    assert (by_name["gis"].encoding, by_name["gis"].decimal) == ("cp1251", ",")
    assert (by_name["mer"].encoding, by_name["mer"].decimal) == ("utf-8-sig", ",")
    assert (by_name["perf"].encoding, by_name["perf"].decimal) == ("cp1251", ",")
    # plastoper.csv stores "269,7600098": the decimal separator is a comma, not a dot.
    assert (by_name["plastoper"].encoding, by_name["plastoper"].decimal) == ("utf-8-sig", ",")
    assert all(f.header_lines == 1 for f in cfg.sources.files)
    # Sources stay where they are: E00 never moves or rewrites them.
    assert all(f.path.startswith("data/Ромашка_сырые/") for f in cfg.sources.files)
    assert cfg.paths.raw == "data/Ромашка_сырые"
```

- [ ] **Step 2: Убедиться, что тесты падают**

```bash
uv run pytest tests/unit/test_config.py -q
```
Ожидается: `ModuleNotFoundError: so_recon.config.load`.

- [ ] **Step 3: Реализовать `schema.py`**

```python
"""Typed configuration schema (SPEC 19.1: versioned YAML validated by typed schemas)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from so_recon.paths import validate_relative_path


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


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
        return validate_relative_path(v)


class SourceFileSpec(StrictModel):
    """A raw source file. E00 opens it read-only: it is never moved or rewritten."""

    name: str
    path: str
    encoding: Literal["utf-8", "utf-8-sig", "cp1251"]
    delimiter: Literal[",", ";"]
    decimal: Literal[".", ","]
    header_lines: int = Field(default=1, ge=0)
    required: bool = True

    @field_validator("path")
    @classmethod
    def _relative(cls, v: str) -> str:
        return validate_relative_path(v)


class SourcesConfig(StrictModel):
    files: list[SourceFileSpec]

    @field_validator("files")
    @classmethod
    def _unique_names(cls, v: list[SourceFileSpec]) -> list[SourceFileSpec]:
        names = [f.name for f in v]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"duplicate source names: {duplicates}")
        return v


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
        return validate_relative_path(v)


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

`PathsConfig` теперь существует, поэтому метод можно типизировать. Импорт — только под `TYPE_CHECKING`: `so_recon.config.schema` импортирует `so_recon.paths` в рантайме, и обычный импорт создал бы цикл.

В начало `paths.py`, рядом с остальными импортами:
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from so_recon.config.schema import PathsConfig
```

И внутрь `ProjectPaths`, сразу после classmethod `default`:
```python
    @classmethod
    def from_config(cls, root: Path, cfg: PathsConfig) -> ProjectPaths:
        root = root.resolve()
        return cls(
            root=root,
            raw=resolve_within_root(root, cfg.raw),
            interim=resolve_within_root(root, cfg.interim),
            processed=resolve_within_root(root, cfg.processed),
            artifacts=resolve_within_root(root, cfg.artifacts),
            reports=resolve_within_root(root, cfg.reports),
            configs=resolve_within_root(root, cfg.configs),
            julia=resolve_within_root(root, cfg.julia),
        )
```

> Каждый путь проходит `resolve_within_root`, поэтому конфигурация не может вывести ни один каталог проекта за пределы repository root — ни через `..`, ни через symlink (инвариант I4).

- [ ] **Step 6: Создать `configs/project.yml`**

Значения `encoding`/`delimiter`/`decimal` взяты из `DATA_AUDIT.md` §3 и **проверены по байтам** исходных файлов (см. таблицу «Проверенные контракты исходных файлов»). Пути указывают на исходное расположение: E00 ничего не перемещает.

```yaml
# SO-RECON project configuration. Validated by so_recon.config.schema.ProjectConfig.
# All paths are relative to the repository root. No personal absolute paths.
# Raw sources are immutable: E00 opens them read-only and never moves or rewrites them.
spec_version: "3.0"
config_version: "E00.2"
project_name: SO-RECON

paths:
  raw: data/Ромашка_сырые
  interim: data/interim
  processed: data/processed
  artifacts: artifacts
  reports: reports
  configs: configs
  julia: julia

sources:
  files:
    - name: coords
      path: data/Ромашка_сырые/coords.csv
      encoding: utf-8
      delimiter: ","
      decimal: "."
      header_lines: 1
    - name: gis
      path: data/Ромашка_сырые/gis.csv
      encoding: cp1251
      delimiter: ";"
      decimal: ","
      header_lines: 1
    - name: mer
      path: data/Ромашка_сырые/mer.csv
      encoding: utf-8-sig
      delimiter: ";"
      decimal: ","
      header_lines: 1
    - name: perf
      path: data/Ромашка_сырые/perf.csv
      encoding: cp1251
      delimiter: ";"
      decimal: ","
      header_lines: 1
    # plastoper.csv stores values such as "269,7600098" -> decimal separator is a comma.
    - name: plastoper
      path: data/Ромашка_сырые/plastoper.csv
      encoding: utf-8-sig
      delimiter: ";"
      decimal: ","
      header_lines: 1

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

- [ ] **Step 7: Прогнать тесты, lint, format, mypy**

```bash
uv run pytest tests/unit/test_config.py tests/unit/test_paths.py -q && uv run ruff check . && uv run ruff format --check . && uv run mypy
```
Ожидается: все тесты проходят, ruff/mypy чисты.

- [ ] **Step 8: Commit**

```bash
git add src/so_recon/config src/so_recon/paths.py configs tests/unit/test_config.py && git rm --cached configs/.gitkeep --ignore-unmatch && git commit -m "feat(e00): add typed project configuration schema, loader and configs/project.yml

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Artifact registry (immutable `ArtifactRef`)

**Files:**
- Create: `src/so_recon/registry/artifact.py`
- Test: `tests/unit/test_artifact.py`

**Interfaces:**
- Consumes: `sha256_bytes`, `sha256_file` (Task 2); `write_bytes_atomic`, `write_json_atomic` (Task 2); `ProjectPaths` (Task 3); `StrictModel` (Task 4).
- Produces:
  - `class ArtifactImmutabilityError(RuntimeError)` — попытка перезаписать существующий артефакт другим содержимым.
  - `class ArtifactRef(StrictModel)`: `artifact_id: str` (равен `sha256` — контентная адресация), `path: str` (posix относительно repo root), `sha256: str`, `size_bytes: int`, `media_type: str`, `schema_version: str`, `producer_run_id: str`, `parent_artifact_ids: list[str]`, `created_at: str`.
  - `register_artifact(path, paths, *, schema_version, producer_run_id, media_type, parent_artifact_ids=(), now) -> ArtifactRef` — для файлов, записанных сторонними библиотеками (Parquet).
  - `write_artifact(path, data: bytes, paths, *, schema_version, producer_run_id, media_type, parent_artifact_ids=(), now) -> ArtifactRef` — атомарная запись с проверкой immutability: если файл существует и его содержимое совпадает — операция идемпотентна; если отличается — `ArtifactImmutabilityError`, старое содержимое остаётся нетронутым.
  - `write_json_artifact(path, obj, paths, *, schema_version, producer_run_id, parent_artifact_ids=(), now) -> ArtifactRef` — то же для детерминированного JSON (`media_type="application/json"`).

**Замечания ревизии 2, реализуемые здесь:** №4 (`ArtifactRef`, immutability результатов), №12 (атомарность).

- [ ] **Step 1: Написать падающие тесты**

`tests/unit/test_artifact.py`:
```python
import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from so_recon.paths import ProjectPaths
from so_recon.registry.artifact import (
    ArtifactImmutabilityError,
    register_artifact,
    write_artifact,
    write_json_artifact,
)

NOW = datetime(2026, 9, 13, 12, 0, tzinfo=UTC)


def _paths(tmp_path: Path) -> ProjectPaths:
    return ProjectPaths.default(tmp_path)


def test_write_artifact_returns_full_lineage_ref(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    ref = write_artifact(
        tmp_path / "reports" / "x.bin", b"payload", paths,
        schema_version="1", producer_run_id="run-1", media_type="application/octet-stream",
        parent_artifact_ids=["parent-sha"], now=NOW,
    )
    assert ref.sha256 == hashlib.sha256(b"payload").hexdigest()
    assert ref.artifact_id == ref.sha256
    assert ref.path == "reports/x.bin"
    assert ref.size_bytes == 7
    assert ref.schema_version == "1"
    assert ref.producer_run_id == "run-1"
    assert ref.parent_artifact_ids == ["parent-sha"]
    assert ref.created_at == "2026-09-13T12:00:00+00:00"


def test_rewriting_identical_content_is_idempotent(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    target = tmp_path / "reports" / "x.bin"
    a = write_artifact(target, b"same", paths, schema_version="1", producer_run_id="r1",
                       media_type="application/octet-stream", now=NOW)
    b = write_artifact(target, b"same", paths, schema_version="1", producer_run_id="r2",
                       media_type="application/octet-stream", now=NOW)
    assert a.sha256 == b.sha256
    assert b.producer_run_id == "r2"
    assert target.read_bytes() == b"same"


def test_rewriting_with_different_content_is_refused(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    target = tmp_path / "reports" / "x.bin"
    write_artifact(target, b"first", paths, schema_version="1", producer_run_id="r1",
                   media_type="application/octet-stream", now=NOW)
    with pytest.raises(ArtifactImmutabilityError):
        write_artifact(target, b"second", paths, schema_version="1", producer_run_id="r2",
                       media_type="application/octet-stream", now=NOW)
    assert target.read_bytes() == b"first", "the existing artifact must survive a refused write"


def test_write_json_artifact_is_deterministic(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    a = write_json_artifact(tmp_path / "reports" / "a.json", {"b": 1, "a": 2}, paths,
                            schema_version="1", producer_run_id="r1", now=NOW)
    b = write_json_artifact(tmp_path / "reports" / "b.json", {"a": 2, "b": 1}, paths,
                            schema_version="1", producer_run_id="r1", now=NOW)
    assert a.sha256 == b.sha256
    assert a.media_type == "application/json"


def test_register_artifact_hashes_existing_file(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    target = tmp_path / "artifacts" / "t.parquet"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"parquet-bytes")
    ref = register_artifact(target, paths, schema_version="1", producer_run_id="r1",
                            media_type="application/vnd.apache.parquet", now=NOW)
    assert ref.sha256 == hashlib.sha256(b"parquet-bytes").hexdigest()
    assert ref.path == "artifacts/t.parquet"


def test_artifact_outside_repository_root_is_refused(tmp_path: Path) -> None:
    paths = ProjectPaths.default(tmp_path / "repo")
    with pytest.raises(ValueError):
        write_artifact(tmp_path / "elsewhere.bin", b"x", paths, schema_version="1",
                       producer_run_id="r1", media_type="application/octet-stream", now=NOW)
```

- [ ] **Step 2: Убедиться, что тесты падают**

```bash
uv run pytest tests/unit/test_artifact.py -q
```
Ожидается: `ModuleNotFoundError: so_recon.registry.artifact`.

- [ ] **Step 3: Реализовать `artifact.py`**

```python
"""Immutable, content-addressed artifact references (SPEC 19.12, invariants I2/I3).

An artifact is identified by the SHA-256 of its bytes. Writing the same bytes to the same
path again is idempotent; writing different bytes to an existing path is refused, so a
result that has been handed to a downstream stage can never change underneath it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from so_recon.config.schema import StrictModel
from so_recon.paths import ProjectPaths
from so_recon.registry.atomic import write_bytes_atomic
from so_recon.registry.hashing import sha256_bytes, sha256_file

JSON_MEDIA_TYPE = "application/json"


class ArtifactImmutabilityError(RuntimeError):
    """An existing artifact would have been overwritten with different content."""


class ArtifactRef(StrictModel):
    artifact_id: str
    path: str
    sha256: str
    size_bytes: int
    media_type: str
    schema_version: str
    producer_run_id: str
    parent_artifact_ids: list[str]
    created_at: str


def _ref(
    *,
    repo_relative: str,
    digest: str,
    size_bytes: int,
    media_type: str,
    schema_version: str,
    producer_run_id: str,
    parent_artifact_ids: Sequence[str],
    now: datetime,
) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=digest,
        path=repo_relative,
        sha256=digest,
        size_bytes=size_bytes,
        media_type=media_type,
        schema_version=schema_version,
        producer_run_id=producer_run_id,
        parent_artifact_ids=list(parent_artifact_ids),
        created_at=now.isoformat(),
    )


def register_artifact(
    path: Path,
    paths: ProjectPaths,
    *,
    schema_version: str,
    producer_run_id: str,
    media_type: str,
    parent_artifact_ids: Sequence[str] = (),
    now: datetime,
) -> ArtifactRef:
    """Describe a file that another library already wrote (for example Parquet)."""
    repo_relative = paths.relative(path)
    return _ref(
        repo_relative=repo_relative,
        digest=sha256_file(path),
        size_bytes=path.stat().st_size,
        media_type=media_type,
        schema_version=schema_version,
        producer_run_id=producer_run_id,
        parent_artifact_ids=parent_artifact_ids,
        now=now,
    )


def write_artifact(
    path: Path,
    data: bytes,
    paths: ProjectPaths,
    *,
    schema_version: str,
    producer_run_id: str,
    media_type: str,
    parent_artifact_ids: Sequence[str] = (),
    now: datetime,
) -> ArtifactRef:
    repo_relative = paths.relative(path)  # also proves containment inside the repository
    digest = sha256_bytes(data)
    if path.exists():
        existing = sha256_file(path)
        if existing != digest:
            raise ArtifactImmutabilityError(
                f"refusing to overwrite {repo_relative}: on disk {existing}, new {digest}"
            )
    else:
        write_bytes_atomic(path, data)
    return _ref(
        repo_relative=repo_relative,
        digest=digest,
        size_bytes=len(data),
        media_type=media_type,
        schema_version=schema_version,
        producer_run_id=producer_run_id,
        parent_artifact_ids=parent_artifact_ids,
        now=now,
    )


def write_json_artifact(
    path: Path,
    obj: object,
    paths: ProjectPaths,
    *,
    schema_version: str,
    producer_run_id: str,
    parent_artifact_ids: Sequence[str] = (),
    now: datetime,
) -> ArtifactRef:
    payload = (
        json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")
    return write_artifact(
        path,
        payload,
        paths,
        schema_version=schema_version,
        producer_run_id=producer_run_id,
        media_type=JSON_MEDIA_TYPE,
        parent_artifact_ids=parent_artifact_ids,
        now=now,
    )
```

> Важно: `paths.relative(path)` вызывается до любой записи, поэтому попытка записать артефакт вне repository root отклоняется `PathEscapeError` (наследует `ValueError`) и файл не создаётся.

- [ ] **Step 4: Прогнать тесты, lint, format, mypy**

```bash
uv run pytest tests/unit/test_artifact.py -q && uv run ruff check . && uv run ruff format --check . && uv run mypy
```
Ожидается: `6 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/so_recon/registry/artifact.py tests/unit/test_artifact.py && git commit -m "feat(e00): add immutable content-addressed artifact refs with lineage fields

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Run registry (run ID, run record, lineage) и логирование

**Files:**
- Create: `src/so_recon/registry/gitinfo.py`, `src/so_recon/registry/run.py`, `src/so_recon/logging_setup.py`
- Test: `tests/unit/test_run_registry.py`, `tests/unit/test_logging_setup.py`

**Interfaces:**
- Consumes: `sha256_bytes`, `sha256_file`, `sha256_json` (Task 2); `write_json_atomic` (Task 2); `ProjectPaths` (Task 3); `ProjectConfig`, `config_hash`, `resolved_config_dict`, `StrictModel` (Task 4); `ArtifactRef` (Task 5).
- Produces (`so_recon.registry.gitinfo`): `git_commit(root) -> str | None`, `git_is_dirty(root) -> bool | None`.
- Produces (`so_recon.registry.run`):
  - `RunStatus = Literal["RUNNING", "PASS", "FAIL"]`
  - `UNAVAILABLE = "unavailable"` — значение полей, когда конфигурация не загрузилась.
  - `make_run_id(command, config_hash, git_commit, now) -> str` — формат `YYYYmmddTHHMMSSZ-<command>-<8 hex>`; при коллизии каталога `RunContext.start` добавляет суффикс `-01` … `-99`, после чего `FileExistsError`.
  - `environment_lock_hash(paths) -> str` — `sha256_json` по `{"uv.lock": h, "julia/Manifest.toml": h, "julia/.julia-version": h}`; отсутствующий файл даёт строку `"missing"`.
  - `class RunRecord(StrictModel)` — поля: `run_id`, `command`, `argv: list[str]`, `created_at`, `finished_at: str | None`, `git_commit: str | None`, `git_dirty: bool | None`, `spec_version`, `config_version`, `resolved_config_hash`, `environment_lock_hash`, `python_version`, `julia_version: str | None`, `jutul_version: str | None`, `jutuldarcy_version: str | None`, `model_checkpoint_hash: str | None`, `schema_versions: dict[str, str]`, `raw_input_hashes: dict[str, str]`, `parent_run_ids: list[str]`, `status: RunStatus`, **`outputs: dict[str, ArtifactRef]`**, `notes: list[str]`.
  - `class RunContext` (`run_id`, `run_dir`, `record`) с `start(...)`, `update(**fields)`, `add_output(key, ref)`, `finish(status, outputs=None, notes=())`, `write()`.
  - `RunContext.start(*, command, argv, cfg: ProjectConfig | None, paths, parent_run_ids=(), raw_input_hashes=None, schema_versions=None, now=None)` — **`cfg` может быть `None`**: это degraded-режим, нужный для замечания №5 (записать FAIL, когда конфигурация не загрузилась). При `cfg is None` `resolved_config.json` не пишется, `config_version` и `resolved_config_hash` равны `UNAVAILABLE`.
- Produces (`so_recon.logging_setup`): `configure_logging(run_id, log_file=None, level=logging.INFO) -> logging.Logger`.

**Замечания ревизии 2, реализуемые здесь:** №4 (`outputs` — `ArtifactRef`), №5 (degraded-режим для FAIL без конфигурации), №12 (атомарная запись `run.json` и `resolved_config.json`).

- [ ] **Step 1: Написать падающие тесты**

`tests/unit/test_run_registry.py`:
```python
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from so_recon.config.schema import ProjectConfig, SourcesConfig
from so_recon.paths import ProjectPaths
from so_recon.registry import run as run_module
from so_recon.registry.artifact import write_json_artifact
from so_recon.registry.run import UNAVAILABLE, RunContext, environment_lock_hash, make_run_id

NOW = datetime(2026, 9, 13, 12, 0, 0, tzinfo=UTC)


def _cfg() -> ProjectConfig:
    return ProjectConfig(spec_version="3.0", config_version="t.1", sources=SourcesConfig(files=[]))


def test_make_run_id_format_and_determinism() -> None:
    a = make_run_id("smoke", "c" * 64, "abc123", NOW)
    b = make_run_id("smoke", "c" * 64, "abc123", NOW)
    assert a == b
    assert re.fullmatch(r"20260913T120000Z-smoke-[0-9a-f]{8}", a)
    assert make_run_id("smoke", "d" * 64, "abc123", NOW) != a
    assert make_run_id("smoke", "c" * 64, None, NOW) != a


def test_environment_lock_hash_marks_missing_files(tmp_path: Path) -> None:
    paths = ProjectPaths.default(tmp_path)
    h_missing = environment_lock_hash(paths)
    (tmp_path / "uv.lock").write_text("lock")
    h_with_lock = environment_lock_hash(paths)
    assert h_missing != h_with_lock
    assert h_with_lock == environment_lock_hash(paths)
    (tmp_path / "julia").mkdir()
    (tmp_path / "julia" / ".julia-version").write_text("1.12.7\n")
    assert environment_lock_hash(paths) != h_with_lock


def test_run_context_writes_lineage_files(tmp_path: Path) -> None:
    paths = ProjectPaths.default(tmp_path)
    ctx = RunContext.start(
        command="smoke", argv=["so-recon", "smoke"], cfg=_cfg(), paths=paths,
        raw_input_hashes={"coords": "a" * 64}, now=NOW,
    )
    assert ctx.run_dir == paths.runs / ctx.run_id
    record = json.loads((ctx.run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "RUNNING"
    assert record["spec_version"] == "3.0"
    assert record["resolved_config_hash"]
    assert record["raw_input_hashes"] == {"coords": "a" * 64}
    assert record["created_at"] == "2026-09-13T12:00:00+00:00"
    resolved = json.loads((ctx.run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    assert resolved["config_version"] == "t.1"

    ref = write_json_artifact(ctx.run_dir / "thing.json", {"x": 1}, paths,
                              schema_version="1", producer_run_id=ctx.run_id, now=NOW)
    ctx.update(julia_version="1.12.7")
    ctx.finish("PASS", outputs={"thing": ref}, notes=["ok"])
    record = json.loads((ctx.run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "PASS"
    assert record["julia_version"] == "1.12.7"
    assert record["outputs"]["thing"]["sha256"] == ref.sha256
    assert record["outputs"]["thing"]["producer_run_id"] == ctx.run_id
    assert record["finished_at"] is not None


def test_run_context_without_config_still_records_a_run(tmp_path: Path) -> None:
    """Degraded mode: a config that failed to load must not prevent a FAIL record."""
    paths = ProjectPaths.default(tmp_path)
    ctx = RunContext.start(command="manifest", argv=["so-recon", "manifest"], cfg=None,
                           paths=paths, now=NOW)
    ctx.finish("FAIL", notes=["config error: boom"])
    record = json.loads((ctx.run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "FAIL"
    assert record["config_version"] == UNAVAILABLE
    assert record["resolved_config_hash"] == UNAVAILABLE
    assert any("boom" in n for n in record["notes"])
    assert not (ctx.run_dir / "resolved_config.json").exists()


def test_update_rejects_an_invalid_status(tmp_path: Path) -> None:
    """Lineage must never be silently corrupt: an unvalidated update would write a bogus
    status straight into run.json."""
    paths = ProjectPaths.default(tmp_path)
    ctx = RunContext.start(command="x", argv=[], cfg=_cfg(), paths=paths, now=NOW)
    with pytest.raises(ValidationError):
        ctx.update(status="PASSED")  # not a member of RunStatus
    on_disk = json.loads((ctx.run_dir / "run.json").read_text(encoding="utf-8"))
    assert on_disk["status"] == "RUNNING", "a refused update must not reach the file"


def test_missing_git_metadata_is_recorded_as_a_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bare null git_commit cannot be told apart from a broken git; say which."""
    paths = ProjectPaths.default(tmp_path)
    monkeypatch.setattr(run_module, "git_commit", lambda root: None)
    ctx = RunContext.start(command="x", argv=[], cfg=_cfg(), paths=paths, now=NOW)
    assert ctx.record.git_commit is None
    assert any("git commit unavailable" in n for n in ctx.record.notes)


def test_failed_start_leaves_no_empty_run_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invariant I6: a claimed directory with no run.json is a run that happened and
    cannot be recorded. The claim must be released instead."""
    paths = ProjectPaths.default(tmp_path)

    def boom(path: Path, obj: object, **kwargs: object) -> bytes:
        raise OSError("disk full")

    monkeypatch.setattr(run_module, "write_json_atomic", boom)
    with pytest.raises(OSError, match="disk full"):
        RunContext.start(command="x", argv=[], cfg=_cfg(), paths=paths, now=NOW)
    assert list(paths.runs.iterdir()) == [], "the claimed run directory was not released"


def test_run_context_suffixes_same_second_runs(tmp_path: Path) -> None:
    paths = ProjectPaths.default(tmp_path)
    first = RunContext.start(command="x", argv=[], cfg=_cfg(), paths=paths, now=NOW)
    second = RunContext.start(command="x", argv=[], cfg=_cfg(), paths=paths, now=NOW)
    third = RunContext.start(command="x", argv=[], cfg=_cfg(), paths=paths, now=NOW)
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
```

- [ ] **Step 2: Убедиться, что тесты падают**

```bash
uv run pytest tests/unit/test_run_registry.py tests/unit/test_logging_setup.py -q
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
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from so_recon import SPEC_VERSION
from so_recon.config.load import config_hash, resolved_config_dict
from so_recon.config.schema import ProjectConfig, StrictModel
from so_recon.paths import ProjectPaths
from so_recon.registry.artifact import ArtifactRef
from so_recon.registry.atomic import write_json_atomic
from so_recon.registry.gitinfo import git_commit, git_is_dirty
from so_recon.registry.hashing import sha256_bytes, sha256_file, sha256_json

RunStatus = Literal["RUNNING", "PASS", "FAIL"]

#: Marker used when a run has to be recorded before the configuration could be loaded.
UNAVAILABLE = "unavailable"

LOCK_FILES = ("uv.lock", "julia/Manifest.toml", "julia/.julia-version")


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
    runs_root.mkdir(parents=True, exist_ok=True)
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
    jutul_version: str | None = None
    jutuldarcy_version: str | None = None
    model_checkpoint_hash: str | None = None
    schema_versions: dict[str, str]
    raw_input_hashes: dict[str, str]
    parent_run_ids: list[str]
    status: RunStatus
    outputs: dict[str, ArtifactRef]
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
        argv: Sequence[str],
        cfg: ProjectConfig | None,
        paths: ProjectPaths,
        parent_run_ids: Sequence[str] = (),
        raw_input_hashes: dict[str, str] | None = None,
        schema_versions: dict[str, str] | None = None,
        now: datetime | None = None,
    ) -> RunContext:
        """Open a run. cfg may be None so that a configuration failure can still be
        recorded as a FAIL run (invariant I6)."""
        now = now or datetime.now(UTC)
        cfg_hash = config_hash(cfg) if cfg is not None else UNAVAILABLE
        commit = git_commit(paths.root)
        dirty = git_is_dirty(paths.root)
        notes: list[str] = []
        if commit is None:
            # A silent null commit is indistinguishable from "no git installed", "not a
            # repository" and "git failed" — and SPEC 19.12 exists precisely to make a run
            # traceable. Say so in the record rather than leaving a bare null.
            notes.append("git commit unavailable: not a git repository, or git could not be run")
        if dirty:
            notes.append("working tree was dirty at run start")
        base_id = make_run_id(command, cfg_hash, commit, now)
        run_id, run_dir = _claim_run_dir(paths.runs, base_id)
        record = RunRecord(
            run_id=run_id,
            command=command,
            argv=list(argv),
            created_at=now.isoformat(),
            git_commit=commit,
            git_dirty=dirty,
            spec_version=SPEC_VERSION,
            config_version=cfg.config_version if cfg is not None else UNAVAILABLE,
            resolved_config_hash=cfg_hash,
            environment_lock_hash=environment_lock_hash(paths),
            python_version=platform.python_version(),
            schema_versions=dict(schema_versions or {}),
            raw_input_hashes=dict(raw_input_hashes or {}),
            parent_run_ids=list(parent_run_ids),
            status="RUNNING",
            outputs={},
            notes=notes,
        )
        ctx = cls(run_id=run_id, run_dir=run_dir, record=record)
        try:
            ctx.write()
        except BaseException:
            # A claimed directory holding no run.json is a run that happened but cannot be
            # recorded — exactly what invariant I6 forbids. Release the claim instead.
            shutil.rmtree(run_dir, ignore_errors=True)
            raise
        # run.json now exists, so a failure below is recorded by the caller's FAIL path
        # rather than vanishing.
        if cfg is not None:
            write_json_atomic(run_dir / "resolved_config.json", resolved_config_dict(cfg))
        return ctx

    def write(self) -> None:
        write_json_atomic(self.run_dir / "run.json", self.record.model_dump(mode="json"))

    def update(self, **fields: Any) -> None:
        """Merge fields into the record, re-validating the result.

        model_copy(update=...) skips validation, and pydantic then serialises a bad value in
        warn mode rather than raising — so a wrong status or a malformed output would be
        written into run.json silently. Static typing does not close this: values reaching
        here can arrive through Any-typed boundaries such as parsed JSON or subprocess
        output. Lineage is the one thing that must not be quietly corrupt.
        """
        self.record = RunRecord.model_validate({**self.record.model_dump(), **fields})
        self.write()

    def add_output(self, key: str, ref: ArtifactRef) -> None:
        self.update(outputs={**self.record.outputs, key: ref})

    def finish(
        self,
        status: RunStatus,
        outputs: dict[str, ArtifactRef] | None = None,
        notes: Sequence[str] = (),
    ) -> None:
        self.update(
            status=status,
            outputs={**self.record.outputs, **(outputs or {})},
            notes=[*self.record.notes, *notes],
            finished_at=datetime.now(UTC).isoformat(),
        )
```

> `update` перепроверяет запись через `model_validate`. Это осознанно дороже, чем `model_copy(update=...)`: последний не валидирует поля, а pydantic затем сериализует некорректное значение в режиме warn, не поднимая исключения — то есть неверный `status` или испорченный `outputs` молча попал бы в `run.json`. Статическая типизация тут не спасает: значения могут приходить через границы с типом `Any` (разобранный JSON, вывод subprocess). Тип `Any` в сигнатуре `update` осознан.

- [ ] **Step 5: Реализовать `logging_setup.py`**

```python
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
```

> Фильтр стоит на обработчиках, а не на логгере, поэтому записи дочерних логгеров (`so_recon.child`) тоже получают `run_id`.

- [ ] **Step 6: Прогнать тесты, lint, format, mypy**

```bash
uv run pytest tests/unit/test_run_registry.py tests/unit/test_logging_setup.py -q && uv run ruff check . && uv run ruff format --check . && uv run mypy
```
Ожидается: `10 passed`, без ошибок.

- [ ] **Step 7: Commit**

```bash
git add src/so_recon/registry/gitinfo.py src/so_recon/registry/run.py src/so_recon/logging_setup.py tests/unit/test_run_registry.py tests/unit/test_logging_setup.py && git commit -m "feat(e00): add run registry with artifact-ref lineage, degraded FAIL runs and logging

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---
### Task 7: Source manifest исходных CSV

**Files:**
- Create: `src/so_recon/registry/source_manifest.py`
- Test: `tests/unit/test_source_manifest.py`

**Interfaces:**
- Consumes: `SourcesConfig`, `SourceFileSpec`, `StrictModel` (Task 4); `ProjectPaths`, `resolve_within_root` (Task 3); `write_bytes_atomic` (Task 2); `ArtifactRef`, `write_artifact` (Task 5).
- Produces:
  - `MANIFEST_SCHEMA_VERSION = "2"` — версия 2, потому что схема изменилась относительно ревизии 1 (`line_count` → `physical_line_count` + `data_rows`, удалены timestamps).
  - `class MissingSourceError(FileNotFoundError)` — атрибут `missing: list[str]`.
  - `hash_and_count(path, chunk_size=1 << 20) -> tuple[str, int, int]` — `(sha256, size_bytes, physical_line_count)` за один проход в режиме `"rb"`. `physical_line_count` = число `\n` плюс 1, если файл непуст и не заканчивается `\n`.
  - `class SourceEntry(StrictModel)`: `name`, `path`, `sha256`, `size_bytes`, `physical_line_count`, `data_rows`, `encoding`, `delimiter`, `decimal`, `header_lines`, `required`.
  - `class SourceManifest(StrictModel)` — **детерминированный**: `manifest_version: Literal["2"]`, `spec_version: str`, `config_version: str`, `sources: list[SourceEntry]`. Никаких timestamps, `run_id` или git-состояния (инвариант I5). Метод `hashes() -> dict[str, str]`.
  - `class SourceManifestStamp(StrictModel)` — **run-scoped, не версионируется**: `manifest_version`, `manifest_sha256`, `run_id`, `created_at`, `git_commit: str | None`, `git_dirty: bool | None`.
  - `build_source_manifest(sources, paths, *, config_version) -> SourceManifest` — пути раскрываются через `paths.resolve(spec.path)` (containment, инвариант I4); все отсутствующие `required` собираются и бросается один `MissingSourceError`; `required=False` пропускаются с записью в лог. `data_rows = max(physical_line_count - header_lines, 0)`.
  - `manifest_bytes(manifest) -> bytes` — детерминированный JSON (`indent=2`, `sort_keys=True`, `ensure_ascii=False`) + `\n`.
  - `write_source_manifest(manifest, paths, *, run_dir, published_path, producer_run_id, now) -> tuple[ArtifactRef, Path]` — пишет **immutable** артефакт `run_dir/source_manifest.json` через `write_artifact`, затем публикует байт-в-байт идентичную копию в `published_path` (`reports/manifests/source_manifest.json`) через `write_bytes_atomic`. Возвращает ref артефакта запуска и путь публикации.
  - `write_manifest_stamp(stamp, path) -> None` — атомарная запись run-scoped штампа.
  - `load_source_manifest(path) -> SourceManifest`.

**Замечания ревизии 2, реализуемые здесь:** №1 (`physical_line_count` / `data_rows`), №2 (никакого `mv`; файлы открываются только на чтение), №4 (immutable артефакт запуска), №10 (детерминированный коммитируемый manifest, timestamps — в run-scoped штампе), №12 (атомарность).

> **Почему две копии.** `artifacts/runs/<run_id>/source_manifest.json` — immutable результат конкретного запуска (замечание №4). `reports/manifests/source_manifest.json` — публикуемая детерминированная копия тех же байт, которая коммитится; её изменение означает реальное изменение данных или конфигурации и видно в `git diff` (замечание №10). История прежних версий хранится в git, поэтому публикация не нарушает immutability.

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
    MANIFEST_SCHEMA_VERSION,
    MissingSourceError,
    build_source_manifest,
    hash_and_count,
    load_source_manifest,
    manifest_bytes,
    write_source_manifest,
)

NOW = datetime(2026, 9, 13, 12, 0, tzinfo=UTC)


def _spec(name: str, required: bool = True, header_lines: int = 1) -> SourceFileSpec:
    return SourceFileSpec(
        name=name, path=f"data/raw/{name}.csv", encoding="utf-8", delimiter=";", decimal=",",
        header_lines=header_lines, required=required,
    )


def _paths(tmp_path: Path) -> ProjectPaths:
    paths = ProjectPaths.default(tmp_path)
    paths.ensure_dirs()
    paths.raw.mkdir(parents=True, exist_ok=True)  # tests create the source dir explicitly
    return paths


def test_hash_and_count_trailing_newline(tmp_path: Path) -> None:
    p = tmp_path / "f.csv"
    p.write_bytes(b"a;b\r\n1;2\r\n")
    sha, size, physical = hash_and_count(p, chunk_size=2)
    assert sha == hashlib.sha256(b"a;b\r\n1;2\r\n").hexdigest()
    assert (size, physical) == (10, 2)


def test_hash_and_count_without_trailing_newline(tmp_path: Path) -> None:
    """plastoper.csv ends without a newline: the last partial line still counts."""
    p = tmp_path / "f.csv"
    p.write_bytes(b"a;b\r\n1;2\r\n3;4")
    sha, size, physical = hash_and_count(p, chunk_size=2)
    assert sha == hashlib.sha256(b"a;b\r\n1;2\r\n3;4").hexdigest()
    assert (size, physical) == (13, 3)


def test_hash_and_count_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "f.csv"
    p.write_bytes(b"")
    assert hash_and_count(p) == (hashlib.sha256(b"").hexdigest(), 0, 0)


def test_manifest_splits_physical_lines_from_data_rows(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    (paths.raw / "mer.csv").write_bytes(b"\xef\xbb\xbfh1;h2\n1;2\n3;4\n")
    (paths.raw / "tail.csv").write_bytes(b"h1;h2\n1;2")  # no trailing newline
    sources = SourcesConfig(files=[_spec("mer"), _spec("tail")])
    m = build_source_manifest(sources, paths, config_version="t")
    by_name = {s.name: s for s in m.sources}
    assert (by_name["mer"].physical_line_count, by_name["mer"].data_rows) == (3, 2)
    assert (by_name["tail"].physical_line_count, by_name["tail"].data_rows) == (2, 1)
    assert by_name["mer"].header_lines == 1


def test_manifest_has_no_timestamps_or_git_state(tmp_path: Path) -> None:
    """Invariant I5: the committed manifest must be byte-identical across runs."""
    paths = _paths(tmp_path)
    (paths.raw / "mer.csv").write_bytes(b"h\n1\n")
    sources = SourcesConfig(files=[_spec("mer")])
    first = manifest_bytes(build_source_manifest(sources, paths, config_version="t"))
    second = manifest_bytes(build_source_manifest(sources, paths, config_version="t"))
    assert first == second
    payload = json.loads(first)
    assert payload["manifest_version"] == MANIFEST_SCHEMA_VERSION
    assert set(payload) == {"manifest_version", "spec_version", "config_version", "sources"}


def test_missing_required_files_raise_with_full_list(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    sources = SourcesConfig(files=[_spec("mer"), _spec("gis"), _spec("opt", required=False)])
    with pytest.raises(MissingSourceError) as exc:
        build_source_manifest(sources, paths, config_version="t")
    assert exc.value.missing == ["data/raw/mer.csv", "data/raw/gis.csv"]


def test_optional_missing_file_is_skipped(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    (paths.raw / "mer.csv").write_bytes(b"1\n")
    sources = SourcesConfig(files=[_spec("mer"), _spec("opt", required=False)])
    m = build_source_manifest(sources, paths, config_version="t")
    assert [s.name for s in m.sources] == ["mer"]


def test_sources_are_never_modified(tmp_path: Path) -> None:
    """Invariant I1: hashing must not touch mtime, size or content of the sources."""
    paths = _paths(tmp_path)
    src = paths.raw / "mer.csv"
    src.write_bytes(b"h\n1\n")
    before = (src.read_bytes(), src.stat().st_size, src.stat().st_mtime_ns)
    build_source_manifest(SourcesConfig(files=[_spec("mer")]), paths, config_version="t")
    after = (src.read_bytes(), src.stat().st_size, src.stat().st_mtime_ns)
    assert before == after
    assert sorted(p.name for p in paths.raw.iterdir()) == ["mer.csv"]


def test_write_publishes_identical_bytes_and_roundtrips(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    (paths.raw / "mer.csv").write_bytes(b"h\n1\n")
    m = build_source_manifest(SourcesConfig(files=[_spec("mer")]), paths, config_version="t")
    run_dir = paths.runs / "run-1"
    run_dir.mkdir(parents=True)
    published = paths.manifests / "source_manifest.json"
    ref, published_path = write_source_manifest(
        m, paths, run_dir=run_dir, published_path=published, producer_run_id="run-1", now=NOW
    )
    assert published_path == published
    assert (run_dir / "source_manifest.json").read_bytes() == published.read_bytes()
    assert ref.sha256 == hashlib.sha256(published.read_bytes()).hexdigest()
    assert ref.producer_run_id == "run-1"
    assert load_source_manifest(published) == m
```

- [ ] **Step 2: Убедиться, что тесты падают**

```bash
uv run pytest tests/unit/test_source_manifest.py -q
```
Ожидается: `ModuleNotFoundError: so_recon.registry.source_manifest`.

- [ ] **Step 3: Реализовать `source_manifest.py`**

```python
"""Manifest of raw source files: SHA-256, size and line counts (SPEC 7.1, 19.12).

The sources are immutable: every file is opened read-only and is never moved, renamed
or rewritten (invariant I1). The manifest itself is deterministic — it carries no
timestamps, run ids or git state, so a repeated gate run produces no git diff
(invariant I5). Those run-scoped facts live in SourceManifestStamp instead.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal

from so_recon import SPEC_VERSION
from so_recon.config.schema import SourcesConfig, StrictModel
from so_recon.paths import ProjectPaths
from so_recon.registry.artifact import ArtifactRef, write_artifact
from so_recon.registry.atomic import write_bytes_atomic, write_json_atomic

log = logging.getLogger(__name__)

MANIFEST_SCHEMA_VERSION = "2"
MANIFEST_MEDIA_TYPE = "application/json"


class MissingSourceError(FileNotFoundError):
    def __init__(self, missing: Sequence[str]) -> None:
        super().__init__(f"required source files are missing: {list(missing)}")
        self.missing = list(missing)


def hash_and_count(path: Path, chunk_size: int = 1 << 20) -> tuple[str, int, int]:
    """Return (sha256, size_bytes, physical_line_count) in a single read-only pass.

    physical_line_count counts newline bytes plus a final partial line when the file
    does not end with a newline. It is a physical count, not a parsed-record count:
    E00 does not parse CSV, so newlines inside quoted fields are not accounted for.
    """
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
    physical = newlines + (1 if size > 0 and last != b"\n" else 0)
    return digest.hexdigest(), size, physical


class SourceEntry(StrictModel):
    name: str
    path: str
    sha256: str
    size_bytes: int
    physical_line_count: int
    data_rows: int
    encoding: str
    delimiter: str
    decimal: str
    header_lines: int
    required: bool


class SourceManifest(StrictModel):
    manifest_version: Literal["2"] = "2"
    spec_version: str
    config_version: str
    sources: list[SourceEntry]

    def hashes(self) -> dict[str, str]:
        return {s.name: s.sha256 for s in self.sources}


class SourceManifestStamp(StrictModel):
    """Run-scoped facts kept out of the committed manifest (invariant I5)."""

    manifest_version: str
    manifest_sha256: str
    run_id: str
    created_at: str
    git_commit: str | None
    git_dirty: bool | None


def build_source_manifest(
    sources: SourcesConfig, paths: ProjectPaths, *, config_version: str
) -> SourceManifest:
    entries: list[SourceEntry] = []
    missing: list[str] = []
    for spec in sources.files:
        full = paths.resolve(spec.path)  # validates form and containment (invariant I4)
        if not full.is_file():
            if spec.required:
                missing.append(spec.path)
            else:
                log.warning("optional source %s not found at %s; skipped", spec.name, spec.path)
            continue
        sha, size, physical = hash_and_count(full)
        data_rows = max(physical - spec.header_lines, 0)
        log.info(
            "hashed %s: %d bytes, %d physical lines, %d data rows",
            spec.name, size, physical, data_rows,
        )
        entries.append(
            SourceEntry(
                name=spec.name,
                path=spec.path,
                sha256=sha,
                size_bytes=size,
                physical_line_count=physical,
                data_rows=data_rows,
                encoding=spec.encoding,
                delimiter=spec.delimiter,
                decimal=spec.decimal,
                header_lines=spec.header_lines,
                required=spec.required,
            )
        )
    if missing:
        raise MissingSourceError(missing)
    return SourceManifest(
        spec_version=SPEC_VERSION, config_version=config_version, sources=entries
    )


def manifest_bytes(manifest: SourceManifest) -> bytes:
    payload = manifest.model_dump(mode="json")
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def write_source_manifest(
    manifest: SourceManifest,
    paths: ProjectPaths,
    *,
    run_dir: Path,
    published_path: Path,
    producer_run_id: str,
    now: datetime,
) -> tuple[ArtifactRef, Path]:
    """Write the immutable run artifact, then publish byte-identical committed copy."""
    payload = manifest_bytes(manifest)
    ref = write_artifact(
        run_dir / "source_manifest.json",
        payload,
        paths,
        schema_version=MANIFEST_SCHEMA_VERSION,
        producer_run_id=producer_run_id,
        media_type=MANIFEST_MEDIA_TYPE,
        now=now,
    )
    write_bytes_atomic(published_path, payload)
    return ref, published_path


def write_manifest_stamp(stamp: SourceManifestStamp, path: Path) -> None:
    write_json_atomic(path, stamp.model_dump(mode="json"))


def load_source_manifest(path: Path) -> SourceManifest:
    return SourceManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
```

- [ ] **Step 4: Прогнать тесты, lint, format, mypy**

```bash
uv run pytest tests/unit/test_source_manifest.py -q && uv run ruff check . && uv run ruff format --check . && uv run mypy
```
Ожидается: `9 passed`.

- [ ] **Step 5: Проверить источники на месте, ничего не перемещая**

Единственная разрешённая операция над исходниками — чтение. Зафиксировать хеши **на исходном пути**:

```bash
shasum -a 256 data/Ромашка_сырые/*.csv
```
Ожидаемые значения для двух малых файлов известны заранее: `coords.csv` → `3298fda637a5f36d2ecbd5ec014e037a95bbd9631973db488c6dcbc1ade78996`, `plastoper.csv` → `129c32e7deb51a06ede288be21b1d878a36fb0f7edf9ce52b591b445674f934b`. Если значение отличается — **остановиться и сообщить**: исходные данные изменились, это не задача E00.

Проверить, что каталог источников не изменялся (ни `mv`, ни новых файлов):
```bash
ls -1 data/Ромашка_сырые/
```
Ожидается ровно пять файлов: `coords.csv`, `gis.csv`, `mer.csv`, `perf.csv`, `plastoper.csv`. Каталог `data/raw/` **не создаётся**.

- [ ] **Step 6: Commit**

```bash
git add src/so_recon/registry/source_manifest.py tests/unit/test_source_manifest.py && git commit -m "feat(e00): add deterministic raw source manifest with physical lines and data rows

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: Детерминированный synthetic fixture и `case.json` (Python-часть)

**Files:**
- Create: `src/so_recon/synthetic/fixture.py`
- Test: `tests/unit/test_fixture.py`

**Interfaces:**
- Consumes: `SmokeFixtureConfig` (Task 4); `canonical_json`, `sha256_json`, `sha256_bytes` (Task 2); `write_bytes_atomic`, `write_json_atomic` (Task 2).
- Produces:
  - `FIXTURE_SCHEMA_VERSION = "1"`, `CASE_SCHEMA_VERSION = "1"`.
  - `@dataclass(frozen=True) class SmokeFixture`: `wells: pa.Table`, `well_month: pa.Table`, `content_hash: str`, `seed: int`.
  - `build_smoke_fixture(cfg) -> SmokeFixture` — детерминирован по `cfg.seed`; `wells`: `well_id: str`, `role: str` (`injector`/`producer`), `x_m: float`, `y_m: float`; `well_month`: `well_id: str`, `month: str` (`YYYY-MM-01`), `liquid_m3: float`, `injection_m3: float`; объёмы неотрицательны и округлены до 6 знаков; у injector `liquid_m3 = 0`, у producer `injection_m3 = 0`.
  - `fixture_content_hash(wells, well_month) -> str` — `sha256_json({"wells": ..., "well_month": ...})`, не зависит от кодирования Parquet.
  - `write_smoke_fixture(fixture, out_dir) -> dict[str, Path]` — `wells.parquet`, `well_month.parquet`, `fixture_meta.json`.
  - `build_smoke_case(cfg, fixture) -> dict[str, Any]` — **единственный вход Julia**. Полностью выводится из `cfg` и `fixture`; содержит `schema_version`, `seed`, `fixture_content_hash`, `grid`, `rock`, `fluids`, `initial`, `schedule`, `controls`. `controls.injected_pore_volume_fraction` выводится из объёмов fixture: `round(0.25 + 0.5 * total_injection / (total_injection + total_liquid), 9)`, что помещает значение в интервал `(0.25, 0.75]`; при нулевой сумме объёмов — `ValueError`.
  - `case_bytes(case) -> bytes` — `canonical_json(case).encode("utf-8")`; именно эти байты хеширует и Python, и Julia.
  - `write_smoke_case(case, path) -> bytes` — атомарная запись `case_bytes`, возвращает записанные байты.

**Замечания ревизии 2, реализуемые здесь:** №3 (Python формирует `case.json` — вход для Julia), №12 (атомарность).

- [ ] **Step 1: Написать падающие тесты**

`tests/unit/test_fixture.py`:
```python
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from so_recon.config.schema import SmokeFixtureConfig
from so_recon.registry.hashing import canonical_json
from so_recon.synthetic.fixture import (
    CASE_SCHEMA_VERSION,
    build_smoke_case,
    build_smoke_fixture,
    case_bytes,
    fixture_content_hash,
    write_smoke_case,
    write_smoke_fixture,
)


def test_fixture_is_deterministic_for_same_seed() -> None:
    a = build_smoke_fixture(SmokeFixtureConfig(seed=7, n_wells=4, n_months=3))
    b = build_smoke_fixture(SmokeFixtureConfig(seed=7, n_wells=4, n_months=3))
    assert a.content_hash == b.content_hash
    assert a.wells.to_pylist() == b.wells.to_pylist()


def test_fixture_changes_with_seed() -> None:
    assert (
        build_smoke_fixture(SmokeFixtureConfig(seed=7)).content_hash
        != build_smoke_fixture(SmokeFixtureConfig(seed=8)).content_hash
    )


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
    assert sorted(set(fx.well_month["month"].to_pylist())) == [
        "2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"
    ]


def test_write_fixture_roundtrips_and_records_hash(tmp_path: Path) -> None:
    fx = build_smoke_fixture(SmokeFixtureConfig(seed=3, n_wells=2, n_months=2))
    files = write_smoke_fixture(fx, tmp_path / "fixture")
    assert fixture_content_hash(pq.read_table(files["wells"]), pq.read_table(files["well_month"])) == fx.content_hash
    meta = json.loads(files["meta"].read_text(encoding="utf-8"))
    assert meta["content_hash"] == fx.content_hash
    assert meta["seed"] == 3


def test_case_is_deterministic_and_derived_from_the_fixture() -> None:
    cfg = SmokeFixtureConfig(seed=5, n_wells=4, n_months=3, nx=7, n_steps=4)
    fx = build_smoke_fixture(cfg)
    case = build_smoke_case(cfg, fx)
    assert case["schema_version"] == CASE_SCHEMA_VERSION
    assert case["fixture_content_hash"] == fx.content_hash
    assert case["grid"]["nx"] == 7
    assert case["schedule"]["n_steps"] == 4
    frac = case["controls"]["injected_pore_volume_fraction"]
    assert 0.25 < frac <= 0.75
    assert build_smoke_case(cfg, build_smoke_fixture(cfg)) == case


def test_case_changes_when_the_fixture_changes() -> None:
    a_cfg = SmokeFixtureConfig(seed=5, n_wells=4, n_months=3)
    b_cfg = SmokeFixtureConfig(seed=6, n_wells=4, n_months=3)
    a = build_smoke_case(a_cfg, build_smoke_fixture(a_cfg))
    b = build_smoke_case(b_cfg, build_smoke_fixture(b_cfg))
    assert case_bytes(a) != case_bytes(b)


def test_case_bytes_are_canonical_and_hashable(tmp_path: Path) -> None:
    cfg = SmokeFixtureConfig(seed=5, n_wells=2, n_months=2)
    case = build_smoke_case(cfg, build_smoke_fixture(cfg))
    path = tmp_path / "case.json"
    written = write_smoke_case(case, path)
    assert written == path.read_bytes()
    assert written == canonical_json(case).encode("utf-8")
    # This is the exact digest Julia is required to return as input_sha256.
    assert hashlib.sha256(path.read_bytes()).hexdigest() == hashlib.sha256(written).hexdigest()


def test_case_rejects_a_fixture_with_no_volumes() -> None:
    cfg = SmokeFixtureConfig(seed=1, n_wells=2, n_months=1)
    fx = build_smoke_fixture(cfg)
    import pyarrow as pa

    empty = pa.table({
        "well_id": pa.array([], pa.string()), "month": pa.array([], pa.string()),
        "liquid_m3": pa.array([], pa.float64()), "injection_m3": pa.array([], pa.float64()),
    })
    with pytest.raises(ValueError):
        build_smoke_case(cfg, type(fx)(wells=fx.wells, well_month=empty, content_hash="h", seed=1))
```

- [ ] **Step 2: Убедиться, что тесты падают**

```bash
uv run pytest tests/unit/test_fixture.py -q
```
Ожидается: `ModuleNotFoundError: so_recon.synthetic.fixture`.

- [ ] **Step 3: Реализовать `fixture.py`**

```python
"""Deterministic synthetic fixture and the JutulDarcy case description for the E00 smoke.

Not a physical model: two tiny tables (wells, well_month) generated from a seeded PCG64
stream, hashed by canonical JSON content so the hash is independent of Parquet encoding.
The fixture then determines case.json, which is the only input Julia reads — that makes
the smoke end-to-end: Julia returns the SHA-256 of the bytes it actually read.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from so_recon.config.schema import SmokeFixtureConfig
from so_recon.registry.atomic import write_bytes_atomic, write_json_atomic
from so_recon.registry.hashing import canonical_json, sha256_json

FIXTURE_SCHEMA_VERSION = "1"
CASE_SCHEMA_VERSION = "1"

_START_YEAR = 2020
_START_MONTH = 1

# Fixed case geometry and fluid properties. Only the injection fraction is data-derived.
_CELL_DX_M = 50.0
_CELL_DY_M = 50.0
_CELL_DZ_M = 10.0
_PERMEABILITY_DARCY = 0.1
_POROSITY = 0.25
_WATER_DENSITY = 1000.0
_OIL_DENSITY = 850.0
_INITIAL_PRESSURE_BAR = 150.0
_INITIAL_SW = 0.2
_PRODUCER_BHP_BAR = 100.0
_STEP_DAYS = 30.0


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
    write_json_atomic(
        meta_path,
        {
            "schema_version": FIXTURE_SCHEMA_VERSION,
            "seed": fixture.seed,
            "content_hash": fixture.content_hash,
            "n_rows": {"wells": fixture.wells.num_rows, "well_month": fixture.well_month.num_rows},
        },
    )
    return {"wells": wells_path, "well_month": wm_path, "meta": meta_path}


def build_smoke_case(cfg: SmokeFixtureConfig, fixture: SmokeFixture) -> dict[str, Any]:
    """Derive the JutulDarcy case description from the configuration and the fixture."""
    total_injection = float(sum(fixture.well_month["injection_m3"].to_pylist()))
    total_liquid = float(sum(fixture.well_month["liquid_m3"].to_pylist()))
    total = total_injection + total_liquid
    if total <= 0.0:
        raise ValueError("fixture carries no volumes; cannot derive an injection target")
    fraction = round(0.25 + 0.5 * (total_injection / total), 9)
    return {
        "schema_version": CASE_SCHEMA_VERSION,
        "seed": cfg.seed,
        "fixture_content_hash": fixture.content_hash,
        "grid": {"nx": cfg.nx, "dx_m": _CELL_DX_M, "dy_m": _CELL_DY_M, "dz_m": _CELL_DZ_M},
        "rock": {"permeability_darcy": _PERMEABILITY_DARCY, "porosity": _POROSITY},
        "fluids": {
            "water_density_kg_m3": _WATER_DENSITY,
            "oil_density_kg_m3": _OIL_DENSITY,
        },
        "initial": {
            "pressure_bar": _INITIAL_PRESSURE_BAR,
            "water_saturation": _INITIAL_SW,
            "oil_saturation": round(1.0 - _INITIAL_SW, 9),
        },
        "schedule": {"n_steps": cfg.n_steps, "dt_days": _STEP_DAYS},
        "controls": {
            "injected_pore_volume_fraction": fraction,
            "producer_bhp_bar": _PRODUCER_BHP_BAR,
        },
    }


def case_bytes(case: dict[str, Any]) -> bytes:
    """The exact bytes Julia reads and hashes back as input_sha256."""
    return canonical_json(case).encode("utf-8")


def write_smoke_case(case: dict[str, Any], path: Path) -> bytes:
    payload = case_bytes(case)
    write_bytes_atomic(path, payload)
    return payload
```

- [ ] **Step 4: Прогнать тесты, lint, format, mypy**

```bash
uv run pytest tests/unit/test_fixture.py -q && uv run ruff check . && uv run ruff format --check . && uv run mypy
```
Ожидается: `8 passed`. Если mypy жалуется на `pa.Table` — override для `pyarrow` задан в `pyproject.toml`, тип станет `Any`; это допустимо.

- [ ] **Step 5: Commit**

```bash
git add src/so_recon/synthetic/fixture.py tests/unit/test_fixture.py && git commit -m "feat(e00): add deterministic smoke fixture and derived case.json input

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---
### Task 9: Julia-окружение с JutulDarcy и сквозной smoke-скрипт

**Files:**
- Create: `julia/Project.toml`, `julia/Manifest.toml` (генерируется), `julia/.julia-version`, `julia/README.md`, `julia/smoke/smoke_case.jl`

**Interfaces:**
- Produces: команда
  `julia --project=julia --startup-file=no julia/smoke/smoke_case.jl --case <case.json> --out <result.json>`
  пишет JSON с ключами: `status` (`"ok"`), `input_sha256` (SHA-256 **байт прочитанного `case.json`**), `case_schema_version`, `julia_version`, `jutuldarcy_version`, `jutul_version`, `nx`, `n_steps`, `cumulative_oil_m3` (>0), `cumulative_water_injected_m3` (>0), `mean_so_final` (в (0,1) и меньше 0.8), `wall_time_s`. Код возврата 0. При ошибке — JSON `{"status":"error","message":...,"wall_time_s":...}` и код возврата 1.
- Produces: `julia/.julia-version` — **точная project-pinned версия Julia**.

**Замечания ревизии 2, реализуемые здесь:** №3 (Julia читает `case.json` и возвращает `input_sha256`), №7 (точная project-pinned версия Julia, без утверждений об upstream stable), №11.

> **Формулировка о версии.** Версия Julia в `julia/.julia-version` — это версия, **зафиксированная проектом** и использованная для разрешения `julia/Manifest.toml`. План не утверждает, что она является актуальным upstream stable-релизом. Обновление выполняется осознанно: новая версия → новый `Manifest.toml` → новая строка в `reports/environment_report.md`.

- [ ] **Step 1: Установить Julia через juliaup и зафиксировать точную версию**

Julia и `juliaup` в системе отсутствуют. Установка:

```bash
brew install juliaup && juliaup add 1.12 && julia +1.12 --version
```
Если `brew` недоступен — официальная альтернатива: `curl -fsSL https://install.julialang.org | sh -s -- --yes --default-channel 1.12`, затем перезапустить shell.

Узнать **точный** patch-релиз, который дал канал `1.12`, и закрепить именно его:

```bash
JULIA_EXACT="$(julia +1.12 --version | awk '{print $3}')" && echo "exact: $JULIA_EXACT" && juliaup add "$JULIA_EXACT" && juliaup default "$JULIA_EXACT" && julia --version && mkdir -p julia && printf '%s\n' "$JULIA_EXACT" > julia/.julia-version && cat julia/.julia-version
```
Ожидается: `julia --version` печатает ровно то же значение, что записано в `julia/.julia-version`. Это значение используется дальше как project-pinned версия.

- [ ] **Step 2: Создать `julia/Project.toml` и разрешить зависимости**

UUID зависимостей проставляет сам `Pkg.add`, поэтому создаётся заготовка только с `[compat]`, а пакеты добавляются командой. `SHA` — stdlib, нужен для `input_sha256`.

```bash
mkdir -p julia/smoke && printf 'name = "SOReconEnv"\nuuid = "0f9b3f1e-7c4a-4a7e-9a3d-5e00e00e0001"\nversion = "0.0.1"\n\n[deps]\n\n[compat]\nJSON = "0.21, 1"\nJutul = "0.4"\nJutulDarcy = "0.3"\njulia = "1.12"\n' > julia/Project.toml && julia --project=julia --startup-file=no -e 'using Pkg; Pkg.add(["JutulDarcy", "Jutul", "JSON", "SHA"]); Pkg.precompile(); Pkg.status()'
```
Ожидается (5–15 минут на первый precompile): `Pkg.status()` показывает `JutulDarcy v0.3.x`, `Jutul v0.4.x`, `JSON`, `SHA`. Появился `julia/Manifest.toml`.

Проверить, что `Manifest.toml` записал ту же версию Julia, что зафиксирована:
```bash
grep -m1 '^julia_version' julia/Manifest.toml && cat julia/.julia-version
```
Ожидается: значения совпадают. Если нет — вернуться к Step 1 и переключить default.

- [ ] **Step 3: Написать `julia/smoke/smoke_case.jl`**

```julia
# E00 smoke: tiny 1D oil-water JutulDarcy case. Not a verification test (that is E05);
# it proves the locked Julia environment runs end to end and produces deterministic
# scalar summaries. The single input is case.json, written by Python; the script returns
# the SHA-256 of the bytes it actually read, so the Python side can prove the round trip.
using JutulDarcy, Jutul, JSON, SHA

function parse_cli(args::Vector{String})
    opts = Dict{String,String}()
    i = 1
    while i <= length(args)
        key = args[i]
        startswith(key, "--") || error("unexpected argument $(key)")
        i + 1 <= length(args) || error("missing value for $(key)")
        opts[key[3:end]] = args[i + 1]
        i += 2
    end
    haskey(opts, "case") || error("--case <case.json> is required")
    haskey(opts, "out") || error("--out <file.json> is required")
    return opts
end

function read_case(path::AbstractString)
    raw = read(path)                       # exact bytes on disk
    input_sha = bytes2hex(sha256(raw))
    case = JSON.parse(String(copy(raw)))   # copy: String(::Vector{UInt8}) takes ownership
    return case, input_sha
end

function run_smoke(case::Dict)
    Darcy, bar, kg, meter, day = si_units(:darcy, :bar, :kilogram, :meter, :day)

    grid = case["grid"]
    rock = case["rock"]
    fluids = case["fluids"]
    initial = case["initial"]
    schedule = case["schedule"]
    controls = case["controls"]

    nx = Int(grid["nx"])
    nsteps = Int(schedule["n_steps"])

    g = CartesianMesh(
        (nx, 1, 1),
        (nx * Float64(grid["dx_m"]), Float64(grid["dy_m"]), Float64(grid["dz_m"])) .* meter,
    )
    domain = reservoir_domain(
        g,
        permeability = Float64(rock["permeability_darcy"]) * Darcy,
        porosity = Float64(rock["porosity"]),
    )
    injector = setup_vertical_well(domain, 1, 1, name = :Injector)
    producer = setup_vertical_well(domain, nx, 1, name = :Producer)

    rhoWS = Float64(fluids["water_density_kg_m3"])kg / meter^3
    rhoOS = Float64(fluids["oil_density_kg_m3"])kg / meter^3
    sys = ImmiscibleSystem((AqueousPhase(), LiquidPhase()), reference_densities = [rhoWS, rhoOS])
    model = setup_reservoir_model(domain, sys, wells = [injector, producer])
    parameters = setup_parameters(model)
    state0 = setup_reservoir_state(
        model,
        Pressure = Float64(initial["pressure_bar"])bar,
        Saturations = [Float64(initial["water_saturation"]), Float64(initial["oil_saturation"])],
    )

    dt = fill(Float64(schedule["dt_days"])day, nsteps)
    pv = pore_volume(model, parameters)
    inj_rate = Float64(controls["injected_pore_volume_fraction"]) * sum(pv) / sum(dt)
    i_ctrl = InjectorControl(TotalRateTarget(inj_rate), [1.0, 0.0], density = rhoWS)
    p_ctrl = ProducerControl(BottomHolePressureTarget(Float64(controls["producer_bhp_bar"])bar))
    forces = setup_reservoir_forces(model, control = Dict(:Injector => i_ctrl, :Producer => p_ctrl))

    wd, states, t = simulate_reservoir(
        state0, model, dt, parameters = parameters, forces = forces, info_level = -1
    )

    orat = wd[:Producer, :orat]      # surface oil rate, negative for production
    wrat_inj = wd[:Injector, :wrat]  # surface water rate, positive for injection
    cum_oil = -sum(orat .* dt)
    cum_winj = sum(wrat_inj .* dt)
    so_final = states[end][:Saturations][2, :]
    mean_so = sum(so_final .* pv) / sum(pv)

    return Dict(
        "status" => "ok",
        "case_schema_version" => string(case["schema_version"]),
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
    t0 = time()
    result = try
        opts = parse_cli(args)
        case, input_sha = read_case(opts["case"])
        r = run_smoke(case)
        r["input_sha256"] = input_sha
        r["wall_time_s"] = time() - t0
        out = opts["out"]
        mkpath(dirname(abspath(out)))
        open(out, "w") do io
            write(io, JSON.json(r))   # JSON.json exists in both JSON.jl 0.21 and 1.x
        end
        r
    catch err
        r = Dict(
            "status" => "error",
            "message" => sprint(showerror, err),
            "wall_time_s" => time() - t0,
        )
        # Best effort: still report through --out when it was parsed successfully.
        try
            opts = parse_cli(args)
            mkpath(dirname(abspath(opts["out"])))
            open(opts["out"], "w") do io
                write(io, JSON.json(r))
            end
        catch
            println(stderr, r["message"])
        end
        r
    end
    return result["status"] == "ok" ? 0 : 1
end

exit(main(ARGS))
```

- [ ] **Step 4: Запустить скрипт вручную и проверить инварианты и `input_sha256`**

```bash
mkdir -p artifacts/tmp && printf '{"controls":{"injected_pore_volume_fraction":0.5,"producer_bhp_bar":100.0},"fixture_content_hash":"manual","fluids":{"oil_density_kg_m3":850.0,"water_density_kg_m3":1000.0},"grid":{"dx_m":50.0,"dy_m":50.0,"dz_m":10.0,"nx":20},"initial":{"oil_saturation":0.8,"pressure_bar":150.0,"water_saturation":0.2},"rock":{"permeability_darcy":0.1,"porosity":0.25},"schedule":{"dt_days":30.0,"n_steps":12},"schema_version":"1","seed":1}' > artifacts/tmp/case.json && julia --project=julia --startup-file=no julia/smoke/smoke_case.jl --case artifacts/tmp/case.json --out artifacts/tmp/smoke.json; echo "exit=$?"; cat artifacts/tmp/smoke.json
```
Ожидается: `exit=0`; в JSON `status: "ok"`, `cumulative_oil_m3 > 0`, `cumulative_water_injected_m3 > 0`, `0 < mean_so_final < 0.8`.

Сверить `input_sha256` с независимым вычислением:
```bash
shasum -a 256 artifacts/tmp/case.json | awk '{print $1}' && python3 -c "import json;print(json.load(open('artifacts/tmp/smoke.json'))['input_sha256'])"
```
Ожидается: две одинаковые строки. Это и есть доказательство сквозного прохода (замечание №3).

Если API-вызов не найден (например, изменилось имя `setup_vertical_well`, `pore_volume` или индексация `wd[:Producer, :orat]`), свериться с документацией установленной версии: `julia --project=julia -e 'using JutulDarcy; println(@doc setup_vertical_well)'`, поправить **только имя вызова**, не меняя физики кейса, и зафиксировать правку в `julia/README.md`.

- [ ] **Step 5: Проверить детерминизм двумя запусками**

```bash
julia --project=julia --startup-file=no julia/smoke/smoke_case.jl --case artifacts/tmp/case.json --out artifacts/tmp/smoke_2.json && python3 -c "
import json
a=json.load(open('artifacts/tmp/smoke.json')); b=json.load(open('artifacts/tmp/smoke_2.json'))
assert a['input_sha256']==b['input_sha256'], 'input_sha256 differs'
for k in ('cumulative_oil_m3','cumulative_water_injected_m3','mean_so_final'):
    rel=abs(a[k]-b[k])/max(1.0,abs(a[k])); print(k, a[k], b[k], rel); assert rel <= 1e-6, k
print('DETERMINISTIC')"
```
Ожидается: `DETERMINISTIC`.

- [ ] **Step 6: Написать `julia/README.md`**

```markdown
# Julia environment (SO-RECON)

`Project.toml` + `Manifest.toml` + `.julia-version` — зафиксированное окружение проекта.
Единственный operational forward backend — JutulDarcy (SPEC 10.1). Пакет `SOReconSimulator/`
появится в E05 и будет подключён в это окружение через `Pkg.develop`.

## Версия Julia

`.julia-version` содержит **точную project-pinned версию**: именно на ней разрешён
`Manifest.toml`. Это фиксация проекта, а не утверждение о том, какая версия является
текущим upstream stable-релизом. Несовпадение версии запущенной Julia, `Jutul` или
`JutulDarcy` с этим lock приводит к `FAIL` в `so-recon smoke`.

Установка pinned-версии:

    juliaup add "$(cat julia/.julia-version)"
    juliaup default "$(cat julia/.julia-version)"

Установка пакетов:

    julia --project=julia --startup-file=no -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

## Smoke-кейс E00

Не верификация физики — она в E05. Вход — `case.json`, сформированный Python;
скрипт возвращает `input_sha256` прочитанных байт, что делает проверку сквозной.

    julia --project=julia --startup-file=no julia/smoke/smoke_case.jl \
        --case artifacts/tmp/case.json --out artifacts/tmp/smoke.json

Обновление зависимостей выполняется только осознанно (`Pkg.update`) с коммитом нового
`Manifest.toml`, обновлением `.julia-version` и новой строкой в `reports/environment_report.md`.
```

- [ ] **Step 7: Убедиться, что артефакты не содержат личных путей, и закоммитить**

```bash
! grep -n "/Users/" julia/Manifest.toml && uv run pytest tests/test_no_absolute_paths.py -q && git add julia && git commit -m "feat(e00): add locked Julia environment and end-to-end smoke case reading case.json

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

> Если `grep` найдёт личный путь в `Manifest.toml` (такое бывает при `Pkg.develop` на локальный каталог) — удалить соответствующую `dev`-зависимость и переразрешить окружение. В E00 `dev`-зависимостей быть не должно.

---

### Task 10: Python-мост к Julia

**Files:**
- Create: `src/so_recon/simulator/julia_bridge.py`
- Test: `tests/unit/test_julia_bridge.py`, `tests/integration/test_julia_smoke.py`

**Interfaces:**
- Consumes: `StrictModel`, `JuliaConfig` (Task 4); `ProjectPaths` (Task 3); скрипт Task 9.
- Produces:
  - `class JuliaNotFoundError(RuntimeError)`, `class JuliaRunError(RuntimeError)`.
  - `find_julia(explicit=None) -> Path` — порядок: `explicit` → env `SO_RECON_JULIA` → `shutil.which("julia")` → `~/.juliaup/bin/julia`; иначе `JuliaNotFoundError`.
  - `class JuliaLauncher(Protocol)`: `launch(self, script: Path, args: list[str], out_path: Path) -> None`.
  - `class SubprocessJuliaLauncher`: `__init__(julia_exe, project, timeout_s)`; `launch` выполняет `[julia_exe, f"--project={project}", "--startup-file=no", str(script), *args, "--out", str(out_path)]`; ненулевой код → `JuliaRunError` с хвостом stderr (последние 4000 символов).
  - `class JuliaSmokeResult(StrictModel)`: `status: Literal["ok"]`, `input_sha256: str`, `case_schema_version: str`, `julia_version: str`, `jutuldarcy_version: str`, `jutul_version: str`, `nx: int`, `n_steps: int`, `cumulative_oil_m3: float`, `cumulative_water_injected_m3: float`, `mean_so_final: float`, `wall_time_s: float`.
  - `run_julia_smoke(launcher, script, case_path, out_path, *, expected_input_sha256) -> JuliaSmokeResult` — вызывает `launch(script, ["--case", str(case_path)], out_path)`, читает JSON; `status != "ok"` → `JuliaRunError(message)`; **`input_sha256 != expected_input_sha256` → `JuliaRunError`** (Julia прочитала не тот вход).
  - `default_launcher(paths, cfg, julia_exe=None) -> SubprocessJuliaLauncher`.

**Замечания ревизии 2, реализуемые здесь:** №3 (сверка `input_sha256`), №5 (`JuliaNotFoundError` поднимается так, что вызывающий может записать FAIL).

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

CASE_BYTES = b'{"schema_version":"1"}'

OK_PAYLOAD: dict[str, object] = {
    "status": "ok",
    "case_schema_version": "1",
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
        out_path.write_text(json.dumps(self.payload), encoding="utf-8")


def _case(tmp_path: Path) -> tuple[Path, str]:
    import hashlib

    p = tmp_path / "case.json"
    p.write_bytes(CASE_BYTES)
    return p, hashlib.sha256(CASE_BYTES).hexdigest()


def test_run_julia_smoke_parses_result_and_passes_the_case(tmp_path: Path) -> None:
    case_path, sha = _case(tmp_path)
    launcher = FakeLauncher({**OK_PAYLOAD, "input_sha256": sha})
    res = run_julia_smoke(
        launcher, tmp_path / "s.jl", case_path, tmp_path / "out.json", expected_input_sha256=sha
    )
    assert res.cumulative_oil_m3 == 1234.5
    assert res.input_sha256 == sha
    assert launcher.calls[0][1] == ["--case", str(case_path)]


def test_run_julia_smoke_rejects_a_mismatched_input_hash(tmp_path: Path) -> None:
    """End-to-end guard: Julia must prove it read exactly the case Python wrote."""
    case_path, sha = _case(tmp_path)
    launcher = FakeLauncher({**OK_PAYLOAD, "input_sha256": "f" * 64})
    with pytest.raises(JuliaRunError, match="input_sha256"):
        run_julia_smoke(
            launcher, tmp_path / "s.jl", case_path, tmp_path / "out.json", expected_input_sha256=sha
        )


def test_run_julia_smoke_raises_on_error_status(tmp_path: Path) -> None:
    case_path, sha = _case(tmp_path)
    launcher = FakeLauncher({"status": "error", "message": "boom", "wall_time_s": 0.1})
    with pytest.raises(JuliaRunError, match="boom"):
        run_julia_smoke(
            launcher, tmp_path / "s.jl", case_path, tmp_path / "out.json", expected_input_sha256=sha
        )


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
    fake.write_text('#!/bin/sh\necho "$@" > "$(dirname "$0")/argv.txt"\necho fatal >&2\nexit 3\n')
    fake.chmod(0o755)
    launcher = SubprocessJuliaLauncher(fake, tmp_path / "proj", timeout_s=30)
    with pytest.raises(JuliaRunError, match="fatal"):
        launcher.launch(tmp_path / "s.jl", ["--case", "c.json"], tmp_path / "o.json")
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
from so_recon.synthetic.fixture import build_smoke_case, build_smoke_fixture, write_smoke_case
from so_recon.registry.hashing import sha256_bytes

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.julia
def test_real_julia_smoke_case_runs_end_to_end(tmp_path: Path) -> None:
    cfg = load_project_config(ROOT / "configs" / "project.yml")
    paths = ProjectPaths.from_config(ROOT, cfg.paths)
    if not (paths.julia / "Manifest.toml").is_file():
        pytest.skip("julia/Manifest.toml missing; run make setup-julia")
    try:
        launcher = default_launcher(paths, cfg.julia)
    except JuliaNotFoundError:
        pytest.skip("julia executable not found")

    fixture = build_smoke_fixture(cfg.smoke)
    case = build_smoke_case(cfg.smoke, fixture)
    case_path = tmp_path / "case.json"
    payload = write_smoke_case(case, case_path)

    res = run_julia_smoke(
        launcher,
        paths.root / cfg.julia.smoke_script,
        case_path,
        tmp_path / "smoke.json",
        expected_input_sha256=sha256_bytes(payload),
    )
    assert res.status == "ok"
    assert res.input_sha256 == sha256_bytes(payload)
    assert res.case_schema_version == case["schema_version"]
    assert res.nx == cfg.smoke.nx and res.n_steps == cfg.smoke.n_steps
    assert res.cumulative_oil_m3 > 0
    assert res.cumulative_water_injected_m3 > 0
    assert 0.0 < res.mean_so_final < 0.8
    assert res.jutuldarcy_version.startswith("0.3.")
```

- [ ] **Step 2: Убедиться, что тесты падают**

```bash
uv run pytest tests/unit/test_julia_bridge.py -q
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
    """No usable julia executable was found."""


class JuliaRunError(RuntimeError):
    """Julia failed, timed out, or returned a result that does not match its input."""


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
    input_sha256: str
    case_schema_version: str
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
    launcher: JuliaLauncher,
    script: Path,
    case_path: Path,
    out_path: Path,
    *,
    expected_input_sha256: str,
) -> JuliaSmokeResult:
    launcher.launch(script, ["--case", str(case_path)], out_path)
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    if payload.get("status") != "ok":
        raise JuliaRunError(f"julia smoke failed: {payload.get('message', payload)}")
    result = JuliaSmokeResult.model_validate(payload)
    if result.input_sha256 != expected_input_sha256:
        raise JuliaRunError(
            "julia read a different input: "
            f"input_sha256 {result.input_sha256} != expected {expected_input_sha256}"
        )
    return result


def default_launcher(
    paths: ProjectPaths, cfg: JuliaConfig, julia_exe: str | None = None
) -> SubprocessJuliaLauncher:
    return SubprocessJuliaLauncher(
        julia_exe=find_julia(julia_exe),
        project=paths.resolve(cfg.project),
        timeout_s=cfg.timeout_s,
    )
```

- [ ] **Step 4: Прогнать unit-тесты, затем integration-тест с реальной Julia**

```bash
uv run pytest tests/unit/test_julia_bridge.py -q && uv run ruff check . && uv run ruff format --check . && uv run mypy
```
Ожидается: `6 passed`.

```bash
uv run pytest tests/integration/test_julia_smoke.py -q -m julia
```
Ожидается: `1 passed` (не `skipped`; если skipped — Julia не найдена, вернуться к Task 9 Step 1).

- [ ] **Step 5: Commit**

```bash
git add src/so_recon/simulator/julia_bridge.py tests/unit/test_julia_bridge.py tests/integration/test_julia_smoke.py && git commit -m "feat(e00): add python-julia bridge with case input and input_sha256 round-trip check

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 11: Environment report и версии из lock

**Files:**
- Create: `src/so_recon/environment/report.py`
- Test: `tests/unit/test_environment_report.py`

**Interfaces:**
- Consumes: `StrictModel` (Task 4); `ProjectPaths` (Task 3); `sha256_file` (Task 2); `environment_lock_hash` (Task 6); `git_commit`, `git_is_dirty` (Task 6); `find_julia`, `JuliaNotFoundError` (Task 10); `write_artifact`, `ArtifactRef` (Task 5); `write_bytes_atomic`, `write_json_atomic` (Task 2).
- Produces:
  - `ENVIRONMENT_SCHEMA_VERSION = "2"`.
  - `parse_julia_manifest(path) -> tuple[str | None, dict[str, str]]` — `(julia_version, {package: version})` из `Manifest.toml` (формат v2: `data["julia_version"]`, `data["deps"][name][0]["version"]`); отслеживаются `JutulDarcy`, `Jutul`, `JSON`.
  - `class LockedVersions(StrictModel)`: `julia_pinned: str | None` (из `julia/.julia-version`), `julia_manifest: str | None`, `packages: dict[str, str]`.
  - `read_locked_versions(paths) -> LockedVersions`.
  - `check_locked_versions(locked, *, julia_version, jutul_version, jutuldarcy_version) -> list[str]` — **список расхождений**; непустой список означает FAIL (замечание №9). Отсутствующий lock — тоже расхождение (`"... lock missing"`), потому что без lock воспроизводимость не доказана.
  - `class EnvironmentReport(StrictModel)` — **детерминированный**, без timestamps и git-состояния: `schema_version`, `os`, `arch`, `python_version`, `uv_version: str | None`, `uv_lock_sha256: str | None`, `julia_pinned_version: str | None`, `julia_executable_version: str | None`, `julia_manifest_version: str | None`, `julia_manifest_sha256: str | None`, `julia_packages: dict[str, str]`, `environment_lock_hash: str`.
  - `class EnvironmentStamp(StrictModel)` — run-scoped: `schema_version`, `report_sha256`, `run_id`, `created_at`, `git_commit: str | None`, `git_dirty: bool | None`.
  - `VersionProbe = Callable[[list[str]], str | None]`, `subprocess_probe(cmd) -> str | None`.
  - `collect_environment(paths, *, probe=subprocess_probe) -> EnvironmentReport` — **без параметра `now`**: отчёт не содержит времени.
  - `render_markdown(report) -> str`, `report_json_bytes(report) -> bytes`, `report_markdown_bytes(report) -> bytes`.
  - `write_environment_report(report, paths, *, run_dir, md_path, json_path, producer_run_id, now) -> tuple[ArtifactRef, ArtifactRef]` — immutable артефакты в `run_dir`, публикация байт-в-байт в `reports/`.
  - `write_environment_stamp(stamp, path) -> None`.

**Замечания ревизии 2, реализуемые здесь:** №9 (сверка версий с lock), №10 (детерминированный отчёт, timestamps в run-scoped штампе), №12, №7 (`julia_pinned_version`).

- [ ] **Step 1: Написать падающие тесты**

`tests/unit/test_environment_report.py`:
```python
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from so_recon.environment.report import (
    ENVIRONMENT_SCHEMA_VERSION,
    check_locked_versions,
    collect_environment,
    parse_julia_manifest,
    read_locked_versions,
    render_markdown,
    report_json_bytes,
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


def _probe(cmd: list[str]) -> str | None:
    if cmd[0] == "uv":
        return "uv 0.12.7 (Homebrew)\n"
    if cmd[0].endswith("julia"):
        return "julia version 1.12.7\n"
    return None


def _with_locks(tmp_path: Path) -> ProjectPaths:
    (tmp_path / "uv.lock").write_text("lock")
    (tmp_path / "julia").mkdir(exist_ok=True)
    (tmp_path / "julia" / "Manifest.toml").write_text(MANIFEST)
    (tmp_path / "julia" / ".julia-version").write_text("1.12.7\n")
    return ProjectPaths.default(tmp_path)


def test_parse_julia_manifest(tmp_path: Path) -> None:
    p = tmp_path / "Manifest.toml"
    p.write_text(MANIFEST)
    julia_version, pkgs = parse_julia_manifest(p)
    assert julia_version == "1.12.7"
    assert pkgs == {"JSON": "1.1.2", "Jutul": "0.4.40", "JutulDarcy": "0.3.11"}


def test_locked_versions_match_is_empty(tmp_path: Path) -> None:
    locked = read_locked_versions(_with_locks(tmp_path))
    assert locked.julia_pinned == "1.12.7"
    assert check_locked_versions(
        locked, julia_version="1.12.7", jutul_version="0.4.40", jutuldarcy_version="0.3.11"
    ) == []


@pytest.mark.parametrize(
    ("julia", "jutul", "darcy", "needle"),
    [
        ("1.12.8", "0.4.40", "0.3.11", "julia"),
        ("1.12.7", "0.4.41", "0.3.11", "Jutul"),
        ("1.12.7", "0.4.40", "0.3.12", "JutulDarcy"),
    ],
)
def test_version_drift_against_lock_is_a_mismatch(
    tmp_path: Path, julia: str, jutul: str, darcy: str, needle: str
) -> None:
    """Amendment 9: drift from the lock is a failure, not an informational note."""
    locked = read_locked_versions(_with_locks(tmp_path))
    mismatches = check_locked_versions(
        locked, julia_version=julia, jutul_version=jutul, jutuldarcy_version=darcy
    )
    assert mismatches
    assert any(needle in m for m in mismatches)


def test_missing_lock_is_a_mismatch(tmp_path: Path) -> None:
    locked = read_locked_versions(ProjectPaths.default(tmp_path))
    mismatches = check_locked_versions(
        locked, julia_version="1.12.7", jutul_version="0.4.40", jutuldarcy_version="0.3.11"
    )
    assert any("missing" in m for m in mismatches)


def test_collect_environment_with_and_without_lock_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SO_RECON_JULIA", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    paths = ProjectPaths.default(tmp_path)
    rep = collect_environment(paths, probe=_probe)
    assert rep.uv_lock_sha256 is None
    assert rep.julia_manifest_version is None
    assert rep.julia_pinned_version is None
    assert rep.julia_executable_version is None
    assert rep.uv_version == "0.12.7"

    rep2 = collect_environment(_with_locks(tmp_path), probe=_probe)
    assert rep2.uv_lock_sha256 is not None
    assert rep2.julia_manifest_version == "1.12.7"
    assert rep2.julia_pinned_version == "1.12.7"
    assert rep2.julia_packages["JutulDarcy"] == "0.3.11"
    assert rep2.environment_lock_hash != rep.environment_lock_hash


def test_report_is_deterministic_and_carries_no_time_or_git_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invariant I5: repeated gate runs must not produce a git diff."""
    monkeypatch.delenv("SO_RECON_JULIA", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    paths = _with_locks(tmp_path)
    first = report_json_bytes(collect_environment(paths, probe=_probe))
    second = report_json_bytes(collect_environment(paths, probe=_probe))
    assert first == second
    payload = json.loads(first)
    assert payload["schema_version"] == ENVIRONMENT_SCHEMA_VERSION
    for forbidden in ("created_at", "git_commit", "git_dirty", "run_id"):
        assert forbidden not in payload


def test_render_and_write_publishes_identical_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SO_RECON_JULIA", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    paths = _with_locks(tmp_path)
    rep = collect_environment(paths, probe=_probe)
    md = render_markdown(rep)
    assert "# Environment report" in md
    assert "environment_lock_hash" in md
    assert "created_at" not in md
    run_dir = paths.runs / "run-1"
    run_dir.mkdir(parents=True)
    md_path = paths.reports / "environment_report.md"
    json_path = paths.manifests / "environment.json"
    md_ref, json_ref = write_environment_report(
        rep, paths, run_dir=run_dir, md_path=md_path, json_path=json_path,
        producer_run_id="run-1", now=datetime(2026, 9, 13, tzinfo=UTC),
    )
    assert md_path.read_text(encoding="utf-8") == md
    assert (run_dir / "environment_report.md").read_bytes() == md_path.read_bytes()
    assert (run_dir / "environment.json").read_bytes() == json_path.read_bytes()
    assert json.loads(json_path.read_text(encoding="utf-8"))["python_version"] == rep.python_version
    assert md_ref.producer_run_id == "run-1" and json_ref.producer_run_id == "run-1"
```

- [ ] **Step 2: Убедиться, что тесты падают**

```bash
uv run pytest tests/unit/test_environment_report.py -q
```
Ожидается: `ModuleNotFoundError: so_recon.environment.report`.

- [ ] **Step 3: Реализовать `report.py`**

```python
"""Environment report: versions and lock hashes (SPEC 19.12; STAGES E00 output).

The report is deterministic: it carries no timestamps, run ids or git state, so a repeated
gate run leaves no git diff (invariant I5). Those facts live in EnvironmentStamp, which is
written into the run directory and never committed.
"""

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
from so_recon.registry.artifact import ArtifactRef, write_artifact
from so_recon.registry.atomic import write_bytes_atomic, write_json_atomic
from so_recon.registry.gitinfo import git_commit, git_is_dirty
from so_recon.registry.hashing import sha256_file
from so_recon.registry.run import environment_lock_hash
from so_recon.simulator.julia_bridge import JuliaNotFoundError, find_julia

ENVIRONMENT_SCHEMA_VERSION = "2"
TRACKED_JULIA_PACKAGES = ("JutulDarcy", "Jutul", "JSON")
JULIA_VERSION_FILE = ".julia-version"

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


class LockedVersions(StrictModel):
    """Versions the project is pinned to. Runtime drift from these is a failure."""

    julia_pinned: str | None
    julia_manifest: str | None
    packages: dict[str, str]


def read_locked_versions(paths: ProjectPaths) -> LockedVersions:
    pin_file = paths.julia / JULIA_VERSION_FILE
    pinned = pin_file.read_text(encoding="utf-8").strip() if pin_file.is_file() else None
    manifest = paths.julia / "Manifest.toml"
    manifest_version: str | None = None
    packages: dict[str, str] = {}
    if manifest.is_file():
        manifest_version, packages = parse_julia_manifest(manifest)
    return LockedVersions(
        julia_pinned=pinned or None, julia_manifest=manifest_version, packages=packages
    )


def check_locked_versions(
    locked: LockedVersions, *, julia_version: str, jutul_version: str, jutuldarcy_version: str
) -> list[str]:
    """Return the drift between the running stack and the lock. Non-empty means FAIL."""
    mismatches: list[str] = []
    expected_julia = locked.julia_pinned or locked.julia_manifest
    if expected_julia is None:
        mismatches.append("julia version lock missing (julia/.julia-version and Manifest.toml)")
    elif expected_julia != julia_version:
        mismatches.append(f"julia version: running {julia_version}, locked {expected_julia}")
    if locked.julia_pinned and locked.julia_manifest and locked.julia_pinned != locked.julia_manifest:
        mismatches.append(
            f"julia lock is inconsistent: .julia-version {locked.julia_pinned}, "
            f"Manifest.toml {locked.julia_manifest}"
        )
    for name, running in (("Jutul", jutul_version), ("JutulDarcy", jutuldarcy_version)):
        expected = locked.packages.get(name)
        if expected is None:
            mismatches.append(f"{name} version lock missing in julia/Manifest.toml")
        elif expected != running:
            mismatches.append(f"{name} version: running {running}, locked {expected}")
    return mismatches


class EnvironmentReport(StrictModel):
    schema_version: str = ENVIRONMENT_SCHEMA_VERSION
    os: str
    arch: str
    python_version: str
    uv_version: str | None
    uv_lock_sha256: str | None
    julia_pinned_version: str | None
    julia_executable_version: str | None
    julia_manifest_version: str | None
    julia_manifest_sha256: str | None
    julia_packages: dict[str, str]
    environment_lock_hash: str


class EnvironmentStamp(StrictModel):
    """Run-scoped facts kept out of the committed report (invariant I5)."""

    schema_version: str
    report_sha256: str
    run_id: str
    created_at: str
    git_commit: str | None
    git_dirty: bool | None


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
    out = probe([str(exe), "--version"])  # "julia version 1.12.7"
    if not out:
        return None
    parts = out.split()
    return parts[2] if len(parts) >= 3 else None


def collect_environment(
    paths: ProjectPaths, *, probe: VersionProbe = subprocess_probe
) -> EnvironmentReport:
    uv_lock = paths.root / "uv.lock"
    manifest = paths.julia / "Manifest.toml"
    locked = read_locked_versions(paths)
    return EnvironmentReport(
        os=f"{platform.system()} {platform.release()}",
        arch=platform.machine(),
        python_version=platform.python_version(),
        uv_version=_second_token(probe(["uv", "--version"])),
        uv_lock_sha256=sha256_file(uv_lock) if uv_lock.is_file() else None,
        julia_pinned_version=locked.julia_pinned,
        julia_executable_version=_julia_exe_version(probe),
        julia_manifest_version=locked.julia_manifest,
        julia_manifest_sha256=sha256_file(manifest) if manifest.is_file() else None,
        julia_packages=locked.packages,
        environment_lock_hash=environment_lock_hash(paths),
    )


def render_markdown(report: EnvironmentReport) -> str:
    rows = [
        ("schema_version", report.schema_version),
        ("os", report.os),
        ("arch", report.arch),
        ("python_version", report.python_version),
        ("uv_version", report.uv_version),
        ("uv_lock_sha256", report.uv_lock_sha256),
        ("julia_pinned_version", report.julia_pinned_version),
        ("julia_executable_version", report.julia_executable_version),
        ("julia_manifest_version", report.julia_manifest_version),
        ("julia_manifest_sha256", report.julia_manifest_sha256),
        ("environment_lock_hash", report.environment_lock_hash),
    ]
    lines = [
        "# Environment report",
        "",
        "Детерминированный отчёт: без timestamps и git-состояния, чтобы повторный gate",
        "не создавал diff. Время запуска и git-состояние — в `artifacts/runs/<run_id>/environment_stamp.json`.",
        "",
        "| key | value |",
        "|---|---|",
    ]
    lines += [f"| `{k}` | `{v}` |" for k, v in rows]
    lines += ["", "## Julia packages (locked)", "", "| package | version |", "|---|---|"]
    lines += [f"| `{k}` | `{v}` |" for k, v in sorted(report.julia_packages.items())]
    return "\n".join(lines) + "\n"


def report_json_bytes(report: EnvironmentReport) -> bytes:
    payload = report.model_dump(mode="json")
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def report_markdown_bytes(report: EnvironmentReport) -> bytes:
    return render_markdown(report).encode("utf-8")


def write_environment_report(
    report: EnvironmentReport,
    paths: ProjectPaths,
    *,
    run_dir: Path,
    md_path: Path,
    json_path: Path,
    producer_run_id: str,
    now: datetime,
) -> tuple[ArtifactRef, ArtifactRef]:
    """Immutable run artifacts first, then byte-identical published copies."""
    md_payload = report_markdown_bytes(report)
    json_payload = report_json_bytes(report)
    md_ref = write_artifact(
        run_dir / "environment_report.md", md_payload, paths,
        schema_version=ENVIRONMENT_SCHEMA_VERSION, producer_run_id=producer_run_id,
        media_type="text/markdown", now=now,
    )
    json_ref = write_artifact(
        run_dir / "environment.json", json_payload, paths,
        schema_version=ENVIRONMENT_SCHEMA_VERSION, producer_run_id=producer_run_id,
        media_type="application/json", now=now,
    )
    write_bytes_atomic(md_path, md_payload)
    write_bytes_atomic(json_path, json_payload)
    return md_ref, json_ref


def write_environment_stamp(stamp: EnvironmentStamp, path: Path) -> None:
    write_json_atomic(path, stamp.model_dump(mode="json"))


def build_environment_stamp(
    paths: ProjectPaths, *, report_sha256: str, run_id: str, now: datetime
) -> EnvironmentStamp:
    return EnvironmentStamp(
        schema_version=ENVIRONMENT_SCHEMA_VERSION,
        report_sha256=report_sha256,
        run_id=run_id,
        created_at=now.isoformat(),
        git_commit=git_commit(paths.root),
        git_dirty=git_is_dirty(paths.root),
    )
```

- [ ] **Step 4: Прогнать тесты, lint, format, mypy**

```bash
uv run pytest tests/unit/test_environment_report.py -q && uv run ruff check . && uv run ruff format --check . && uv run mypy
```
Ожидается: `9 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/so_recon/environment/report.py tests/unit/test_environment_report.py && git commit -m "feat(e00): add deterministic environment report and lock-version drift check

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---
### Task 12: Runner, сценарий smoke и CLI `so-recon`

**Files:**
- Create: `src/so_recon/runner.py`, `src/so_recon/smoke.py`, `src/so_recon/cli.py`, `tests/conftest.py`
- Test: `tests/unit/test_runner.py`, `tests/unit/test_smoke.py`, `tests/unit/test_cli.py`

**Interfaces:**
- Consumes: всё из Tasks 2–11.
- Produces (`so_recon.runner`):
  - `CommandBody = Callable[[RunContext, logging.Logger], tuple[RunStatus, list[str]]]`
  - `execute_run(*, command, argv, cfg, paths, body, schema_versions=None, parent_run_ids=()) -> RunContext` — открывает `RunContext`, настраивает логирование в `run_dir/run.log`, выполняет `body`. **Любое** исключение из `body` перехватывается: полный traceback уходит в `run.log`, в `notes` попадает `exception: <Type>: <msg>`, запись закрывается со `status="FAIL"`. Функция никогда не пробрасывает исключение наружу — гарантия инварианта I6. Возвращает всегда закрытый (`PASS`/`FAIL`) контекст.
- Produces (`so_recon.smoke`):
  - `EXPECTED_FILENAME = "smoke_expected.json"`, `EXPECTED_SCHEMA_VERSION = "2"`.
  - `class SmokeExpectation(StrictModel)` — **без timestamps** (инвариант I5): `schema_version`, `fixture_content_hash`, `case_sha256`, `nx`, `n_steps`, `cumulative_oil_m3`, `cumulative_water_injected_m3`, `mean_so_final`, `julia_version`, `jutul_version`, `jutuldarcy_version`.
  - `compare_with_expected(expected, *, fixture_hash, case_sha256, result, rel_tol) -> list[str]` — хеши и целые сравниваются точно; float — `abs(a-b) <= rel_tol*max(1,|b|)`; **версии Julia/Jutul/JutulDarcy сравниваются точно и расхождение является mismatch** (замечание №9), а не заметкой.
  - `smoke_body(cfg, paths, launcher_factory, freeze_expected) -> CommandBody` — сценарий: fixture → `case.json` → Julia → сверка с lock → сверка с ожиданиями. `launcher_factory` вызывается **внутри** тела, поэтому отсутствие Julia даёт FAIL-запись (замечание №5).
  - `run_smoke(*, cfg, paths, argv, launcher_factory, freeze_expected=False) -> RunContext` — `execute_run` со `smoke_body`.
- Produces (`so_recon.cli`):
  - `main(argv=None, *, launcher_factory=None) -> int`.
  - Общие флаги: `--config PATH`, `--root PATH`.
  - `so-recon manifest`, `so-recon env-report`, `so-recon smoke [--freeze-expected] [--julia PATH]`. Код возврата 0 при `PASS`, 1 при `FAIL`, 2 при неверных аргументах (argparse).

**Замечания ревизии 2, реализуемые здесь:** №3 (сквозной сценарий), №4 (`outputs` — `ArtifactRef`), №5 (FAIL при любом исключении, включая отсутствие Julia и ошибку конфигурации), №9 (несовпадение с lock → FAIL), №10 (`smoke_expected.json` без timestamps), №12.

> **Единственный случай без run-записи** — невозможность определить repository root: писать некуда. Он обрабатывается печатью в stderr и кодом 1, и зафиксирован в отчёте E00 как известное ограничение.

- [ ] **Step 1: Написать `tests/conftest.py`**

```python
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

PROJECT_YAML = """
spec_version: "3.0"
config_version: "test.1"
paths:
  raw: data/sources
sources:
  files:
    - {name: a, path: data/sources/a.csv, encoding: utf-8, delimiter: ",", decimal: "."}
    - {name: b, path: data/sources/b.csv, encoding: cp1251, delimiter: ";", decimal: ","}
smoke: {seed: 11, n_wells: 2, n_months: 2, nx: 5, n_steps: 2, rel_tol: 1.0e-6}
julia: {}
"""

JULIA_MANIFEST = """
julia_version = "1.12.7"
manifest_format = "2.0"

[[deps.JSON]]
version = "1.1.2"

[[deps.Jutul]]
version = "0.4.40"

[[deps.JutulDarcy]]
version = "0.3.11"
"""


@pytest.fixture
def tmp_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A minimal repository layout: marker files, config, two fake immutable sources."""
    monkeypatch.delenv("SO_RECON_ROOT", raising=False)
    monkeypatch.delenv("SO_RECON_JULIA", raising=False)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    (tmp_path / "src" / "so_recon").mkdir(parents=True)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "project.yml").write_text(PROJECT_YAML, encoding="utf-8")
    (tmp_path / "data" / "sources").mkdir(parents=True)
    (tmp_path / "data" / "sources" / "a.csv").write_bytes(b"x,y\n1,2\n")
    (tmp_path / "data" / "sources" / "b.csv").write_bytes(b"x;y\r\n1;2\r\n")
    (tmp_path / "julia" / "smoke").mkdir(parents=True)
    (tmp_path / "julia" / "smoke" / "smoke_case.jl").write_text("# fake\n")
    (tmp_path / "julia" / "Manifest.toml").write_text(JULIA_MANIFEST, encoding="utf-8")
    (tmp_path / "julia" / ".julia-version").write_text("1.12.7\n", encoding="utf-8")
    (tmp_path / "uv.lock").write_text("lock\n")
    return tmp_path


@pytest.fixture
def fake_launcher_factory() -> Callable[..., Any]:
    """Build a launcher that echoes a fixed payload, optionally hashing the real case file."""

    class FakeLauncher:
        def __init__(self, payload: dict[str, Any], *, echo_input_sha: bool = True) -> None:
            self.payload = payload
            self.echo_input_sha = echo_input_sha
            self.calls = 0
            self.cases: list[Path] = []

        def launch(self, script: Path, args: list[str], out_path: Path) -> None:
            self.calls += 1
            case_path = Path(args[args.index("--case") + 1])
            self.cases.append(case_path)
            payload = dict(self.payload)
            if self.echo_input_sha and "input_sha256" not in payload:
                import hashlib

                payload["input_sha256"] = hashlib.sha256(case_path.read_bytes()).hexdigest()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(payload), encoding="utf-8")

    return FakeLauncher
```

- [ ] **Step 2: Написать падающие тесты `tests/unit/test_runner.py`**

```python
import json
import logging
from pathlib import Path

from so_recon.config.load import load_project_config
from so_recon.paths import ProjectPaths
from so_recon.registry.run import RunContext, RunStatus
from so_recon.runner import execute_run


def _setup(tmp_project: Path) -> tuple[object, ProjectPaths]:
    cfg = load_project_config(tmp_project / "configs" / "project.yml")
    return cfg, ProjectPaths.from_config(tmp_project, cfg.paths)


def test_execute_run_records_pass(tmp_project: Path) -> None:
    cfg, paths = _setup(tmp_project)

    def body(ctx: RunContext, log: logging.Logger) -> tuple[RunStatus, list[str]]:
        log.info("working")
        return "PASS", ["done"]

    ctx = execute_run(command="x", argv=["so-recon", "x"], cfg=cfg, paths=paths, body=body)
    record = json.loads((ctx.run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "PASS"
    assert record["notes"] == ["done"]
    assert (ctx.run_dir / "run.log").is_file()


def test_execute_run_converts_any_exception_into_a_fail_record(tmp_project: Path) -> None:
    """Invariant I6: an unexpected exception must still leave a FAIL run record."""
    cfg, paths = _setup(tmp_project)

    def body(ctx: RunContext, log: logging.Logger) -> tuple[RunStatus, list[str]]:
        raise RuntimeError("unexpected boom")

    ctx = execute_run(command="x", argv=["so-recon", "x"], cfg=cfg, paths=paths, body=body)
    record = json.loads((ctx.run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "FAIL"
    assert any("unexpected boom" in n for n in record["notes"])
    assert "RuntimeError" in (ctx.run_dir / "run.log").read_text(encoding="utf-8")


def test_execute_run_works_without_config(tmp_project: Path) -> None:
    paths = ProjectPaths.default(tmp_project)

    def body(ctx: RunContext, log: logging.Logger) -> tuple[RunStatus, list[str]]:
        return "FAIL", ["config error"]

    ctx = execute_run(command="x", argv=[], cfg=None, paths=paths, body=body)
    assert json.loads((ctx.run_dir / "run.json").read_text(encoding="utf-8"))["status"] == "FAIL"
```

- [ ] **Step 3: Написать падающие тесты `tests/unit/test_smoke.py`**

```python
import json
from pathlib import Path
from typing import Any

from so_recon.config.load import load_project_config
from so_recon.paths import ProjectPaths
from so_recon.simulator.julia_bridge import JuliaNotFoundError, JuliaSmokeResult
from so_recon.smoke import EXPECTED_FILENAME, SmokeExpectation, compare_with_expected, run_smoke

OK: dict[str, Any] = {
    "status": "ok", "case_schema_version": "1", "julia_version": "1.12.7",
    "jutuldarcy_version": "0.3.11", "jutul_version": "0.4.40", "nx": 5, "n_steps": 2,
    "cumulative_oil_m3": 100.0, "cumulative_water_injected_m3": 150.0,
    "mean_so_final": 0.6, "wall_time_s": 1.0,
}


def _setup(tmp_project: Path) -> tuple[Any, ProjectPaths]:
    cfg = load_project_config(tmp_project / "configs" / "project.yml")
    return cfg, ProjectPaths.from_config(tmp_project, cfg.paths)


def _expectation(**over: Any) -> SmokeExpectation:
    base: dict[str, Any] = {
        "schema_version": "2", "fixture_content_hash": "h", "case_sha256": "c",
        "nx": 5, "n_steps": 2, "cumulative_oil_m3": 100.0,
        "cumulative_water_injected_m3": 150.0, "mean_so_final": 0.6,
        "julia_version": "1.12.7", "jutul_version": "0.4.40", "jutuldarcy_version": "0.3.11",
    }
    base.update(over)
    return SmokeExpectation(**base)


def _result(**over: Any) -> JuliaSmokeResult:
    return JuliaSmokeResult.model_validate({**OK, "input_sha256": "i", **over})


def test_compare_with_expected_tolerances() -> None:
    exp = _expectation()
    assert compare_with_expected(
        exp, fixture_hash="h", case_sha256="c",
        result=_result(cumulative_oil_m3=100.0 + 5e-5), rel_tol=1e-6,
    ) == []
    mism = compare_with_expected(
        exp, fixture_hash="h", case_sha256="c",
        result=_result(cumulative_oil_m3=101.0), rel_tol=1e-6,
    )
    assert any("cumulative_oil_m3" in m for m in mism)


def test_compare_detects_input_drift() -> None:
    exp = _expectation()
    assert any(
        "fixture_content_hash" in m
        for m in compare_with_expected(exp, fixture_hash="x", case_sha256="c", result=_result(), rel_tol=1e-6)
    )
    assert any(
        "case_sha256" in m
        for m in compare_with_expected(exp, fixture_hash="h", case_sha256="x", result=_result(), rel_tol=1e-6)
    )


def test_version_drift_is_a_mismatch_not_a_note() -> None:
    """Amendment 9: a JutulDarcy bump must fail the smoke, not be logged as a note."""
    exp = _expectation()
    mism = compare_with_expected(
        exp, fixture_hash="h", case_sha256="c",
        result=_result(jutuldarcy_version="0.3.12"), rel_tol=1e-6,
    )
    assert any("jutuldarcy_version" in m for m in mism)


def test_run_smoke_fails_without_expected_then_freezes_then_passes(
    tmp_project: Path, fake_launcher_factory: Any
) -> None:
    cfg, paths = _setup(tmp_project)
    launcher = fake_launcher_factory(OK)

    ctx = run_smoke(cfg=cfg, paths=paths, argv=["smoke"], launcher_factory=lambda: launcher)
    assert ctx.record.status == "FAIL"
    assert any("freeze-expected" in n for n in ctx.record.notes)

    ctx2 = run_smoke(cfg=cfg, paths=paths, argv=["smoke"], launcher_factory=lambda: launcher,
                     freeze_expected=True)
    assert ctx2.record.status == "PASS"
    expected_path = paths.configs / EXPECTED_FILENAME
    frozen = json.loads(expected_path.read_text(encoding="utf-8"))
    assert frozen["nx"] == 5 and frozen["cumulative_oil_m3"] == 100.0
    assert "frozen_at" not in frozen, "the committed expectation must carry no timestamp"

    ctx3 = run_smoke(cfg=cfg, paths=paths, argv=["smoke"], launcher_factory=lambda: launcher)
    assert ctx3.record.status == "PASS"
    assert ctx3.record.julia_version == "1.12.7"
    assert ctx3.record.jutuldarcy_version == "0.3.11"
    assert (ctx3.run_dir / "fixture" / "wells.parquet").is_file()
    assert (ctx3.run_dir / "case.json").is_file()
    assert (ctx3.run_dir / "julia_smoke.json").is_file()
    assert (ctx3.run_dir / "run.log").is_file()
    assert ctx3.record.outputs["case"].path.endswith("case.json")
    assert ctx3.record.outputs["case"].sha256 == launcher_case_sha(ctx3.run_dir)


def launcher_case_sha(run_dir: Path) -> str:
    import hashlib

    return hashlib.sha256((run_dir / "case.json").read_bytes()).hexdigest()


def test_run_smoke_detects_numeric_drift(tmp_project: Path, fake_launcher_factory: Any) -> None:
    cfg, paths = _setup(tmp_project)
    run_smoke(cfg=cfg, paths=paths, argv=["smoke"],
              launcher_factory=lambda: fake_launcher_factory(OK), freeze_expected=True)
    drifted = fake_launcher_factory({**OK, "mean_so_final": 0.61})
    ctx = run_smoke(cfg=cfg, paths=paths, argv=["smoke"], launcher_factory=lambda: drifted)
    assert ctx.record.status == "FAIL"
    assert any("mean_so_final" in n for n in ctx.record.notes)


def test_run_smoke_fails_when_versions_drift_from_the_lock(
    tmp_project: Path, fake_launcher_factory: Any
) -> None:
    """Amendment 9: running JutulDarcy differs from julia/Manifest.toml -> FAIL."""
    cfg, paths = _setup(tmp_project)
    run_smoke(cfg=cfg, paths=paths, argv=["smoke"],
              launcher_factory=lambda: fake_launcher_factory(OK), freeze_expected=True)
    drifted = fake_launcher_factory({**OK, "jutuldarcy_version": "0.3.99"})
    ctx = run_smoke(cfg=cfg, paths=paths, argv=["smoke"], launcher_factory=lambda: drifted)
    assert ctx.record.status == "FAIL"
    assert any("JutulDarcy" in n or "jutuldarcy" in n for n in ctx.record.notes)


def test_run_smoke_records_fail_when_julia_is_missing(tmp_project: Path) -> None:
    """Amendment 5: a missing Julia executable must still produce a FAIL run record."""
    cfg, paths = _setup(tmp_project)

    def factory() -> Any:
        raise JuliaNotFoundError("julia executable not found")

    ctx = run_smoke(cfg=cfg, paths=paths, argv=["smoke"], launcher_factory=factory)
    assert ctx.record.status == "FAIL"
    record = json.loads((ctx.run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "FAIL"
    assert any("julia executable not found" in n for n in record["notes"])


def test_run_smoke_records_fail_when_julia_reports_an_error(
    tmp_project: Path, fake_launcher_factory: Any
) -> None:
    cfg, paths = _setup(tmp_project)
    broken = fake_launcher_factory(
        {"status": "error", "message": "solver blew up", "wall_time_s": 0.0}, echo_input_sha=False
    )
    ctx = run_smoke(cfg=cfg, paths=paths, argv=["smoke"], launcher_factory=lambda: broken)
    assert ctx.record.status == "FAIL"
    assert any("solver blew up" in n for n in ctx.record.notes)


def test_run_smoke_detects_a_tampered_case_file(tmp_project: Path, fake_launcher_factory: Any) -> None:
    """End-to-end guard: Julia must return the hash of the bytes it actually read."""
    cfg, paths = _setup(tmp_project)
    liar = fake_launcher_factory({**OK, "input_sha256": "f" * 64}, echo_input_sha=False)
    ctx = run_smoke(cfg=cfg, paths=paths, argv=["smoke"], launcher_factory=lambda: liar)
    assert ctx.record.status == "FAIL"
    assert any("input_sha256" in n for n in ctx.record.notes)
```

- [ ] **Step 4: Написать падающие тесты `tests/unit/test_cli.py`**

```python
import json
from pathlib import Path
from typing import Any

import pytest

from so_recon.cli import main

OK: dict[str, Any] = {
    "status": "ok", "case_schema_version": "1", "julia_version": "1.12.7",
    "jutuldarcy_version": "0.3.11", "jutul_version": "0.4.40", "nx": 5, "n_steps": 2,
    "cumulative_oil_m3": 100.0, "cumulative_water_injected_m3": 150.0,
    "mean_so_final": 0.6, "wall_time_s": 1.0,
}


def _only_run(tmp_project: Path) -> dict[str, Any]:
    runs = sorted((tmp_project / "artifacts" / "runs").iterdir())
    assert len(runs) == 1
    return json.loads((runs[0] / "run.json").read_text(encoding="utf-8"))


def test_manifest_command_writes_manifest_and_run_record(tmp_project: Path) -> None:
    assert main(["--root", str(tmp_project), "manifest"]) == 0
    manifest = json.loads(
        (tmp_project / "reports" / "manifests" / "source_manifest.json").read_text(encoding="utf-8")
    )
    assert [s["name"] for s in manifest["sources"]] == ["a", "b"]
    assert manifest["sources"][0]["physical_line_count"] == 2
    assert manifest["sources"][0]["data_rows"] == 1
    record = _only_run(tmp_project)
    assert record["command"] == "manifest"
    assert record["status"] == "PASS"
    assert set(record["raw_input_hashes"]) == {"a", "b"}
    assert record["outputs"]["source_manifest"]["schema_version"] == "2"


def test_manifest_is_byte_identical_on_a_second_run(tmp_project: Path) -> None:
    """Amendment 10: repeated runs must not change the committed manifest."""
    published = tmp_project / "reports" / "manifests" / "source_manifest.json"
    assert main(["--root", str(tmp_project), "manifest"]) == 0
    first = published.read_bytes()
    assert main(["--root", str(tmp_project), "manifest"]) == 0
    assert published.read_bytes() == first


def test_manifest_command_fails_on_missing_source(tmp_project: Path) -> None:
    (tmp_project / "data" / "sources" / "b.csv").unlink()
    assert main(["--root", str(tmp_project), "manifest"]) == 1
    record = _only_run(tmp_project)
    assert record["status"] == "FAIL"
    assert any("data/sources/b.csv" in n for n in record["notes"])


def test_broken_config_still_produces_a_fail_run_record(tmp_project: Path) -> None:
    """Amendment 5: a configuration error must not leave the run unrecorded."""
    (tmp_project / "configs" / "project.yml").write_text("spec_version: \"9.9\"\n", encoding="utf-8")
    assert main(["--root", str(tmp_project), "manifest"]) == 1
    record = _only_run(tmp_project)
    assert record["status"] == "FAIL"
    assert record["config_version"] == "unavailable"
    assert any("config" in n.lower() for n in record["notes"])


def test_env_report_command(tmp_project: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("PATH", str(tmp_project))
    monkeypatch.setenv("HOME", str(tmp_project))
    assert main(["--root", str(tmp_project), "env-report"]) == 0
    assert (tmp_project / "reports" / "environment_report.md").is_file()
    env = json.loads(
        (tmp_project / "reports" / "manifests" / "environment.json").read_text(encoding="utf-8")
    )
    assert env["uv_lock_sha256"] is not None
    assert env["julia_pinned_version"] == "1.12.7"
    assert "created_at" not in env


def test_smoke_command_freeze_then_pass(
    tmp_project: Path, fake_launcher_factory: Any, capsys: Any
) -> None:
    launcher = fake_launcher_factory(OK)
    factory = lambda paths, cfg, julia: launcher  # noqa: E731
    assert main(["--root", str(tmp_project), "smoke", "--freeze-expected"],
                launcher_factory=factory) == 0
    assert main(["--root", str(tmp_project), "smoke"], launcher_factory=factory) == 0
    out = capsys.readouterr().out
    assert "status=PASS" in out
    assert "artifacts/runs/" in out


def test_smoke_command_returns_1_on_fail(tmp_project: Path, fake_launcher_factory: Any) -> None:
    launcher = fake_launcher_factory(OK)
    assert main(["--root", str(tmp_project), "smoke"],
                launcher_factory=lambda p, c, j: launcher) == 1


def test_smoke_command_without_julia_returns_1_and_records_fail(
    tmp_project: Path, monkeypatch: Any
) -> None:
    # PATH and HOME are redirected so the developer's real julia cannot be discovered:
    # this test is about the FAIL record, not about running a simulation.
    monkeypatch.setenv("PATH", str(tmp_project))
    monkeypatch.setenv("HOME", str(tmp_project))
    assert main(["--root", str(tmp_project), "smoke", "--julia", str(tmp_project / "absent")]) == 1
    record = _only_run(tmp_project)
    assert record["status"] == "FAIL"
    assert any("julia" in n.lower() for n in record["notes"])


def test_unknown_command_returns_2(tmp_project: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--root", str(tmp_project), "nope"])
    assert exc.value.code == 2
```

- [ ] **Step 5: Убедиться, что тесты падают**

```bash
uv run pytest tests/unit/test_runner.py tests/unit/test_smoke.py tests/unit/test_cli.py -q
```
Ожидается: `ModuleNotFoundError` для `so_recon.runner`, `so_recon.smoke`, `so_recon.cli`.

- [ ] **Step 6: Реализовать `runner.py`**

```python
"""One place that guarantees invariant I6: every command run leaves a run record.

Whatever the body raises — a missing source, a missing Julia executable, a bug — the
traceback goes to run.log, the message goes to run.json notes, and the record is closed
with status FAIL. execute_run never propagates an exception from the body.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

from so_recon.config.schema import ProjectConfig
from so_recon.logging_setup import configure_logging
from so_recon.paths import ProjectPaths
from so_recon.registry.run import RunContext, RunStatus

CommandBody = Callable[[RunContext, logging.Logger], tuple[RunStatus, list[str]]]


def execute_run(
    *,
    command: str,
    argv: Sequence[str],
    cfg: ProjectConfig | None,
    paths: ProjectPaths,
    body: CommandBody,
    schema_versions: dict[str, str] | None = None,
    parent_run_ids: Sequence[str] = (),
) -> RunContext:
    ctx = RunContext.start(
        command=command,
        argv=argv,
        cfg=cfg,
        paths=paths,
        schema_versions=schema_versions,
        parent_run_ids=parent_run_ids,
    )
    log = configure_logging(ctx.run_id, ctx.run_dir / "run.log")
    try:
        status, notes = body(ctx, log)
    except Exception as exc:  # noqa: BLE001 - deliberate: every failure becomes a FAIL record
        log.exception("command %s failed", command)
        ctx.finish("FAIL", notes=[f"exception: {type(exc).__name__}: {exc}"])
    else:
        ctx.finish(status, notes=notes)
    finally:
        for handler in logging.getLogger("so_recon").handlers:
            handler.flush()
    return ctx
```

- [ ] **Step 7: Реализовать `smoke.py`**

```python
"""E00 smoke scenario: deterministic fixture -> case.json -> JutulDarcy -> frozen expectations."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

from so_recon.config.schema import ProjectConfig, StrictModel
from so_recon.environment.report import check_locked_versions, read_locked_versions
from so_recon.paths import ProjectPaths
from so_recon.registry.artifact import register_artifact, write_artifact, write_json_artifact
from so_recon.registry.atomic import write_bytes_atomic
from so_recon.registry.hashing import sha256_bytes
from so_recon.registry.run import RunContext, RunStatus
from so_recon.runner import CommandBody, execute_run
from so_recon.simulator.julia_bridge import (
    JuliaLauncher,
    JuliaSmokeResult,
    run_julia_smoke,
)
from so_recon.synthetic.fixture import (
    CASE_SCHEMA_VERSION,
    FIXTURE_SCHEMA_VERSION,
    build_smoke_case,
    build_smoke_fixture,
    case_bytes,
    write_smoke_fixture,
)

EXPECTED_FILENAME = "smoke_expected.json"
EXPECTED_SCHEMA_VERSION = "2"
SCHEMA_VERSIONS = {
    "smoke_fixture": FIXTURE_SCHEMA_VERSION,
    "smoke_case": CASE_SCHEMA_VERSION,
    "julia_smoke_result": "1",
    "smoke_expected": EXPECTED_SCHEMA_VERSION,
    "run_record": "1",
}

LauncherFactory = Callable[[], JuliaLauncher]


class SmokeExpectation(StrictModel):
    """Frozen expectations. Committed, therefore deterministic: no timestamps (I5)."""

    schema_version: str = EXPECTED_SCHEMA_VERSION
    fixture_content_hash: str
    case_sha256: str
    nx: int
    n_steps: int
    cumulative_oil_m3: float
    cumulative_water_injected_m3: float
    mean_so_final: float
    julia_version: str
    jutul_version: str
    jutuldarcy_version: str


def _close(a: float, b: float, rel_tol: float) -> bool:
    return abs(a - b) <= rel_tol * max(1.0, abs(b))


def compare_with_expected(
    expected: SmokeExpectation,
    *,
    fixture_hash: str,
    case_sha256: str,
    result: JuliaSmokeResult,
    rel_tol: float,
) -> list[str]:
    mismatches: list[str] = []
    for label, got, exp in (
        ("fixture_content_hash", fixture_hash, expected.fixture_content_hash),
        ("case_sha256", case_sha256, expected.case_sha256),
        # A version bump changes the numbers it is compared against, so it is a failure,
        # not a note: reproducibility is only claimed against the locked stack.
        ("julia_version", result.julia_version, expected.julia_version),
        ("jutul_version", result.jutul_version, expected.jutul_version),
        ("jutuldarcy_version", result.jutuldarcy_version, expected.jutuldarcy_version),
    ):
        if got != exp:
            mismatches.append(f"{label}: got {got}, expected {exp}")
    for name in ("nx", "n_steps"):
        got_i, exp_i = getattr(result, name), getattr(expected, name)
        if got_i != exp_i:
            mismatches.append(f"{name}: got {got_i}, expected {exp_i}")
    for name in ("cumulative_oil_m3", "cumulative_water_injected_m3", "mean_so_final"):
        got_f, exp_f = getattr(result, name), getattr(expected, name)
        if not _close(got_f, exp_f, rel_tol):
            mismatches.append(f"{name}: got {got_f!r}, expected {exp_f!r} (rel_tol={rel_tol})")
    return mismatches


def smoke_body(
    *,
    cfg: ProjectConfig,
    paths: ProjectPaths,
    launcher_factory: LauncherFactory,
    freeze_expected: bool,
) -> CommandBody:
    def body(ctx: RunContext, log: logging.Logger) -> tuple[RunStatus, list[str]]:
        now = datetime.now(UTC)

        fixture = build_smoke_fixture(cfg.smoke)
        files = write_smoke_fixture(fixture, ctx.run_dir / "fixture")
        for key, file_path in files.items():
            ctx.add_output(
                f"fixture_{key}",
                register_artifact(
                    file_path, paths,
                    schema_version=FIXTURE_SCHEMA_VERSION, producer_run_id=ctx.run_id,
                    media_type=(
                        "application/vnd.apache.parquet"
                        if file_path.suffix == ".parquet"
                        else "application/json"
                    ),
                    now=now,
                ),
            )
        log.info("fixture built: seed=%d hash=%s", fixture.seed, fixture.content_hash)

        case = build_smoke_case(cfg.smoke, fixture)
        payload = case_bytes(case)
        case_sha = sha256_bytes(payload)
        case_ref = write_artifact(
            ctx.run_dir / "case.json", payload, paths,
            schema_version=CASE_SCHEMA_VERSION, producer_run_id=ctx.run_id,
            media_type="application/json", now=now,
        )
        ctx.add_output("case", case_ref)
        log.info("case written: sha256=%s", case_sha)

        # The factory runs here on purpose: a missing Julia must become a FAIL record.
        launcher = launcher_factory()

        julia_out = ctx.run_dir / "julia_smoke.json"
        result = run_julia_smoke(
            launcher,
            paths.resolve(cfg.julia.smoke_script),
            ctx.run_dir / "case.json",
            julia_out,
            expected_input_sha256=case_sha,
        )
        ctx.update(
            julia_version=result.julia_version,
            jutul_version=result.jutul_version,
            jutuldarcy_version=result.jutuldarcy_version,
        )
        ctx.add_output(
            "julia_smoke",
            write_artifact(
                julia_out, julia_out.read_bytes(), paths,
                schema_version="1", producer_run_id=ctx.run_id,
                media_type="application/json", parent_artifact_ids=[case_ref.artifact_id], now=now,
            ),
        )
        log.info(
            "julia smoke ok: oil=%.6g winj=%.6g mean_so=%.6f (%.1fs)",
            result.cumulative_oil_m3, result.cumulative_water_injected_m3,
            result.mean_so_final, result.wall_time_s,
        )

        # Drift from the locked Julia stack is a failure (amendment 9).
        lock_mismatches = check_locked_versions(
            read_locked_versions(paths),
            julia_version=result.julia_version,
            jutul_version=result.jutul_version,
            jutuldarcy_version=result.jutuldarcy_version,
        )
        if lock_mismatches:
            for m in lock_mismatches:
                log.error("locked version mismatch: %s", m)
            return "FAIL", lock_mismatches

        expected_path = paths.configs / EXPECTED_FILENAME
        if freeze_expected:
            expectation = SmokeExpectation(
                fixture_content_hash=fixture.content_hash,
                case_sha256=case_sha,
                nx=result.nx,
                n_steps=result.n_steps,
                cumulative_oil_m3=result.cumulative_oil_m3,
                cumulative_water_injected_m3=result.cumulative_water_injected_m3,
                mean_so_final=result.mean_so_final,
                julia_version=result.julia_version,
                jutul_version=result.jutul_version,
                jutuldarcy_version=result.jutuldarcy_version,
            )
            ref = write_json_artifact(
                ctx.run_dir / EXPECTED_FILENAME, expectation.model_dump(mode="json"), paths,
                schema_version=EXPECTED_SCHEMA_VERSION, producer_run_id=ctx.run_id,
                parent_artifact_ids=[case_ref.artifact_id], now=now,
            )
            write_bytes_atomic(expected_path, (ctx.run_dir / EXPECTED_FILENAME).read_bytes())
            ctx.add_output("smoke_expected", ref)
            return "PASS", [f"expected frozen to {paths.relative(expected_path)}"]

        if not expected_path.is_file():
            return "FAIL", [
                f"expected file missing: {paths.relative(expected_path)}; "
                "run with --freeze-expected"
            ]

        expected = SmokeExpectation.model_validate(
            json.loads(expected_path.read_text(encoding="utf-8"))
        )
        mismatches = compare_with_expected(
            expected, fixture_hash=fixture.content_hash, case_sha256=case_sha,
            result=result, rel_tol=cfg.smoke.rel_tol,
        )
        if mismatches:
            for m in mismatches:
                log.error("smoke mismatch: %s", m)
            return "FAIL", mismatches
        return "PASS", []

    return body


def run_smoke(
    *,
    cfg: ProjectConfig,
    paths: ProjectPaths,
    argv: Sequence[str],
    launcher_factory: LauncherFactory,
    freeze_expected: bool = False,
) -> RunContext:
    paths.ensure_dirs()
    return execute_run(
        command="smoke",
        argv=argv,
        cfg=cfg,
        paths=paths,
        schema_versions=SCHEMA_VERSIONS,
        body=smoke_body(
            cfg=cfg, paths=paths, launcher_factory=launcher_factory,
            freeze_expected=freeze_expected,
        ),
    )
```

> Файлы fixture регистрируются как артефакты через `register_artifact`, поэтому `wells.parquet` и `well_month.parquet` попадают в lineage `run.json` с собственными `sha256` (замечание №4).

- [ ] **Step 8: Реализовать `cli.py`**

```python
"""so-recon command line: manifest | env-report | smoke.

Every subcommand runs inside execute_run, so any failure — including a configuration
error or a missing Julia executable — leaves a run.json with status FAIL (invariant I6).
The single exception is a repository root that cannot be located: there is nowhere to write.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from so_recon.config.load import load_project_config
from so_recon.config.schema import JuliaConfig, ProjectConfig
from so_recon.environment.report import (
    ENVIRONMENT_SCHEMA_VERSION,
    build_environment_stamp,
    collect_environment,
    report_json_bytes,
    write_environment_report,
    write_environment_stamp,
)
from so_recon.paths import ProjectPaths, RepoRootNotFoundError, find_repo_root
from so_recon.registry.hashing import sha256_bytes
from so_recon.registry.run import RunContext, RunStatus
from so_recon.registry.source_manifest import (
    MANIFEST_SCHEMA_VERSION,
    SourceManifestStamp,
    build_source_manifest,
    manifest_bytes,
    write_manifest_stamp,
    write_source_manifest,
)
from so_recon.runner import execute_run
from so_recon.simulator.julia_bridge import JuliaLauncher, default_launcher
from so_recon.smoke import run_smoke

LauncherFactory = Callable[[ProjectPaths, JuliaConfig, str | None], JuliaLauncher]


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


def _manifest_body(
    cfg: ProjectConfig, paths: ProjectPaths
) -> Callable[[RunContext, logging.Logger], tuple[RunStatus, list[str]]]:
    def body(ctx: RunContext, log: logging.Logger) -> tuple[RunStatus, list[str]]:
        now = datetime.now(UTC)
        manifest = build_source_manifest(cfg.sources, paths, config_version=cfg.config_version)
        ref, published = write_source_manifest(
            manifest, paths, run_dir=ctx.run_dir,
            published_path=paths.manifests / "source_manifest.json",
            producer_run_id=ctx.run_id, now=now,
        )
        write_manifest_stamp(
            SourceManifestStamp(
                manifest_version=MANIFEST_SCHEMA_VERSION,
                manifest_sha256=ref.sha256,
                run_id=ctx.run_id,
                created_at=now.isoformat(),
                git_commit=ctx.record.git_commit,
                git_dirty=ctx.record.git_dirty,
            ),
            ctx.run_dir / "source_manifest_stamp.json",
        )
        ctx.update(raw_input_hashes=manifest.hashes())
        ctx.add_output("source_manifest", ref)
        log.info("source manifest published to %s", paths.relative(published))
        return "PASS", []

    return body


def _env_report_body(
    paths: ProjectPaths,
) -> Callable[[RunContext, logging.Logger], tuple[RunStatus, list[str]]]:
    def body(ctx: RunContext, log: logging.Logger) -> tuple[RunStatus, list[str]]:
        now = datetime.now(UTC)
        report = collect_environment(paths)
        md_ref, json_ref = write_environment_report(
            report, paths, run_dir=ctx.run_dir,
            md_path=paths.reports / "environment_report.md",
            json_path=paths.manifests / "environment.json",
            producer_run_id=ctx.run_id, now=now,
        )
        write_environment_stamp(
            build_environment_stamp(
                paths, report_sha256=sha256_bytes(report_json_bytes(report)),
                run_id=ctx.run_id, now=now,
            ),
            ctx.run_dir / "environment_stamp.json",
        )
        ctx.update(
            julia_version=report.julia_manifest_version,
            jutul_version=report.julia_packages.get("Jutul"),
            jutuldarcy_version=report.julia_packages.get("JutulDarcy"),
            schema_versions={"environment_report": ENVIRONMENT_SCHEMA_VERSION},
        )
        ctx.add_output("environment_report", md_ref)
        ctx.add_output("environment_json", json_ref)
        log.info("environment report written")
        return "PASS", []

    return body


def _report(ctx: RunContext, paths: ProjectPaths) -> int:
    print(
        f"run_id={ctx.run_id} status={ctx.record.status} run_dir={paths.relative(ctx.run_dir)}"
    )
    for note in ctx.record.notes:
        print(f"  {note}")
    return 0 if ctx.record.status == "PASS" else 1


def _record_startup_failure(
    root: Path, command: str, argv: Sequence[str], exc: Exception
) -> int:
    """Configuration could not be loaded: still leave a FAIL record (invariant I6)."""
    paths = ProjectPaths.default(root)

    def body(ctx: RunContext, log: logging.Logger) -> tuple[RunStatus, list[str]]:
        return "FAIL", [f"config error: {exc}"]

    ctx = execute_run(command=command, argv=argv, cfg=None, paths=paths, body=body)
    print(f"config error: {exc}", file=sys.stderr)
    return _report(ctx, paths)


def main(argv: Sequence[str] | None = None, *, launcher_factory: LauncherFactory | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(raw_argv)
    full_argv = ["so-recon", *raw_argv]

    try:
        root = args.root.resolve() if args.root else find_repo_root()
    except RepoRootNotFoundError as exc:
        # The only path with no run record: there is no repository to write into.
        print(f"cannot locate repository root: {exc}", file=sys.stderr)
        return 1

    try:
        cfg = load_project_config(args.config or (root / "configs" / "project.yml"))
        paths = ProjectPaths.from_config(root, cfg.paths)
    except Exception as exc:  # noqa: BLE001 - any config failure must still be recorded
        return _record_startup_failure(root, args.command, full_argv, exc)

    if args.command == "smoke":
        factory = launcher_factory or default_launcher
        ctx = run_smoke(
            cfg=cfg, paths=paths, argv=full_argv,
            launcher_factory=lambda: factory(paths, cfg.julia, args.julia),
            freeze_expected=args.freeze_expected,
        )
        return _report(ctx, paths)

    paths.ensure_dirs()
    body = _manifest_body(cfg, paths) if args.command == "manifest" else _env_report_body(paths)
    ctx = execute_run(command=args.command, argv=full_argv, cfg=cfg, paths=paths, body=body)
    return _report(ctx, paths)


if __name__ == "__main__":
    raise SystemExit(main())
```

> `default_launcher(paths, cfg, julia_exe)` уже имеет нужную сигнатуру из Task 10, поэтому подходит на роль `LauncherFactory` без обёртки.

- [ ] **Step 9: Прогнать все тесты, lint, format, mypy**

```bash
uv run pytest -q -m "not julia" && uv run ruff check . && uv run ruff format --check . && uv run mypy
```
Ожидается: все тесты проходят, ruff и mypy без ошибок. При замечаниях `ruff format` — выполнить `uv run ruff format .` и перепроверить.

- [ ] **Step 10: Commit**

```bash
git add src/so_recon/runner.py src/so_recon/smoke.py src/so_recon/cli.py tests/conftest.py tests/unit/test_runner.py tests/unit/test_smoke.py tests/unit/test_cli.py && git commit -m "feat(e00): add run executor guaranteeing FAIL records, smoke scenario and so-recon CLI

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 13: Прогон gate, заморозка ожиданий, отчёт `reports/stages/E00.md`

**Files:**
- Create: `configs/smoke_expected.json`, `reports/manifests/source_manifest.json`, `reports/manifests/environment.json`, `reports/environment_report.md` (генерируются), `reports/stages/E00.md`
- Modify: `README.md` (при необходимости уточнить команды)

**Interfaces:**
- Consumes: CLI (Task 12), `scripts/gate.sh` (Task 1).
- Produces: статус этапа E00 и разрешённые входы для E01/E05.

**Замечания ревизии 2, реализуемые здесь:** №1 (сверка `physical_line_count`/`data_rows`), №6 (запуск gate и gate-clean), №10 (доказательство отсутствия git-diff при повторном gate), №11.

- [ ] **Step 1: Заморозить ожидания smoke на реальной Julia**

```bash
uv run so-recon smoke --freeze-expected && cat configs/smoke_expected.json
```
Ожидается: `status=PASS`; файл содержит `fixture_content_hash`, `case_sha256`, `cumulative_oil_m3`, `mean_so_final`, `julia_version`, `jutul_version`, `jutuldarcy_version` и **не содержит** временных меток.

- [ ] **Step 2: Повторный прогон без заморозки должен пройти**

```bash
uv run so-recon smoke && ls artifacts/runs | tail -3
```
Ожидается: `status=PASS`, без заметок о расхождениях.

Проверить, что `configs/smoke_expected.json` не изменился повторным запуском:
```bash
git status --porcelain -- configs/smoke_expected.json; echo "(empty above = unchanged)"
```

- [ ] **Step 3: Сформировать manifest и environment report на реальных данных**

```bash
uv run so-recon manifest && uv run so-recon env-report && cat reports/manifests/source_manifest.json && cat reports/environment_report.md
```

Сверить счётчики с проверенной таблицей (раздел «Проверенные контракты исходных файлов»):

| файл | physical_line_count | data_rows |
|---|---:|---:|
| coords.csv | 4241 | 4240 |
| gis.csv | 107176 | 107175 |
| mer.csv | 1388595 | 1388594 |
| perf.csv | 33236 | 33235 |
| plastoper.csv | 3573 | 3572 |

Автоматическая сверка:
```bash
python3 - <<'PY'
import json
expected = {"coords": (4241, 4240), "gis": (107176, 107175), "mer": (1388595, 1388594),
            "perf": (33236, 33235), "plastoper": (3573, 3572)}
m = json.load(open("reports/manifests/source_manifest.json", encoding="utf-8"))
bad = []
for s in m["sources"]:
    got = (s["physical_line_count"], s["data_rows"])
    if got != expected[s["name"]]:
        bad.append((s["name"], got, expected[s["name"]]))
print("MISMATCHES:", bad if bad else "none")
PY
```
`data_rows` должны совпасть с колонкой «Строк» в `DATA_AUDIT.md` §2. Если число отличается — **не править код**: зафиксировать расхождение в `E00.md` как замечание и обосновать (физический счёт против разобранных записей).

Убедиться, что источники не тронуты:
```bash
ls -1 data/Ромашка_сырые/ && test ! -e data/raw && echo "sources untouched, no data/raw created"
```

- [ ] **Step 4: Зафиксировать сгенерированные детерминированные артефакты в git**

Их нужно закоммитить **до** gate, чтобы шаг проверки детерминизма в `gate.sh` сравнивал отслеживаемые файлы.

```bash
git add configs/smoke_expected.json reports/manifests/source_manifest.json reports/manifests/environment.json reports/environment_report.md && git commit -m "chore(e00): freeze smoke expectations and publish source/environment manifests

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] **Step 5: Полный gate на чистом Python-окружении**

```bash
make gate 2>&1 | tail -40
```
Ожидается: все восемь шагов проходят, последняя строка — `== gate PASS ... ==`, включая `deterministic artifacts unchanged (no git pollution)`. Дополнительно убедиться вручную:
```bash
git status --porcelain -- uv.lock julia/Manifest.toml julia/.julia-version reports configs; echo "(empty above = no git pollution)"
```

- [ ] **Step 6: Gate на чистом Julia depot**

```bash
make gate-clean 2>&1 | tail -20
```
Ожидается: `== gate PASS ... ==`. Шаг долгий (полная загрузка и precompile JutulDarcy во временный depot). Если он не укладывается в разумное время или падает по сети — **не считать это провалом E00**: зафиксировать в `E00.md` как ограничение со статусом `PASS_WITH_LIMITATIONS` и указать причину.

- [ ] **Step 7: Написать `reports/stages/E00.md`**

Заполнить фактическими значениями из `artifacts/runs/<run_id>/run.json`, `reports/manifests/source_manifest.json`, `reports/environment_report.md`, `configs/smoke_expected.json`. Шаблон:

```markdown
# E00 — Основание проекта и воспроизводимое окружение

**Статус:** PASS | PASS_WITH_LIMITATIONS  (выбрать по результату Steps 5–6)
**Дата:** <YYYY-MM-DD>
**SPEC.md:** 3.0 · **STAGES.md:** 3.0 · **config_version:** E00.2
**Git commit кода:** <sha последнего коммита кода; сам отчёт коммитится следом>

## 1. Входные артефакты и hashes

Источник: `reports/manifests/source_manifest.json` (детерминированный; run-штамп — в `artifacts/runs/<run_id>/source_manifest_stamp.json`).

| файл | sha256 | байт | physical_line_count | data_rows |
|---|---|---:|---:|---:|
| data/Ромашка_сырые/coords.csv | <sha> | <n> | 4241 | 4240 |
| data/Ромашка_сырые/gis.csv | <sha> | <n> | 107176 | 107175 |
| data/Ромашка_сырые/mer.csv | <sha> | <n> | 1388595 | 1388594 |
| data/Ромашка_сырые/perf.csv | <sha> | <n> | 33236 | 33235 |
| data/Ромашка_сырые/plastoper.csv | <sha> | <n> | 3573 | 3572 |

`physical_line_count` — физические строки (включая последнюю строку без перевода строки в `plastoper.csv`); `data_rows = physical_line_count - 1` (один заголовок). Сверка с `DATA_AUDIT.md` §2 (колонка «Строк» = `data_rows`): <совпадает / перечислить расхождения>.

Исходники **не перемещались и не изменялись**: остаются в `data/Ромашка_сырые/`, открываются только на чтение.

## 2. Окружение

Источник: `reports/environment_report.md` (детерминированный; время и git-состояние — в `artifacts/runs/<run_id>/environment_stamp.json`).

- Python <ver>, uv <ver>, `uv.lock` sha256 `<...>`
- Julia <ver> — **project-pinned** (`julia/.julia-version`), не утверждение об upstream stable
- `julia/Manifest.toml` sha256 `<...>`, JutulDarcy <ver>, Jutul <ver>
- environment_lock_hash `<...>`
- ОС/арх: <...>

## 3. Выполненные команды

    uv run so-recon smoke --freeze-expected   # однократно, run_id <...>
    uv run so-recon smoke                     # run_id <...>, PASS
    uv run so-recon manifest                  # run_id <...>, PASS
    uv run so-recon env-report                # run_id <...>, PASS
    make gate                                 # PASS
    make gate-clean                           # PASS | не выполнен (причина)

## 4. Результаты проверок

- Smoke сквозной: Python пишет `case.json` (sha256 `<...>`), Julia возвращает `input_sha256`, значения совпали.
- Smoke fixture: content_hash `<...>` (seed 20260913, 4 скважины × 12 месяцев).
- JutulDarcy smoke (1D, nx=20, 12 шагов по 30 сут): cumulative_oil_m3 = <...>, cumulative_water_injected_m3 = <...>, mean_so_final = <...>; повторный прогон в допуске rel_tol = 1e-6.
- Версии Julia/Jutul/JutulDarcy совпадают с lock (расхождение дало бы FAIL).
- Тесты: <N> passed, 0 failed; integration-тест Julia выполнен (не skipped).
- Lint/типизация: `ruff check`, `ruff format --check`, `mypy --strict` чисты.
- Guard абсолютных путей: `tests/test_no_absolute_paths.py` PASS.
- Детерминизм: повторный `make gate` не изменил ни один отслеживаемый файл.
- FAIL-запись: проверена тестами для отсутствующего источника, битой конфигурации, отсутствующей Julia и ошибки решателя.

## 5. Созданные артефакты

- `pyproject.toml`, `uv.lock`, `.python-version`, `Makefile`, `scripts/gate.sh`, `scripts/gate_clean.sh`, `README.md`
- `julia/Project.toml`, `julia/Manifest.toml`, `julia/.julia-version`, `julia/smoke/smoke_case.jl`
- `configs/project.yml`, `configs/smoke_expected.json`
- `src/so_recon/` (paths, config, registry, environment, synthetic, simulator, runner, smoke, cli)
- `reports/manifests/source_manifest.json`, `reports/manifests/environment.json`, `reports/environment_report.md`
- `artifacts/runs/<run_id>/` (локально, не в git)

## 6. Ограничения

- CI не настроен (нет remote; precompile JutulDarcy в CI дорог). Роль чистой установки выполняют `make gate` и `make gate-clean`.
- Julia-smoke — проверка работоспособности окружения, не верификация физики (E05).
- `physical_line_count` — физический счёт строк; переводы строк внутри закавыченных полей не учитываются, потому что E00 не парсит CSV. Реальное число записей определяется в E01.
- Единственный случай без run-записи — недоступный repository root: писать некуда.
- GNU Make 3.81 не поддерживает `.SHELLFLAGS`, поэтому `pipefail` реализован в `scripts/gate.sh`.
- Расхождения документации: `STAGES.md` §8 ссылается на `docs/README_SO_RECON.md`, фактический файл `docs/README.md`. <другие найденные>

## 7. Что передаётся следующим этапам

- E01: `ProjectPaths`, `ProjectConfig`/`SourceFileSpec` (encoding/delimiter/decimal/header_lines для пяти файлов), `RunContext`/`execute_run`, `ArtifactRef`, `source_manifest.json` как эталон входов, `hash_and_count`.
- E05: `julia/` окружение с зафиксированным JutulDarcy, `SubprocessJuliaLauncher`/`JuliaLauncher` как точка расширения, контракт обмена `case.json` → результат с `input_sha256`.
- Все этапы: правило «один запуск = один `run_id` + `run.json` + `resolved_config.json` + `run.log`», immutable `ArtifactRef` и разделение детерминированных коммитируемых артефактов и run-штампов.
```

- [ ] **Step 8: Финальные проверки и коммит отчёта**

```bash
uv run pytest -q && uv run ruff check . && uv run ruff format --check . && uv run mypy && git add reports/stages/E00.md README.md && git commit -m "docs(e00): add E00 stage report with verified hashes, line counts and gate results

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

Убедиться, что `git status` чист:
```bash
git status --short
```

---

## Self-Review

**Spec coverage (STAGES E00 «Основная работа» → задачи):**
- структура Python/Julia-проекта и typed configuration → Tasks 1, 4, 9;
- lock-файлы Python, Julia, JutulDarcy → Tasks 1 (`uv.lock`), 9 (`Manifest.toml`, `.julia-version`, compat 0.3);
- каталоги raw/interim/processed/artifacts/reports → Task 3 (`ProjectPaths`), Task 4 (`PathsConfig`); источники остаются на исходном пути;
- manifest исходных файлов, configs и environment → Task 7 (source), Task 6 (`resolved_config.json`, `run.json`), Task 11 (environment);
- минимальный synthetic fixture и smoke test → Tasks 8, 9, 10, 12;
- единый CLI/run ID и правила логирования → Tasks 6, 12.

**Ключевые выходы:** `pyproject.toml`+`uv.lock` (T1), `Project.toml`+`Manifest.toml`+`.julia-version` (T9), каркас каталогов (T1), `source_manifest.json` (T7/T13), `environment_report.md` (T11/T13), `reports/stages/E00.md` (T13).

**Gate:** чистая установка → `make gate` (T1, T13) и `make gate-clean` (T1, T13); детерминированный fixture → T8 + T9 Step 5 + `smoke_expected.json` (T13); версии и hashes → `run.json`, manifests (T6, T7, T11); нет абсолютных путей → `tests/test_no_absolute_paths.py` (T3) и валидаторы путей (T3/T4).

**SPEC 19.12 поля lineage в `RunRecord`:** raw hashes ✓, schema versions ✓, git commit ✓, environment lock hash ✓, Julia/Jutul/JutulDarcy версии ✓, checkpoint hash (поле, `None`) ✓, resolved config hash ✓, parent artifact IDs (в `ArtifactRef.parent_artifact_ids`) ✓, timestamp ✓, command/run ID ✓.

**Покрытие 12 замечаний ревизии 2:**

| № | Замечание | Где реализовано | Чем проверено |
|---:|---|---|---|
| 1 | `plastoper` decimal `","`; `physical_line_count` 3573 / `data_rows` 3572 | T4 (`configs/project.yml`, `header_lines`), T7 (`SourceEntry`) | `test_repo_config_file_is_valid_and_matches_audited_contracts`, `test_hash_and_count_without_trailing_newline`, `test_manifest_splits_physical_lines_from_data_rows`, T13 Step 3 |
| 2 | Удалён разрушительный `mv`; источники неизменяемы | T3 (`ensure_dirs` не создаёт raw), T4 (пути), T7 (только `"rb"`) | `test_ensure_dirs_creates_runtime_dirs_but_not_sources`, `test_sources_are_never_modified`, T13 Step 3 |
| 3 | Сквозной smoke: `case.json` → Julia → `input_sha256` | T8 (`build_smoke_case`), T9 (`read_case`), T10 (сверка), T12 | `test_case_bytes_are_canonical_and_hashable`, `test_run_julia_smoke_rejects_a_mismatched_input_hash`, `test_run_smoke_detects_a_tampered_case_file`, T9 Step 4 |
| 4 | `ArtifactRef` + immutability | T5 (`artifact.py`), T6 (`outputs`) | `test_rewriting_with_different_content_is_refused`, `test_run_context_writes_lineage_files` |
| 5 | FAIL при любом исключении, включая отсутствие Julia | T6 (degraded ctx), T12 (`execute_run`, factory внутри тела) | `test_execute_run_converts_any_exception_into_a_fail_record`, `test_run_smoke_records_fail_when_julia_is_missing`, `test_broken_config_still_produces_a_fail_run_record` |
| 6 | Gate: pipefail/tee, `ruff format --check`, `gate-clean` с временным depot | T1 (`scripts/gate.sh`, `scripts/gate_clean.sh`) | T1 Step 10 (`bash -n`), T13 Steps 5–6 |
| 7 | Точные версии Python и Julia; Julia — project-pinned | T1 (`.python-version` 3.13.2), T9 (`.julia-version`, README) | T1 Step 8, T9 Step 1–2, `test_collect_environment...` (`julia_pinned_version`) |
| 8 | Запрет `..`, Windows absolute/UNC, symlink-escape | T3 (`validate_relative_path`, `resolve_within_root`), T4 (валидаторы) | `test_validate_relative_path_rejects` (16 кейсов), `test_resolve_within_root_rejects_symlink_escape`, `test_dangerous_paths_config_values_are_rejected` |
| 9 | Несовпадение с lock → FAIL | T11 (`check_locked_versions`), T12 (`compare_with_expected`, вызов в `smoke_body`) | `test_version_drift_against_lock_is_a_mismatch`, `test_version_drift_is_a_mismatch_not_a_note`, `test_run_smoke_fails_when_versions_drift_from_the_lock` |
| 10 | Детерминированные manifests отдельно от run-timestamps | T7 (`SourceManifest`/`SourceManifestStamp`), T11 (`EnvironmentReport`/`EnvironmentStamp`), T12 (`SmokeExpectation` без `frozen_at`), T1 (проверка в gate) | `test_manifest_has_no_timestamps_or_git_state`, `test_report_is_deterministic_and_carries_no_time_or_git_state`, `test_manifest_is_byte_identical_on_a_second_run`, T13 Step 5 |
| 11 | Нет личных абсолютных путей в командах плана | все задачи (команды из корня репозитория) | `tests/test_no_absolute_paths.py`, ручная вычитка плана |
| 12 | Атомарная запись `run.json` и manifests | T2 (`atomic.py`), используется в T5–T12 | `test_failed_write_leaves_no_debris_and_keeps_old_content`, `test_write_atomic_leaves_no_temporary_files` |

**Placeholder scan:** значения `<...>` присутствуют только в шаблоне отчёта Task 13 Step 7 и обозначают фактические результаты прогона; все кодовые шаги содержат полный код.

**Type consistency:** `ProjectPaths.from_config(root, cfg.paths)` — T3 определение, T4/T10/T12 вызовы; `RunContext.start(command=, argv=, cfg=, paths=, ...)` — T6 сигнатура, T12 вызовы через `execute_run`; `run_julia_smoke(launcher, script, case_path, out_path, *, expected_input_sha256)` — T10 определение, T10/T12 вызовы; поля `JuliaSmokeResult` совпадают с JSON-ключами `smoke_case.jl` (T9) и с `SmokeExpectation` (T12); `check_locked_versions(locked, *, julia_version, jutul_version, jutuldarcy_version)` — T11 определение, T12 вызов; `hash_and_count` возвращает `(sha, size, physical_line_count)` — T7 тесты и код согласованы; `write_artifact(path, data, paths, *, schema_version, producer_run_id, media_type, parent_artifact_ids, now)` — T5 определение, T7/T11/T12 вызовы.
