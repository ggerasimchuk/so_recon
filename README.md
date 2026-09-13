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
