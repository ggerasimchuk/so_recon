# SO-RECON

Физически ограниченная байесовская реконструкция текущей нефтенасыщенности.
Действующие требования: [SPEC 4.0](docs/SPEC.md), [этапы E00–E14](docs/STAGES.md),
[вычислительные профили](docs/COMPUTE_PROFILES.md).
Полный [указатель документации](docs/README.md) и [переход с E00](docs/DECISIONS.md).

E00 сохраняется в выполненном объёме 3.0. Его код, текущий `configs/project.yml`,
smoke и исторические manifests ещё относятся к 3.0; документация 4.0 не меняет их
задним числом. E01 реализует совместимость версии/lineage, persistent Julia worker,
ресурсные ограничения и учебные физические synthetic cases. Актуальная приёмка и
ограничения указаны в [отчёте E01](reports/stages/E01.md); работа над E02 ведётся отдельно.

Первый полный цикл с ML и SMC предусмотрен в E03. Полевая канонизация данных —
E04; основные научные сравнения — E11. В [STAGES §7](docs/STAGES.md) приведён
готовый запрос на план E01. Присланные `*_v2.md` остаются материалами обсуждения;
планы следуют согласованным документам в `docs/`.

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
`run.log` и артефактами запуска. Исторические файлы E00 в `reports/` и `configs/`
детерминированы. Отчёт E01 содержит время и идентификаторы реальных запусков и обновляется
при повторной проверке.

### E01: forward-synthetic

`--root` и `--config` — глобальные и идут **до** подкоманды; порядок не менялся.

    uv run so-recon --config configs/e01.yml forward --case PATH
    uv run so-recon --config configs/e01.yml forward-resume --case PATH --restart PATH
    uv run so-recon --config configs/e01.yml verify-physics --suite {p0|p1|bo}
    uv run so-recon --config configs/e01.yml verify-physics --suite p0 \
        --resume-ledger artifacts/runs/<run_id>/ledgers/<job>.json [--replay]
    uv run so-recon --config configs/e01.yml synthetic-p1 --seeds 41 42 43 44 45
    uv run so-recon --config configs/e01.yml benchmark-forward --case PATH --warm-runs 5
    uv run so-recon --config configs/e01.yml e01-report --runs PATH [PATH ...]

    make e01-p0                   # зарегистрированный P0 suite (бюджет 10 минут)
    make e01-p1                   # зарегистрированный P1 suite (бюджет 1 час)
    make e01-gate                 # регрессия + suites + валидаторы + stage-отчёт
    make e01-report RUNS="artifacts/runs/<run_id> ..."

`PATH` — **фактический** путь: файл, который у оператора есть, или путь, который напечатала
предыдущая команда. Никогда не имя будущего артефакта и никогда не выдуманный run id.

Код возврата: `0` только если произошло то, о чём просили. Записанный checkpoint, частичный
отчёт или опубликованный мир сами по себе не являются успехом. Исчерпанный бюджет —
`2` со списком незапущенных jobs; проваленная обязательная проверка — `1`.

Запланированная матрица (23 job P0, 17 P1, 4 BO) описана в `configs/e01_jobs.json`; пороги
зафиксированы в `configs/e01_tolerances.yml` и hash порогов проверяется перед каждым запуском.
Итог стадии — `reports/stages/E01.md`, собранный **только** из опубликованных артефактов.
Фигуры (`reports/figures/*.png`) читаются из опубликованных HDF5/Parquet; подпись на каждой:
«synthetic truth; exploratory; educational OW».

`--resume-ledger` принимает путь ledger, который напечатала предыдущая команда. Сессия
читает опубликованный `e01_suite.json` того запуска: группа, все запланированные jobs которой
вернулись ожидаемым статусом и все проверки которой PASS, переносится только после
сверки текущих исходников, конфигурации, окружения и хешей всех результатов и не
перезапускается; остальные выполняются **в прежнем порядке плана**. Forward, чей model hash и
digest опубликованного case-манифеста уже имеют COMPLETE-запись на ledger родителя, не входит
в solver повторно. Проваленная проверка никогда не переносится. `--replay` отключает и то, и
другое и перезапускает всё. Задания с `cache_bypass` выполняются заново. Старые отчёты
без привязки входов и результатов не используются для пропуска расчётов.

`scripts/e01_gate.sh` запускает регрессию, зарегистрированные suites, stage-валидаторы
(`--fail-on-skip`: SKIP в матрице gate — это не PASS, план 12.9) и пересобирает
`reports/stages/E01.md` из тех же артефактов. Frozen inputs проверяются после завершения шагов,
включая провалившийся шаг; принудительное завершение оболочки не считается успешным gate. Он намеренно **не** вызывает `scripts/gate.sh`: тот удаляет `.venv` и
сверяет lock-файлы с HEAD. Полный foundation gate выполняется отдельно, после коммита
изменений зависимостей (E01 добавляет Matplotlib). Детерминированный отчёт окружения E00
не перезаписывается: каждая E01-сессия пишет свои lock-хэши и свой хост в
`e01_environment.json` внутри своего run-каталога.

Проверка разрешения five-spot сохраняет результат исходной пары 16/48, а также
проверяет заранее выбранную пару 112/144 при прежних допусках. Положительный результат
этой пары не подтверждает точность грубой сетки или неоднородных P1/E02 моделей.
Два дополнительных расчёта явно учтены сверх исторического списка заданий.
Методика и границы приёмки: [завершение E01](docs/superpowers/plans/2026-09-15-e01-completion.md).

### E02: probabilistic inverse

Чистая математическая проверка не запускает Julia:

    uv run so-recon --config configs/e02.yml verify-inverse --suite math
    uv run so-recon --config configs/e02.yml inverse-budget --experiment reduced-v1
    uv run so-recon --config configs/e02.yml inverse-budget --experiment e02-t1-v1-s141
    make gate-e02-math

Native-команды fail-closed и требуют `E02_E01_REPORT` — путь к принятому `E01.json`,
пересобранному на финальном physics tree. `N64` при 24 уровнях и двух moves имеет верхнюю
границу 3136 новых forward-вызовов и поэтому не гарантированно помещается в P1 cap 2000;
checkpoint означает продолжимый расчёт, но не завершённый posterior. Текущий авторитетный
статус E02 находится в [stage-отчёте](reports/stages/E02.md).
