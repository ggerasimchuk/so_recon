# Julia environment (SO-RECON)

`Project.toml` + `Manifest.toml` + `.julia-version` — зафиксированное окружение проекта.
Единственный operational forward backend — JutulDarcy (SPEC 10.1). Пакет `SOReconSimulator/`
появится в E05 и будет подключён в это окружение через `Pkg.develop`.

`julia/Project.toml` — **environment project, а не пакет**: в нём нет `name`/`uuid`/`version`.
Это существенно, см. отклонение 3 ниже.

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

## Отклонения от исходного плана (Task 9)

Зафиксированное окружение: Julia `1.12.7`, `JutulDarcy v0.3.11`, `Jutul v0.4.31`,
`JSON v1.8.0`, `SHA v0.7.0` (stdlib).

1. **`julia/Project.toml`: порядок `Pkg.add` vs `[compat]`.** Черновик Project.toml из плана
   объявлял `[compat]` для `JSON`/`Jutul`/`JutulDarcy` при пустом `[deps]`. На установленной
   версии Pkg (Julia 1.12.7) это — ошибка валидации проекта ("Compat `JSON` not listed in
   `deps`..."), потому что чтение `Project.toml` теперь строго проверяет, что каждая запись
   `[compat]` соответствует пакету в `[deps]`/`weakdeps`/`extras`. Исправление: сначала
   `Pkg.add(["JutulDarcy","Jutul","JSON","SHA"])` на файле без предварительного `[compat]`
   (кроме `julia = "1.12"`, которая для чтения `Project.toml` — особый случай, не требующий
   записи в `[deps]`), затем `Pkg.compat(...)` заполнил `[compat]` версиями, разрешёнными
   `Pkg.add` (`JSON = "1.8.0"`, `Jutul = "0.4.31"`, `JutulDarcy = "0.3.11"`, `SHA = "0.7.0"`),
   а не диапазонами из плана. `julia = "1.12"` добавлена в `[compat]` вручную, так как
   `Pkg.add` её не проставляет.

   **Уточнение (fix round 1).** Записи без `=` в `[compat]` Pkg — это **не** точные пины, а
   caret-диапазоны: `"1.8.0"` означает `[1.8.0, 2.0.0)`, `"0.4.31"` — `[0.4.31, 0.5.0)`. Так
   изначально записанный `Pkg.compat(...)` только задавал нижнюю границу, и последующий
   `Pkg.add`/`Pkg.develop` (например, подключение `SOReconSimulator/` в E05) мог бы молча
   переразрешить `JutulDarcy` на другой `0.3.x` или `JSON` на `1.9.x` без изменения
   `Project.toml`, противореча заявлению о зафиксированном окружении. Исправлено: `JSON`,
   `Jutul`, `JutulDarcy` и `SHA` теперь записаны с префиксом `=` (`JSON = "=1.8.0"` и т.д.) —
   это настоящие точные пины, и `Pkg.resolve()` не меняет разрешённые версии. Запись
   `julia = "1.12"` **остаётся диапазоном** (таково соглашение Pkg для ключа `julia` —
   он не поддерживает `=`-префикс так же); точный patch Julia фиксируется отдельно, в
   `.julia-version`, а не в `[compat]`.

2. **`run_smoke(case::Dict)` → `run_smoke(case::AbstractDict)`.** Установленная `JSON v1.8.0`
   парсит JSON-объект в `JSON.Object{String,Any}`, а не в `Dict` — это `AbstractDict`, но не
   `Dict`, поэтому сигнатура `run_smoke(case::Dict)` из плана никогда не подходит и диспетчеризация
   падает во время выполнения (`MethodError`). Исправление ограничено именем/типом аннотации
   аргумента: `::Dict` заменено на `::AbstractDict`. Обе исторические семантики (`JSON.jl 0.21`
   возвращал `Dict`, `JSON.jl 1.x` возвращает `JSON.Object`) совместимы с `AbstractDict`.
   Физика и структура кейса не менялись. Остальные аннотации (`parse_cli(args::Vector{String})`,
   `main(args::Vector{String})`) не менялись — `ARGS` действительно `Vector{String}`.

Вызовы `setup_vertical_well`, `pore_volume` и индексация `wd[:Producer, :orat]` /
`wd[:Injector, :wrat]` в установленной версии `JutulDarcy v0.3.11` работают без изменений —
переименований API этих вызовов не потребовалось.

3. **`Project.toml` перестал быть пакетом (найдено в Task 13, Step 5).** Черновик объявлял
   `name = "SOReconEnv"`, `uuid`, `version`, но каталога `julia/src/` с модулем не создавал.
   Для Pkg это пакет без исходника: `Pkg.precompile()` (шаг 2/8 `scripts/gate.sh` и цель
   `make setup-julia`) падал с `Missing source file for Base.PkgId(... "SOReconEnv")`, а
   `Manifest.toml` содержал фантомную запись `[[deps.SOReconEnv]] path = "."`. Ошибка не
   всплывала раньше, потому что `so-recon smoke` использует `--project=julia` только для
   загрузки зависимостей и сам проект не прекомпилирует.

   Исправление: `name`/`uuid`/`version` удалены — `julia/` теперь обычный environment
   project, каким он и описан выше. После `Pkg.resolve()` в `Manifest.toml` изменились
   ровно две вещи: `project_hash` и удаление записи `[[deps.SOReconEnv]]`. **Ни одна версия
   пакета не изменилась** — точные `=`-пины в `[compat]` этого не позволяют. Следствие:
   `julia_manifest_sha256` и `environment_lock_hash` в `reports/environment_report.md`
   пересчитаны; результаты smoke (`cumulative_oil_m3`, `mean_so_final`) не изменились.
