# DECISIONS.md — переход SO-RECON 3.0 → 4.0 без перезапуска проекта

**Дата:** 13 сентября 2026 г. **Статус:** решения согласованной редакции [SPEC.md](SPEC.md) и [STAGES.md](STAGES.md).

## 1. Решение и область изменения

Сохраняется существующий проект и основание E00. В SPEC 4.0 интегрированы малый сквозной опыт, conditional NSF, явный SMC bridge, выбор полевой физики по sensitivity и вычислительные профили из присланных материалов. Из SPEC 3.0 сохранены научные обязательства по state/localization/volume/field validation, provenance и верификации. Это изменение требований будущих этапов; оно само не реализует worker, ML, SMC или полевую модель.

Входные материалы: действовавшие `docs/SPEC.md`/`docs/STAGES.md` 3.0 и присланные `SPEC_v2.md`, `STAGES_v2.md`, `COMPUTE_PROFILES_v2.md` с внутренней версией 4.0. Присланные файлы не редактируются и не являются нормативными копиями. Единственные действующие документы находятся в `docs/`; текущий вход в проект — корневой README.

Дерево перед согласованием: Git commit `23f84a3611ab466b2ae4f43c0df291f7ee0c42c6`. SPEC 3.0 доступна в истории Git по этому commit; её SHA-256 — `5904e5f5f85070da5d6cbac5ab03f0a35d22fab5ac522f9f4409f62c55903eb8`. Старые ссылки на разделы внутри E00 кода/плана относятся к этой версии. Ни старые run records, ни frozen smoke expectations, ни исходные CSV не переписываются при согласовании документов.

При финальной сверке также учтены отдельные коммиты завершения E00 `5e70ab0` и `aa78e96`: уточнены комментарий CLI и отчёт о gate, без изменения runtime/config 3.0. Они сохраняются и не требуют повторного проектирования foundation.

## 2. Что принято, сохранено и уточнено

| Область | Решение 4.0 | Причина и граница |
|---|---|---|
| Цель и роль ML | Скрытая текущая So из физической вероятностной модели; learned inverse density предлагает параметры | Центральная роль ML уже была в 3.0; выигрыш проверяется при конечном бюджете |
| Последовательность | E01–E03 дают малую физику, inverse reference и полный ML+SMC цикл; E07 повторяет его на полеподобном пилоте | Снижение риска до большого корпуса; ранние результаты exploratory |
| E00 | Сохранён в исходном объёме; новые runtime/resource требования входят в E01.0 | Не требовать задним числом полного posterior schema или persistent worker от foundation |
| ML family | Conditional NSF как основной выбор; FMPE — альтернатива через ADR/benchmark | Аналитическая change-of-variables density удобна для repeated correction; превосходство над FMPE не доказано |
| Encoder | Малый graph-temporal старт; summary/temporal-set обязательны для сравнения | Graph остаётся в final model только при измеримой пользе |
| SMC | Bridge от defensive mixture r к p0·L; полные веса/MH; residual сохранён | Старый prior-tempering с начальным p0/r также математически допустим; выбран один operational алгоритм |
| Likelihood | Bounded-bin Student-t с явным noise generator и временной зависимостью | Это новая модель ошибок, поэтому E02 имеет отдельный normalization/recovery/coverage gate |
| Физика | OW для P0/P1; PhysicsDecision по gas/PVT/Pc для поля | Отсутствие измерений газа не доказывает его несущественность; BO без защищаемого PVT тоже не гарантия |
| История | Soi до разработки отдельно от S0 начала расчёта | Нельзя обнулить предысторию при сокращении окна или переиспользовать state другой частицы |
| Даты | t_ref выбирается по CompletenessReport; 2020 остаётся кандидатом; 2024 — поздний условный продукт | Смена даты не выбирается по качеству карты; правила в SPEC §4.2.1 |
| Основная метрика | **Сохранена MAE из 3.0**; RMSE дополнительная и ранняя exploratory | Не менять основной научный критерий только из-за новой редакции текста |
| Локализация и объёмы | Отдельные C_LOCALIZATION/C_VOLUME/C_FIELD и resolution tests | Снижение средней ошибки So не доказывает правильный выбор зон или нефтяной инвентарь |
| Пороги | Проектные пороги 3.0 сохранены с явными знаменателями, support и statistical uncertainty | Не измеренные свойства поля; изменение только до закрытого теста с ADR |
| ES-MDA | Сильный практический approximate baseline; отдельный Gaussian-compatible control | Не выдавать standard ES-MDA за точный sampler bounded Student-t target |
| Геология/карты | Сохранён residual, общий spatial support, collector presence отдельно от So, quantile-of-sum | Исключает искусственное сужение uncertainty и ошибки агрегации |
| Хранилище | Parquet + HDF5 shards и native restart | Канонические/крупные scientific arrays ещё не реализованы; foundation JSON/Parquet сохраняются, обязательная миграция в Zarr/NetCDF не нужна |
| Пути/инфраструктура | Существующие `julia/`, `src/so_recon/`, `artifacts/runs/`, `docs/superpowers/plans/` | Нет переезда в `sim/julia/`, второго registry или новой платформы |
| Ограничения | Resumable jobs, полная стоимость, последовательные тяжёлые фазы | 24 часа — выбираемый лимит сессии, не обещанный срок исследования |

Сохранённые требования 3.0 перенесены по смыслу: старые §§7/19 → SPEC §§6/17 (данные и provenance); §§10/20 → §§8/23 (физика и тесты); §§16/17 → §§14/15/23 (научная проверка, разрешение и отказ от ложной точности); §§15/18 → §§15/16 (карты/решения/дополнительные измерения). Нумерация больше не совпадает. Удалены некорректные универсальные правила вроде обязательного повторного Jacobian физического renderer или вывода prior dominance только из малого сдвига среднего. Сохранённая архивная формулировка не конкурирует с 4.0.

## 3. Совместимость E00: факты и задачи следующего плана

Основание проверено чтением текущего кода, конфигурации и [отчёта E00](../reports/stages/E00.md). Его PASS относится к прежним требованиям. Настоящий документ не заявляет повторного полного `make gate` или проверки будущих возможностей.

| Что уже есть | Свидетельство | Как использовать дальше |
|---|---|---|
| Python/Julia locked environment | `pyproject.toml`, `uv.lock`, `julia/Project.toml`, `julia/Manifest.toml`, версии runtime в соответствующих файлах | Сохранить; новые зависимости добавлять по потребителю, затем проверять smoke |
| Typed paths/config и CLI | `src/so_recon/paths.py`, `config/`, `cli.py` | Расширять существующие interfaces |
| Artifact/run registry и журналирование | `registry/`, `runner.py`, `logging_setup.py` | Сохранять immutable outputs, resolved config, hashes и failure-lineage |
| Source/environment manifests | `registry/source_manifest.py`, `environment/report.py`, `reports/manifests/` | Не подменять физический счёт строк полным ETL; machine stamps остаются отдельными |
| Один процесс Julia на smoke-запуск | `simulator/julia_bridge.py`: `SubprocessJuliaLauncher.launch` использует `subprocess.run` | Допустимо для E00 smoke; многократные physics jobs в E01 получают отдельный persistent launcher |
| Реальный детерминированный 1D smoke | `julia/smoke/smoke_case.jl`, `configs/smoke_expected.json`, integration test | Сохранить как regression foundation; это ещё не physics verification matrix |
| Версия 3.0 в runtime | `src/so_recon/__init__.py`, `config/schema.py`, `configs/project.yml` | Совместимость 4.0 требует кода, не только редактирования YAML |

### 3.1. E01.0 — узкое расширение foundation внутри E01

План E01 должен включать следующие задачи до серий physics jobs:

1. **Версии и lineage.** `ProjectConfig.spec_version` сейчас `Literal["3.0"]`, а runtime `SPEC_VERSION` также 3.0. `RunContext.start` маркирует run через константу, не через cfg. При добавлении 4.0 нужно согласовать package/config validation и весь путь `run.json`/source manifests. Legacy 3.0 smoke/config читается с исходным смыслом; новые 4.0 запуски не должны маркироваться 3.0, а старые — 4.0. Один version bump без анализа источника stamp недостаточен.
2. **Проверки совместимости.** Загрузка старого конфига, принятие нового, отклонение неизвестной версии; соответствие config и run/source metadata; корректный FAIL при ошибке до загрузки cfg; сохранение frozen E00 smoke. Новый тип артефакта получает schema version, старый не изменяется задним числом. Обновление детерминированных отчётов выполняется отдельно и объясняется, не используется для сокрытия регрессии.
3. **Минимальные контракты потребителя.** В E01 достаточно CaseBundle/ForwardResult/job descriptor с controls, units, hashes, status и restart metadata. DensitySchema/ThetaRecord вводятся в E02, proposal metadata — E03, а PosteriorBundle — вместе с inference. Не создавать в E00 все будущие классы без потребителя.
4. **Persistent worker.** На основе текущего поиска/запуска Julia реализовать процесс с несколькими изолированными jobs, job_id/result matching, timeout/exception handling и cleanup. Старый launcher можно сохранить для smoke. Проверить разные последовательные cases и отсутствие скрытого состояния между ними. Новые модули размещаются внутри `julia/` и текущего Python simulator package.
5. **Ресурсы.** До серий jobs добавить profile/config для RAM/disk/wall/forward limits, hardware/device probe и корректную классификацию resource failure. Реальные limits берутся из COMPUTE_PROFILES и фактического probe. Полный SMC checkpoint здесь не нужен: в E01 сохраняются completed job/results; beta/weights/RNG реализуются с SMC в E02.
6. **Прежняя проверка foundation.** Запустить подходящие существующие tests и smoke после реальных изменений. Не переписывать `smoke_expected.json`, чтобы скрыть несовместимость; изменение физики fixture требует отдельного решения. Добавить проверки новых contracts/worker/guards по SPEC §23. В этой документальной миграции исходный код не меняется.

Блок E01.0 не является отдельным длительным этапом или требованием заново реализовать E00. Его размер определяется минимальными потребностями E01; ML packages, весь ETL, MAP/ES-MDA и full posterior machinery сюда не входят. Если текущая работа над E00 уже добавила часть возможностей, следующий агент подтверждает их по коду/тестам и не дублирует.

### 3.2. Разделение статусов

Документация: SPEC/STAGES/COMPUTE_PROFILES **4.0**. Историческое основание и текущий runtime: **3.0 / E00.2**, пока не исполнен блок перехода. Это объявленная граница, а не скрытое противоречие. Можно писать план E01 сейчас; нельзя заявлять, что существующий executable уже выполняет SPEC 4.0.

Исторический [план E00](superpowers/plans/2026-09-13-e00-foundation.md) и [отчёт](../reports/stages/E00.md) не редактируются при принятии этой редакции. Корневые `*_v2.md` сохраняют роль присланных вариантов. Завершение текущего E00, согласование его Git-изменений и исполнение E01 остаются отдельными действиями.

## 4. Условия готовности следующего плана

План E01 читает этот документ, SPEC §§8–9, 17–18, 23, STAGES E01 и профили P0/P1; использует существующее дерево и включает только необходимые расширения foundation. Его конечные артефакты — проверенный малый forward adapter, synthetic truth, restart и runtime report. Он не обещает уже восстановленную inverse So: это E02/E03.

Последовательность остальных этапов и таблица соответствия старым номерам находятся в STAGES. Метрики/пороги E11 задаются SPEC §14.3; конкретные priors, выбранная дата, support, costs и baseline configs фиксируются в будущих E05–E10 до закрытой проверки. Отсутствующие полевые сведения перечисляются как inputs/gates, а не выдуманные defaults.
