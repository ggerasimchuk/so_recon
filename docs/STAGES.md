# STAGES.md — этапность реализации SO-RECON

**Версия:** 3.0  
**Дата:** 13 сентября 2026 г.  
**Нормативная спецификация:** `SPEC.md`  
**Назначение:** последовательность разработки, проверки и научной валидации проекта «Локализация остаточных запасов нефти с применением методов машинного обучения».

---

# 1. Как читать этот документ

`SPEC.md` отвечает на вопрос **«что именно строится и почему»**.  
`STAGES.md` отвечает на вопрос **«в какой последовательности это реализовывать»**.

Это не подробный программный план. Для каждого этапа далее создаётся отдельный implementation plan с конкретными файлами, функциями, тестами и командами. Нельзя просить агента реализовать весь проект одним запросом: сначала завершается и принимается текущий этап, затем пишется план следующего.

Статусы этапа:

- `NOT_RUN` — работа не начиналась;
- `IN_PROGRESS` — выполняется;
- `PASS` — все обязательные выходы и проверки завершены;
- `PASS_WITH_LIMITATIONS` — этап пригоден для ограниченного продолжения, ограничения перечислены явно;
- `FAIL` — критический gate не пройден, зависимые этапы заблокированы.

Для каждого этапа создаётся отчёт `reports/stages/E##.md`, в котором фиксируются:

- версия `SPEC.md` и конфигурации;
- входные артефакты и их hashes;
- выполненные команды;
- результаты тестов и численных проверок;
- созданные артефакты;
- найденные ограничения;
- статус этапа;
- что разрешено передать следующему этапу.

---

# 2. Проект в пяти крупных фазах

| Фаза | Смысл | Этапы | Что появляется в конце |
|---|---|---|---|
| I. Основание и данные | Сделать данные причинно и физически корректными | `E00–E03` | Канонические таблицы, история подключений, пилот и замороженный эксперимент |
| II. Физическая модель | Построить проверенный forward model и обычную инверсию | `E04–E07` | Геологический ensemble, JutulDarcy-модель, likelihood и сильные baselines |
| III. Обучаемая инверсия | Создать synthetic truth, обучить ML и выполнить exact correction | `E08–E10` | Graph-temporal FMPE proposal и физически проверенный posterior |
| IV. Научное доказательство | Проверить метод на закрытой синтетике и на реальной истории | `E11–E12` | Ответ, восстанавливается ли `So`, помогает ли ML и что переносится на поле |
| V. Инженерный результат | Выпустить карты, сценарии бурения и материалы ВКР/статьи | `E13–E14` | Карты с uncertainty, ranking кандидатов, VoI и воспроизводимый release |

Самая важная логика:

```text
данные
  → геология и подключения
  → проверенный физический симулятор
  → обычная физическая инверсия
  → synthetic truth и observability
  → ML posterior proposal
  → exact physical correction
  → закрытая проверка
  → реальная реконструкция
  → карты и инженерные сценарии
```

Нейросеть начинается только после того, как физическая задача уже формально определена и существует baseline без ML.

---

# 3. Граф зависимостей

```text
                         ┌──────────────→ E05 JutulDarcy verification ───────┐
E00 Foundation ─→ E01 Data ─→ E02 History ─→ E03 Experiment lock ─→ E04 Geology
                                                │                            │
                                                └────────────────────────────┴─→ E06 Integrated forward
                                                                                  │
                                                                                  v
                                                                            E07 Inverse baselines
                                                                                  │
                                                                                  v
                                                                            E08 Synthetic corpus
                                                                                  │
                                                                                  v
                                                                            E09 ML proposal
                                                                                  │
                                                                                  v
                                                                            E10 Exact posterior
                                                                                  │
                                                                                  v
                                                                            E11 Closed synthetic test
                                                                                  │
                                                                                  v
                                                                            E12 Real-field inference
                                                                                  │
                                                                                  v
                                                                            E13 Maps and support
                                                                                  │
                                                                                  v
                                                                            E14 Decisions and release
```

`E05` можно вести параллельно с `E01–E04` на искусственных моделях. Интеграция с реальным сектором начинается только в `E06`.

---

# 4. Контрольные вехи

## M1 — данные готовы (`E03`)

Известно, какие записи допустимы, какие скважины и слои входят в пилот, какие даты и режимы используются, что скрыто в тесте и какой результат считается успехом.

## M2 — физический контур готов (`E07`)

JutulDarcy воспроизводимо считает сектор, сохраняет баланс и restart, likelihood определена, а задача уже решается обычными методами без ML.

## M3 — полный метод готов (`E10`)

Нейросеть предлагает распределение параметров, exact SMC корректирует его по физической likelihood, все финальные частицы имеют полноценный forward run.

## M4 — научный вывод получен (`E12`)

На закрытой синтетике измерено качество восстановления истинной `So`; на реальных данных выполнены backtests и получен допустимый posterior либо доказано ограничение наблюдаемости.

## M5 — выпуск готов (`E14`)

Сформированы карты на поддержанном масштабе, контрфактические сценарии, материалы статьи/ВКР и воспроизводимый пакет.

---

# 5. Подробные этапы

## E00. Основание проекта и воспроизводимое окружение

**Цель:** создать техническую основу, на которой все последующие расчёты воспроизводятся и имеют полную lineage.

**Основная работа:**

- создать структуру Python/Julia-проекта и typed configuration;
- зафиксировать Python, Julia, JutulDarcy и зависимости lock-файлами;
- определить каталоги raw/interim/processed/artifacts/reports;
- реализовать manifest исходных файлов, configs и environment;
- создать минимальный synthetic fixture и smoke test;
- определить единый CLI/run ID и правила логирования.

**Ключевые выходы:**

- `pyproject.toml` и lock-файл Python;
- `Project.toml` и `Manifest.toml` Julia;
- каркас `src/so_recon/`, `julia/`, `tests/`, `configs/`, `reports/`;
- `source_manifest.json`;
- `environment_report.md`;
- `reports/stages/E00.md`.

**Gate:** чистая установка выполняет один детерминированный fixture; версии и hashes сохраняются; код не зависит от личных абсолютных путей.

**Не входит:** чтение всех промысловых данных, построение модели пласта, обучение ML.

---

## E01. Специализированный ETL и независимый аудит пяти CSV

**Зависимость:** `E00`.

**Цель:** получить канонические таблицы без потери смысла и без молчаливого исправления исходных данных.

**Основная работа:**

- реализовать отдельные parsers для `coords.csv`, `gis.csv`, `mer.csv`, `perf.csv`, `plastoper.csv`;
- учесть разную ширину строк GIS, кодировки, десятичную запятую, BOM и хвостовые разделители;
- сохранить исходный ID и нормализованный ID без объединения отдельных стволов;
- привести единицы и разделить физический ноль, отсутствие определения и ошибку;
- построить словари колонок, пластов и источников координат;
- сверить строки, ключи, даты, объёмы, массы и пересечения фондов.

**Ключевые выходы:**

- `data/interim/well_registry.parquet`;
- `data/interim/geology_intervals.parquet`;
- `data/interim/well_month.parquet`;
- `data/interim/completion_events.parquet`;
- `data/interim/coordinate_registry.parquet`;
- `data_dictionary.md`, `formation_dictionary.yml`, `identity_crosswalk.parquet`;
- обновлённый machine-readable data audit;
- `reports/stages/E01.md`.

**Gate:** каждое удаление или преобразование объяснимо; контрольные суммы МЭР совпадают с источником в заданном допуске; известные ловушки чтения покрыты golden tests.

**Не входит:** трактовка `So_log` как текущей `So`, разнесение МЭР по пластам, выбор пилотного участка.

---

## E02. История подключений, управления и наблюдений

**Зависимость:** `E01`.

**Цель:** восстановить, что было открыто, закрыто и эксплуатировалось в каждый момент, а также строго разделить controls и observations.

**Основная работа:**

- реализовать completion state machine по событиям перфорации, дострела, перестрела, отключения и изоляции;
- объединять перекрывающиеся интервалы без двойного учёта толщины;
- восстановить помесячную активность пластовых групп и её uncertainty;
- отдельно учесть изменения списков пластов после июня 2021 года;
- построить месячные controls: жидкость, закачка, режим и часы работы;
- построить observations: oil/water split, watercut, GIS observations и quality masks;
- создать availability ledger с physical time, available time и ingest time;
- реализовать leakage tests для временных и new-well режимов.

**Ключевые выходы:**

- `completion_state_month.parquet`;
- `control_schedule.parquet`;
- `observation_registry.parquet`;
- `availability_ledger.parquet`;
- `completion_scenarios.parquet`;
- `history_qc.md`;
- `reports/stages/E02.md`.

**Gate:** месячные controls воспроизводят исходные объёмы; закрытый интервал не даёт поток; пропуск, нулевой поток и отсутствие коллектора различаются; будущая фазовая информация не попадает в control.

**Не входит:** history matching, выбор параметров prior, ML-признаки.

---

## E03. Выбор пилота и фиксация научного эксперимента

**Зависимость:** `E02`.

**Цель:** до разработки моделей определить область, даты, разбиения, информационные режимы и критерии успеха.

**Основная работа:**

- выбрать пилотный сектор по покрытию данных и динамической информативности, а не по будущему качеству модели;
- определить `core`, `buffer`, внешние скважины и возможный граничный обмен;
- зафиксировать девять слоёв Д1 и допустимые агрегаты верх/низ/суммарный Д1;
- выбрать основную дату состояния 31.12.2020 и правила сценария 31.12.2024;
- зафиксировать temporal backtest, spatial blocks и held-out well families;
- определить `pre_drill` и `post_drill_pre_flow` режимы новых скважин;
- создать parent-level split для синтетики;
- закрыть hashes наборов `inverse_test`, `end_to_end_confirmatory` и field holdout;
- зафиксировать primary metrics, budgets и gates до model selection.

**Ключевые выходы:**

- `sector_definition.yml`;
- `core_buffer_geometry.*`;
- `split_manifest.json`;
- `experiment_protocol.md`;
- `closed_set_manifest.json`;
- `reports/sector_selection.md`;
- `reports/stages/E03.md`.

**Gate:** выбранный сектор имеет достаточный объём наблюдаемой истории; все будущие данные явно разрешены или запрещены; тест нельзя менять без новой версии протокола.

**Не входит:** подбор лучшей архитектуры по закрытому тесту, финальное место бурения.

---

## E04. Условный геологический ensemble и начальное состояние

**Зависимости:** `E01`, `E03`.

**Цель:** построить не одну гладкую карту, а физически допустимое распределение статических моделей, задающее `PV`, `S_oi` и геологическую неопределённость.

**Основная работа:**

- построить структурные поверхности и вертикальную схему девяти пластов;
- задать collector probability, effective thickness, porosity и permeability;
- сохранить статистические связи ФЕС и мелкомасштабную вариабельность;
- определить ВНК, водяную область, lateral boundaries и альтернативные geometry families;
- построить bias-aware prior начальной нефтенасыщенности;
- задать сценарии `Sorw`, PVT и относительных проницаемостей с provenance;
- сформировать низко- и высокочастотные компоненты статической неопределённости;
- выполнить spatial cross-validation и prior geometry checks.

**Ключевые выходы:**

- `geology/grid_definition.*`;
- `geology/static_ensemble.zarr`;
- `geology/family_registry.parquet`;
- `geology/prior_config.yml`;
- `geology/volume_balance.parquet`;
- `reports/geology_validation.md`;
- `reports/stages/E04.md`.

**Gate:** поверхности не пересекаются; составные пласты не создают двойной `PV`; ансамбль сохраняет допустимые распределения и покрывает наблюдаемую геологию; uncertainty не заменена одной интерполяцией.

**Не входит:** подгонка геологии по production history, финальный posterior.

---

## E05. Адаптер JutulDarcy и верификация прямой физики

**Зависимость:** `E00`; может выполняться параллельно `E01–E04`.

**Цель:** получить надёжный operational forward simulator до подключения реального сектора.

**Основная работа:**

- реализовать интерфейсы `build_case`, `run_case`, restart и adjoint gradient;
- проверить однофазные и двухфазные малые fixtures;
- проверить producer/injector controls без раскрытия будущего phase split;
- проверить completion masks, shut/stop и смену ролей;
- реализовать месячные интегралы нефти, воды и закачки;
- проверить component/material balance, grid/time convergence и restart equivalence;
- проверить adjoint gradients конечными разностями;
- классифицировать numerical failure и physical infeasibility.

**Ключевые выходы:**

- `src/so_recon/simulator/`;
- `julia/SOReconSimulator/`;
- набор simulation fixtures;
- `simulator_manifest.json`;
- `reports/simulator_verification.md`;
- `reports/stages/E05.md`.

**Gate:** balance, restart, monthly integrals, controls и gradients проходят установленные критерии; частичный output не принимается как успешный расчёт.

**Не входит:** ручная подстройка реального объекта, ML, финальная производительность.

---

## E06. Интегрированная физическая модель пилотного сектора

**Зависимости:** `E02`, `E04`, `E05`.

**Цель:** связать геологию, скважины, историю и JutulDarcy в первый полный forward model реального сектора.

**Основная работа:**

- преобразовать каждый static realization в Jutul grid/rock model;
- подключить скважины и completion schedule;
- задать PVT, ОФП, initial state, aquifer/boundary families и controls;
- реализовать full-history run и restart на целевых датах;
- построить prior predictive ensemble;
- сравнить диапазон рассчитанных watercut/phase volumes с реальной историей;
- диагностировать structural mismatch по скважинам, периодам и model families;
- измерить стоимость forward run и пригодность параллелизма/cache.

**Ключевые выходы:**

- `simulation_case_registry.parquet`;
- `prior_predictive_ensemble.zarr`;
- `prior_predictive_metrics.parquet`;
- `reports/integrated_forward.md`;
- `reports/prior_predictive.md`;
- `reports/stages/E06.md`.

**Gate:** сектор устойчиво считается; баланс соблюдён; prior predictive envelope хотя бы частично покрывает ключевые реальные отклики. Если покрытие отсутствует системно, исправляются `E04–E06`, а не увеличивается сеть.

**Не входит:** posterior conditioning и утверждение о текущей `So`.

---

## E07. Вероятностная обратная задача и same-physics baselines

**Зависимость:** `E06`.

**Цель:** полностью определить inverse problem и решить её сильными методами без обучаемого proposal.

**Основная работа:**

- определить низкоразмерную латентную переменную `z`, nuisance `ξ`, transforms и Jacobians;
- реализовать deterministic decoder физических параметров;
- задать normalized likelihood для динамики, GIS bias и completion uncertainty;
- проверить likelihood на synthetic residuals и известных параметрах;
- реализовать `B0` static prior;
- реализовать `B1` multi-start adjoint MAP;
- реализовать `B2` ES-MDA;
- реализовать `B3` prior-start tempered SMC;
- сравнить методы на одинаковой физике и одинаковом observation contract.

**Ключевые выходы:**

- `parameter_registry.yml`;
- `prior_and_transforms`;
- `observation_model.yml`;
- `likelihood_diagnostics.parquet`;
- `baseline_results/{B0,B1,B2,B3}/`;
- `reports/inverse_baselines.md`;
- `reports/stages/E07.md`.

**Gate:** toy inverse problems восстанавливаются; densities конечны и согласованы; B1–B3 выполняют full forward runs; posterior predictive проверен; существует реальный baseline, относительно которого измеряется вклад ML.

**Не входит:** обучение нейросети, выдача финальной карты.

---

## E08. Synthetic corpus и исследование наблюдаемости

**Зависимость:** `E07`.

**Цель:** создать честную обучающую и проверочную среду, где истинная `So` известна и можно измерить, что вообще восстанавливается из данных данного типа.

**Основная работа:**

- сэмплировать parent cases из геологических и физических families;
- запускать JutulDarcy и сохранять истинные states/parameters;
- генерировать доступные `G, U, Y, events, masks`, соответствующие реальным CSV;
- моделировать шум, пропуски, epoch bias и completion errors;
- включить ID, held-out и structural-stress families;
- разделить train/validation/test только на parent-level;
- реализовать простую summary-SBI baseline `B4`;
- построить observability curves, ambiguity pairs и resolution diagnostics;
- заморозить confirmatory corpus до разработки окончательной ML-модели.

**Ключевые выходы:**

- `synthetic/corpus.zarr`;
- `synthetic/parent_manifest.parquet`;
- `synthetic/split_manifest.json`;
- защищённые truth artifacts;
- `baseline_results/B4/`;
- `reports/observability_development.md`;
- `reports/stages/E08.md`.

**Gate:** нет parent leakage; входы не богаче реальных данных; truth скрыта от модели; baseline-метрики воспроизводимы; подтверждено, на каком масштабе задача хотя бы потенциально наблюдаема.

**Не входит:** открытие confirmatory test для выбора архитектуры.

---

## E09. Graph-temporal FMPE posterior proposal

**Зависимость:** `E08`.

**Цель:** обучить центральную ML-модель, которая по истории новой задачи предлагает многомодальное распределение физических параметров.

**Основная работа:**

- реализовать temporal encoder месячных историй и masks;
- реализовать set pooling и information graph скважин;
- реализовать conditional Flow Matching Posterior Estimator;
- обеспечить согласованные `sample` и `log_prob`;
- обучить на synthetic train и выбирать модель только по validation;
- выполнить curriculum и targeted simulation при выявленном OOD;
- проверить toy multimodality, SBC, coverage и posterior predictive;
- выполнить абляции: без графа, без temporal encoder, summaries-only;
- сохранить несколько seeds/checkpoints и calibration model.

**Ключевые выходы:**

- `models/fmpe/`;
- `ml_checkpoint_manifest.json`;
- `normalization_and_schema.json`;
- `proposal_metrics.parquet`;
- `reports/ml_training.md`;
- `reports/ml_ablations.md`;
- `reports/stages/E09.md`.

**Gate:** proposal не использует hidden truth; sample/log_prob согласованы; coverage приемлема на validation; пустая история не даёт ложной уверенности; graph остаётся только при доказанном вкладе.

**Не входит:** использование сырого ML posterior как конечной карты реального объекта.

---

## E10. Exact posterior correction: defensive tempered SMC

**Зависимости:** `E07`, `E09`.

**Цель:** превратить ML proposal в физически и вероятностно корректный posterior.

**Основная работа:**

- смешать ML proposal с prior в defensive proposal;
- вычислять `log prior`, `log proposal` и exact Jutul likelihood;
- реализовать adaptive likelihood tempering;
- реализовать ESS-based resampling и ancestry;
- реализовать rejuvenation, включая adjoint-assisted MALA для непрерывных параметров;
- обрабатывать дискретные geology/completion families;
- сохранять все numerical failures и rejected particles;
- выполнять exact forward run для каждой финальной частицы;
- сравнить learned-proposal SMC с prior-start `B3` при одинаковых budgets.

**Ключевые выходы:**

- `src/so_recon/inference/`;
- `posterior_ensemble` schema;
- `smc_diagnostics.parquet`;
- `failure_registry.parquet`;
- `reports/exact_inference_verification.md`;
- `reports/stages/E10.md`.

**Gate:** аналитические и toy tests весов пройдены; ESS/max-weight/mode gates соблюдены; все финальные частицы имеют successful exact forward; ML не изменяет target distribution, а только эффективность предложения.

**Не входит:** открытие закрытого confirmatory test ради исправления метода.

---

## E11. Закрытая synthetic-валидация всей системы

**Зависимость:** `E10`.

**Цель:** один раз проверить научные гипотезы на случаях с известной истинной `So`.

**Основная работа:**

- открыть замороженные `inverse_test` и `end_to_end_confirmatory`;
- сравнить `B0–B4` и SO-RECON при одинаковой физике;
- сравнить fixed wall-clock и fixed simulation budgets;
- измерить ошибки `So`, `PV·So`, мобильного объёма и зональных рангов;
- проверить calibration, coverage, ESS, mode retention и OOD;
- провести structural-stress tests;
- определить resolution kernel и минимальный допустимый report support;
- присвоить статусы H1–H3 и выбрать Outcome A/B/C/D для synthetic evidence;
- после просмотра test не менять метод без создания нового confirmatory family.

**Ключевые выходы:**

- `confirmatory_predictions/`;
- `confirmatory_metrics.parquet`;
- `resolution_report.md`;
- `hypothesis_status.md`;
- `reports/end_to_end_validation.md`;
- `reports/stages/E11.md`.

**Gate:** заранее заданные primary criteria рассчитаны полностью; показаны успешные и неуспешные cases; разрешение итоговых карт определяется тестом, а не размером grid.

**Развилка:**

- состояние восстанавливается — переход к полевой реконструкции;
- динамика прогнозируется, но `So` неоднозначна — перейти к Outcome C и VoI;
- даже агрегированные зоны неразличимы — перейти к Outcome D, не выпускать ложную карту.

---

## E12. Реальная реконструкция и полевые backtests

**Зависимость:** `E11`.

**Цель:** применить замороженный метод к реальному сектору и проверить его там, где существуют наблюдаемые будущие отклики.

**Основная работа:**

- оценить OOD реального контекста относительно synthetic corpus;
- при допустимом OOD выполнить posterior conditioning до 31.12.2020;
- провести заранее выбранный temporal backtest;
- провести spatial/held-out-well tests;
- провести new-well tests в режимах `pre_drill` и `post_drill_pre_flow`;
- сравнить posterior predictive watercut/oil fractions с реальностью;
- выполнить чувствительность к geology, boundary, completion, PVT/relperm и GIS bias;
- при допустимости построить сценарный posterior на 31.12.2024;
- определить, какие зоны informed, prior-dominated или unsupported.

**Ключевые выходы:**

- `field_posterior_2020/`;
- `field_backtest_predictions.parquet`;
- `new_well_backtests.parquet`;
- `field_sensitivity_budget.parquet`;
- `field_posterior_2024_scenarios/` при прохождении условий;
- `reports/field_validation.md`;
- `reports/stages/E12.md`.

**Gate:** реальные predictive checks не противоречат истории; OOD и sensitivity отражены; отсутствие независимой истинной `So` не маскируется процентной «точностью поля».

**Не входит:** ручное исправление карт по фактической поздней воде после завершения test.

---

## E13. Карты состояния, uncertainty, support и abstention

**Зависимость:** `E12`.

**Цель:** преобразовать posterior particles в инженерно читаемые пространственные продукты без создания дополнительной информации интерполяцией.

**Основная работа:**

- агрегировать physical states на масштабе, подтверждённом `E11`;
- построить `So q10/q50/q90` и exceedance probabilities;
- рассчитать текущий нефтенасыщенный объём и объём выше `Sorw` по particles;
- построить карты sweep, posterior width и variance reduction;
- классифицировать support A/B/C/D;
- отметить prior-dominated, boundary-sensitive и completion-sensitive зоны;
- разложить uncertainty по источникам;
- реализовать `NO_LOCAL_QUANTITATIVE_ESTIMATE` по правилам `SPEC.md`;
- проверить quantile-of-sum и spatial aggregation.

**Ключевые выходы:**

- `maps/state_ensemble.zarr`;
- `maps/maps.nc`;
- `maps/map_cells.parquet`;
- `maps/support_class.parquet`;
- `map_method_card.md`;
- `uncertainty_budget.md`;
- `reports/stages/E13.md`.

**Gate:** каждая карта восходит к exact posterior particles; unsupported области не заполнены искусственными значениями; детализация не выше resolution test; единицы и объёмы закрывают баланс.

**Не входит:** объявление объёма выше `Sorw` утверждёнными извлекаемыми запасами.

---

## E14. Контрфактические сценарии, VoI и воспроизводимый выпуск

**Зависимость:** `E13`.

**Цель:** проверить практическую полезность локализации и собрать окончательный результат ВКР/статьи.

**Основная работа:**

- отфильтровать геологически и технологически допустимые точки-кандидаты;
- для каждой posterior particle выполнить paired base/intervention runs;
- считать дополнительную нефть и воду по всему сектору, а не только дебит новой скважины;
- проверить влияние на соседние скважины и конкуренцию за дренирование;
- оценить устойчивость ranking между model families и policies;
- реализовать abstention при смене знака эффекта или неустойчивом ранге;
- рассчитать value of information для давления, PLT и текущей `So`;
- выполнить clean synthetic end-to-end reproducibility run;
- подготовить figures, tables, model/data cards, thesis chapters и article outline;
- сформировать private scientific archive и public synthetic reproducibility package.

**Ключевые выходы:**

- `candidate_registry.parquet`;
- `counterfactual_results.parquet`;
- `candidate_ranking.parquet`;
- `data_acquisition_priorities.parquet`;
- `reports/decision_stability.md`;
- `reports/final_research_report.md`;
- release manifest и воспроизводимый synthetic example;
- материалы статьи и ВКР;
- `reports/stages/E14.md`.

**Gate:** ranking основан на paired physics runs; incremental production считается по сектору; риски и ограничения видны рядом с эффектом; все основные утверждения связаны с заранее определённым тестом.

---

# 6. Главные stop/go gates

## G1 — после E03: существует ли пригодный пилот?

Если нет сектора с достаточной историей, геологией и режимными изменениями, проект не переходит к масштабной реализации. Выбирается другой сектор или научная постановка ограничивается observability/VoI.

## G2 — после E06: работает ли физическая модель до ML?

Если prior predictive не покрывает реальную динамику и обнаружен системный mismatch, исправляются геология, границы, completion model или PVT/ОФП. Нейросеть не используется для маскировки ошибки forward model.

## G3 — после E07: существует ли честная baseline-инверсия?

Если B1–B3 не могут стабильно работать даже на toy и synthetic cases, ML не начинается. Сначала исправляется inverse formulation.

## G4 — после E11: восстанавливается ли скрытая `So` на synthetic truth?

Если нет, нельзя заявлять локализацию на реальном объекте. Допустимы Outcome C или D: predictive model, ambiguity analysis и программа дополнительных исследований.

## G5 — после E12: поддерживается ли реальный результат данными?

Если реальный context OOD, posterior prior-dominated или boundary/completion uncertainty определяет вывод, выпускается агрегированный или abstained result, а не точная карта.

---

# 7. Что можно выполнять параллельно

- `E05` можно выполнять параллельно `E01–E04`, используя synthetic fixtures.
- В `E04` можно параллельно разрабатывать structural geometry и conditional property ensembles, но общий `PV` проверяется совместно.
- В `E08` generation workers могут работать параллельно после заморозки parent manifests.
- В `E09` обучение разных seeds и абляций параллельно, но test остаётся закрытым.
- В `E12` sensitivity families могут рассчитываться параллельно после заморозки field method.
- В `E14` paired candidate simulations параллелятся по particles и candidates с общими manifests.

Нельзя параллельно выполнять зависимый этап на неподтверждённых интерфейсах, если затем результаты будут использоваться как научные evidence.

---

# 8. Как просить Codex писать планы

Для каждого этапа используется отдельный запрос. Базовая форма:

```text
Прочитай файлы /docs/README_SO_RECON.md, /docs/DATA_AUDIT.md,
/docs/RESEARCH.md, /docs/SPEC.md и /docs/STAGES.md.

Подготовь подробный implementation plan только для этапа E##.
Учитывай входные артефакты и gate этапа.
Не меняй архитектуру и критерии из SPEC.md.
Не реализуй последующие этапы.
План должен содержать структуру файлов, интерфейсы, тесты,
команды проверки, ожидаемые артефакты и отчёт reports/stages/E##.md.
```

После утверждения плана этап реализуется отдельно. Следующий план пишется только после появления отчёта и статуса текущего этапа.

Рекомендуемая первая последовательность запросов:

```text
1. Напиши план E00.
2. Реализуй E00 и проверь gate.
3. Напиши план E01 на фактическом состоянии репозитория.
4. Реализуй E01 и проверь gate.
5. Продолжай по одному этапу.
```

Не следует сразу просить «реализовать E00–E14»: агент потеряет границы интерфейсов, проверки и научные stop/go решения.

---

# 9. Минимальная логика итогового результата

Первое поле текущей `So` появляется только после `E12`, а публикуемая карта — после `E13`.

До этого проект последовательно доказывает:

1. данные прочитаны правильно;
2. история подключений не содержит временной утечки;
3. геологический объём физически допустим;
4. симулятор решает задачу и сохраняет баланс;
5. inverse problem работает без ML;
6. скрытая `So` восстанавливается на synthetic truth;
7. ML действительно улучшает точность или стоимость;
8. exact correction не допускает неподдержанных ML-состояний;
9. реальная история согласуется с posterior;
10. детализация карты соответствует фактической разрешающей способности данных.

Если цепочка обрывается, проект переходит на предусмотренный честный результат Outcome B, C или D, а не продолжает строить визуально убедительную, но недоказанную карту.
