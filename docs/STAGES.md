# STAGES.md — последовательность реализации SO-RECON

**Редакция 4.0 · 13.09.2026 · local-first.** Основа — SPEC.md, ресурсные пределы — COMPUTE_PROFILES.md. Это этапность и критерии приёмки, а не подробный план написания кода. Каждый этап получает отдельный план и отдельную проверку реализации.

## 1. Что изменилось в порядке работ

Полный малый метод появляется **до сложной полевой модели**. Сначала E00–E03 доказывают, что физический solver, likelihood, learned density, SMC и оценка So действительно работают вместе. Затем E04–E07 проверяют перенос постановки на структуру ваших данных. Только после этого создаются большие корпуса, усиливаются inference/baselines и выполняется закрытое исследование.

Первая синтетика — **E01**. Первая обратная задача с известной So — **E02**. Первая обученная сеть с физической коррекцией и сравнением истинной/восстановленной So — **E03**. Не нужно ждать E09/E10, чтобы увидеть результат метода.

Старые E01–E14 не совпадают с этой последовательностью. **E00 сохраняется в выполненном объёме SPEC 3.0**; статус подтверждается его историческим отчётом, а не переименованием версии. Недостающие runtime/worker/resource возможности добавляются внутри E01. Сопоставление кода и требований — [DECISIONS.md](DECISIONS.md). Новый план E00 и повторное создание основания не требуются.

## 2. Общая последовательность

| Этап | Результат | Зависимости | Основной профиль |
|---|---|---|---|
| **E00** | Существующее окружение, CLI, registry и Julia smoke по 3.0 | Исторический отчёт | Foundation smoke |
| **E01** | Совместимость foundation с 4.0, ресурсы, Jutul adapter и малая forward-синтетика | E00 | P0_VERIFY → P1_LOOP |
| **E02** | Likelihood, prior и корректная малая inverse-задача без ML | E01 | P0_VERIFY / P1_LOOP |
| **E03** | **Первый полный ML+SMC цикл, So truth и решение о продолжении** | E02 | P1_LOOP |
| **E04** | Проверенные канонические данные пяти CSV | E00 + совместимость E01.0 для запусков 4.0; далее независимо от physics E01–E03 | Data-only |
| **E05** | История подключений, пригодный пилот, режимы знания и источники | E04 | P2_PILOT design |
| **E06** | Геологический prior, полевая физика и PhysicsDecision | E01, E05 | P2_PILOT |
| **E07** | **Полеподобный малый сквозной опыт и новый budget gate** | E03, E06 | P2_PILOT |
| **E08** | Основной corpus и замороженный research protocol | E07 | P2/P3 |
| **E09** | Research NSF/encoder, development ablations и calibration | E08 | P3_RESEARCH |
| **E10** | Усиленная SMC-коррекция, adjoint и сильные baselines | E02, E06, E09 | P2/P3 |
| **E11** | Закрытое synthetic сравнение So, uncertainty и затрат | E09, E10 | P3 + P4 subset |
| **E12** | Реальная реконструкция и полевые backtests | E11; с допустимым статусом результата | P3 |
| **E13** | Карты So/нефти/неопределённости на поддержанном масштабе | E12 | Product-only + selected replays |
| **E14** | Парные инженерные сценарии и воспроизводимый выпуск | E13 | Selected posterior scenarios |

Параллельность допустима по независимым программным задачам, не означает одновременное обучение и несколько solver jobs в 24 ГБ. На одной машине resource scheduler имеет приоритет над количеством доступных агентов.

## 3. Вехи

**M0 / E01:** Jutul считает известный малый мир; получена истинная So, но обратная задача ещё не решена.

**M1 / E03:** весь метод работает на ноутбучном профиле; имеется отчёт «что восстановилось / что неоднозначно / помогает ли ML / сколько это стоит». Это первый момент принятия решения о целесообразности, не результат статьи.

**M2 / E07:** полеподобный пилот с вашей структурой наблюдений проходит малую проверку; получен защищаемый PhysicsDecision либо явно сценарный статус. Теперь можно обосновывать длительные запуски.

**M3 / E11:** на закрытых мирах известен научный результат сравнения ML с сильными методами; положительный или отрицательный вывод подтверждён данными.

**M4 / E14:** результат на реальном объекте имеет допустимую интерпретацию, карты и воспроизводимые материалы статьи/ВКР.

# 4. Карточки этапов

## E00 — существующее основание проекта

**Объём:** окружение, source manifests, typed config/paths, run/artifact registry, CLI/logging, deterministic fixture и сквозной Julia smoke. Выполнялся по 3.0; [план](superpowers/plans/2026-09-13-e00-foundation.md) и [отчёт](../reports/stages/E00.md) остаются историческими документами.

**Текущее свидетельство:** отчёт имеет PASS; проверка кода подтверждает наличие foundation-компонентов. Эта редакция не повторяет и не расширяет задним числом приёмку E00. Если его текущий исполнитель завершает исправления, он завершает именно согласованный объём 3.0; следующий план учитывает финальное дерево и тесты.

**Переиспользование:** `src/so_recon/config`, `paths.py`, `registry`, `runner.py`, `logging_setup.py`, `environment`, `simulator/julia_bridge.py`, `julia/smoke`, lockfiles и существующие tests. Окружение не пересоздаётся ради новой схемы каталогов. Новые зависимости добавляются при появлении потребителя и проверяются с существующим smoke.

**Не требовалось для прежнего E00:** persistent Julia worker, системные RAM/disk guards, полноценные CaseBundle/ThetaRecord/PosteriorBundle, ML, SMC, проверка физики. Их отсутствие не аннулирует foundation. Требования распределены по E01–E03, см. DECISIONS §3.

## E01 — физический adapter и первые synthetic states

**Цель:** получить настоящие малые состояния JutulDarcy. SPEC §§8–9, 17–18; профиль P0→P1.

**Вход:** существующий E00 и его фактическое состояние, [DECISIONS.md](DECISIONS.md), штатные примеры/документация закреплённого Jutul, явно учебные PVT/ОФП. План не зависит от готового ETL или полевых PVT.

**Первый блок плана — E01.0, совместимость основания (часть E01, не новый этап):** сохранить прежний smoke; расширить versioned config/lineage с поддержкой новых 4.0 запусков и чтением исторических 3.0; устранить расхождение cfg.spec_version и runtime stamping во всём пути run/source manifests; добавить минимальные CaseBundle/ForwardResult и ресурсный профиль. Persistent worker и guards создаются здесь перед сериями физических jobs. ThetaRecord/DensitySchema вводятся в E02, PosteriorBundle — вместе с SMC; полные будущие схемы не требуются заранее. Детальные задачи и проверки перехода перечислены в DECISIONS §3.

**Содержание:** persistent Julia worker; OW case; wells/rate/BHP limits; monthly phase integrals; gravity; open/close connections и роль producer/injector; initial state; minimal output и native restart; классификация численных/resource failures; ограниченный retry. Проверки: closed-cell balance, Buckley–Leverett в его области, hydrostatics, two-layer mixing/crossflow, restart equivalence. Создать неоднородный двухслойный P1 generator с sparse static observations и скрытой So. Малый BO capability test фиксируется отдельно и выполняется до PhysicsDecision, не обязан блокировать первый OW цикл.

**Артефакты:** verified OW adapter; P0 fixtures; первые P1 worlds с theta/controls/noiseless outputs/true states; physics report; cold/warm runtime и RAM; restart round-trip result.

**Приёмка:** старый smoke работает без подгонки frozen ожиданий; legacy/new конфиги и lineage имеют согласованные версии; resource/timeout failure не считается success. В физических проверках mock не заменяет решатель; задана жидкость, а не одновременно нефть и вода; месячные интегралы согласованы с балансом; закрытые соединения ведут себя правильно; новый job не наследует скрытое состояние предыдущего; ресурсы измерены; воспроизведение restart соответствует непрерывной траектории.

**Не входит:** вся полевая история, ML, MAP/ES-MDA, полный adjoint, OPM installation как обязательный барьер. Тест на простом мире не доказывает полевую точность.

## E02 — вероятностная обратная задача и ранняя наблюдаемость

**Цель:** проверить, что именно будем обучать и корректировать. SPEC §§5, 7, 10, 13; P0/P1.

**Вход:** E01, известный conditional prior учебного мира.

**Содержание:** DensitySchema и prior/renderer с сохранённым residual; bounded-bin Student-t observation model и **тот же** рекурсивный noise generator; даты/masks/нулевая жидкость; CPU Float64 log-density. Реализовать универсальный SMC bridge с r, которое пока может равняться prior, корректные веса/CESS/resampling, gradient-free MH/pCN, checkpoint beta/ancestry. Проверить analytic Gaussian и bimodal cases, missing-mode defensive mixture и likelihood normalization. На reduced физическом мире 1–4 неизвестных получить независимую reference inversion; на P1 исследовать T1 информативность и T2/T4 неоднозначность.

**Артефакты:** likelihood/noise module; density tests; SMC engine; prior-only/reference posterior; ambiguity examples; предварительные So metrics; budget full inverse.

**Приёмка:** пройдены normalization/generator/recovery проверки SPEC §23 для новой likelihood, включая 0/1, пропуски и date mixtures; sample/log-density consistent; latent/physical Jacobians не посчитаны дважды; beta достигает 1 либо честный incomplete status; toy reference согласуется; на пустом наблюдении нет искусственного сужения; state сравнивается на одном support; техническая ошибка не выдается за неидентифицируемость.

**Не входит:** full LIS, большой encoder, давление-псевдофича, fine-grid production inference. Нельзя требовать, чтобы все зоны маленького мира были идентифицируемы.

## E03 — первый полный ML + physical correction

**Цель:** ответить на вопрос пользователя «делаем ли мы то, что нужно» до крупного расчёта. SPEC §22, §§11–14; P1_LOOP.

**Вход:** E02, малая physics-синтетика и её explicit information model.

**Содержание:** generate стартовые 128 parents с последующим расширением только по learning curve; masks/parent split; маленький graph-temporal encoder и conditional NSF; discrete head для ограниченной гипотезы; NLL training; frozen CPU Float64 proposal; defensive mixture; **тот же** SMC из E02 с r от сети; true So maps, credible widths и одинаковый evaluator. Сравнить static prior, prior-start SMC, raw q и q+SMC; выполнить минимальную summary/temporal-set ablation. T1–T5 добавляются по одному, не все сложности одновременно.

**Артефакты:** обученный малый checkpoint; sample/log_prob/round-trip tests; `EARLY_RESULT.md`; таблица So RMSE, CRPS/width, phase fit, beta/modes/ancestry, cold/online budgets; reproducible command малого полного пути; список unresolved risks.

**Приёмка / G_LOCAL:** PASS_CODE и диагностированные state/ML outcomes по SPEC §22.4. Пример T1 должен показывать работу обратного восстановления, а T2/T4 — сохранение реальной неоднозначности. Прохождение скрипта и снижение NLL сами по себе не дают PROMISING_STATE. Необходим полный ledger, а не лишь время готовой сети.

**Решение:** если код некорректен — исправить; если So не наблюдаема — проверить reference/problem/support; если ML не помогает — один ограниченный development-цикл диагностики и зафиксированный вывод; при перспективности — переход к полеподобному пилоту. Нельзя лечить fail заказом суток аренды.

**Не входит:** утверждение 10% научного преимущества по восьми мирам, полевые запасы, обязательная GPU или полный black-oil. Эти модули не выбрасываются после опыта: дальше меняются profile/config/design, а не создаётся другой pipeline.

## E04 — специализированный ETL исходных CSV

**Цель:** воспроизвести аудит и получить канонические данные с lineage. SPEC §§2, 6; DATA_AUDIT.md.

**Вход:** пять CSV и проверенные hashes; E00. Для запуска с новым spec_version требуется общий блок совместимости E01.0; после него E04 может идти независимо от физических задач E01–E03. Разработка парсеров не требует готового ML или полевого simulator.

**Содержание:** chunked parsers с CP1251/UTF-8/BOM, десятичными запятыми и разной шириной ГИС; хвост PERF; well IDs с суффиксами; units/conditions registry; layer sets; дублеты; missing/zero/invalid; source row provenance; численные суммы/фонд; неоднозначные So/газ не исправляются догадкой. Canonical Parquet/локальная аналитическая БД для агрегатов, без загрузки всех таблиц как Python object-строк в несколько workers.

**Артефакты:** canonical tables, dictionaries, unit-status registry, qc ledger, source manifests, ETL evidence и diff с сохранённым аудитом.

**Приёмка:** совпадают либо объяснены исходные row counts/hashes/суммы; нет потери суффиксов; нет тайного column shift; исправления трассируются; неподтверждённые единицы обозначены; отсутствующие координаты не придуманы. Audit из документа — ориентир, фактическая корректная семантика имеет приоритет.

**Не входит:** фиксированное разнесение фаз по слоям, выбор пилота по качеству history fit, обучение по `Нефтенас.` как target текущей карты.

## E05 — время, подключения, пилот и источники

**Цель:** зафиксировать реальную постановку до подгонки. SPEC §§4–7, 9–10, 14.6.

**Вход:** E04, геологические/эксплуатационные признаки, внешние источники пользователя и подтверждённые открытые данные.

**Содержание:** completion interval state machine; начальные неизвестные подключения при потоке до первой перфорации; post-2021 uncertainty; внутримесячные роли/uptime; датировка ГИС и availability; core/buffer с всеми связанными слоями/скважинами; local frame и вертикальная конвенция; реальные controls и observation support. Выбрать t_ref по CompletenessReport и правилам SPEC §4.2.1, включая кандидата 2020; зафиксировать choice до history matching. Определить as-of/smoothing/conditional/pre-drill режимы. Создать source registry PVT/ОФП/Sor/давления с статусом и задачами закрытия gaps. Определить excluded zones не по их будущему отклику.

**Артефакты:** pilot passport, timeline, controls/observations, uncertainty windows, prior-source ledger, cutoff/split protocol, data gaps, initial P2 budget.

**Приёмка / G_DATA:** пилот связан физически; нет неучтённого отбора из другого пласта; единицы/геометрия достаточны для выбранного статуса; давление источника имеет дату/глубину/единицы; So_log сохраняет собственную дату; будущие данные не попали в as-of. При отсутствии защищаемых PVT допускается полеподобная synthetic development, но field quantitative release остаётся заблокированным/сценарным.

**Не входит:** декларация «2020 точно полный», вымышленное измерение давления, выбор final точки бурения.

## E06 — геологический prior и PhysicsDecision пилота

**Цель:** получить согласованную реальную forward model малого сектора. SPEC §§7–10; P2_PILOT.

**Вход:** E01 adapter, E05 pilot/source/timeline.

**Содержание:** surfaces/order/thickness/NTG/collector; совместный phi/logK prior; реальные присутствующие единицы Д1; условная геология без двойного conditioning; water region, contact/Pc consistency; initial S0/warm-up; boundaries/WI; latent basis+residual, known density. Подключить месячные управления/события. Провести prior predictive и sensitivity OW↔BO/PVT/Pc/grid/buffer; использовать small BO capability при необходимости. Зафиксировать PhysicsDecision по SPEC §8.1. Собрать P2 forward budget.

**Артефакты:** prior generator/renderer, model_id, P2 CaseBundle, geological holdout report, prior predictive report, PhysicsDecision, runtime/memory/failure logs. Если полевая L существенно отличается от учебной, повторить E02 density/noise/coverage checks перед E07.

**Приёмка / G_PHYS:** PV не удвоен; поверхности не пересечены; flow attribution внутри well model; S0 не подменён virgin field; существенные boundary/phase unknowns отражены. Выбранный physics class верифицирован; альтернативы не отсеяны только за численную сложность. Реальная динамика имеет защищаемое объяснение в prior либо зарегистрирован mismatch.

**Не входит:** безусловное требование BO любой ценой; разрешение OW только потому, что нет давления; увеличение сетки до 80 тыс. по умолчанию.

## E07 — полеподобный сквозной pilot experiment

**Цель:** проверить, что успех малого искусственного мира переносится на структуру ваших данных. SPEC §§12, 14, 18, 22; P2.

**Вход:** E03 полный малый метод, E06 полевая геометрия/prior/physics.

**Содержание:** малый corpus новых conditional worlds с фактическими schedule templates, sparse ГИС, многопластовостью, incomplete events и отсутствующим давлением; обучить/адаптировать q **с проверкой новой латентной семантики**; полный SMC и prior-start baseline; state/support analysis; repeated forward с альтернативными разумными boundaries/ОФП/газом; просчитать всё исследование по фактической стоимости. До доступа к закрытым будущим данным prior не подстраивается по ним.

**Артефакты:** `PILOT_RESULT.md`, поля synthetic truth/reconstruction, support/ambiguity, measurement priorities, PhysicsDecision confirmation, updated BudgetReport, решение о разрешённом масштабе P3.

**Приёмка / второй gate:** полезная наблюдаемость хотя бы на инженерных зонах, отсутствие сильной скрытой miscalibration, объяснённые model discrepancies, реализуемый budget. Лучшая tiny model не считается автоматически подходящей для 75 лет и девяти слоёв. При провале сначала устраняется причина; большой corpus/аренда не назначаются как универсальное решение.

**Не входит:** окончательная статья, просмотр locked holdout, гарантированная точность карты 2024.

## E08 — основной corpus и закрытый protocol

**Цель:** создать честную обучающую и проверочную среду. SPEC §12 и §14.

**Вход:** E07, зафиксированные priors/physics, модель наблюдений, basis, support-selection rule, budget.

**Содержание:** independent parent generation; отдельные train/development/locked-ID/locked-stress; joint consistency conditional G/latents; реальные controls только как разрешённые design templates; noise/missingness/events как на поле; отдельный truth storage; sharded generation/checkpoint. Fine-grid/different-relperm/missing-event tests определены до закрытого запуска. Adaptive sampling включается только с известной density и weights; initial baseline corpus prior-sampled.

**Артефакты:** corpus manifest и hashes, dataset loader, generation/budget ledger, locked manifests, `ResearchProtocol` со всеми C_STATE/C_CAL/C_VOLUME/C_LOCALIZATION/C_FIELD/C_COST из SPEC §14.3, точными метриками/порогами/CI/стоимостью, power-and-cost justification. Primary — MAE, RMSE дополнительная; truth anomaly masks и правила выбора зон определены до оценки.

**Приёмка:** дочерние views не разделены между folds; true K/So/подключения не попали в context; corpus не состоит из шумовых копий одной real MAP; все failures учтены; число test/SMC calls включено в budget. Long-run профили не активируются без явного выбора пользователя.

**Не входит:** автоматические 8 192 дорогих расчёта, обещание завершить исследование за 24 часа, настройка test family по удачной карте.

## E09 — исследовательский encoder и NSF

**Цель:** получить пригодный для коррекции proposal с известной density. SPEC §11.

**Вход:** E08; малая реализация E03.

**Содержание:** training/early stopping по development; graph-temporal versus temporal-set/summary ablations; NLL и при adaptive sampling корректирующие веса; discrete probability calibration; residual prior; permutation/masks/causal-prefix tests; CPU/MPS parity; несколько seeds; density tails/multimodality/OOD diagnostics. Freeze operational Float64 q и context.

**Артефакты:** checkpoint/config/scalers, training history, density tests, development comparison, selected encoder ADR, OOD/reference diagnostics, full training cost.

**Приёмка:** q.sample и q.log_prob согласованы; q нормально обусловлена G/design; нет double Jacobian; пустой context не приводит к беспричинной уверенности; selection не использует closed test; graph оставлен только при обоснованной пользе. Скорость сырой сети не называется точностью So.

**Не входит:** окончательная карта из raw q, обязательный параллельный FMPE, увеличение depth без диагностики corpus.

## E10 — research inference, adjoint и сильные baselines

**Цель:** довести ранний inference до уровня научного сравнения. SPEC §§13–14.

**Вход:** E02 SMC/likelihood, E06 физика, E09 q. Разработку gradient adapter можно вести раньше на согласованных interfaces.

**Содержание:** defensive bridge/CESS/weights; replicated starts, global/discrete/local/pCN moves; cache; finite-budget statuses; полноценная remainder exploration. Chain-rule monthly likelihood adjoint по SPEC §13.3.1, finite-difference checks, MALA с MH. B1 prior-SMC, B2 multi-start MAP, B3 настроенный localized ES-MDA и Gaussian-compatible control; B4 representation baseline. Аналитическая и reduced-physics calibration всей системы; measured forward-equivalent cost.

**Артефакты:** research inference module, gradient report, baseline configs, toy/reference/SBC results, convergence diagnostics, benchmark ledger. Настройка baselines выполняется на development по правилам E08, её стоимость учитывается; окончательные конфиги всех методов замораживаются до E11. Результаты E09 могут потребовать повторной development-настройки/обучения, но не открытия locked test.

**Приёмка / G_DENSITY:** корректны target/proposal/reference measures; новые q не меняются внутри bridge; градиенты соответствуют месячной нормированной L, не иной MSE; partial beta не posterior; failures не выброшены. При неподдержанном adjoint используется корректный kernel и честно фиксируется ограничение MAP; не писать, что baseline реализован, если он заменён другим.

**Не входит:** обязательный full HMC/NUTS, непрозрачная cross-language end-to-end autodiff, финальные claims до E11.

## E11 — закрытая synthetic проверка

**Цель:** проверить state accuracy и вклад ML, а не просто работоспособность программы. SPEC §14; P3/P4.

**Вход:** замороженные E08–E10, независимые locked worlds, resources approved.

**Содержание:** paired independent-world comparison B0–B4/M по заявленным бюджетам; primary MAE So и supporting RMSE, CRPS/coverage/width, C_VOLUME, C_LOCALIZATION/false positives/resolution и ранг зон/regret; controls/phase fit отдельно; structural OOD, unknown events, gas-sensitive worlds; grid/time refinement subset; multiple seeds; cold-start/amortized costs; failures и невосстановимые случаи.

**Артефакты:** confirmatory results и immutable raw metrics, world-level CI/effect sizes, uncertainty calibration, rank/support results, code/config/data hashes, scientific outcome A/B/C/D.

**Приёмка:** выполнен зарегистрированный протокол, а не обязательно положительный результат. Для каждого scientific claim отдельно применяются C_* из SPEC §14.3; улучшение средней MAE не заменяет C_LOCALIZATION/C_VOLUME; если критерий не достигнут, вывод ограничивается наблюдаемой пользой или отрицательным результатом. Cells не используются как тысячи независимых test cases. Изменение метода после открытия требует нового confirmatory набора.

**Не входит:** выбор «лучшего seed», сдвиг support/таргета ради 10%, утверждение полевой ошибки So из synthetic RMSE.

## E12 — реальная реконструкция и backtests

**Цель:** применить замороженный метод в проверяемом информационном режиме. SPEC §§4.3, 14.6, 19.

**Вход:** E11 с корректным scientific status, реальные CaseBundle и source completeness.

**Содержание:** strict as-of rolling cutoffs/conditional forecasts, pre-drill/post-drill-pre-flow проверки без собственной будущей геологии, smoothing отдельно; model/prior/controls frozen на каждый опыт; retrospective t_ref и late 2024 при допустимых источниках; completion/PVT/boundary/OOD sensitivity; posterior convergence и предсказательные интервалы.

**Артефакты:** PosteriorBundle, field backtest report, map eligibility/status, data/physics assumptions, unresolved measurements.

**Приёмка / G_FIELD:** отсутствует leakage; C_FIELD из SPEC §14.3 оценён с paired uncertainty, а data completeness поддерживает заявленный режим; gas/completion/geometry sensitivity не скрыта. При отсутствии прямой So полевой RMSE не выдуман. Если E11 показал отсутствие ML-преимущества, метод не называется ML-enhanced accuracy только из-за успешной полевой подгонки.

**Не входит:** количественная карта вне поддержанного data/physics domain; рекомендация бурить по одной median So.

## E13 — пространственные продукты

**Цель:** корректно превратить частицы в инженерные карты. SPEC §15.

**Вход:** E12, общий report support, posterior weights/states и geometry uncertainty.

**Содержание:** So mean/quantiles/width, collector presence, remaining oil, oil above Sorw, kr_o diagnostic; conservative geometry intersections; quantile-of-sum; supported/prior-dominated/unresolved masks; representative physical realizations; inventory provenance; GIS export только при подтверждённой CRS.

**Артефакты:** state/particle arrays, maps and zone tables, support/method cards, uncertainty/sensitivity report, source provenance.

**Приёмка:** суммарные объёмы совпадают с физическими particle inventories; PV=0 не превращается в So=0; сумма cell quantiles не использована как zone quantile; mean cube не объявлен физической траекторией; неопределённость и дата видны пользователю.

**Не входит:** ручная дорисовка нефти, новая supervised модель запасов, сертифицированные извлекаемые запасы без решения/экономики.

## E14 — решения и воспроизводимый выпуск

**Цель:** показать практический смысл локализации и собрать статью/ВКР. SPEC §16, §21.

**Вход:** E13, admissible candidates и policy/constraints.

**Содержание:** небольшое число заранее определённых candidates; парные base/intervention расчёты одинаковых posterior worlds; BHP/facility constraints; incremental oil **сектора**, вода и перераспределение отбора; устойчивость ранга; при неопределённости ranked measurement needs. Quantitative nested EVSI — дополнительный опыт по бюджету, не обязательный барьер завершения всего проекта. Итоговая документация, reproducible synthetic command, figures/metrics/provenance и изложение ограничений.

**Артефакты:** candidate scenario table, paired outcomes, admissible recommendation или запрос доизучения; method/data/model cards; release manifest; структура статьи/ВКР с реальными результатами.

**Приёмка / G_DECISION:** известны политики, единицы/физика и uncertainty; собственный дебит новой скважины не принят за весь дополнительный эффект; chemical/thermal EOR не симулируются ручным уменьшением Sor; research findings воспроизводимы; приватные CSV не опубликованы. Наличие работающего pipeline не выдаётся за подтверждённый положительный научный результат.

# 5. Правила работы с агентами

Каждый план использует действующие SPEC/STAGES 4.0 и матрицу SPEC §23. Каждый план содержит: цель; требования SPEC с номерами разделов; входные artifacts; разрешённый profile и budget; изменения интерфейсов; тесты с ожидаемыми наблюдаемыми результатами; provenance; что не входит; факторы блокировки; rollback/checkpoint. Подробный порядок файлов/функций пишет агент-планировщик, а не этот документ.

После исполнения нужен отчёт с реально выполненными командами, exit status, созданными artifacts, проверками и ограничениями. Формулировка «тесты должны пройти» не является приёмкой. Независимая проверка смотрит математическую постановку и утечки, а не только linter.

Один этап не меняет prior/likelihood/target ради удобства кода. ADR нужен для научных изменений. Изменение batch/числа workers в границах профиля — ресурсное решение с записью config; изменение сетки/physics/basis — новая model version с повторной validation.

# 6. Переход со старой этапности

| Прежний блок | Куда попадает в 4.0 |
|---|---|
| E00 основание | Сохранённый E00 3.0; недостающая совместимость/resources/worker — E01.0 |
| E01–E03 ETL / timeline / pilot | E04–E05 |
| E04 геологический prior | E06 |
| E05 simulator verification | E01 + полевая проверка E06 |
| E06 интегрированный сектор | E06–E07 |
| E07 inverse/baselines | Малый E02; research E10 |
| E08 corpus | Малый E03/E07; research E08 |
| E09 ML | Малый E03/E07; research E09 |
| E10 correction | Малый E02–E03; research E10 |
| E11–E14 science/field/maps/decisions | E11–E14, с новыми gates |

Нельзя дать исполнителю старый план E03 и новый STAGES, не объяснив смену значения номера. Во всех новых планах указывается `spec_version=4.0` и полный заголовок этапа.

# 7. Следующий запрос агенту

> Прочитай README.md, docs/README.md, docs/SPEC.md, docs/STAGES.md, docs/COMPUTE_PROFILES.md и docs/DECISIONS.md; если в checkout есть AGENTS.md, прочитай его. Сверь фактический код, Git-состояние, старый план и reports/stages/E00.md. Напиши отдельный план **E01 — физический adapter и первые synthetic states** для SO-RECON 4.0. Сохрани готовое основание E00. В начале E01 включи блок совместимости версии/lineage, минимальных контрактов, persistent worker и ресурсных guards по DECISIONS §3; не создавай foundation заново. Используй существующие julia/, src/so_recon/ и artifacts/runs/. Затем запланируй физические проверки P0 и первые P1 worlds. Для каждой задачи укажи требования SPEC, входы, артефакты, тесты/допуски и budget. Не приступай к реализации и не включай обучение/SMC/полевой ETL в E01.

Новый план сохраняется в `docs/superpowers/plans/` с датой, номером и полным названием этапа. Аналогично планируются E02 и далее только в их объёме. Непройденные зависимости отражаются как входные gates: будущий план не объявляет отсутствующий artifact существующим.

## 7.1. Статусы и приёмка

E00 сохраняет исторические SPEC/STAGES 3.0. Новые планы и отчёты используют 4.0 и матрицу SPEC §23. По состоянию на согласование документации E01–E14 — NOT_RUN: написанный контракт не является реализованным модулем. Само планирование E01 разрешено; запуск следующего этапа зависит от приёмки предыдущего. Исполнитель существующего E00 может закончить его текущие исправления без перехода к E01.

Статусы этапа: NOT_RUN, IN_PROGRESS, PASS, PASS_WITH_LIMITATIONS, FAIL. Отчёт `reports/stages/E##.md` содержит версию требований, входные hashes, commit/dirty state, команды с exit status, тесты/ресурсы, output artifacts, ограничения и точное решение о зависимостях. Для исследовательских этапов technical status и C_* scientific outcomes различаются по SPEC §23.2; отрицательный scientific результат не является поводом подгонять протокол.
