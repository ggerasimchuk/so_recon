# E03 — первый сквозной learned inverse: conditional NSF → defensive SMC → physical So

**Дата плана:** 16.09.2026.  
**Проект:** SO-RECON. **Репозиторий:** `ggerasimchuk/so_recon`. **spec_version:** 4.0. **Предлагаемая config_version:** E03.1.  
**Изученная ветка:** `e02-probabilistic-inverse`.  
**Изученный HEAD:** `1810ec4af6234ffb718461220a98dce852da2351`.  
**Статус документа:** план будущей реализации, не отчёт о выполненных экспериментах.

Репозиторий не изменялся. При подготовке плана не выполнялись обучение, Julia-прогоны, тесты, коммиты и создание PR. Изучены спецификация, этапность, вычислительные профили, планы E01/E02 и дополнения E01, актуальные отчёты, основные контракты, реализация генератора/renderer/target/SMC/evaluator и релевантные тесты. Gitignored-артефакты недоступны: результаты ниже — сведения сохранённых отчётов, а не независимая перепроверка чисел по массивам.

## 0. Решение и исходное состояние

**Продолжать стоит — до ограниченного, проверяемого E03. Переходить сразу к большому корпусу или полевой карте пока рано.** На следующем этапе необходимо проверить не привлекательность архитектуры, а два утверждения: восстанавливается ли скрытая So на информативном синтетическом мире и помогает ли обученное предложение при ограниченном бюджете.

### 0.1. Что действительно имеется в изученном снимке

| Компонент | Подтверждённое состояние по коду/отчёту | Следствие для E03 |
|---|---|---|
| E01 OW | Актуальный `reports/stages/E01.md`: `PASS_WITH_LIMITATIONS`, 13 обязательных OW-проверок пройдены; BO не входит в актуальный принятый scope | Переиспользовать worker, forward, баланс и restart, не переписывать solver |
| Дискретизация | Принятая reference-пара 112/144 не сертифицирует сетку 16×16. Coarse 16/48 FAIL сохранён | Разделить in-model результат и устойчивость к иной дискретизации |
| E02 reduced reference | Диагностический протокол сходится на 257 узлах | Не повторять разработку quadrature без изменения её зависимостей |
| Reduced SMC | Исходная проверка N64 не прошла по Q05; отдельно зарегистрированная N128-диагностика прошла. Итог `PASS_WITH_LIMITATIONS` | Не переписывать историю как «N64 прошёл»; учитывать конечную MC-точность |
| Noise recovery | 200 независимых noise seeds, фиксированный физический банк, PASS | Это не calibration So на 200 независимых геологиях |
| T2/T4 | Truth/control/ambiguity setup gates пройдены | Это разрешение считать posterior, а не готовый posterior |
| E02 в целом | `FAIL — P1_NATIVE_MATRIX_NOT_RUN` | Native dependency ещё не закрыта |
| ML | В дереве нет готового ML-пакета; в `pyproject.toml` ещё нет Torch/NSF-зависимостей | E03 действительно добавляет обучаемый модуль, а не только отчёт |

E01 completion amendment описывает более ранние запуски, включая BO. Для конкретной зависимости использовать свежий машинный evidence на соответствующем коде, а не переносить допуск со старого коммита. Исторические результаты сохраняются.

### 0.2. Конкретные места, которые нельзя механически перенести в E03

1. `generate_dynamic_history(..., noise=FIXED_NOISE_THETA)` имеет фиксированное значение по умолчанию. Для обучающего корпуса передавать явно **noise, восстановленный из той же theta, которая записана как метка**. Иначе координаты sigma/rho будут размечены одним законом, а Y получен другим.
2. `run_physical_smc` сейчас создаёт `PhysicalTarget(prior, prior, ...)`. Новый runner должен передавать реальную defensive mixture и в target, и в `infer`/`continue_inference`.
3. `PhysicalTarget.fingerprint` включает proposal. Для проверки SAME TARGET нужен отдельный **scientific target identity**, исключающий proposal, seed, N и формат вывода. Существующий runtime/checkpoint fingerprint не ослаблять.
4. `_state_summary` использует truth PV для агрегирования всех частиц. Для инженерной зональной So использовать PV каждой частицы; truth PV оставить весами оценки ошибки. Старый fixed-truth-PV diagnostic допустим отдельно, с правильным названием.
5. `_state_summary` считает основную ошибку по поклеточной медиане. В E03 зафиксировать posterior mean как основной estimator; median/quantiles оставить дополнительными продуктами.
6. Начальный ансамбль learned SMC имеет закон r, а не p0. Его нельзя подписывать «static prior», как можно было сделать при r=p0 в E02.
7. `report_zone_matrix` содержит пересекающиеся слои, квадранты и remote-зоны. Их нельзя все одновременно включать в один PV-взвешенный primary score.
8. T4 не имеет тот же остаток, что T1: n_v=11, n_residual=6; T1/T2 имеют n_residual=4. Не обрезать/дополнять theta для удобства batch.
9. Native integration test E02 отдельно проверяет COMPLETE/beta=1; это не полноценный convergence/calibration gate.
10. `_truth_checks` в physical runner допускает median step balance 1e-4, тогда как SPEC для smooth verification указывает 1e-5. Для нового E03 scope зафиксировать нормативный порог 1e-5 в конфиге; не наследовать более мягкое число без документированного основания для конкретного класса задач. Это замечание к правилам приёмки, не утверждение, что сохранённые прогоны нарушили 1e-5.

## 1. Цель, гипотезы и границы

### 1.1. Цель

Получить воспроизводимый путь:

`доступные G/U/geometry → conditional prior → theta → JutulDarcy → noisy Y + hidden truth → graph-temporal NSF → defensive SMC → физический ансамбль So → ошибки / неопределённость / стоимость`.

Каждая публикуемая реализация So рассчитывается JutulDarcy. Сеть не предсказывает raster So как замену физической модели.

### 1.2. Научные гипотезы

**H_STATE.** На T1 динамическая история снижает ошибку зональной So относительно условного статического prior.

**H_ML.** При одном и том же posterior target обученное предложение улучшает достижимое качество So/CRPS при заданном online-бюджете либо снижает стоимость достижения сопоставимого качества. Идеальный posterior с ML и без ML одинаков; проверяется эффективность конечного расчёта, не «новая информация из сети».

**H_AMBIGUITY.** После коррекции модель не демонстрирует необоснованную уверенность в распределении нефти между слоями T2 и в удалённой зоне T4.

**H_GRAPH — вторичная.** Графовое взаимодействие полезнее temporal-set/summary при сопоставимых данных и затратах. Отрицательный результат по H_GRAPH не опровергает H_ML.

### 1.3. Что не входит

Полевой ETL, текущая карта Ромашкинского объекта, неопределённые полевые PVT, BO-аттестация, adjoint/MALA, MAP/ES-MDA, neural forward surrogate, production UI, рекомендации бурить, крупный гиперпараметрический поиск. MAP/ES-MDA остаются последующими обязательными research comparisons, но не блокируют первый E03.

Первый native smoke может иметь меньше миров/частиц, чем scientific matrix, однако он не получает `PROMISING_STATE` или `PROMISING_ML` только за техническое завершение.

## 2. Зависимость E03.0: закончить E02, не переоткрывая весь проект

### 2.1. Разрешённая последовательность

Чистые ML-contract/unit задачи можно разрабатывать на закреплённой основе параллельно завершению E02. Дорогой корпус и научное сравнение E03 разрешаются после машинно проверенного допуска E02.

Из `configs/e02_experiments.json` сохранить четыре постановки:

- `e02-t1-v1-s141`;
- `e02-t1-v1-s142`;
- `e02-t2-v2-s143`;
- `e02-t4-v1-s144`.

Для них завершить зарегистрированную native matrix N32/N64 × inference seeds 11/12: **16 posterior runs**. Это оставшаяся матрица E02, не новый произвольный benchmark E03. Проверить актуальность T1 setup и совместимость ранее опубликованного setup T2/T4. Старую T2-v1 сохранить как CONTROL_INFEASIBLE, не включать её в положительную статистику.

Проверить state diagnostics и существующий convergence screen. При нехватке бюджета сохранить checkpoint/ledger; не менять beta вручную и не ослаблять пороги. При необходимости N128 — отдельная заранее зарегистрированная диагностика с новым бюджетом, а не «исправленный результат N64».

### 2.2. Новый dependency gate

`validation/e03_dependency.py` реализует `require_e02_inverse(...)`. Он читает существующие ArtifactRef и проверяет:

- принятый OW evidence через `require_e01_ow`;
- reduced reference и accepted reduced-SMC diagnostic;
- noise-recovery report с корректным scope;
- physical setup, завершённые SMC, comparison/convergence artifacts;
- hashes физических исходников, lockfiles, конфигов, наблюдений и родителей;
- наличие всех обязательных файлов, а не только текст `PASS` в Markdown.

Отсутствие gitignored evidence даёт объяснимый отказ допуска. Нельзя создавать заглушки с фальшивыми метриками. Unit-тесты валидатора используют явно тестовые fixtures, не scientific evidence.

## 3. Зафиксированный физический эксперимент E03

### 3.1. Базовый масштаб

Основной P1: 16×16×2 = 512 ячеек, существующая физическая область и 36 месяцев. Существующие sparse log(k), календарь, geometry и controls сохраняются по design. Никакого увеличения до поля в этом плане.

Изученные density layouts:

| Design | v | z_perp | s |
|---|---|---|---|
| T1 | 8 geology + sigma/rho/log_bias = 11 | 4 geology modes | {0,1}: существующие kz/kx hypotheses |
| T2-v2 | 11 | 4 geology modes | {0} |
| T4-v1 | 11 | 5 geology modes + 1 remote-state mode = 6 | {0} |

Размеры читаются из DensitySchema/PriorContext, а не из названия `p1-conditional-12` и не из общего magic number.

### 3.2. T3: одно ограниченное добавление

Добавить новый versioned E03 design, не расширяя задним числом Literal старых E02 records. Он использует E01 geology/geometry и две объявленные гипотезы межпластовой связи. Нижнее подключение выбранного producer открывается на границе 18-го месяца; этот календарь известен обоим методам. Конкретная s не раскрывается encoder.

Чтобы минимальный E03 не сводился только к неизвестной K при заведомо известном So0, T3 содержит один независимый initial-state residual u~N(0,1):

`Sw0 = Swc + 0.10 * Phi(u)`.

Здесь 0.10 — **проектная настройка учебного T3**, не значение месторождения. Проверяется `Swc + 0.10 < 1 - Sor`. При текущем учебном Swc=0.2 это Sw0 в (0.2,0.3); ни clipping, ни ручное изменение конечной So не используются. Перед серийным запуском проходят balance и closed-transient preflight, аналогичный по смыслу T4.

У T3 n_v=11, n_residual=5: четыре прежних geology modes и u. Весь residual остаётся в target и SMC. Основные OFP остаются фиксированными; неизвестный So0 закрывает минимальное требование state uncertainty. Расширение неизвестных OFP — отдельная последующая версия, не скрытое изменение модели.

### 3.3. T5: mismatch отдельно от in-model результата

Первая обязательная T5-версия — paired coarse/fine truth на заранее выбранных двух evaluation parents T1. Inference остаётся на 16×16×2. Fine truth — отдельный bounded P4-type diagnostic, предпочтительно 48×48×2: нечётное сгущение позволяет сохранить физические координаты скважин в существующей cell-centred постановке.

Сохраняются физическая область, латентная геология как функция координат, радиус/длина/число соединений, controls, dates и fluids. WI пересчитывается; соединение нельзя размножать на все дочерние горизонтальные ячейки. Крупные support-зоны задаются в физических координатах. Fine truth и coarse inference имеют разные model hashes.

Fine child относится к тому же parent/split. Он не является дополнительным независимым миром. Если fine run не укладывается в измеренный бюджет, это NOT_RUN/resource limitation, а не PASS. Дешёвый T5 с ошибкой даты события можно добавить отдельным зарегистрированным diagnostic, но он не сертифицирует сеточную точность.

### 3.4. Один совместимый model class

В E03 общий learned блок имеет размер 11; остаток и categorical support различаются по schema. Shared encoder/NSF допускается для явно перечисленных совместимых layouts. Metadata design/schema используется для routing и masks, но не как embedding seed/world_id/truth-family.

У разных G законно разные basis_hash. Модель не должна требовать один basis_hash для всего корпуса. Binding конкретного proposal к конкретному context обязан строго проверять соответствие theta/basis. Порядок и знаки basis modes детерминированы; семантическая смена базиса требует новой версии.

## 4. Входы, выходы и provenance

### 4.1. Разделение трёх потоков данных

| Поток | Содержимое | Кто читает |
|---|---|---|
| Inference input | Geometry, sparse G с supports/uncertainty, U, известные события/constraints, Y/masks, cutoff, разрешённые prior descriptors | Encoder, prior, target |
| Training labels | theta: s/v/z_perp и её schema/basis; ссылки на generator law | Training loader; labels не входят в feature tensor |
| Evaluation truth | Полные физические поля, states, actual theta, fine truth, checks | Только evaluator и генератор |

Не передавать encoder `ForwardResult`, raw `CaseBundle` с истинными rock/initial arrays, truth paths, random seeds, полные K/phi/So/p, actual achieved hidden rates и истинные неизвестные соединения. Из широкого `context.design` строить typed allowlist, не скармливать весь dictionary.

BHP setpoint в T2-v2 — известный control, не измеренное давление. Для него QL не подставляется из truth как «входной дебит». В tensor отдельно кодировать `control_kind`, `control_value`, физическую единицу/нормирование и маску применимости.

### 4.2. Новые минимальные records

Создать в `ml/contracts.py`:

- `CorpusManifest`: experiment/version, parent rows, derived views, split, design distribution, expected/complete/failed counts, schema/basis references, hashes.
- `ContextSpec`/`ContextBatch`: разрешённые features, masks, geometry/edges, cutoff, units, feature order, per-world identity metadata вне tensors.
- `TrainingManifest`: corpus/scaler/model/config hashes, train seed, optimizer/device, selected epoch, metrics, costs, checkpoint references.
- `ProposalManifest`: frozen weights/config/scaler hashes, supported layouts, dtype/backend, architecture version, context-builder version.
- `ComparisonProtocol`: parents, methods, budgets, seeds, primary support/estimator/time, gates, preregistration hash.

Scientific target identity и E03 report schema можно расположить в `validation/e03_protocol.py`. Не создавать второй ArtifactRef, RunContext, CaseBundle, ThetaRecord или registry.

### 4.3. Scientific target identity

Включить prior/renderer/latent semantics, G/information/cutoff, observation values/masks/role, likelihood/operator/noise definitions, geometry/controls/fluids/boundary, физический solver/discretization.

Исключить learned proposal, epsilon, training seed, N, inference seed и выбор diagnostic output. Изменение output request всё равно меняет execution/cache identity, но не обязано менять математический posterior.

Перед B1/ML сравнением проверять равенство scientific target identity. Existing `PhysicalTarget.fingerprint`, включающий proposal, сохраняется как runtime/checkpoint identity.

### 4.4. Артефакты

Использовать существующие `artifacts/runs/<run_id>/`, JSON/Parquet/HDF5, immutable publication и parent ArtifactRef. Минимум:

`corpus_manifest.json`, `labels.parquet`, `context/*`, `truth/*`, `training.json`, `checkpoint/*`, `proposal_manifest.json`, `smc checkpoint`, `posterior_bundle.json`, `comparison.parquet`, `e03_report.json`, `budget_report.json`, figures и `reports/stages/E03.md`.

Точные подпути определяются существующими writers. В metadata: inspected/source commit и dirty state, actual lock hashes, command/resolved config, schema/model/renderer/basis/feature/scaler/checkpoint hashes, parent IDs, random streams, dtype/device, information cutoff, units. Fingerprints не включают случайный absolute path машины. В tensor не попадают hashes/IDs как признаки.

Отдельно сохранять `log_p0`, `log_q`, `log_r`, `log_L` для финальных частиц. Старый TargetEvaluation с log_r не ломать ради этого: допустим отдельный versioned density-diagnostics artifact, связанный с theta hash.

Raw q ensemble не публиковать под видом скорректированного PosteriorBundle. Отдельный `ensemble_kind=raw_proposal` обязателен.

## 5. Математика

Пусть C содержит только разрешённые G, geometry, U, observed Y и cutoff; theta=(s,v,z).

### 5.1. Prior и target

`p(theta | Y,G,U) ∝ p0(theta | G) L(Y | F(theta,U), theta_noise)`.

Все плотности считаются относительно `counting_x_latent_lebesgue`. В текущих whitened layouts v и z имеют стандартный нормальный prior, а s — объявленные categorical probabilities. Renderer Jacobian в эти latent densities не добавляется. Статический G, уже использованный в conditioning, второй раз в L не умножается.

### 5.2. Обученное предложение

`q_phi(s,v,z | C) = q_phi(s | C) q_phi(v | s,C) p0(z | v,s,G)`.

Для текущих independent-whitened residual blocks последний множитель — полная N(0,I) размерности n_residual. Это не утверждение, что posterior residual равен prior: история корректирует residual через физический target/SMC.

Если v = T_phi(epsilon; s,C), epsilon~N(0,I), то

`log q_phi(v | s,C) = log N(T_phi^{-1}(v);0,I) + log|det D T_phi^{-1}(v)|`.

Контракт wrapper фиксирует направление transform, чтобы sampling и log_prob не использовали противоположные знаки logdet.

### 5.3. Обучение

Корпус создаётся с theta~p0(theta|G) и Y из той же generative likelihood. Основной loss:

`J(phi) = - (1/J) sum_j [log q_phi(s_j|C_j) + log q_phi(v_j|s_j,C_j)]`.

Residual term не оптимизируется сетью, но включается в operational log_prob. Для нескольких views одного parent вклад нормируется числом views; иначе увеличение noise copies меняет веса миров. В первой версии нет adaptive proposal rounds, SNPE correction, MAP labels или reweighting «по хорошести симуляции».

### 5.4. Defensive bridge

`r = (1-epsilon) q_phi + epsilon p0`, начальная epsilon=0.10, фиксируется до inference.

`pi_beta(theta) ∝ r(theta)^(1-beta) [p0(theta)L(theta)]^beta`.

Для перехода beta→beta':

`log w'_i = log w_i + (beta'-beta) [log p0(theta_i) + log L(theta_i) - log r(theta_i)]`.

При beta=1 target один и тот же для learned/prior start. При beta<1 это не posterior. Плотность смеси вычисляется через logsumexp; нельзя использовать только density выбранной sampling-компоненты.

Сохраняются E02 CESS/ESS, resampling, global/RW/pCN/discrete MH kernels и полный bridge MH ratio. Epsilon=1 даёт prior-start path. Positive defensive prior mass не гарантирует, что конечные 32 частицы посетили каждую моду.

### 5.5. So, support и объём

Для частицы i и зоны z:

`So_i,z = sum_c A_zc PV_i,c So_i,c / sum_c A_zc PV_i,c`.

Для truth использовать его собственный PV. Posterior estimator:

`So_hat_z = sum_i W_i So_i,z`.

Primary PV-weighted MAE для одного мира:

`MAE = sum_z PV_truth,z |So_hat_z - So_truth,z| / sum_z PV_truth,z`.

RMSE получается заменой абсолютной ошибки на квадрат с последующим корнем. Затем усреднять по независимым worlds с одинаковым весом мира, а не объединять все ячейки в псевдовыборку.

Для OW inventory в surface units:

`N_remaining_i,z = sum_c A_zc PV_i,c So_i,c / Bo_i,c`.

Сначала вычислить сумму каждой частицы, затем weighted quantiles распределения сумм. Не суммировать поклеточные Q05/Q95. Inventory не называется извлекаемыми запасами. Нулевой PV: So не определена, inventory=0, отдельная маска; отсутствие прогноза на положительном truth PV — failure, а не повод исключить зону.

## 6. Корпус и split-протокол

### 6.1. Генерация одного parent

1. Выбрать зарегистрированный design и сформировать доступный G из отдельного context stream.
2. Построить conditional PriorContext; не использовать будущую историю при conditioning/basis/scaler.
3. Независимым truth-latent stream получить новую theta из `GaussianConditionalPrior(context)`.
4. Выполнить `render_theta` и forward; проверить balance, controls, finite states.
5. Получить prediction и вызвать `generate_dynamic_history(..., noise=rendered.noise)` с отдельным history seed.
6. Сохранить разрешённый input, training labels и evaluator truth раздельно; проверить hashes и round-trip theta→physical arrays.

Не использовать E01 `theta.json` напрямую как E02 ThetaRecord. E01 forward worlds не образуют готовый amortized training corpus.

Число неуспешных worlds и причины сохраняются. Нельзя тихо заменить неудобный draw следующим seed. Retry выполняется для той же theta по зафиксированной numerical policy. Неудача renderer/resource — не logL=-inf. При повторяющемся провале prior/controls пересматриваются отдельной версией до нового корпуса.

### 6.2. Размеры

Стартовый scientific corpus — **128 train + 32 development + 8 evaluation независимых parents**, не разбиение 128 на три части. Предлагаемое фиксированное распределение:

| Family | Train | Development | Evaluation |
|---|---:|---:|---:|
| T1 | 64 | 16 | 4 |
| T2 | 24 | 6 | 1 |
| T3 | 16 | 4 | 1 |
| T4 | 24 | 6 | 2 |
| Итого | 128 | 32 | 8 |

T5 — paired views двух заранее указанных T1 evaluation parents; не плюс два независимых наблюдения к статистике. Это exploratory дизайн: четыре T1 parents не дают точного подтверждения 10% выигрыша, а один T2 — калибровки. При перспективном результате заранее увеличить evaluation до 16: T1=8, T2=2, T3=2, T4=4; доплата отражается в бюджете.

Для первого технического thin slice разрешить 8 train / 2 dev / 1 отдельный T1 world. Этот одноразовый smoke namespace не входит в оценку научных gates.

### 6.3. Разбиение и views

Split назначается parent до симуляции. Все noise replicas, control variants, prefixes и fine/coarse children остаются в том же split. E01 seeds 41–45 и E02 seeds 141–144 не объявляются новыми blind evaluation parents.

Обучающие значения watercut берутся из той же канонической bin/report representation, которую оценивает likelihood; более точное скрытое raw value не становится дополнительным входом сети. Missing/reset mechanism задаётся наблюдаемым расписанием/протоколом. Проверить, что state-dependent `observed_valid` из физического prediction не создаёт неучтённый informative-missingness channel: для стартового корпуса маски должны определяться объявленным наблюдаемым дизайном; другой механизм требует отдельного generative/likelihood решения, а не молчаливого пропуска строк.

Сначала один full-history view на parent. Затем ограниченные prefixes 12/24/36 и максимум две noise copies разрешены только с parent weighting. Scaler fit — только train; validation/test не определяют ни feature scaling, ни выбор эпохи, ни thresholds.

После открытия exploratory evaluation корректировка допустима только с новым experiment ID и честным статусом уже просмотренных данных. Confirmatory test статьи создаётся отдельно позже.

## 7. ML-реализация

### 7.1. Features и masks

Well-time tensor хранит controls, watercut/bin information, observation validity, reset, calendar/elapsed time, role и completion-information flags. BHP/liquid-rate/water-rate типы не смешиваются в одну безразмерную колонку без type encoding. Static well/context features — geometry и доступный G с uncertainty/support.

Отличать observed zero, missing, shut, padding, newly opened и unknown completion. Для неизвестной дискретной связи доступны только объявленные гипотезы; true barrier/connectivity matrix не передаётся.

Граф описывает потенциальную связь по geometry/известной структуре. Edge attributes: относительное положение/расстояние и разрешённые layer relations. Не брать conductance из истинной K. Hash, seed и parent ID не являются embeddings.

### 7.2. Три encoders, один density head

- Основной: два causal temporal + edge-conditioned graph блока; width=64, heads=2; masked pooling в context=128.
- Temporal-set: та же temporal обработка без graph message passing, затем permutation-invariant pooling.
- Summary: заранее описанные per-well агрегаты плюс set pooling; не фиксированный порядок скважин.

Все используют одну и ту же categorical/NSF голову и одинаковый dataset contract. Сначала запускается основной путь, затем две небольшие ablations на development. Не обучать крупную GNN/Transformer и не подключать PyG/DGL без измеренной необходимости.

### 7.3. NSF

Выбор реализации: Torch + `nflows`, тонкий typed adapter вокруг rational-quadratic coupling transforms. Совместимость Python 3.13/ARM64 и CPU Float64 проверяется отдельным smoke; проходящие версии фиксируются `uv.lock`. Не обещать совместимость конкретного release без этого теста и не менять Julia packages ради ML.

Старт: 6 coupling transforms, 8 bins, normal base, linear tails; чередующиеся masks/permutations. Например tail_bound=4, min bin width/height/derivative=1e-3 как фиксированные проектные настройки. Dropout=0 и отсутствие batch-dependent normalization облегчают однозначную operational density. В conditioned network подаются encoder context и s; singleton support маскирует только действительно невозможные категории, не «категорию, которой не было в train».

### 7.4. Обучение

Проектный старт: AdamW, lr=1e-3, weight_decay=1e-4, batch=8, максимум 200 epochs, patience=20, gradient norm cap=5. Это настройки эксперимента, не научные пороги. Выбор checkpoint — development parent-averaged NLL; downstream development checks выполняются после выбора ограниченного числа checkpoints, не на каждой эпохе.

CPU Float32 — default. MPS разрешается после отдельного smoke/parity; operational density остаётся CPU Float64. Не нужны AMP, torch.compile, quantization и distributed training.

Обязательные logs: decomposition NLL по categorical/v, разница с prior log density, gradient norms, best/last epoch, parent counts, seed, duration/RSS, failed/nonfinite batches. Падение train loss без dev/generalization/state результата не означает научный успех.

### 7.5. Frozen operational density

`FrozenConditionalNSF` реализует существующий `Density`:

```python
@property
def fingerprint(self) -> str: ...
def sample(self, n: int, rng: np.random.Generator) -> tuple[ThetaRecord, ...]: ...
def log_prob(self, theta: ThetaRecord) -> float: ...
```

При binding зафиксировать checkpoint, scaler, context-builder, allowed context, prior/basis/layout, dtype/backend. Пересчитать encoder context той же CPU Float64 копией, которой будут выполняться sampling и log_prob.

**Весь operational sampling потребляет только переданный NumPy Generator.** Categorical uniforms, normal base и residual draws генерируются им и передаются transform как tensors. Не вызывать library `Flow.sample()` с глобальным Torch RNG, если он не связан с сохранённым SMC RNG. Sampling одной копией и scoring другой запрещены.

Перенос Float32-trained weights в Float64 сам по себе допустим: operational законом является полностью зафиксированная Float64 модель. Проверять самосогласованность именно этого закона, а не требовать побитовой идентичности разным training backends.

Веса экспортировать в непикловый формат (например safetensors + JSON); training checkpoint optimizer может быть дополнительным runtime artifact, но не единственным воспроизводимым scientific checkpoint.

## 8. Порядок реализации и карта файлов

Пути ниже — предлагаемые новые/изменяемые файлы, не утверждение об их наличии. Существующие большие модули не дробить попутно. `__init__.py` создаётся при первом потребителе, не как пустая архитектура заранее.

### Task 00 — E02 dependency closure

**Файлы:** `validation/e03_dependency.py`, `tests/unit/test_e03_dependency.py`; при необходимости точечная сборка E02 report через существующий путь.  
**Сделать:** выполнить §2, проверить native matrix и происхождение.  
**Готово:** dependency validator принимает реальные актуальные evidence и отказывает при отсутствующем/подменённом/незавершённом артефакте. Старые E00/E01 reports не переписаны.

### Task 01 — Protocol и ML dependency smoke

**Файлы:** `config/learning.py`, `ml/contracts.py`, `configs/e03.yml`, `configs/e03_experiments.json`, `configs/e03_tolerances.yml`, `pyproject.toml`, `uv.lock`.  
**Сделать:** зафиксировать scope, family proportions, splits, seeds, supports, estimator, training и resource budgets; установить только нужные ML-зависимости.  
**Тесты:** invalid config/layout, запрет пересечения split, Torch/nflows CPU Float32 backward и CPU Float64 forward/inverse/logdet.  
**Готово:** существующие E00/E02 конфиги читаются по своим версиям; Julia lock неизменён.

### Task 02 — T3 и registry физических designs

**Файлы:** `synthetic/loop_designs.py`, небольшое dispatch-расширение `geology/renderer.py`, новый conditional-context constructor рядом с существующей логикой; tests `test_e03_designs.py`.  
**Сделать:** T3 §3.2 с неизвестным initial-state residual; поддержка T5 physical mapping без изменения исторических designs. Старый renderer строго валидирует layout, новый — отдельно свою версию.  
**Тесты:** изменение каждого residual влияет на объявленную величину; s меняет только заявленную связь; u изменяет initial, но не вписывает final So; months/units/bounds; closed preflight; coarse/fine well coordinates/length/number.  
**Готово:** один реальный T3 truth forward прошёл физические checks; T5 design имеет независимые identities.

### Task 03 — Leakage-safe corpus builder

**Файлы:** `synthetic/inverse_corpus.py`, `ml/dataset.py`, tests `test_inverse_corpus.py`.  
**Сделать:** §6; reuse `GaussianConditionalPrior`, renderer, worker, predictor, `generate_dynamic_history`, registry; явный noise=rendered.noise. On-disk shard loading; parent labels отдельно.  
**Тесты:** parent split наследуется всеми children; label→render round-trip; nuisance/noise совпадает; failed worlds не исчезают; seed streams независимы; context создаётся без truth read; повторная публикация immutable.  
**Готово:** thin-slice корпус читается minibatch-ами и восстанавливается по manifest без Julia.

### Task 04 — Context builder и encoders

**Файлы:** `ml/context.py`, `ml/encoders.py`, `ml/normalization.py`.  
**Сделать:** typed allowlist и три представления §7; scaler train-only.  
**Тесты:** permutation wells/edges, padding invariance, masked-value mutation, zeros≠missing, future-tail mutation не меняет prefix, true s/K/phi/So/pressure/seed injection отвергается; BHP protocol не притворяется QL protocol.  
**Готово:** одинаковые физические inputs при перестановке скважин дают одинаковую density в допуске.

### Task 05 — NSF head и density wrapper

**Файлы:** `ml/flow.py`, `ml/proposal.py`, `ml/checkpoint.py`.  
**Сделать:** categorical + conditional flow + полный residual law, frozen CPU64 export/binding.  
**Тесты:** §9 density/RNG; несовместимый context/checkpoint/scaler/basis отклоняется; T4 имеет все 6 residual coordinates.  
**Готово:** `isinstance(bound_model, Density)`/contract проверка проходит; образцы и log_prob согласованы.

### Task 06 — Training command и CPU checkpoint

**Файлы:** `ml/train.py`, `ml/commands.py`, точечное подключение `cli.py`.  
**Сделать:** bounded training, resume, best checkpoint, costs, CPU64 export.  
**Тесты:** controlled overfit tiny batch; finite gradients; restore optimizer/RNG/minibatch cursor; split isolation; corrupt checkpoint refusal; no automatic native job при import/config validation.  
**Готово:** реальное короткое обучение на thin slice и экспорт; сам факт overfit не повышает scientific status.

### Task 07 — Same-target learned SMC

**Файлы:** `inference/learned_loop.py`, `validation/e03_protocol.py`; минимальные изменения существующих execution helpers вместо копии SMC engine.  
**Сделать:** bound q → DefensiveMixture(prior,q,0.1) → PhysicalTarget(prior,r,...) → infer/continue. Explicit state outputs 0/12/24/36; при необходимости native restart выбранных финальных физических particles сохраняется/воспроизводится с отдельным учётом стоимости.  
**Тесты:** target equality B1/ML; proposal equality runtime; true log_q/log_r; epsilon=1; altered checkpoint rejected; stop/resume learned path.  
**Готово:** один настоящий T1 learned run достигает beta=1 и имеет физические state refs; beta<1 никогда не подписывается posterior.

### Task 08 — Корректный state evaluator

**Файлы:** `validation/ensemble_states.py`, `validation/learned_comparison.py`, reuse `inverse_metrics.py`, `support.py`, `inverse_plots.py`.  
**Сделать:** собственный PV частицы, общий support, mean/MAE, RMSE/CRPS/coverage/width, representative physical realization, inventory quantile-of-sum, source-independent B0.  
**Тесты:** аналитические двухклеточные примеры с разной phi, anticorrelated inventory, zero PV, overlapping support refusal, malformed weights, missing estimates, aggregate geometry and axes.  
**Готово:** evaluator не требует truth для построения operational products; truth нужен только для scoring.

### Task 09 — Первый тонкий полный цикл

**Файлы:** `tests/integration/test_e03_learned_loop.py`, `configs/e03_smoke.yml`.  
**Сделать:** последовательно B0, B1, raw q, q+SMC на одном новом T1. Не подменять physics toy-моделью.  
**Готово:** получен первый отчёт So/errors/cost с реальным checkpoint и графиками. Scientific gates остаются INCONCLUSIVE при столь малой выборке.

### Task 10 — Бounded main corpus и development

**Файлы:** уже созданные corpus/training runners, `validation/proposal_diagnostics.py`.  
**Сделать:** только после BudgetReport построить 128/32/8, обучить основной encoder, минимальные summary/set ablations, две training initialization seeds для проверки устойчивости хотя бы на заранее зафиксированном development subset.  
**Тесты/диагностики:** dev NLL, categorical support, residual sensitivity, shuffle/prefix controls, actual epoch memory/time.  
**Готово:** выбор architecture/checkpoint и protocol заморожен до evaluation; budget включает все попытки выбора модели.

### Task 11 — Paired scientific comparison и T5

**Файлы:** `validation/learned_comparison.py`, `validation/e03_report.py`, `configs/e03_experiments.json`.  
**Сделать:** §10 matrix, frozen metrics и выбранные T5 children. При необходимости отдельно зарегистрировать expansion 128→256; не запускать 512 автоматически.  
**Готово:** все запланированные cells matrix имеют результат или явно указанный incomplete/failure; исходы не отфильтрованы по удачности.

### Task 12 — Stage report и read-only gate

**Файлы:** `scripts/e03_gate.sh`, `tests/unit/test_e03_report.py`, `tests/integration/test_e03_artifacts.py`, `reports/stages/E03.md`, `README.md`.  
**Сделать:** evidence collector читает артефакты, формирует e03_report.json и Markdown; не запускает solver/training при построении отчёта.  
**Готово:** отдельные technical/algorithm/science statuses; рекомендуемый следующий шаг вытекает из gate matrix. Наличие новых файлов и зелёных unit tests не превращается в научный PASS.

## 9. Обязательные тесты E03

Численные значения ниже — предлагаемые начальные test tolerances, замораживаемые до evaluation. Они не заменяют существующие физические допуски.

| Проверка | Что должна поймать | Критерий |
|---|---|---|
| Small transform normalization | Неправильный logdet/sign/tails | 1D reference integration с учтёнными tails: ошибка нормировки ≤1e-6 |
| Round-trip/logdet | Несогласованность sample и score | CPU64 round-trip ≤1e-8 на зарегистрированных центральных/tail points; FD/autograd logdet ≤1e-6 на малой размерности |
| Categorical/residual law | Потеря s или residual | Нормированная разрешённая categorical mass; exact Gaussian residual log density для всех объявленных координат |
| RNG isolation | Global Torch RNG ломает resume | Один NumPy state → те же draws; изменение глобального Torch RNG не меняет operational sample |
| Frozen reload | Иной закон после serialization | Тот же CPU64 backend и checkpoint дают те же draws/log_prob в принятом deterministic tolerance |
| Context permutation | Нейросеть учит порядок скважин | log_prob/context invariant, CPU64 max absolute delta ≤1e-8 |
| Mask/prefix leakage | Чтение padding/future/truth | Изменение скрытых значений не меняет допустимый prefix result; forbidden input отказ |
| Empty-data correction | Learned q выдаётся за posterior | С намеренно неравной prior q и L=1 SMC восстанавливает p0 на независимом toy reference |
| Missed-mode toy | Формальная epsilon прикрывает collapse | Зарегистрированный bimodal/discrete reference с известными mass; сравнение в MC-aware tolerance, без гарантии от одного seed |
| Residual-dependent toy | z сохранён, но не используется в L/MH | L явно зависит от z; corrected residual posterior совпадает с reference, raw q не считается финалом |
| Resume boundaries | Другая stochastic trajectory | Повторить E02 phase-boundary tests с learned q, включая pending draw/MH, budget stop и reload |
| Target identity | Неравные posterior targets в сравнении | Изменение q не меняет scientific identity, изменение physics/data меняет; execution fingerprints остаются строгими |
| Evaluator PV | Truth weighting выдаётся за физическую агрегированную So | Ручной example с разной phi у particles и truth |
| B0 identity | Начальная mixture названа prior | B0 ссылается только на draws из p0; изменение q не меняет выбранный B0 |
| Native physics | Mock даёт ложный сквозной PASS | Настоящий Julia/Jutul, balance/restart/controls, physical output identities |
| Report negative cases | Пропущенные/неуспешные runs превращаются в PASS | Missing, changed hash, beta<1, nonconvergence, missing family, incomplete matrix не дают полный PASS |

Нормировка q доказывается конструкцией и проверяется на малых references; нельзя «проверить интеграл 11-мерной плотности несколькими samples» и объявить его точным. MC reference tolerances фиксируются отдельно и учитывают реальное число независимых starts.

Существующие `test_smc_engine.py`, `test_smc_checkpoint.py`, `test_mh_kernels.py`, `test_defensive_proposal.py`, conditional prior/renderer/likelihood tests остаются регрессионными. Native marker skips перечисляются отдельно: skip не равен физической проверке.

## 10. Матрица сравнений и метрики

### 10.1. Методы

| Метод | Закон / смысл | Физические расчёты |
|---|---|---|
| B0 | Static conditional p0(theta|G) | Нужны для state ensemble; можно reuse именно prior-start initial draws с доказанным происхождением |
| B1 | Prior-start SMC | Существующий exact-target baseline |
| Q | Raw conditional NSF | Отдельные физические реализации q; без posterior claim |
| M | NSF + defensive SMC | Основной proposed method |
| A_summary / A_set | Summary/set NSF + тот же SMC | Ограниченные development ablations; дополнительные evaluation runs только по зарегистрированному бюджету |

Main evaluation: 8 независимых parents, B1/M, N32/N64, inference seeds 11/12. Это **64 SMC runs**, не «128 обучающих симуляций». B0/Q, truth, T5, второй training seed и ablations учитываются дополнительно.

Не запускать все 64 как побочный эффект одной команды без утверждённого BudgetReport. Сначала thin slice, затем небольшая зарегистрированная часть матрицы; partial execution остаётся partial evidence. При отсутствии бюджета на основную матрицу выдаётся INCONCLUSIVE/INCOMPLETE, не уменьшенное число worlds под прежним названием complete.

Первый main checkpoint выбирается на development. Второй training seed проверяется на заранее заданных двух T1 development parents и по одному T2/T4; для положительного устойчивого ML-вывода подтвердить тенденцию на независимой evaluation части либо явно ограничить вывод одним checkpoint/learning curve. Не называть inference repeats независимыми геологиями.

### 10.2. Primary и diagnostics

Primary support: восемь непересекающихся квадрантов по слоям в физических координатах, final month 36. Primary MAE по posterior mean, secondary RMSE. Month 12/24 — temporal diagnostics. Layer aggregates и T4 remote zones — отдельные diagnostics, без повторного учёта в primary score.

Для каждого world/method/seed: MAE, RMSE, weighted CRPS, Q05/Q50/Q95, width и empirical coverage, phase history fit, inventory distribution, s probabilities по **всем допустимым s**, unique ancestors и physical states, beta path, pre-resampling ESS, acceptance по kernel, residual movement, runtime/RSS/disk/failures.

Сохранить карты truth, posterior mean, median, width, error и одной фактической representative particle. Mean/median — summaries, не restartable physical state. Representative выбирается из существующих particles по объявленному правилу, а не генерируется усреднением состояния.

Bootstrap/uncertainty — по parent worlds. При 4 T1/8 total evaluation parents сообщать все индивидуальные значения и широкую неопределённость; не обещать доказанную 90% calibration. T2/T4 — контроль механизмов неоднозначности, а не статистически точная coverage study.

### 10.3. Ambiguity и residual diagnostics

На exact-independent toy likelihood residual posterior обязан оставаться prior. На реальном T4 нельзя требовать буквально неизменной ширины: слабая динамическая информация всё же возможна. Сужение оценивается относительно physical baseline/reference, ambiguity pairs и calibration, а не по правилу «posterior всегда шире prior».

T2 проверяет сохранение возможных layer explanations; текущая почти симметричная физика не даёт права заранее навязать точные веса 0.5/0.5. Такие точные веса тестируются на специально симметричном toy reference.

Отдельно диагностировать, не приходится ли существенная чувствительность L на z_perp, который raw q берёт из prior. Если это причина плохой эффективности, следующий ограниченный development round может изменить learned/residual partition с новой schema и полным сохранением modes. Не начинать с произвольного обнуления residual или увеличения сети.

### 10.4. Стоимость

Сравнивать Pareto quality–cost и достижение сопоставимого качества, а не только одинаковое N. Отдельно считать actual wall/CPU, logical target evaluations, unique forward work и executed cache misses. Общий disk cache не должен делать вторым запущенный метод «бесплатным»: показывать standalone replay cost и фактически потраченную campaign cost.

`C_cold = C_context/corpus + C_preprocess + C_all_training_trials + C_online + C_verification`.

`C_amortized(K) = C_shared / K + C_online`.

Для одного объекта главный результат — cold-start. K допустим только для действительно совместимых следующих задач, где не нужны новые physics, basis/model class или обучение. Break-even существует лишь при положительной online экономии; вычислить `ceil(C_shared / (C_B1_online - C_M_online))`, явно указав допущения. Это расчётный порог, не доказанное число будущих применений.

beta<1 budget points можно отображать как промежуточную диагностику, но не включать как готовые posteriors в точностную кривую. Нельзя удалить incomplete/failed cases перед усреднением и заявить общий выигрыш.

## 11. Acceptance gates и разрешённые решения

### G_DEPENDENCY

E01 OW и E02 native probabilistic scope приняты по действительным артефактам. Исторические ограничения явно сохранены.

### PASS_CODE

Реальная физика, generative noise, normalized q и E02 SMC связаны; analytic density/negative tests пройдены; native balance/controls/restart подтверждены; есть beta=1 physical learned run и работающее resume. Scaler/context/truth separation проверено. Missing evidence не допускает PASS.

### G_MC

Для scientific comparisons применять существующий E02 paired N32/N64 screen как ранний diagnostic: zone median gap≤0.03, family mass gap≤0.15, width gap≤25% при width>0.02, иначе absolute gap≤0.01; N64 across-start median SE≤0.01 и family SE≤0.05. Две реализации дают лишь screen, не строгий глобальный convergence proof.

Дополнить unseen-mode/ambiguity/residual checks. Один ESS=N после resampling не достаточен. При провале фиксировать POSTERIOR_NOT_CONVERGED; расширение N только отдельным зарегистрированным diagnostic, без изменения thresholds задним числом.

### PROMISING_STATE

На T1 воспроизводимое снижение state error. B1 и M оцениваются раздельно: положительный H_STATE без выигрыша M ещё не означает положительный H_ML. Для замороженного основного N64 сначала усреднить RMSE по inference seeds внутри мира, затем одинаково по T1 parents. Ориентир: `1 - mean_world(RMSE_method) / mean_world(RMSE_B0) >= 0.10`. N32 остаётся convergence/cost diagnostic, а не способом выбрать меньшую ошибку после просмотра truth. Начальный exploratory ориентир SPEC — RMSE как минимум на 10% ниже B0; primary MAE и CRPS не должны показывать скрытое ухудшение, все worlds/repeats и uncertainty опубликованы. Если baseline error практически нулевой, относительный выигрыш неинтерпретируем: отдельный absolute-error diagnostic и INCONCLUSIVE по relative gate.

T2/T4 не дают подтверждённой необоснованной уверенности; T5 обсуждается с actual coarse/fine discrepancies. Сильная state error без наблюдаемого предупреждения на mismatch ограничивает переносимость и не скрывается за хорошим history fit.

### PROMISING_ML

M показывает более выгодное accuracy–cost соотношение B1 на одинаковом target и заранее заданных parents/budgets, без потери uncertainty/modes; либо одна ограниченная learning-curve ступень 128→256 последовательно улучшает downstream state/uncertainty–cost результат на development. Одного улучшения NLL для этого недостаточно. Cold-start расходы обязательны. Отдельный удачный seed или ниже training NLL не достаточен.

Early ML outcome может быть efficiency-only, provisional или negative. Graph ablation может проиграть set-модели без провала всего learned-inverse направления.

### G_NUMERICS / G_MISMATCH

Принятие E01 reference pair не подменяет P1 refinement. Сохранить actual coarse/fine So/inventory/monthly discrepancies. Небольшой mismatch не обязан заметно портить результат; крупная ошибочная уверенность без флага — существенное ограничение. Отсутствие T5 не позволяет заявить устойчивость к нему.

### Три слоя статусов

1. Process/run: существующие RUNNING/PASS/FAIL.
2. Algorithm: существующие COMPLETE / INCOMPLETE_BUDGET / POSTERIOR_NOT_CONVERGED / NO_TARGET_SUPPORT / EVALUATION_FAILURE.
3. E03 technical status и scientific outcomes: NOT_RUN/IN_PROGRESS/PASS/PASS_WITH_LIMITATIONS/FAIL отдельно от SUPPORTED/NOT_SUPPORTED/INCONCLUSIVE для H_STATE/H_ML/H_AMBIGUITY.

Технически корректно выполненное исследование может завершиться отрицательным ML-результатом. Нельзя требовать положительную гипотезу как условие честного завершения протокола.

| Ситуация | Следующее действие |
|---|---|
| Нет PASS_CODE | Исправлять конкретный runtime/density/data defect; не масштабировать |
| Нет G_MC | Улучшать/диагностировать sampling; не трактовать collapse как физическую определённость |
| B1 не восстанавливает So на T1 | Разделить недостаток наблюдаемости, ошибку постановки/likelihood, support и MC; GPU не решает эту диагностику |
| B1 полезен, ML проигрывает | Один bounded development round: проверить labels/noise, encoder, proposal coverage, v/z partition и summary/set; не запускать бесконечный HPO |
| STATE перспективен, ML не выигрывает | Сохранить baseline/ambiguity результат; честно ограничить ML-утверждение |
| STATE и ML перспективны, mismatch приемлем | Готовить E07 field-like synthetic pilot; масштаб увеличивать только по измеренному бюджету |
| Хорош только history fit | Не выдавать точную So; исследовать неоднозначность и необходимую дополнительную информацию |

## 12. Ресурсы и stop rules

Переиспользовать `BudgetLedger`, resource probe и `PersistentJuliaWorker`. Стартовые рамки действующего P1: один worker, Julia threads=4, BLAS=1; soft 12 GiB, hard 16 GiB с фактическим резервом ОС не менее 6 GiB; обычная сессия до 3600 s / 2000 новых forward attempts и 5 GiB disk по принятому профилю. Это limits, не обещанная длительность всего E03.

Генерация, обучение и SMC — последовательные тяжёлые фазы. Torch threads измеряются; DataLoader workers=0 сначала. Не держать solver state каждой частицы в RAM. Не суммировать RSS и MPS allocation как независимые объёмы одной unified memory.

До 128-world корпуса измерить cold compile и warm forward для T1/T2/T4/T3, epoch/density cost, peak RSS и bytes/forward. T1 timing нельзя автоматически переносить на сложную family или T5 fine grid. Весь BudgetReport включает setup, 168 coarse truths, дополнительные views, raw q, 64 main SMC runs, ablations, retries, checks, plots и checkpoint/export.

Первый прогноз может использовать `expected N × (1 + beta_levels × moves_per_level)` target evaluations для каждого SMC с поправкой на cache/rejections, но реальные adaptive levels/стоимость берутся из pilot. Внешний суммарный experiment budget обязателен: множество автоматических resume по одному часу не должно обходить запрет на несанкционированный long-run.

При wall/memory/disk отказе закончить безопасную единицу, записать checkpoint и pending work; новые jobs не запускать. Отдельные run intents требуются для полного matrix/расширения корпуса/длительной сессии. Ни этот план, ни генерация отчёта не являются таким разрешением.

## 13. Риски и ограничения

| Риск | Diagnostic | Действие |
|---|---|---|
| Noise labels не совпадают с Y | Сверка NoiseTheta(label) с generator metadata, recovery test | Явный noise argument, новый corpus version при исправлении |
| 128 parents мало для выразительной сети | Train/dev NLL, fixed seed variability, set/summary ablations | Небольшая модель; один обоснованный рост до 256 |
| Learnable v не покрывает важные dynamic directions | v/z sensitivity, residual movement, kernel acceptance | Новая partition только с сохранением всех modes и schema version |
| Небольшой N скрывает posterior modes | Multiple starts, doubled N, ambiguity/missed-mode tests | Не путать narrow posterior с незамеченными модами |
| Статический prior чрезмерно определяет карты | B0, information ablations, model-family sensitivity | Ограничить вывод конкретным prior/support |
| Inverse crime/coarse error | T5 и physical same-support comparison | Не переносить in-model accuracy на поле |
| Controls и achieved response смешаны | BHP/rate masks и input allowlist | Не использовать truth rate как известный input |
| Пересекающиеся support/чужой PV искажают score | Hand-calculated tests | Правильная физическая агрегация, непересекающийся primary support |
| Corpus/SMC дисковая стоимость выше ожидаемой | Bytes/forward, whole-matrix forecast | Stop/resume по лимитам; не удалять требуемый lineage ради PASS |
| Единичный объект не окупает обучение | Cold vs amortized ledger | Разрешить efficiency-only/negative ML conclusion |
| Структурный mismatch не обнаружим по Y | State truth + predictive diagnostics на T5 | Явное ограничение переносимости, не обещание универсального OOD detector |

Синтетическая калибровка доказывает свойства относительно объявленного generative model и его диапазона. Она не доказывает правильность полевого prior, событий, PVT и геологии. Извлекаемые запасы/место бурения потребуют последующих paired intervention simulations и полевых проверок.

## 14. Предлагаемый CLI и порядок передачи исполнителю

Имена ниже — **будущий интерфейс**, сейчас они не заявляются существующими:

```text
so-recon --config configs/e03.yml plan-e03
so-recon --config configs/e03_smoke.yml build-ml-corpus
so-recon --config configs/e03_smoke.yml train-proposal
so-recon --config configs/e03_smoke.yml export-proposal
so-recon --config configs/e03_smoke.yml run-learned-loop
so-recon --config configs/e03.yml compare-learned-inverse
so-recon --config configs/e03.yml validate-e03
```

`plan-e03` показывает dependency state, registered jobs и budget forecast без physics/training. Execution-команды требуют явного выбора зарегистрированного experiment/run intent. `validate-e03` только читает evidence.

Реализовывать сначала Tasks 00–09 до первого физического отчёта, затем Tasks 10–12 по результатам. Каждая задача сдаётся с изменёнными файлами, тестами и конкретным артефактом/проверяемым поведением. Не тратить цикл на ещё одну общую архитектурную спецификацию и не создавать новый solver/SMC/registry.

Git operations этим документом не авторизуются. Работа с ветками/коммитами/PR требует отдельного разрешения пользователя.

## 15. Definition of done

E03 завершён как выполненный технический протокол, когда существующая основа сохранена, новый корпус имеет корректные conditional labels/noise и parent splits, обученная q нормирована и воспроизводима, defensive SMC работает через старый engine, native So сравнивается с truth на фиксированном support, весь бюджет и неуспехи учтены, T2/T4/T5 обработаны с честным scope, а отчёт отдельно отвечает на H_STATE и H_ML.

Положительный результат E03 означает разрешение готовить следующий малый field-like эксперимент. Он не означает доказанную точность карты реального месторождения, доказанную новизну статьи или экономическую эффективность бурения.

**Практическое решение:** продолжать, но следующий крупный результат должен быть измерением качества So и стоимости, а не увеличением объёма инфраструктуры.

## Основания и навигация по изученным исходникам

Все repository references ниже относятся к commit `1810ec4af6234ffb718461220a98dce852da2351`:

- `docs/SPEC.md`: §§11–14, 17–19, 22–23 — q/SMC, state metrics, provenance, first-loop gates.
- `docs/STAGES.md`: E02/E03 и зависимости дальнейших этапов.
- `docs/COMPUTE_PROFILES.md`: P1 sizes, 128/32/8, CPU64 density, resource caps.
- `reports/stages/E01.md`, `reports/stages/E02.md`: актуальные сохранённые evidence summaries.
- `docs/superpowers/plans/2026-09-13-e01-forward-synthetic.md`.
- `docs/superpowers/plans/2026-09-14-e01-physics-review-amendment.md`.
- `docs/superpowers/plans/2026-09-15-e01-completion.md`.
- `docs/superpowers/plans/2026-09-14-e02-probabilistic-inverse.md`.
- `configs/e02_experiments.json`: native parents, N, seeds, старый infeasible design.
- `src/so_recon/geology/density.py`: Density и GaussianConditionalPrior.
- `src/so_recon/geology/renderer.py`: physical transforms, residual, NoiseTheta.
- `src/so_recon/inference/contracts.py`: layouts, latent measure, status semantics.
- `src/so_recon/inference/proposals.py`, `smc.py`, `target.py`: integration seams, bridge, hashes, caches.
- `src/so_recon/synthetic/inverse_worlds.py`, `inverse_designs.py`: truth separation, fixed-noise default, T2/T4 layouts.
- `src/so_recon/validation/physical_smc.py`, `inverse_metrics.py`: current runner/evaluator и необходимые различия E03.
- `tests/unit/test_smc_engine.py`, `tests/integration/test_e02_physical_smc.py`: resume/failure handling и фактический native execution gate.
- `pyproject.toml`: текущие dependencies/typing/tests.

Методические внешние основания: Durkan et al., *Neural Spline Flows*, arXiv:1906.04032; авторский `bayesiains/nflows`; официальная PyTorch documentation, MPS backend / double-precision types. NSF выбран за нормированную вычислимую density и обратимый transform, а не за обещание лучших результатов на данных SO-RECON. MPS не поддерживает Float64; обязательная operational density выполняется на CPU.
