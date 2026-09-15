# E02 — Probabilistic Inverse and Early Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Получить проверенную малую обратную задачу без ML: нормированные prior/likelihood, универсальный SMC, независимый физический reference и карты истинной/восстановленной So с неопределённостью и полным бюджетом.

**Architecture:** Python CPU Float64 отвечает за латентную меру, conditioning, наблюдения, SMC и evaluator; принятый E01 JutulDarcy через один persistent worker рассчитывает каждую физическую реализацию. Observation noise и likelihood используют одну условную bin probability, а SMC принимает нормированное стартовое распределение через интерфейс density, чтобы E03 подключил NSF без замены алгоритма. Truth хранится отдельно от inverse context; незавершённый расчёт сохраняет точное состояние алгоритма и не получает статус готового posterior.

**Tech Stack:** Существующие Python 3.13 / uv / NumPy / Pydantic / PyArrow / HDF5 / pytest, Julia с текущими Project/Manifest и JutulDarcy; SciPy для Student-t, линейной алгебры и quadrature; Matplotlib для статических научных рисунков. Новые версии фиксируются в uv.lock после проверки совместимости, без обновления Julia ради E02.

**Spec:** [SPEC.md](../../SPEC.md), редакция 4.0, §§5, 7, 9–10, 13, 14.2, 15.1.1, 17–18, 22–23; [STAGES.md — E02](../../STAGES.md#e02--вероятностная-обратная-задача-и-ранняя-наблюдаемость); [COMPUTE_PROFILES.md](../../COMPUTE_PROFILES.md); [DECISIONS.md](../../DECISIONS.md). Зависимость: [план E01](2026-09-13-e01-forward-synthetic.md), прежде всего Task 12.

## Global Constraints

Следующие требования процитированы из нормативных документов; уточнения реализации ниже не отменяют их.

- «Новые планы используют полный заголовок этапа и `spec_version=4.0`.» — SPEC §0.
- «Редакция документа не сбрасывает выполненную работу.» — SPEC §0. Исторические E00 config, smoke, reports и records остаются 3.0.
- «Все оставшиеся моды сохраняются»; «Зануление остатка или замена его единственной средней реализацией запрещены.» — SPEC §7.5.
- «Статические сведения, использованные для conditioning prior, не умножаются второй раз как независимая likelihood.» — SPEC §5.
- «Если beta<1, это промежуточное распределение, **не posterior**» — SPEC §13.4.2: сохранять `INCOMPLETE_BUDGET`, не принуждать beta к 1.
- «Стартовая цель — 0,8N» для CESS; «по умолчанию ESS < 0,5N» для resampling — SPEC §13.2.
- «SMC/prior/likelihood | CPU Float64, log-space»; «Solver precision | Float64» — COMPUTE §5.
- «Стартовые пределы проекта: **soft 12 GiB, hard 16 GiB**»; «с резервом не менее 6 GiB для ОС и прочих процессов» — COMPUTE §5.
- «BLAS threads на worker | 1 по умолчанию для исключения вложенного распараллеливания» — COMPUTE §5. Один Julia worker; начальная настройка Julia threads=4.
- P0_VERIFY: 600 s / 64 новых forward; P1_LOOP: 3600 s / 2000 новых forward; disk budget=5 GiB для малой сессии — COMPUTE §§7, 10 и действующие ResourceProfile E01. Лимит включает failed/retry attempts; новый вызов resume не скрывает накопленные затраты.
- «Long-run не запускается агентом автоматически после E00/E01; требуется явный run intent пользователя и пройденные gates.» — SPEC §18.4. Этот документ не запускает никакие solver/training jobs.
- «Full field runs, ML обучение и long-run не запускаются как побочный эффект обновления документации.» — SPEC §23.2.

---

## 1. Статус плана и граница с E01

План подготовлен 14.09.2026 в master `542f94c`; реализация E01 просмотрена в `.worktree/e01-forward-synthetic` на `af0464e` (`feat: generate reproducible two-layer synthetic truth worlds`). В этом снимке есть Tasks 1–11, native test modules, `simulate`, `BudgetLedger`, `CaseBundle`, `ForwardResult` и P1 generator. `reports/stages/E01.md` ещё отсутствует. Наличие кода/тестов здесь **не подтверждает их свежий PASS**: при составлении плана Julia и scientific jobs не запускались. Независимое изменение `.idea/so_field.iml` не относится к этому плану.

Во время проверки плана E01 продвинулся до `2d66d50`: исправлена публикация case/context и проверка принадлежности forward конкретному world. Просмотренный diff ограничен `synthetic/world_io.py` и его unit/integration tests; эта правка не заменяет Task12 acceptance. Сохранить её целиком в основе E02; source-line pointers в таблице ниже оставлены относительно первоначально изученного снимка.

**Планировать E02 уже можно.** Начать Tasks 1–5 и 7–10 (контракты и чистую математику) можно на совместимом E01 checkout, пока его исполнитель завершает Task 12. Native Tasks 6, 11, 13 и итоговая приёмка Task 14 требуют принятого OW scope E01: баланс, физические проверки, native restart, worker isolation, измеренный runtime/RAM и ссылки на evidence. BO Task 13 E01 не блокирует OW E02; BO capability остаётся обязательной отдельной зависимостью до E06 PhysicsDecision.

Перед реализацией создать отдельный E02 worktree через `using-git-worktrees` от **фактического финального E01 commit** либо master после его интеграции. Не менять рабочее дерево занятого E01 агента и не cherry-pick выборочные simulator files. Если E01 ещё меняется, математические задачи можно выполнить от зафиксированного снимка; перед native задачами перенести всю согласованную E01 основу, повторить API compatibility checks и перепривязать evidence. Этот план не разрешает автоматически merge E01 в master.

Задача 12 E01 — не только оформление: без неё нельзя утверждать, что физические входы/выходы и стоимость уже приняты. Оставшиеся defects E01 устраняются до dependent native gate, а не маскируются настройками likelihood или объявлением So ненаблюдаемой.

### Фактические точки интеграции

Пути ниже относительно корня **E02 checkout**; строки относятся к просмотренному `af0464e`, при переносе ориентироваться на имя функции.

| Уже существует | Использовать в E02 |
|---|---|
| `simulator/contracts.py:535`, `:714`, `:782` | `CaseBundle`, `OutputRequest`, `ForwardResult`; ArrayRef с axes/units/hash |
| `simulator/forward.py:739` | `simulate(case, output_request, *, worker, ctx, ledger, solver_config=None) -> ForwardResult` |
| `simulator/case_io.py:504` | `write_case(case, paths, ctx, *, now=None) -> ArtifactRef`; `write_arrays`, `load_array`, `compute_model_hash` |
| `simulator/budget.py:547`, `simulator/worker.py` | Один `BudgetLedger` на session, один `PersistentJuliaWorker`, прежняя retry policy |
| `synthetic/p1.py:159`, `:299`, `:755` | Формула `layer_fields`; проверенный `layer_geology`; `render_p1(seed, design)` — forward generator, **не** conditional prior |
| `synthetic/world_io.py:315` | `build_p1_case(world, paths, ctx)` публикует arrays; схему записи использовать как образец, не обращаться из inference к truth-world |
| `registry/run.py`, `registry/artifact.py`, `runner.py` | `RunContext`, `ArtifactRef`, atomic publication, `execute_run`; не создавать второй registry |

### Научная постановка E02

1. Сначала exact toy mathematics, затем native reduced inverse с **одной** неизвестной (допустимые 1–4), затем P1 T1/T2/T4. T3/T5, обученная q, неизвестные полевые PVT, полный LIS и adjoint относятся к последующим этапам.
2. Для P1 использовать конечную 12-мерную геологическую модель E01, но задать её **условный** закон по sparse log(k). Восемь whitened geology coordinates входят в `v`, четыре в `z_perp`; это полный конечный prior, а не отсечение неизвестного бесконечного поля. В `v` дополнительно входят три nuisance coordinates sigma/rho/log-bias: всего `dim(v)=11`, `dim(z_perp)=4`.
3. Ограниченная дискретная гипотеза `s ∈ {0,1}` задаёт `kz/kx ∈ {0.05,0.001}` с p(s)=0.5. Состав геологии и её размерность одинаковы; s не является ID мира. T1/T4 поддерживают этот prior; T2 использует объявленную симметричную отдельную постановку с фиксированным s.
4. Информация `G` данного раннего эксперимента — только sparse **log(k)** на объявленных supports, sigma=0.2 ln(m²), известные geometry/design/controls. E01 noisy phi исключена из conditioning и scoring явно (`use='heldout_diagnostic'`), потому что совместная nonlinear phi likelihood не даёт используемого ниже аналитического Gaussian conditional. Это информационная абляция, одинаковая для всех методов, а не заявление об использовании всех E01 observations. Старые E01 So_log с Gaussian noise не переименовывать в bounded-bin данные. Новые bounded So_log для проверки дат создавать под новой observation version; P1 основной watercut-only опыт не получает скрытый дополнительный So channel.
5. Для нового E02 parent сначала зафиксировать design/G, затем sample из соответствующего `p0(theta|G)` и получить truth через F. Источник G — отдельный context stream совместного Gaussian log(k) generator; исходная pre-conditioning realization не используется как final truth. Старые пять E01 parents остаются forward evidence, новый experiment ID не превращает их в независимый blind test.
6. `L=1` на пустом наборе обратных наблюдений; при r=p0 результат совпадает с **условным** prior. Это не безусловная N(0,I) в физических коэффициентах: whitening уже содержит G. Gas/pressure не включать; fw, Vo, Vw, total liquid и cumulative production не складывать как независимые measurements.

## 2. Карта файлов и публичные контракты

Каждый новый пакет получает `__init__.py` в своей первой задаче. Существующие большие E01 modules не дробить попутно. Все новые schema models наследуют `StrictModel` (`extra='forbid'`), SHA/relative paths используют имеющиеся типы.

| Задача | Новые файлы / ответственность |
|---|---|
| 1 | `inference/contracts.py`, `config/inference.py`, `validation/e01_dependency.py`: typed records, settings, evidence gate |
| 2 | `geology/density.py`, `geology/conditional.py`, `geology/renderer.py`: mixed density, Gaussian conditioning, full latent renderer |
| 3 | `observation/bins.py`: нормированная bounded-bin Student-t kernel |
| 4 | `observation/history.py`, `observation/noise.py`: masks/reset/time и тот же noise process |
| 5 | `observation/logs.py`: support/date mixture/group bias |
| 6 | `observation/predict.py`, `inference/target.py`, `inference/cache.py`: настоящий F→L и раздельные caches |
| 7 | `inference/weights.py`, `inference/resampling.py`: log weights, beta/CESS, genealogy indices |
| 8 | `inference/proposals.py`, `inference/kernels.py`: defensive density и инвариантные MH transitions |
| 9 | `inference/smc.py`, `inference/checkpoint.py`: engine и точное продолжение |
| 10 | `validation/toy_inverse.py`: независимые reference distributions и calibration tests |
| 11 | `synthetic/reduced_inverse.py`, `validation/reference_inverse.py`: native low-dimensional reference, noise recovery |
| 12 | `validation/support.py`, `validation/inverse_metrics.py`, `validation/inverse_plots.py`: общая геометрия, weighted metrics, рисунки |
| 13 | `synthetic/inverse_worlds.py`, `validation/observability.py`: T1/T2/T4 experiment definitions и ambiguity evidence |
| 14 | `inference/commands.py`, `validation/e02_report.py`, `scripts/e02_gate.sh`, `configs/e02.yml`, `configs/e02_experiments.json`, `reports/stages/E02.md`: CLI, бюджеты и фактическая приёмка |

Общая нотация типов: `F64 = numpy.typing.NDArray[np.float64]`; `I64 = NDArray[np.int64]`. Runtime constructors проверяют размерности, конечность разрешённых полей, сумму probability weights; `-inf` допустим только для log-density вне поддержки, NaN и +inf запрещены. В JSON `-inf` хранить как `null` плюс `in_support=false`, не как невалидный JSON Infinity.

Ключевые records, реализуемые Task 1:

- `ThetaRecord(schema_id: str, s: int, v: tuple[float,...], z_perp: tuple[float,...], basis_hash: str)`. Порядок v: 8 geology, noise_sigma, noise_rho, log_bias. Для toy/reduced schema задаёт свой размер. `log_prior_parts` хранить как проверяемые diagnostics в Particle, не как доверенный вход `log_prob`.
- `DensitySchema(schema_id, n_v, n_residual, families: tuple[int,...], basis_hash, transform_version, measure='counting_x_latent_lebesgue')`. Renderer Jacobian не добавлять в латентную density.
- `PriorContext(schema, n_geology: int, n_state_residual: int, mean: F64, chol: F64, rotation: F64, design: dict, g_hash: str, information_hash: str)`. mean имеет shape `(n_geology,)`, chol/rotation — `(n_geology,n_geology)`. Основной P1/T2: n_geology=12,n_state_residual=0; T4:13/1. Geology vector собирается как `concat(v[:8],z_perp[:n_geology-8])`; независимые initial-state coordinates — оставшийся tail z_perp. Arrays persist через HDF5 refs; shape проверяется при loading. Reduced/toy имеют собственный renderer/schema и не используют P1 slicing.
- `NoiseTheta(sigma: float, rho: float, nu: float, log_bias: float)`; `nu=5`, `sigma=0.015+0.065*Phi(v[8])`, `rho=0.8*Phi(v[9])`, `log_bias=0.03*v[10]`. Единицы sigma/bias — g-space. Для fixed-noise tests `sigma=.03, rho=.4, nu=5, log_bias=0`.
- `HistoryRow(well_id, month_index, raw_value: float|None, bin_index: int|None, quality_group: str, observed_valid: bool, reset: bool, sigma_multiplier: float=1)`. `month_index` — календарный ordinal от начала; positive multiplier задаётся наблюдаемым качеством, не fitted residual.
- `LogRow(observation_id, support_cell_ids, support_weights, bin_index, date_times_s, date_weights, date_group, bias_group, valid, use)`; одинаковая неизвестная дата для нескольких интервалов — один общий `date_group` и одна совместная marginalization, не произведение независимых mixtures. E02 поддерживает одну bias_group на bundle; дополнительная группа требует расширенной noise schema с отдельным anchored latent, не молчаливого совместного bias.
- `ObservationBundle(history: tuple[HistoryRow,...], logs: tuple[LogRow,...], bin_edges_by_group: dict, cutoff_s: float, information_hash: str, observation_hash: str)`; sorted unique `(well_id,month_index)`, observation-role registry (`condition_prior`, `likelihood`, `heldout_diagnostic`).
- `ModelObservations(fw: dict[tuple[str,int],float|None], so_support: dict[tuple[str,float],float], model_hash: str)`.
- `LoglikResult(value: float, terms: tuple[float,...], n_used: int, observation_hash: str)`; no observations → `value=0,terms=(),n_used=0`.
- `TargetEvaluation(theta, log_p0, log_l, log_r, forward_ref: ArtifactRef|None, cache_key: str)`; `Particle(particle_id, ancestor_id, evaluation)`.
- `InferenceConfig(n_particles=32, cess_fraction=.8, resample_fraction=.5, max_beta_steps=24, moves_per_level=2, seed, rw_scale=.25, pcn_scale=.2)`; no hidden adaptation.
- `SMCState(particles, log_weights, beta, level, log_evidence, phase, cursor, rng_state, proposal_hash, target_hash, pending_proposal, pending_log_u)`; `phase ∈ {initialize,reweight,resample,rejuvenate,finalize}`.
- `PosteriorBundle(state_ref, particles_ref, diagnostics_ref, ledger_ref, algorithm_status, beta, convergence_status, physical_state_refs, parent_run_ids)`. Статусы алгоритма SPEC + отдельный `EVALUATION_FAILURE` для runtime/likelihood exceptions; process и science status не смешивать.

В `inference/contracts.py` определить исключения `E01DependencyError`, `RendererNumericalError`, `LikelihoodNumericalError`, `ObservationSupportMismatch`, `UnsupportedObservationError` как subclasses `ValueError`/`RuntimeError` с сохранённой reason; в `weights.py` — `NoTargetSupport(RuntimeError)`. `SMCState` также содержит `algorithm_status` и `diagnostics: dict`. Ни одна ошибка этого списка не конвертируется в физический нулевой likelihood общим catch-all.

Records в этом документе — проектируемые интерфейсы, не готовый код. Каждый code block ниже — точная реализация математического ядра либо тестовый пример, который исполнитель должен встроить и проверить; это не свидетельство выполненного эксперимента.

## 3. План задач

### Task 1: Типизированный inverse input и исполняемая граница E01

**Files:** Create `src/so_recon/inference/__init__.py`, `src/so_recon/inference/contracts.py`, `src/so_recon/config/inference.py`, `src/so_recon/validation/e01_dependency.py`, `tests/unit/test_inverse_contracts.py`, `tests/unit/test_e01_dependency.py`; Modify `src/so_recon/config/schema.py` (optional inference field для 4.0), `pyproject.toml`, `uv.lock`.

**Interfaces:** Consumes existing `StrictModel`, `ArtifactRef`, `ResourceProfile`, `ProjectConfig`. Produces records §2; `require_e01_ow(evidence_path: Path, paths: ProjectPaths) -> dict[str, object]`; invalid/missing/stale evidence raises `E01DependencyError`. Math tests не вызывают этот gate.

- [ ] **1.1 RED: написать проверки размеров, unsupported family, unknown fields, versions и gate.** Минимальный исполняемый пример:

```python
import pytest
from so_recon.inference.contracts import DensitySchema, ThetaRecord

def test_dimension_is_not_silently_truncated():
    schema = DensitySchema(schema_id="test", n_v=2, n_residual=1,
        families=(0, 1), basis_hash="a" * 64, transform_version="identity-1")
    theta = ThetaRecord(schema_id="test", s=0, v=(0.,), z_perp=(1.,),
                        basis_hash="a" * 64)
    with pytest.raises(ValueError, match="dimension"):
        schema.validate_theta(theta)
```

Дополнительные exact cases: 3.0 config + inference → validation error; 4.0 без inference читает E01; theta s=2 → support error; NaN coordinate → error; `evidence.status=PASS` при missing restart row → `E01DependencyError`; evidence SHA не совпадает с artifact → refusal. Fixtures evidence формировать из временных файлов с реальными SHA, не из строки PASS.
- [ ] **1.2 Run RED:** `uv run pytest tests/unit/test_inverse_contracts.py tests/unit/test_e01_dependency.py -q`; expected missing new module/contract assertion, не missing Julia.
- [ ] **1.3 Реализовать records и validator.** `DensitySchema.validate_theta` проверяет schema/basis/family/length; нормализация вероятностей проверяется с `abs(sum(p)-1)<=1e-12`, положительные p не нормализуются молча. Создать протокол evidence `e01-ow-dependency-1`: `accepted_commit`, `code_tree_hash`, `lock_hash`, `ow_status`, `checks` (ID/status/metric/limit/artifact ref), `benchmark_ref`, `restart_ref`, `report_ref`. Adapter читает окончательный машинный отчёт E01 Task 12; если schema там другая, mapping реализовать в этом одном модуле, без выдумывания results.

```python
def validate_theta(self, theta):
    if theta.schema_id != self.schema_id or theta.basis_hash != self.basis_hash:
        raise ValueError("schema/basis mismatch")
    if theta.s not in self.families:
        raise ValueError("family outside support")
    if len(theta.v) != self.n_v or len(theta.z_perp) != self.n_residual:
        raise ValueError("latent dimension mismatch")
```

Gate проверяет mandatory OW matrix из финального E01, в том числе analytic/native physics, five-spot/refinement, restart, controls/boundaries и resource failures. BO NOT_RUN допустим. Новый E02 commit не обязан равняться E01 commit: доказать ancestry и отсутствие необъяснённых изменений в physics/configs/locks; при изменении этих компонентов повторить затронутые native gates и обновить evidence lineage.
- [ ] **1.4 Добавить SciPy при первом потребителе через `uv add scipy`, зафиксировать resolver result; Matplotlib добавить в Task 12.** Здесь dependency addition нужна для density/likelihood, ML frameworks не добавлять. Применить typing override только к пакетам без stubs, не отключать strict project typing.
- [ ] **1.5 Run GREEN:** команды 1.2, `uv run pytest tests/unit/test_config.py tests/unit/test_e01_versions.py -q`, `uv run ruff check .`, `uv run mypy`. Commit: `feat: define E02 inverse contracts and E01 dependency gate` с перечисленными файлами (`git add` только этого task).

### Task 2: Нормированный conditional prior и renderer без потери residual

**Files:** Create `src/so_recon/geology/__init__.py`, `src/so_recon/geology/density.py`, `src/so_recon/geology/conditional.py`, `src/so_recon/geology/renderer.py`, `tests/unit/test_conditional_prior.py`, `tests/unit/test_latent_renderer.py`; Modify `src/so_recon/synthetic/p1.py` только для выделения public deterministic coefficient renderer, с прежней generator version/output regression.

**Interfaces:** Consumes Task 1 schema; existing `layer_fields`, physical constants. Produces `Density.sample(n: int, rng: np.random.Generator) -> tuple[ThetaRecord,...]`, `Density.log_prob(theta) -> float`, `Density.fingerprint: str`; `GaussianConditionalPrior(context)` implements Density. `condition_gaussian(A: F64, y: F64, variance: F64) -> tuple[F64,F64]` returns mean/covariance. `render_theta(theta, context) -> RenderedParameters` returns arrays, `NoiseTheta`, family and version/hash; no file writes/truth reads. `RenderedParameters` dataclass has `arrays: dict[str,F64]`, `noise: NoiseTheta`, `family: int`, `renderer_hash: str`. Из E01 выделить `render_coefficients(coefficients:F64, design:P1Design)->dict[str,F64]`: чистая часть `render_p1`, создающая arrays; прежний `render_p1` вызывает её с тем же RNG draw и сохраняет неизменные world outputs.

- [ ] **2.1 RED: scalar analytic conditioning, density normalization, renderer parity/residual.**

```python
import numpy as np
from so_recon.geology.conditional import condition_gaussian

def test_gaussian_conditioning_has_known_mean_and_variance():
    mean, cov = condition_gaussian(np.array([[1.]]), np.array([2.]), np.array([1.]))
    np.testing.assert_allclose(mean, [1.], atol=1e-12)
    np.testing.assert_allclose(cov, [[.5]], atol=1e-12)
```

Add actual assertions on 20,000 cheap prior draws: whitened means <.04, diagonal variances within .05, s frequency within .02; residual two distinct draws change K/phi; `render(z)` deterministic; old `render_p1` arrays repeat identically after extraction; transform physical k=exp(z) still gives log_prior=normal.logpdf(z), without extra `-z` in latent measure.
- [ ] **2.2 Run RED:** `uv run pytest tests/unit/test_conditional_prior.py tests/unit/test_latent_renderer.py -q`.
- [ ] **2.3 Реализовать exact Gaussian conditioning для sparse log(k).** `a~N(0,I12)`, `G-b=Aa+e`, `e~N(0,R)`. b включает `ln(base_mD*MILLIDARCY_M2)`. A строится из **тех же** cosine modes/weights и support averaging, что E01; conditional mean/covariance:

```python
import numpy as np
from scipy.linalg import cho_factor, cho_solve

def condition_gaussian(A, y, variance):
    A, y, variance = (np.asarray(x, dtype=np.float64) for x in (A, y, variance))
    if A.ndim != 2 or y.shape != (A.shape[0],) or variance.shape != y.shape:
        raise ValueError("conditioning shape mismatch")
    if not all(np.isfinite(x).all() for x in (A, y, variance)) or np.any(variance <= 0):
        raise ValueError("invalid conditioning inputs")
    precision = np.eye(A.shape[1]) + A.T @ (A / variance[:, None])
    factor = cho_factor(precision, lower=True)
    cov = cho_solve(factor, np.eye(A.shape[1]))
    mean = cho_solve(factor, A.T @ (y / variance))
    return mean, (cov + cov.T) / 2
```

Построить orthogonal Q из eigendecomposition `A.T @ R^-1 @ A`, descending eigenvalues; column sign фиксировать по максимальному absolute entry. Для eigenvalue tie с relative tolerance1e-12 канонизировать basis подпространства: projector на tied block, проекции стандартных unit vectors по возрастанию coordinate index, modified Gram–Schmidt, пропуск norm<1e-10; persist полученные Q/eigenvalues/version/hash, при resume не пересчитывать. `a=mean+chol(cov)@Q@concat(v[:8],z_perp)`. Q orthonormal в latent coordinate space; cosine field basis E01 сам по себе **не** объявлять ортонормированным. Все 12 направлений используются. Geology sample/log_prob — standard normals в whitened coordinates; s=.5/.5, nuisance independent standard normals. Jacobian chol/Q относится к физическим a-density только при явном запросе такой density, не добавляется в latent p0.
- [ ] **2.4 Реализовать renderer с проверками инженерного диапазона.** E01 `layer_geology` отказывает на экстремальной K: это не нормированное усечение Gaussian prior. Не rejection-sample «пока красиво» и не возвращать `-inf` по такому исключению; отдавать `RendererNumericalError` с theta hash, останавливать физическую оценку как failure. Если нужен конечный physical support, требуется новая явная prior version с sample/normalizer tests, а не silent clipping. Для штатных draws использовать `layer_fields`, проверку finite/positive, прежнюю связную phi–k формулу, pressure initializer и shape conventions. Пара s определяет только kz/kx; mean/cov G одинаковы у этих двух family, значит p(s|G)=.5.
- [ ] **2.5 Run GREEN:** команда 2.2 + `uv run pytest tests/unit/test_p1_worlds.py -q`. Commit: `feat: condition the full synthetic latent prior on sparse geology`.

### Task 3: Устойчивые bounded-bin Student-t probabilities

**Files:** Create `src/so_recon/observation/__init__.py`, `src/so_recon/observation/bins.py`, `tests/unit/test_bounded_bins.py`.

**Interfaces:** Produces `BinGrid(centers: F64, edges: F64)`; `rounding_grid(step: float) -> BinGrid`; `bin_index(raw: float, grid: BinGrid) -> int`; `log_interval_t(lo, hi, *, mu, sigma, nu) -> float`; `bin_log_probs(grid, *, mu, sigma, nu) -> F64`. Внешние границы exactly 0/1; interiors — midpoints соседних report centers; у 0/1 половинные bins. Все аргументы lo/hi/mu в g-space, `g(f)=arcsin(sqrt(f))`.

- [ ] **3.1 RED: partition normalization, half-bins и extreme tails.**

```python
import numpy as np
import pytest
from scipy.special import logsumexp
from so_recon.observation.bins import rounding_grid, bin_log_probs, bin_index

@pytest.mark.parametrize("mu,sigma", [(0.,.03),(np.pi/2,.03),(-20.,.001),(20.,.001)])
def test_bins_form_a_probability_partition(mu, sigma):
    grid = rounding_grid(.01)
    lp = bin_log_probs(grid, mu=mu, sigma=sigma, nu=5.)
    assert np.isfinite(lp).all()
    assert abs(logsumexp(lp)) < 1e-10
    assert bin_index(0., grid) == 0
    assert bin_index(1., grid) == 100
    assert grid.edges[1] == .005
```

Дополнительно сравнить 11 bins с независимым `scipy.integrate.quad` по t.pdf; sigma≤0, nu≤0, repeated/gapped edges, raw outside [0,1] → ValueError, не clipping. Boundary equality принадлежит правому bin кроме f=1 (последний). Narrow tail error ≤1e-8 в log probability на declared numerical test grid.
- [ ] **3.2 Run RED:** `uv run pytest tests/unit/test_bounded_bins.py -q`.
- [ ] **3.3 Реализовать interval kernel.** Не вычитать две величины около 1 в linear space; negative side через logcdf, positive через logsf; interval spanning zero через CDF difference с `log1p/expm1`.

```python
import numpy as np
from scipy.stats import t

def logdiffexp(a, b):
    if not b < a:
        raise FloatingPointError("unresolved CDF interval")
    return float(a + np.log(-np.expm1(b - a)))

def standardized_interval(lo, hi, nu):
    if not lo < hi:
        raise ValueError("empty interval")
    if lo >= 0:
        return logdiffexp(t.logsf(lo, nu), t.logsf(hi, nu))
    return logdiffexp(t.logcdf(hi, nu), t.logcdf(lo, nu))
```

При coincident floating-point CDF values применять независимую scaled logpdf quadrature на [lo,hi]: anchor=`t.logpdf(clip(0,lo,hi),nu)`; integral=`quad(exp(logpdf(x)-anchor),lo,hi,epsabs=1e-12,epsrel=1e-10,points=[0] if lo<0<hi else None)`; если error/integral>1e-8 либо integral≤0, `LikelihoodNumericalError`, а не floor 1e-300. Разбивать интервалы при резком масштабе вокруг 0; тесты должны ловить непокрытый пик. `log_interval_t` стандартизует lo/hi; bounded probability=`log_interval_t(g(a),g(b))-log_interval_t(0,pi/2)`. Sigma normalizer учтён через CDF; произвольный MSE или дополнительный `-log sigma` к bin probability не добавлять.
- [ ] **3.4 Run GREEN:** команда 3.2; numeric reference uses independent integration, не тот же production helper. Commit: `feat: implement normalized bounded Student-t observation bins`.

### Task 4: Calendar masks, recursive noise и watercut likelihood

**Files:** Create `src/so_recon/observation/history.py`, `src/so_recon/observation/noise.py`, `tests/unit/test_history_likelihood.py`, `tests/unit/test_recursive_noise.py`.

**Interfaces:** Consumes `HistoryRow`, `ModelObservations`, `NoiseTheta`, Task 3 bins. Produces `history_loglik(rows, prediction: dict[tuple[str,int],float|None], noise, grids) -> LoglikResult`; `draw_history(rows, prediction, noise, grids, rng) -> tuple[HistoryRow,...]`. Shared `conditional_mean(predicted: float, previous: tuple[int,float,float]|None, month: int, rho: float, reset: bool) -> float`; previous = month/noisy_center/predicted at last valid observation.

- [ ] **4.1 RED: реальный gap, reset и end points.**

```python
import numpy as np
from so_recon.observation.history import conditional_mean

def test_gap_uses_calendar_distance_and_observed_bin_center():
    g = lambda x: np.arcsin(np.sqrt(x))
    actual = conditional_mean(.4, (2,.8,.2), 5, .5, False)
    assert np.isclose(actual, g(.4) + .5**3 * (g(.8)-g(.2)))
    assert conditional_mean(.4, (2,.8,.2), 5, .5, True) == g(.4)
```

Rows unobserved at months 3/4 не становятся предыдущими. Reset в пропущенном месяце обязан сбросить память перед следующим наблюдением. Well IDs имеют независимые buffers. NaN valid prediction → failure. Наблюдённая positive liquid + predicted dry period не превращается в masked observation: `ObservationSupportMismatch`; после declared feasibility policy это data incompatibility, не permission отбросить строку.
- [ ] **4.2 Run RED:** `uv run pytest tests/unit/test_history_likelihood.py tests/unit/test_recursive_noise.py -q`.
- [ ] **4.3 Реализовать один conditional kernel для generator/scorer.**

```python
import numpy as np

def conditional_mean(predicted, previous, month, rho, reset):
    g = lambda x: np.arcsin(np.sqrt(x))
    if previous is None or reset:
        return float(g(predicted))
    old_month, old_center, old_prediction = previous
    if month <= old_month:
        raise ValueError("non-increasing month")
    return float(g(predicted) + rho**(month-old_month) *
                 (g(old_center)-g(old_prediction)))
```

Generator выбирает bin через `rng.choice(len(lp),p=exp(lp-logsumexp(lp)))`; это exact discrete law приведённой модели, не Gaussian noise с clipping. Использовать выбранный grid.center как published rounded observation и следующий AR residual. Сохранять RNG state/seed/noise version и исходные masks; no valid observation → no draw. Скорер суммирует исходные log probabilities, без renormalization across actual rows. Quality multiplier >0 умножает sigma; групповая sigma не становится отдельным параметром месяца. Пропуски preserve calendar Delta; declared reset сбрасывает AR даже без measured value.
- [ ] **4.4 Проверить совместный закон независимо:** на grid=.5 (3 bins), 3 valid months с gap перебрать 27 последовательностей и получить sum joint probabilities=1±1e-10. 20,000 generated sequences сравнить с enumerated probabilities через deterministic multinomial intervals; rho=0 даёт произведение independent bins. Zero measured liquid/invalid flag даёт отсутствующий fw, не raw=0. Unknown future observations генерировать последовательно.
- [ ] **4.5 Run GREEN:** команда 4.2. Commit: `feat: share the recursive watercut model between noise and likelihood`.

### Task 5: Датированный So_log, геометрический support и group bias

**Files:** Create `src/so_recon/observation/logs.py`, `tests/unit/test_log_observation.py`.

**Interfaces:** Consumes `LogRow`, `NoiseTheta`, Task 3 kernel. Produces `log_date_mixture(log_terms: F64, probabilities: F64) -> float`; `logs_loglik(rows, prediction: ModelObservations, noise, grids) -> LoglikResult`; `draw_logs(rows, prediction, noise, grids, rng) -> tuple[LogRow,...]`.

- [ ] **5.1 RED: mixture не max и не mean(log L).**

```python
import numpy as np
from so_recon.observation.logs import log_date_mixture

def test_dates_are_marginalized():
    result = log_date_mixture(np.log([.1,.8]), np.array([.25,.75]))
    assert np.isclose(result, np.log(.625))
    assert not np.isclose(result, np.log(.8))
```

Также: два интервала с одной date_group используют `logsumexp(log(w_d)+sum_i log L_id)`, а не sum_i mixtures; shared bias draw один на группу; later log не меняет initial Sw; conditioned-G observation не входит повторно в L; unknown date вне requested state axis → refusal.
- [ ] **5.2 Run RED:** `uv run pytest tests/unit/test_log_observation.py -q`.
- [ ] **5.3 Реализовать date integral и anchoring.**

```python
import numpy as np
from scipy.special import logsumexp

def log_date_mixture(log_terms, probabilities):
    if log_terms.shape != probabilities.shape or np.any(probabilities < 0):
        raise ValueError("invalid date mixture")
    if not np.isclose(probabilities.sum(), 1., atol=1e-12, rtol=0):
        raise ValueError("date probabilities must sum to one")
    keep = probabilities > 0
    return float(logsumexp(np.log(probabilities[keep]) + log_terms[keep]))
```

`H_log` — объявленное normalized geometric support, не completion flux weights. Для дат d вычислить mu=`g(So_support[d])+group_bias`, bounded kernel с sigma_log=.03, nu=5, rounding=.01; один bias prior N(0,.03²), затем sensitivity sigma_bias=.015/.06 как **отдельные targets**, не tuning на оценке. Generator сначала выбирает дату по weights (одна на date_group), затем тот же bin law. OW semantic operator ровно So; вход So+Sg/numeric category/газ в этом E02 → `UnsupportedObservationError` с явным future E06 mapping. Invalid QC сохраняется, но не score. Одно numeric значение исключает повторное weak categorical measurement.
- [ ] **5.4 Run GREEN:** команда 5.2 и совместный recovery: известные две даты, 1000 cheap repeated draws, posterior date weights averaged over draws match prior weights within .05 (law of total expectation), bias marginal coverage сравнить с 1D quadrature. Commit: `feat: marginalize log dates with shared anchored interpretation bias`.

### Task 6: Подключить настоящий E01 forward и разделить cache F/L

**Files:** Create `src/so_recon/observation/predict.py`, `src/so_recon/inference/target.py`, `src/so_recon/inference/cache.py`, `tests/unit/test_inverse_target.py`, `tests/unit/test_inverse_cache.py`, `tests/integration/test_e02_forward_target.py`; Modify `src/so_recon/geology/renderer.py` (case publisher).

**Interfaces:** `build_inverse_case(rendered: RenderedParameters, context: PriorContext, paths: ProjectPaths, ctx: RunContext) -> CaseBundle`; `predict_observations(result: ForwardResult, observations: ObservationBundle, paths: ProjectPaths) -> ModelObservations`; `evaluate_loglik(prediction, observations, noise) -> LoglikResult`; `PhysicalTarget(prior, proposal, context, observations, worker, ledger, run_factory).evaluate(theta) -> TargetEvaluation`. `run_factory(command: str, parent_run_ids: tuple[str,...]) -> RunContext` creates a unique child evaluation run through existing registry. Cache `get(key)->ArtifactRef|None`, `put(key,ref)->None`; immutable/hash verified.

- [ ] **6.1 RED unit: cache dependencies и failure mapping.**

```python
from so_recon.inference.cache import likelihood_key

def test_noise_change_invalidates_likelihood_only():
    common = dict(forward_hash="a"*64, observation_hash="b"*64,
                  operator_hash="c"*64)
    assert likelihood_key(**common, noise_hash="d"*64) != \
           likelihood_key(**common, noise_hash="e"*64)
```

`likelihood_key` hashes canonical dict all four keys; invalid SHA refused. Tests alter controls, geometry, saturation, fluids, solver config, requested dates, cutoff and ensure F cache misses; change noise/masks/bin-grid and ensure L misses but F reusable. Reused artifact with corrupt bytes fails SHA check. Failure unit test maps TIMEOUT/RESOURCE_FAILURE/NUMERICAL_FAILURE after retry to exceptions, never `log_l=-inf`.
- [ ] **6.2 Run RED:** `uv run pytest tests/unit/test_inverse_target.py tests/unit/test_inverse_cache.py -q`.
- [ ] **6.3 Реализовать bridge без truth dependency.** Case arrays создаются из theta/context в уникальном child run; не брать `render_p1(seed).arrays` как скрытый template. Public geometry/well/control helpers E01 допустимы. Publisher повторяет actual axes/units, content hashes и `compute_model_hash`. `simulate` получает отдельный ctx на evaluation (его job IDs привязаны к run); worker и ledger общие для всей inference session. Не создавать worker на каждую particle.

```python
def require_complete_forward(result):
    if result.status != "COMPLETE":
        raise ForwardEvaluationError(result.status, result.reason, result.job_id)
    return result
```

`ForwardEvaluationError(status,reason,job_id)` определить в target.py. `UNPHYSICAL`/`CONTROL_INFEASIBLE` можно превратить в support rejection **только** при заранее записанном mathematical support/observed feasibility event. Стартовая policy E02 — остановка с диагностикой на любом не-COMPLETE, чтобы не менять target от численной/инженерной фильтрации. Theta outside declared density support отклоняется до F.

F cache включает physical arrays + design/U/fluids/initial/boundary/grid/time + solver/retry/Julia lock/adapter code + output request; не включать RNG/particle ID/nuisance в semantic key, если они не влияют на F. Ссылку на существующий artifact проверять и регистрировать parent. L key дополнительно включает полный observation bundle, masks/rounding/reset/date-support, noise и operator version. Key не равен только `case.model_hash`: отдельно добавить solver/config/output/version components.
- [ ] **6.4 Реализовать observation extraction.** Брать `monthly.parquet`: фактические поля E01 `oil_prod_m3_sc`, `water_prod_m3_sc`, `liquid_prod_m3_sc`, `fw`, `fw_valid`, `well_id`, `month_index`. Вычислить ratio месячных **produced** интегралов `water_prod_m3_sc/(oil_prod_m3_sc+water_prod_m3_sc)` при valid и проверить parity с stored fw. `water_inj_m3_sc` не входит в этот ratio. Не ratio averages или final rates. Observed mask определяет scored rows, predicted invalid не может удалить observed valid row. Для log quadrature запросить все нужные state_times_s **до** F; arbitrary dates требуют exact native output scheduling или explicit unsupported-date failure, а не линейной интерполяции So. Начальный E02 использует report-edge dates.
- [ ] **6.5 Native RED/GREEN после Task 1 gate:** из accepted reduced/E01 fixture построить две разные theta, пропустить через один worker, проверить конечные states, phase integrals, finite log L, отдельные result refs и неизменность первого. Переоценка первой theta hit cache; изменение только sigma не вызывает новый forward. `uv run pytest tests/integration/test_e02_forward_target.py -m julia -q`. В test invocation включить P0 ledger; 3–5 attempts planned, retries учитываются.
- [ ] **6.6 Run GREEN:** команды 6.2/6.5, E01 output/restart regression только при затронутых API. Commit: `feat: evaluate inverse targets through the verified persistent forward adapter`.

### Task 7: Log weights, CESS, beta schedule и resampling

**Files:** Create `src/so_recon/inference/weights.py`, `src/so_recon/inference/resampling.py`, `tests/unit/test_smc_weights.py`, `tests/unit/test_resampling.py`.

**Interfaces:** `normalize_log_weights(logw:F64)->tuple[F64,float]` (normalized logw, normalizer); `ess(logw)->float`; `cess(logw,ell,delta)->float`; `next_beta(beta,logw,ell,target_fraction)->float`; `bridge_log_target(beta,log_p0,log_l,log_r)->float`; `systematic_resample(logw,rng)->I64`. All finite normalized input assumptions validated; `NoTargetSupport` for all -inf.

- [ ] **7.1 RED:**

```python
import numpy as np
from so_recon.inference.weights import cess, bridge_log_target

def test_cess_at_zero_is_n_even_for_nonuniform_weights():
    lw = np.log([.9,.1])
    assert np.isclose(cess(lw, np.array([-100.,30.]), 0.), 2.)
    assert bridge_log_target(0., -np.inf, -np.inf, -2.) == -2.
```

Add weight increment when r≠prior; ell+10000 shifts log evidence but not normalized weights/CESS; finite positive probabilities including current zero weights; no-support all L=0; monotone beta≤1; resampling expected offspring and ancestry under nonuniform weights (10,000 cheap repeats).
- [ ] **7.2 Run RED:** `uv run pytest tests/unit/test_smc_weights.py tests/unit/test_resampling.py -q`.
- [ ] **7.3 Реализовать stable formulas.**

```python
import numpy as np
from scipy.special import logsumexp

def cess(logw, ell, delta):
    n = len(logw)
    if delta == 0:
        return float(n)
    active = np.isfinite(logw)
    x = delta * ell[active]
    lw = logw[active]
    a, b = logsumexp(lw + x), logsumexp(lw + 2*x)
    if not np.isfinite(a):
        raise NoTargetSupport("all incremental target weights are zero")
    return float(n * np.exp(2*a-b))

def bridge_log_target(beta, log_p0, log_l, log_r):
    if beta == 0:
        return float(log_r)
    if beta == 1:
        return float(log_p0 + log_l)
    return float((1-beta)*log_r + beta*(log_p0+log_l))
```

`ell=log_p0+log_l-log_r` для particles in support r; +inf/NaN diagnostic failure. При частицах L=0 возникает CESS jump при delta→0+: сначала удалить нулевую target mass через корректное reweight/resample с evidence increment, документировать шаг на минимальной положительной beta, либо вернуть `NO_TARGET_SUPPORT`/stalled-support diagnostic, если невозможно соблюсти policy; **не** бесконечно делать beta=0. Стартовая реализация выбирает диагностированный stop `EVALUATION_FAILURE(reason='CESS_SUPPORT_DISCONTINUITY')` вместо изменения CESS threshold.

`next_beta`: если CESS(1-beta)≥.8N, вернуть **exact 1.0**; иначе bisection delta∈(0,1-beta), tolerance 1e-10, max 80 iterations. Не способен продвинуться ≥1e-12 → diagnostic stop с сохранением state. Increment logZ=`logsumexp(old_logw+delta*ell)`; normalize до resampling. Нельзя потерять marginal likelihood constant от усечения Student-t.
- [ ] **7.4 Systematic resampling:** `u0=rng.random()/N`, positions=`u0+arange(N)/N`, `cdf=cumsum(exp(logw));cdf[-1]=1`, indices=`searchsorted(cdf,positions,side='right')`. Сохранить indices и исходные ancestor IDs; новые logw=-log N. Если weights не normalized/finite except -inf — refusal, а не masking. Trigger только ESS<.5N; CESS не путать с ESS.
- [ ] **7.5 Run GREEN:** команда 7.2. Commit: `feat: implement the prior proposal corrected SMC weight algebra`.

### Task 8: Defensive mixture и gradient-free инвариантные transitions

**Files:** Create `src/so_recon/inference/proposals.py`, `src/so_recon/inference/kernels.py`, `tests/unit/test_defensive_proposal.py`, `tests/unit/test_mh_kernels.py`.

**Interfaces:** `DefensiveMixture(prior:Density, proposal:Density, epsilon:float)` implements Density; E02 operational r=prior, epsilon mixture tested with analytic q. `Move(proposed:ThetaRecord, log_reverse_minus_forward:float, kernel:str)`; `propose_global(theta,r,rng)->Move`; `propose_rw(theta,scale,rng)->Move`; `propose_pcn(theta,scale,rng)->Move`; `propose_family(theta,schema,rng)->Move`; `mh_log_accept(old:TargetEvaluation,new:TargetEvaluation,beta:float,move:Move)->float`. Fixed kernel probabilities (.25,.35,.25,.15); if block absent choose a normalized fixed per-schema vector before run and hash it.

- [ ] **8.1 RED: q missed mode, asymmetric transitions, pCN при r≠p0.**

```python
import numpy as np
from so_recon.inference.kernels import pcn_reverse_minus_forward

def test_pcn_proposal_ratio_has_correct_sign():
    # Gaussian reference density: log phi(old)-log phi(new).
    assert np.isclose(pcn_reverse_minus_forward(np.array([0.]), np.array([2.])), 2.)
```

Add detailed-balance equality on finite 3-state transition with asymmetric proposal probabilities; mixture integrate to1, draw frequencies match density; epsilon=0 refused (defensive class), epsilon=1 exactly prior; family dimension mismatch refuses local switch. pCN at beta=0 must preserve r even when r≠prior, not always accept.
- [ ] **8.2 Run RED:** `uv run pytest tests/unit/test_defensive_proposal.py tests/unit/test_mh_kernels.py -q`.
- [ ] **8.3 Реализовать mixture и complete MH ratio.**

```python
import numpy as np

def mixture_log_prob(log_q, log_p0, epsilon):
    if not 0 < epsilon <= 1:
        raise ValueError("epsilon must be in (0,1]")
    if epsilon == 1:
        return float(log_p0)
    return float(np.logaddexp(np.log1p(-epsilon)+log_q, np.log(epsilon)+log_p0))

def pcn_reverse_minus_forward(old, new):
    return float(.5 * (new @ new - old @ old))
```

Global move reverse/forward=`log r(old)-log r(new)`; rw in latent v symmetric; pCN only full z_perp block `new=sqrt(1-h²)*old+h*N(0,I)`, reference term above; family uniform among other supported categories gives symmetric proposal if same number categories, else include actual reverse probability. Full log acceptance=`min(0,log pi_beta(new)-log pi_beta(old)+log_reverse_minus_forward)`. Nonfinite new target -inf legitimately reject; numerical F error suspends engine, не MH reject. Proposal global draws **whole** theta including residual and s; no hidden per-family copy of truth.
- [ ] **8.4 Invariance tests:** Gaussian target chain 30,000 steps with fixed seeds after 5,000 burn-in: means within .06 and variances within .08; correlated 2D target, beta=.37 mixed r; 3-state exact balance abs error<1e-12. Proposal scales/kernel weights фиксировать в config; adaptation не делать по текущей state без Hastings correction.
- [ ] **8.5 Run GREEN:** команда 8.2. Commit: `feat: add defensive proposals and invariant gradient free rejuvenation`.

### Task 9: Универсальный SMC engine и точный checkpoint/resume

**Files:** Create `src/so_recon/inference/smc.py`, `src/so_recon/inference/checkpoint.py`, `tests/unit/test_smc_engine.py`, `tests/unit/test_smc_checkpoint.py`.

**Interfaces:** `infer(target, proposal:Density, schema:DensitySchema, config:InferenceConfig, checkpoint_dir:Path, stop_requested:Callable[[],bool]) -> SMCState`; `continue_inference(state:SMCState, target, proposal, schema, config, checkpoint_dir, stop_requested)->SMCState`; `save_state(state,path,paths,ctx)->ArtifactRef`; `load_state(ref,paths,expected_hashes:dict[str,str])->SMCState`. State carries algorithm_status as defined Task1. Both infer/resume use same internal phase machine; no rerun initialization on resume.

- [ ] **9.1 RED: прекращение на каждой atomic boundary.**

```python
import numpy as np
from so_recon.inference.weights import normalize_log_weights

def test_all_zero_target_is_not_reset_to_uniform():
    from so_recon.inference.weights import NoTargetSupport
    import pytest
    with pytest.raises(NoTargetSupport):
        normalize_log_weights(np.array([-np.inf, -np.inf]))
```

Engine tests use `GaussianToyTarget` from Task10 (реализовать helper раньше при исполнении Task9): run uninterrupted, stop after initial particle7 / reweight / resample / MH proposal3 / acceptance decision, then resume. Assert exact theta, weights, beta sequence, RNG, ancestry, logZ and proposal stream equality in same locked CPU environment. Incompatible proposal/noise/basis/solver hash → refusal; partial shard no committed manifest → previous checkpoint readable.
- [ ] **9.2 Run RED:** `uv run pytest tests/unit/test_smc_engine.py tests/unit/test_smc_checkpoint.py -q`.
- [ ] **9.3 Реализовать state machine.**

```text
initialize: draw a pending full theta from r; save RNG/pending; evaluate; append; advance cursor
reweight: compute ell; choose beta; update logZ/logweights; save old/new beta and diagnostics
resample: record pre-ESS; draw/save indices; replace particles; retain root ancestry; save
rejuvenate: for each sweep/particle choose fixed-law kernel; save proposed theta and log_u;
            evaluate pending proposal; accept by full MH; advance cursor; save
finalize: if beta==1 keep final physical refs; record diagnostics; return COMPLETE candidate
```

Только после набора всех N initial particles установить beta=0, log_weights=-log(N), log_evidence=0 и перейти к reweight. Partial population не получает веса завершённого ensemble. На каждом level нужны только cached logL для неизменных частиц; evaluations новых proposals идут по одному. `COMPLETE` здесь означает кандидат на convergence evaluation Tasks11/13/14; это ещё не scientific PASS.

При r=p0 и truly empty observation bundle: sample normalized conditional prior, beta=1, logL=0, no F calls, `physical_state_refs=()`; это probability test bundle, карты и physical posterior export запрещены без states. При r≠p0 всё равно выполнить full prior/proposal bridge даже с empty L.

На каждом этапе проверять stop before new expensive call. `pending_proposal` и uniform/log_u сохранять **до** F: crash после дорогого evaluation использует cache, повторение не меняет draw. Незаписанная acceptance не теряет старую particle. Init interruption сохраняет partial population и cursor; не normalize её как completed N. max_beta_steps/time/count/memory → `INCOMPLETE_BUDGET` с фактической beta. Numerical exceptions → `EVALUATION_FAILURE`; no-support → `NO_TARGET_SUPPORT`.
- [ ] **9.4 Durable artifacts:** particle metadata Parquet, compact arrays HDF5, manifest JSON; write temp→fsync/atomic rename→register manifest last через E00 atomic/registry. Include complete theta and density components, phase/cursor/RNG/ancestry, target/proposal/model/basis/config/code/lock hashes, consumed cost and pending jobs. Manifest references all shard hashes. Never overwrite accepted snapshots; new checkpoint references previous. Resume validates bytes and versions before using cache. Disk failure preserves last committed state and returns nonzero.
- [ ] **9.5 Run GREEN:** команда 9.2; failure injection every phase; no uncontrolled real sleeps/OOM. Commit: `feat: persist and resume the full SMC phase machine`.

### Task 10: Независимые analytic posterior и density acceptance

**Files:** Create `src/so_recon/validation/toy_inverse.py`, `tests/unit/test_toy_inverse.py`, `tests/unit/test_density_calibration.py`; finish helper dependency declared Task9.

**Interfaces:** `GaussianToyTarget(prior_mean,prior_variance,y,observation_variance,proposal)` with `.evaluate(theta)->TargetEvaluation`; `gaussian_reference(m0,v0,y,vy)->tuple[float,float,float]` returns mean,var,logZ; `BimodalToyTarget` uses normalized p0 mixture; `toy_suite(seed:int,n_particles:int)->dict[str,object]`. No Julia/native process.

- [ ] **10.1 RED: Gaussian exact posterior и evidence.**

```python
import numpy as np
from so_recon.validation.toy_inverse import gaussian_reference

def test_reference_is_analytic():
    mean, variance, logz = gaussian_reference(0., 1., 2., 1.)
    assert mean == 1.
    assert variance == .5
    assert np.isclose(logz, -.5*np.log(4*np.pi)-1.)
```

- [ ] **10.2 Run RED:** `uv run pytest tests/unit/test_toy_inverse.py tests/unit/test_density_calibration.py -q`.
- [ ] **10.3 Реализовать exact oracle и disconnected target.**

```python
import numpy as np
from scipy.stats import norm

def gaussian_reference(m0, v0, y, vy):
    variance = 1/(1/v0+1/vy)
    mean = variance*(m0/v0+y/vy)
    return float(mean), float(variance), float(norm.logpdf(y, m0, np.sqrt(v0+vy)))
```

Bimodal: s∈{0,1}, p0(s)=(.7,.3), x|s~N((-3,+3)[s],.5²), y=0 with L=N(y;x,2²). Posterior s weights analytic via `p0(s)*N(0;mu_s,4.25)`; within family conjugate posterior. Missed-mode q gives only s=0, same continuous density there; r=.8q+.2p0 gives >0 s=1. Use 64 particles, 20 replicated starts; evaluate recovered family probability, not just pooled mean. Add r≠prior Gaussian shifted q; flawed likelihood-only weights must fail reference comparison.
- [ ] **10.4 Register numeric acceptance:** 20 seeds×N64, pooled posterior mean difference≤.08 sd_reference, variance relative difference≤.12, mean estimated Z relative error≤.10 with reported across-run MC SE (average Z, not average logZ as unbiased evidence). Family weight abs error≤.06 and no systematic missed s=1; publish per-run failures. At N32/N64 use paired seed report; thresholds fixed before runs. 200 cheap Gaussian prior-predictive repetitions: по32 posterior draws для ranks (resample по финальным weights, ancestry/dependence disclose), randomized rank ties и 90% intervals. Coverage 95% Wilson interval содержит .9; independent exact Gaussian posterior draws дают отдельный reference rank/coverage control. Не выдавать rank histogram correlated SMC descendants за independent-draw calibration proof. Failure to meet MC criteria → diagnose/add recorded cheap repetitions, no tuning seeds or deleting outcomes.
- [ ] **10.5 Empty-data & measure tests:** r=p0 keeps conditional moments/family mass/residual; r≠p0 converges to prior. Testing posterior at fixed y does **not** replace repeated prior-predictive calibration. No-support and invalid-transform assertions are deterministic and must pass irrespective of MC variability.
- [ ] **10.6 Run GREEN:** команда 10.2 + all pure E02 tests. Commit: `test: verify SMC density calibration against analytic references`.

### Task 11: Независимая reduced native inversion и noise recovery

**Files:** Create `src/so_recon/synthetic/reduced_inverse.py`, `src/so_recon/validation/reference_inverse.py`, `tests/unit/test_reference_quadrature.py`, `tests/integration/test_e02_reduced_inverse.py`.

**Interfaces:** `ReducedDesign` StrictModel; `build_reduced_case(z:float, design:ReducedDesign, paths,ctx)->CaseBundle`; `quadrature_reference(nodes:F64, prior_log_density:F64, log_likelihood:F64, integration_weights:F64)->dict[str,F64|float]`; `run_reduced_reference(design,observations,worker,ledger,run_factory)->ArtifactRef`. Reference calls exact same F/observation model but does **not** use SMC beta, weights helpers or particles for its integration.

- [ ] **11.1 RED: quadrature catches omitted grid weights.**

```python
import numpy as np
from so_recon.validation.reference_inverse import quadrature_reference

def test_nonuniform_quadrature_weights_are_used():
    out = quadrature_reference(np.array([0.,1.]), np.zeros(2), np.zeros(2),
                               np.array([.25,.75]))
    assert np.isclose(out["mean"], .75)
```

- [ ] **11.2 Run RED:** `uv run pytest tests/unit/test_reference_quadrature.py -q`.
- [ ] **11.3 Зарегистрировать настоящий reduced case.** `shape=(16,1,1)`, `extent=(160,10,10)m`, phi=.25, K=(100,100,5)mD, initial Sw=.2, same OW fluids E01; pressure initialized from native hydrostatic helper; injector cell0, producer cell15, BHP 30/5MPa, 12 calendar months from 2000-01-01. Frozen `e02-reduced-corey-v1` с rate=2m³_sc/day дал `DESIGN_NOT_INFORMATIVE`: native truth seed701 имел watercut range `0.0010195`. Новый design ID `e02-reduced-corey-v2` меняет только rate на 4m³_sc/day; независимый diagnostic forward дал range `0.63248`, поэтому observations/reference регенерируются под v2. One uncertain latent z~N(0,1) sets oil Corey exponent `n_o=1.5+2*Phi(z)`; n_w=2, Sorw=.2, Swc=.2. This parameter changes phase transport; do not use absolute K scale as the only “identifiable” unknown in fixed-rate displacement. Truth z from seed701, fixed-noise `.03/.4/5`, report watercut bins=.01; save truth state at all 13 report edges. Reduced schema `(n_v=1,n_residual=0,families=(0,))`; log_density is N(z), no double physical Jacobian.

Before inverse, run native balance and actual water breakthrough check on this design. If twelve months do not span informative response, record `DESIGN_NOT_INFORMATIVE`; a revised control/horizon needs new design ID before observations/reference are regenerated. Do not pretend changed design is same experiment.
- [ ] **11.4 Independent quadrature:** initial nested trapezoid nodes linspace(-7,7,17), then33 (17+16=33 F jobs), then65 and129 **only** with remaining session budget/run intent. Use physical outputs cache at repeated nodes; priors and integration widths remain explicit. Formula:

```python
import numpy as np
from scipy.special import logsumexp

def quadrature_reference(nodes, prior_log_density, log_likelihood, integration_weights):
    logmass = prior_log_density + log_likelihood + np.log(integration_weights)
    logz = float(logsumexp(logmass))
    weights = np.exp(logmass-logz)
    mean = float(weights @ nodes)
    return {"weights": weights, "mean": mean,
            "variance": float(weights @ (nodes-mean)**2), "logz": logz}
```

Original prior is full Gaussian: account omitted |z|>7 mass `2*norm.sf(7)`, not silently renormalize it to a different prior. Because all bin likelihood factors≤1, unnormalized omitted evidence≤tail mass; require tail_mass / estimated_Z_lower <1e-6. Estimated_Z_lower uses quadrature refinement error margin. Successive grids require posterior mean change<.02 prior sd, q05/q50/q95 change<.03, relative Z change<.02. If sharp likelihood fails, reference status `UNRESOLVED_REFERENCE`, no ground-truth posterior claim.

  **Execution checkpoint 15.09.2026:** на окончательном physics tree `32fcf2f` native truth и все 33 reference nodes завершились (`34/64` P0 forwards, `53.7224 s` summed native wall, peak process-tree RSS `1,548,042,240` bytes). Design check для `e02-reduced-corey-v2` остался `INFORMATIVE`, но сравнение 17→33 узлов дало mean `0.877669→0.752864`, max quantile change `0.4375`, relative evidence change `0.328363` и tail/evidence lower `1.58481e-08`. Статус `UNRESOLVED_REFERENCE` сохранён в `artifacts/runs/20260915T133942Z-e02-reduced-reference-9c485ce3/reduced_reference.json`; run status `FAIL`. Порог не расширять и posterior ground truth не объявлять. Следующий разрешённый reference batch — nested 65/129 с повторным использованием уже рассчитанных узлов и единым наследуемым ledger; до его сходимости dependent пункты 11.5, 11.6 и 13.5 остановлены по fail-closed правилу плана.

  **Refinement checkpoint 15.09.2026:** commit `d650b7c` добавил schema `e02-reduced-reference-2`, проверяемое наследование ledger и fail-closed reuse по reference artifact, producer run, case/model hash, native result и completed ledger entry. Nested 65-node run переиспользовал 33 узла и добавил 32; он остался `UNRESOLVED_REFERENCE` (mean change `0.0559271`, max quantile change `0.21875`, relative evidence change `0.459770`, tail ratio `1.06442e-08`). Nested 129-node run переиспользовал 65 узлов и добавил 64; mean change `0.00108289` и tail ratio `6.15894e-09` прошли, но max quantile change `0.109375` и relative evidence change `0.0343119` не прошли. Финальный authority — `artifacts/runs/20260915T141509Z-e02-reduced-reference-cc795699/reduced_reference.json`; inherited chain содержит 130 complete forwards, `192.312 s` wall и `488.045 s` CPU. Текущий pre-registered horizon исчерпан на 129 узлах: 257 nodes/adaptive quadrature не запускать без отдельного изменения плана и run intent; пункты 11.5, 11.6 и 13.5 остаются заблокированы.
- [ ] **11.5 Native SMC reference comparison:** two seeds 11/12 at32 and64, every run separately budgeted; same prior/L/U/support and output dates. Posterior mean differs from converged quadrature by ≤max(.05 prior sd,3 across-start MC SE), quantiles≤max(.08,3 MC SE), report evidence/modes/beta/unique ancestors. beta1 alone not pass; reference grid states produce independent truth/reference So distribution on same evaluator support. Native attempts max64 in P0: first reference stage 1 truth+33 nodes=34, retries may consume remainder; SMC physical comparisons use P1 limits if needed, not an unannounced larger P0.
- [ ] **11.6 Noise recovery/calibration:** fixed physical prediction bank reuse for 200 independent recursive noisy observations, fresh noise seeds 8000–8199. For sigma/rho recovery use 2D deterministic quadrature over the declared transformed-prior coordinates, with grid refinement, no new F for noise-only changes; true sigma/rho drawn from their declared prior. Check repeated coverage and boundary-bin frequencies, record Wilson intervals and rho/sigma correlation. Include missing month3, reset7, dry observed masks, exact0/1 realizations, and Task5 date/bias mixture on saved report states. 200 *noise* replicates are conditional on this F, not 200 independent geological worlds; full system repeated-physical-world coverage deferred to E03/E11 and stated as limitation.
- [ ] **11.7 Run GREEN:** unit command11.2 then `uv run pytest tests/integration/test_e02_reduced_inverse.py -m julia -q` under accepted gate/ledger; incomplete native run produces artifact/nonzero instead of skipping assertions. Commit: `test: validate reduced physical inversion against independent quadrature`.

### Task 12: Общий report support, weighted So metrics и научные рисунки

**Files:** Create `src/so_recon/validation/support.py`, `src/so_recon/validation/inverse_metrics.py`, `src/so_recon/validation/inverse_plots.py`, `tests/unit/test_inverse_support.py`, `tests/unit/test_inverse_metrics.py`, `tests/unit/test_inverse_plots.py`; Modify `pyproject.toml`, `uv.lock` (Matplotlib).

**Interfaces:** `aggregate_state(intersection_pv:F64[zone,cell], so:F64[cell], bo:F64[cell], sorw:F64[cell]) -> ZoneState`; `ZoneState(pv,so,n_rem,n_above_sor,valid_geometry)` arrays; `weighted_quantile(values:F64, weights:F64, probabilities:F64)->F64`; `weighted_crps(values,weights,truth)->float`; `compare_so(estimate:F64,truth:F64,truth_pv:F64)->dict[str,float]`; `render_inverse_figures(products:dict,output_dir:Path)->tuple[Path,...]`. Rectilinear cell/zone intersections explicit; invalid geometry is separate from zero PV.

- [ ] **12.1 RED: PV weighting, zero PV и quantile-of-sum.**

```python
import numpy as np
from so_recon.validation.support import aggregate_state

def test_zone_so_uses_pore_volume():
    zone = aggregate_state(np.array([[1.,3.],[0.,0.]]), np.array([.2,.6]),
                           np.ones(2), np.full(2,.2))
    assert np.isclose(zone.so[0], .5)
    assert np.isnan(zone.so[1])
    assert zone.n_rem[1] == 0
```

Two equally weighted particles inventories cells `[0,10]` and `[10,0]`: zone Q95=10 although sum cell Q95=20. Tests unequal particle weights [.9,.1], tied values, conditional collector probability, invalid geometry (NaN output, not inventory0), truth/estimate different grids with equal conservative aggregate, output assets generated from real supplied arrays.
- [ ] **12.2 Run RED:** `uv run pytest tests/unit/test_inverse_support.py tests/unit/test_inverse_metrics.py tests/unit/test_inverse_plots.py -q`.
- [ ] **12.3 Реализовать conservative reducer.** Rectilinear intersection volume=`prod(max(0,min(cell_hi,zone_hi)-max(cell_lo,zone_lo)))`; multiply by particle phi (and NTG only if not already encoded in active effective volume). For each particle: PV_A=sum(intersection_pv), So_A=sum(pv*So)/PV_A, N_rem=sum(pv*So/Bo), N_above=sum(pv*max(So-Sorw,0)/Bo). Unit standard m³ for inventory; N_above is not recoverable reserves. Empty PV -> missing So/inventory0; malformed geometry -> invalid for all outputs.

```python
import numpy as np

def compare_so(estimate, truth, truth_pv):
    valid = np.isfinite(truth) & (truth_pv > 0)
    if not valid.any() or not np.isfinite(estimate[valid]).all():
        raise ValueError("no comparable valid support or missing estimate")
    w = truth_pv[valid] / truth_pv[valid].sum()
    error = estimate[valid] - truth[valid]
    return {"mae": float(w @ np.abs(error)), "rmse": float(np.sqrt(w @ error**2))}
```

Weighted quantile convention: inverse empirical CDF, stable sort and cumulative **particle** weights, first cdf≥q; normalize conditioned weights only for So when collector present, report collector probability separately. CRPS=`sum_i wi|xi-y| - .5sum_ij wi wj|xi-xj|`. Coverage and width use Q05/Q95, same support; loss primary PV-weighted MAE, RMSE supplementary. Truth support weights are fixed across method comparison; estimate cannot remove difficult zones by changing PV or mask. Missing estimate on valid truth support counts invalid comparison, never improved metric.
- [ ] **12.4 Publish figures from numeric artifacts:** each layer/date start/12/24/36 (reduced uses0/6/12), true So; weighted median; signed error; Q95-Q05; prior→posterior width ratio; one actual physical posterior realization nearest weighted median in zone space; producer fw observed bins/weighted predictive curves; beta/CESS/ESS-before-resampling, ancestors/family weights; cumulative cost/RAM. Quantile map is marked ensemble statistic, not physical trajectory. Same color scale across true/recovered panel, saturation [0,1], layer/depth/date/unit labels; pressure diagnostic separated from observation. If beta<1, plots prominently `INTERMEDIATE beta=...`, no posterior label.
- [ ] **12.5 Run GREEN:** command12.2; render synthetic unequal-value fixture and inspect PNG for swapped layers, axes, labels/colorbars, clipped text. Add `uv add matplotlib` only here; use Agg and static figures, no dashboard. Commit: `feat: evaluate and plot inverse states on conservative shared support`.

### Task 13: P1 T1/T2/T4 и проверяемая ранняя наблюдаемость

**Files:** Create `src/so_recon/synthetic/inverse_worlds.py`, `src/so_recon/validation/observability.py`, `tests/unit/test_inverse_worlds.py`, `tests/integration/test_e02_observability.py`, `configs/e02_experiments.json`.

**Interfaces:** `make_inverse_world(design_id:str,seed:int,paths,ctx)->tuple[PriorContext,ObservationBundle,ArtifactRef]` final ArtifactRef truth is available only to generator/evaluator; `run_observability(experiment:dict,worker,ledger,run_factory)->ArtifactRef`; `compare_pair(observations:ObservationBundle,prediction_a,prediction_b,states_a,states_b,support)->dict`. `InferenceInput` persist whitelist context/G/U/observations/schema/basis; no truth ref, theta, full K/phi/p, seed-derived renderer lookup.

- [ ] **13.1 RED: leakage and parent identity.**

```python
from so_recon.synthetic.inverse_worlds import inference_payload

def test_truth_cannot_enter_an_inverse_payload():
    import pytest
    with pytest.raises(ValueError, match="truth"):
        inference_payload({"context_ref": "context.json", "truth_ref": "truth/states.h5"})
```

`inference_payload(payload:dict)->dict` reject unexpected keys recursively; allowlisted schema from §2. Test numerical G influences conditioning; hidden theta perturbation cannot alter context except through explicitly regenerated observations; raw seed/parent ID used only for lineage, excluded from learned feature payload E03. Mask selection/design does not depend on truth/reconstruction quality.
- [ ] **13.2 Run RED:** `uv run pytest tests/unit/test_inverse_worlds.py -q`.
- [ ] **13.3 Fix experiment definitions before native runs.** Four independent exploratory parents: T1 seeds141/142, T2 seed143, T4 seed144. G/noise/latent/schedule streams via named SeedSequence children, all hashes fixed before scoring. Use T1 first and register outcome even if uninformative. Main T1: E01 16×16×2, 36 months, existing four well columns/control policy/fluids; Task2 conditional prior and s hypotheses, no future pressure. G uses E01 eight logk supports, phi heldout, watercut bins=.01; no additional So measurements in primary inference. Observation/noise seeds independent from final truth latent draw.

T2: two layers with equal thickness, equal base K=80mD/phi relation, identical horizontal design, `kz/kx=1e-4` — объявленный low-crossflow case на существующем permeability input E01; both layers completed in same wells, layer-symmetric controls without P2 lower-layer closure and common pressure reference. Same finite cosine family in each layer, G logk supports observe equal-weight layer-average **log k** (A symmetric); conditional Gaussian law preserves exchange symmetry. Contrasting layer coefficients arise from residual/latent draw, not hand-painted So. s fixed; report swapped-layer physical pair through F: преобразовать theta→physical coefficients a, переставить два layer blocks, решить `z=Q.T@solve(chol,a_swapped-mean)`, затем разложить обратно v/z_perp с теми же nuisance. Перестановка первых/последних whitened coordinates сама по себе не является перестановкой слоёв. Gravity/compressibility/crossflow/well constraints могут нарушать точную симметрию, поэтому сравниваются actual outputs. Pre-register acceptable ambiguity evidence mean absolute fw difference≤.01 AND at least one layer-zone |So_a-So_b|≥.05 at month36; otherwise `AMBIGUITY_NOT_DEMONSTRATED`, revise design under new ID. Не добавлять несуществующие face-multiplier/gravity API ради этого fixture.

T4: base 16×16×2 but all well columns in western half: I1=(2,2),I2=(2,13),P1=(6,2),P2=(6,13). Eastern quadrant x-index≥12,y-index≥8 is predefined report zone. Создать mask m=0 при x≤9, m=1/3 на x10, m=2/3 на x11, m=1 при x≥12; умножать K на .5/.1/.01 в x10/x11/x≥12 соответственно, в остальной сетке multiplier1. Это часть **обоих** renderer: truth и inference.

T4 schema `e02-t4-17d`: n_v=11,n_residual=6,n_geology=13,n_state_residual=1. Первые12 physical geology coefficients — прежние layer cosine fields; coefficient13 добавляет `a13*m*cos(pi*y_normalized)` в shared geological field обоих слоёв; эта же новая мода меняет phi по прежней logistic формуле. K contrast multiplier — известный design, добавляется в logk offset b, новая мода добавляется column в A. Conditional Gaussian работает с13 dimensions; `a=mean+chol@Q@concat(v[:8],z_perp[:5])`. Последний residual `z_perp[5]` независимо задаёт `Sw=.2+.25*Phi(z_perp[5])*m` в обоих слоях. Все17 coordinates имеют определённую density; не фиксировать remote So0 к truth.

Initial pressure — тот же oil-connected hydrostatic initializer при 15MPa datum, допустимый начальный transient S0 с пространственно неоднородным Sw. Сохранять interpretation `synthetic_nonvirgin_initial_state`, не equilibrium/virgin Soi claim. Native closed preflight проверяет finite states, Sw/So bounds, component mass conservation и воспроизводимость transient, **не** отсутствие движения: при общей oil pressure вода может перераспределяться. Все starts используют этот declared initializer. Если transient/reference fails, исправить физическую постановку до inverse. No G supports in eastern quadrant; неизвестное So0 необходимо, поскольку при точно известной Sw=.2 и отсутствии вытеснения узкий posterior So=.8 сам по себе не был бы ложной уверенностью.
- [ ] **13.4 Реализовать pair metrics и controls.**

```python
import numpy as np

def ambiguity_metrics(fw_a, fw_b, so_a, so_b):
    return {"mean_absolute_fw_gap": float(np.mean(np.abs(fw_a-fw_b))),
            "max_absolute_so_gap": float(np.max(np.abs(so_a-so_b)))}
```

Actual pair comparison uses only mutually observed fw rows and common report support; if none, status no-data instead of empty mean. T4 pair: hold all coordinates fixed except remote So0 at -1/+1, both F trajectories, check above thresholds on remote zone; proximity to wells alone is not support classification. Inspect prior/posterior remote width and coverage under true simulation, compare residual-frozen diagnostic separately and label wrong approximation. No hand changes to final states.
- [ ] **13.5 Run planned inference comparisons:** static conditional prior, prior-start SMC32 seeds11/12 and SMC64 seeds11/12; same G/U/noise for all, physical states for prior included in costs (initial SMC particles may be reused as prior ensemble when identical distribution/IDs are disclosed). max24 levels,2 moves each; every run one P1 budget, no concurrent worker. Data and benchmark setup 4 truth forwards + pair replays recorded separately. Parent count4 here is early E02 development coverage, **not** the E03 training128/dev32/evaluation8 corpus. E03 expands protocol explicitly.

Convergence screen fixed: replicated zone median difference≤.03, family mass gap≤.15, width relative difference≤.25 when baseline width>.02; otherwise absolute width gap≤.01. Report across-start spread/MC uncertainty separately; estimated median SE>.01 or family-mass SE>.05 означает insufficient precision и не расширяет допуск автоматически. Два starts — exploratory screen с неточной оценкой SE, не доказательство mixing; сравнение32→64 и независимый reduced reference обязательны. Return `POSTERIOR_NOT_CONVERGED` on unresolved required zones/modes even at beta1. More particles needs new budget forecast, not fewer residuals. T1 state outcome based on MAE/RMSE/prior error, no mandatory improvement claim manufactured; T2/T4 require credible ambiguity demonstration or explicit experimental failure. No bootstrap treats neighboring cells as worlds.
- [ ] **13.6 Run GREEN:** unit13.2; `uv run pytest tests/integration/test_e02_observability.py -m julia -q` runs bounded declared native case(s), resume via Task14 command for remaining manifest jobs. Produce all-world table including failed/incomplete/design-not-informative rows. Commit: `feat: test informative and ambiguous physical inverse worlds`.

### Task 14: CLI, budget forecast, stage evidence и передача E03

**Files:** Create `src/so_recon/inference/commands.py`, `src/so_recon/validation/e02_report.py`, `configs/e02.yml`, `scripts/e02_gate.sh`, `tests/unit/test_e02_cli.py`, `tests/unit/test_e02_report.py`, `reports/stages/E02.md`; Modify `src/so_recon/cli.py`, `src/so_recon/config/schema.py`, `Makefile`, `README.md`, `docs/README.md`; finalize `configs/e02_experiments.json`.

**Interfaces:** `run_e02(cfg,paths,*,suite:str,experiment:str|None,resume:Path|None)->RunContext`; `build_e02_report(run_dirs:tuple[Path,...],paths)->dict`; `render_e02_report(report:dict)->str`; `e02_exit_code(algorithm_status:str,convergence_status:str)->int`. CLI thin dispatch through `execute_run`, subcommands below. No catch-all successful return on checkpoint write.

- [ ] **14.1 RED CLI/status:**

```python
from so_recon.inference.commands import e02_exit_code

def test_checkpoint_is_not_completed_inference():
    assert e02_exit_code("INCOMPLETE_BUDGET", "NOT_RUN") == 2
    assert e02_exit_code("COMPLETE", "FAIL") == 2
    assert e02_exit_code("COMPLETE", "PASS") == 0
```

Report rejects PASS for beta<1, missing native/reference/generator checks, missing ancestor diagnostics or broken artifact checksum. `argparse` unknown flags/profile/experiment→nonzero + failure lineage. Failed scientific improvement does not make technically correct probability code falsely pass observability: distinct stage and outcome fields.
- [ ] **14.2 Run RED:** `uv run pytest tests/unit/test_e02_cli.py tests/unit/test_e02_report.py -q`.
- [ ] **14.3 CLI/config implement with global arg order preserved.**

```bash
uv run so-recon --config configs/e02.yml verify-inverse --suite math
uv run so-recon --config configs/e02.yml inverse-budget --experiment reduced-v2
uv run so-recon --config configs/e02.yml verify-inverse --suite reduced
uv run so-recon --config configs/e02.yml inverse-budget --experiment e02-t1-v1
uv run so-recon --config configs/e02.yml inverse-p1 --experiment e02-t1-v1 --seed 141 --particles 32 --inference-seed 11
```

Also define `inverse-resume --checkpoint <path printed by last run>` and `e02-report --runs <one or more actual run paths>`; these metavariables are CLI argument descriptions, not literal files. `verify-inverse --suite math` never starts Julia. `--suite reduced` uses explicit P0/P1 experiment session from manifest; default run is first bounded reduced stage, not all refinements/replicates. `inverse-p1` executes exactly requested parent/seed/N, no automatic launch of the whole matrix. `inverse-budget` is read-only and uses E01 final benchmark; absent measurement yields `BUDGET_UNMEASURED`, not guessed seconds.

`configs/e02.yml`: spec_version4.0, config_version E02.1, inherited paths/sources/julia conventions, ResourceProfile preset P0_VERIFY by default; inference fields from §2 and allowlisted experiment manifest; P1 command explicitly selects validated P1_LOOP preset with unchanged hard caps. `configs/e02_experiments.json` stores phase/job IDs, theta/prior/observation/design/schema versions, seeds, N, requested dates, tolerances, required gates, sequential order, expected new calls and dependency refs. No personal path or fabricated E01 run ID in tracked config.
- [ ] **14.4 Forecast before native inverse.** Each ordinary run bound, without cache/rejections/retries: `N_initial + N * moves_per_level * max_beta_steps`; N32→1568, N64→3136, **so N64 at24 levels cannot fit 2000 forwards in worst case**. Keep mathematical settings and return budget stop; do not claim guaranteed completion. Budget predictor separately shows measured expected accepted proposals/cache hits and worst case; use initial N32 and observed actual beta levels to authorize N64 session, or checkpoint/resume under explicit run intent. Do not silently reduce transitions/convergence just to claim PASS.

Reserve wall for compile/checkpoint/I/O; projected remaining cost uses E01 measured warm distribution and successful E02 timings, with safety factor1.5, difficult prior draws and retry rows included. One hour is a cap per session, not promise of full E02 completion. `T_total=T_compile+sum(actual F wall)+density+IO+retry`; no division by CPU core count. Include native reference grid, truth, prior ensembles, pair replays, validation, failed attempts and noise-only work, not only “online SMC”. Keep attempts and unique physical models separate, report F/L cache hit counts.
- [ ] **14.5 Gate commands & assets:**

```bash
uv run pytest tests/unit tests/test_no_absolute_paths.py -q
uv run ruff check .
uv run ruff format --check .
uv run mypy
make smoke
uv run so-recon --config configs/e02.yml verify-inverse --suite math
```

Native checks are selected bounded jobs from E02 manifest after require_e01_ow, not unbounded `pytest tests` rerunning every E01 native case. `scripts/e02_gate.sh` uses `set -euo pipefail`, accepts `--suite math|reduced|p1` and actual resume path; failed command exits before printing pass. Existing `make gate`/`make gate-clean` keep E00 semantics; add `make gate-e02-math` and explicit `gate-e02-reduced` targets instead of silently enlarging old smoke.
- [ ] **14.6 Build `reports/stages/E02.md` solely from verified artifacts.** Initial status NOT_RUN. Final report: E01 dependency matrix/ref/accepted code, E02 commit+dirty/code/lock hashes, exact commands/exit status, prior/information model incl excluded phi and no old Gaussian So relabel, bin/generator/date/measure tests, toy and native reference errors, beta/logZ/ESS/ancestry/modes and starts/doubling, all T1/T2/T4 outcomes, state metrics/support/PNG refs, full costs and remaining jobs. Publish machine JSON and Parquet; Markdown is readable view, not authority for fake numeric success.

Stage `PASS` requires normalized likelihood/generator/date recovery, sample/log_prob/measure consistency, reference-supported SMC, native reduced inversion, completed P1 T1/T2/T4 evidence and meaningful convergence diagnostics. `PASS_WITH_LIMITATIONS` permits documented educational OW/low parent count/conditional noise calibration/BO not yet run; **does not** waive failed math, failed mandatory OW, missing reference, beta<1 or failed required convergence. Genuine unobservability can be a valid scientific result, provided the control/reference and ambiguity evidence support it. T1 no improvement with reliable reference is scientific NOT_SUPPORTED/INCONCLUSIVE and a G_LOCAL warning to E03; technical/runtime failure is never classified as unobservability.
- [ ] **14.7 E03 handoff contract:** provide frozen DensitySchema/PriorContext examples, `Density.sample/log_prob/fingerprint`, r interface, same `infer` and checkpoint format, likelihood/noise API, evaluator/support, all-world manifests and measured N32/N64 budget. E03 q predicts v/s and samples full residual from declared conditional reference; no second SMC implementation. Preserve planned extensions unknown ОФП/So0/T3/T5, repeated physical-world calibration and full ML+SMC comparisons; E02 does not claim those complete.
- [ ] **14.8 Run GREEN:** unit14.2, math gate, completed registered native matrix within separately authorized sessions, checksum audit, visual PNG inspection, `git diff --check`. Commit: `feat: expose and report the bounded E02 inverse workflow`. Commit only relevant source/tests/config/docs; do not add private data, large generated solver states or unrelated IDE edits.

## 4. Порядок, независимые работы и стоп-условия

Logical order: 1→2; 1→3→4/5; 1→7→8→9→10; 2+4+5→6; 6+10→11; 1→12; 2+6+9+11+12→13; all→14. Task10 Gaussian helper may be created with Task9 tests, with final analytic acceptance in10. Tasks3–5/7–10 can progress while E01 report is unfinished. Independent mathematical implementation does not require running another Julia process while E01 owns the physical worker.

Stop dependent native execution on missing/stale E01 evidence, normalization failure, inconsistent latent measure, incomplete output dates, unclassified F failure or unconverged reference. Save diagnostic artifacts and continue unrelated code/tests where possible. Time/memory/disk stops preserve remaining job manifest; they do not authorize higher budgets. No implementation work is executed by creating this plan.

## 5. Spec coverage и проверка плана

| Требование | Задачи / свидетельство |
|---|---|
| E01 dependency, legacy3.0/new4.0, resource failure lineage | 1,6,14 |
| Conditional prior, explicit G, no double counting, residual, mixed latent measure | 2,5,8,10,13 |
| Bounded normalized Student-t,0/1, tails, noise generator, gap/reset/dry | 3,4,10,11 |
| Date quadrature, geometric support, common anchored bias | 5,6,11 |
| Full r bridge, CESS/ESS, constants, resampling/genealogy, MH/pCN | 7–10 |
| Bimodal/discrete evidence, defensive missed mode, empty data | 8–10 |
| Exact resume RNG/pending/phase/hash, no beta-forcing | 6,9,14 |
| Independent 1–4D native reference; technology vs identifiability | 11,13 |
| PV-weighted MAE/RMSE, coverage/CRPS, quantile-of-sum, zeroPV, invalid geometry | 12 |
| T1 information,T2 layer ambiguity,T4 remote uncertainty, physical states | 13 |
| Replicated starts,N doubling, final states and model-based map limits | 9,11–14 |
| Cost forecasts, all attempts, no implicit long-run | 1,6,11,13,14 |
| Shared usable E03 interfaces, no premature ML/field claim | 2,8,9,12,14 |

Ограниченные части общей SPEC, не критерии полного исполнения E02: full LIS; native adjoint/MALA; MAP/ES-MDA research baselines; training corpus/NSF; T3/T5 full loop; полевые channels/ETL/PVT; произвольные unstructured intersections и конечные production maps. Их сохранённые владельцы E03/E06–E14 перечислены выше и в STAGES; нельзя считать их реализованными этим планом.

Self-review исполнителя документа: проверить все task interfaces и Files; каждую code-related задачу снабдить concrete RED/GREEN; проверить отсутствие незаданных symbols; `MONTHLY_SCHEMA` сверена с просмотренным E01 и её поля приведены в Task6. Новые outputs/statuses/header относятся к планируемой реализации. Синтаксическая проверка code blocks не доказывает математическую/физическую приёмку.

## 6. Источники для numerical implementation

Приоритет имеет закреплённая локальная версия пакета. Для этого плана проверены официальные API: [SciPy Student-t](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html) предоставляет logcdf/logsf/ppf/isf; [SciPy logsumexp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html) — устойчивое суммирование log masses. Они не заменяют independent normalization tests.

Методические первоисточники: [Zhou, Johansen, Aston — adaptive SMC/model comparison](https://arxiv.org/abs/1303.3123), [Cotter, Roberts, Stuart, White — pCN/function-space MCMC](https://arxiv.org/abs/1202.0709). Конкретный bridge и численные пороги этого проекта задаёт SPEC; переносить likelihood-only pCN acceptance на произвольное r нельзя.
