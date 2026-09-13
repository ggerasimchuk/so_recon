# E01 — физический adapter и первые synthetic states Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Получить проверенный малый oil–water оператор JutulDarcy, воспроизводимые двухслойные P1-миры с истинной So, native restart и измеренными затратами, сохранив историческую совместимость E00.

**Architecture:** Расширяем существующие config/registry/CLI; один долгоживущий Julia worker последовательно исполняет изолированные задания из immutable manifests. Python отвечает за контракты, ресурсы, календарь, синтетический дизайн и отчёты; JutulDarcy — за состояния, скважины и фазовые потоки. Обмен состоит из JSON descriptors, Parquet таблиц, HDF5 summaries и отдельно штатных restart-файлов закреплённого Jutul.

**Tech Stack:** Существующие Python `>=3.13,<3.14`, uv, Pydantic 2, NumPy, PyArrow, pytest, Ruff, mypy; Julia 1.12, Jutul `=0.4.31`, JutulDarcy `=0.3.11`, JSON `=1.8.0`. Новые непосредственные потребители: Python h5py, psutil и Matplotlib для scientific arrays, системных измерений и отчёта; Julia HDF5 для переносимого обмена. Точные новые версии фиксируются lockfiles после совместимого smoke; обновление solver-пакетов не входит в план.

**Spec:** [SPEC.md 4.0](../../SPEC.md), прежде всего §§8–9, 17–18, 23; [STAGES.md, E01](../../STAGES.md); [DECISIONS.md §3](../../DECISIONS.md); [COMPUTE_PROFILES.md, P0/P1](../../COMPUTE_PROFILES.md).

## Global Constraints

Ниже нормативные формулировки перенесены из действующих документов; численные настройки fixtures в следующих разделах являются решениями этого плана, а не измеренными свойствами месторождения.

- «Новые планы используют полный заголовок этапа и `spec_version=4.0`.» — SPEC §0.
- «Исторические E00 records сохраняют 3.0.» — SPEC §17.4.1.
- «Единственный operational backend — JutulDarcy» — SPEC §8.1.
- «Добывающая скважина получает **суммарный стандартный жидкостный расход**, а не два независимо предписанных фазовых расхода.» — SPEC §9.1.
- «Нагнетательная — стандартный расход воды.» — SPEC §9.1.
- «Snapshot в конце месяца не заменяет это интегрирование.» — SPEC §9.3.
- «HDF5 summary не объявляется полноценным restart без round-trip test.» — SPEC §17.1.
- «дублирующий registry не создаётся.» — SPEC §17.4.1.
- «Память задаётся в **GiB = 2^30 байт**»; «soft 12 GiB, hard 16 GiB»; «резервом не менее 6 GiB для ОС и прочих процессов» — COMPUTE §5.
- Julia workers: **1**; потоки Julia: **4** как начальная настройка; BLAS threads: **1**; solver precision: **Float64** — COMPUTE §5.
- P0_VERIFY: **1–128 ячеек**, **1–2 слоя**, **0–3 скважины**, **1–12 отчётных интервалов**, **10 минут**, **64** новых forward за сессию — COMPUTE §§2, 10.
- P1_LOOP: **256–1 024 ячейки**, **по умолчанию 16×16×2=512**, **4–8 скважин**, **36 месяцев по умолчанию**, **1 час**, **2 000** новых forward за сессию — COMPUTE §§2, 10.
- «Бюджет диска по умолчанию: 5 GiB для малой сессии» — COMPUTE §7.
- «На smooth verification cases начальные допуски: cumulative component-balance relative error ≤1e−3 и median per-step ≤1e−5» — SPEC §23.1; целевой более строгий контроль из §8.3 сохраняется отдельно.
- «Resource failure и numerical failure не дают физический нулевой likelihood без анализа.» — SPEC §18.4.
- «Long-run не запускается агентом автоматически после E00/E01; требуется явный run intent пользователя и пройденные gates.» — SPEC §18.4.
- «Full field runs, ML обучение и long-run не запускаются как побочный эффект обновления документации.» — SPEC §23.2.

---

## 1. Исходное состояние и граница этапа

План составлен 13.09.2026 по чистому `master`, HEAD `3c0c8b3353963441d26a224f541cf0061ea7fdd6`. Прочитаны исходники foundation, конфиг, Julia lock и исторический `reports/stages/E00.md`. Исторический отчёт фиксирует PASS E00 и 176 tests на принятом дереве; при составлении этого плана тесты/physics jobs повторно не запускались. Перед исполнением сверить HEAD/status и применимость указанных мест изменения.

Существуют `RunContext`, `ArtifactRef`, атомарные writers, относительные `ProjectPaths`, `execute_run`, CLI, `find_julia`, отдельный subprocess smoke, frozen `configs/smoke_expected.json`. В `ProjectConfig.spec_version` и runtime `SPEC_VERSION` пока только 3.0; `RunContext.start` и `build_source_manifest` stamp-ят глобальную константу. Persistent worker, физическая verification matrix и HDF5 exchange отсутствуют.

**E01.0 = Tasks 1–4:** совместимость версий, минимальные контракты, ресурсы и persistent transport. Это часть E01, а не повтор E00. До её приёмки разрешены unit/transport проверки, но не серии физических jobs.

**E01 physics = Tasks 5–12:** OW модель, расписание, интегралы, restart/failure policy, физическая верификация, P1 generator и итоговый runner/benchmark. **Task 13** — отдельно учитываемый ранний BO capability; его отсутствие не блокирует первый OW цикл, но блокирует заявление о проверенной BO-конфигурации и должно быть закрыто до PhysicsDecision.

В E01 нет ETL пяти CSV, полевого prior/PVT, ThetaRecord/DensitySchema, likelihood, MAP/ES-MDA/SMC, обучения сети, posterior-карт и обязательной установки OPM. `theta.json` в synthetic world — параметры известного генератора, не будущий вероятностный `ThetaRecord`. Первые 5 P1-миров и benchmark — exploratory engineering fixtures; не training corpus 128/32/8 и не закрытая оценка статьи.

**Правило Git:** текущий запрос разрешает подготовку плана. Ниже commit checkpoints — рекомендуемые границы будущего исполнения; `git add/commit`, новая ветка, merge/push выполняются только при соответствующей авторизации. Сейчас код и Git-история не меняются. При исполнении изоляция оформляется через using-git-worktrees, если она нужна и разрешена; новый worktree не создаёт новое окружение по умолчанию.

## 2. Карта файлов и ответственности

Все пути ниже относительно корня репозитория; номера строк — ориентиры прочитанного дерева, перед правкой найти символ. Новые пакеты получают `__init__.py` только при первом потребителе.

| Файлы | Ответственность | Task |
|---|---|---|
| `src/so_recon/__init__.py`, `config/schema.py`, `config/load.py`, `registry/run.py`, `registry/source_manifest.py`, `cli.py` | Версии конфигов и lineage без изменения исторических записей | 1 |
| `src/so_recon/simulator/contracts.py`, `case_io.py` | Typed CaseBundle/ForwardResult/JobDescriptor, проверка файлов и shapes | 2 |
| `src/so_recon/config/resources.py`, `environment/resources.py`, `simulator/budget.py` | Профили, hardware probe, watchdog, budget ledger | 3 |
| `src/so_recon/simulator/worker.py`, `julia/worker/main.jl` | Жизненный цикл одного процесса, job matching, cleanup | 4 |
| `julia/adapter/SOReconAdapter.jl`, `model.jl`, `julia/verification/fixtures.jl` | Единый entrypoint, OW model, маленькие физические fixtures | 5 |
| `src/so_recon/simulator/schedule.py`, `julia/adapter/controls.jl` | Календарные интервалы, roles, BHP limits, completion masks | 6 |
| `julia/adapter/outputs.jl`, `src/so_recon/simulator/results.py`, `validation/balance.py` | Accepted-substep extraction, monthly volumes, HDF5/Parquet, независимый баланс | 7 |
| `julia/adapter/restart.jl`, `src/so_recon/simulator/forward.py` | Native round-trip, output policy, retries и результаты незавершённых jobs | 8 |
| `julia/verification/analytic.jl`, `src/so_recon/validation/physics.py` | Closed cell/PVT, BL, hydrostatic checks | 9 |
| `julia/verification/operations.jl`, `refinement.jl` | Mixing/crossflow, roles, boundary/aquifer, five-spot, grid/time convergence | 10 |
| `src/so_recon/synthetic/p1.py`, `world_io.py` | 2-layer renderer, sparse observations, отделение truth, world provenance | 11 |
| `src/so_recon/simulator/commands.py`, `validation/e01_report.py`, `validation/plots.py` | CLI orchestration, benchmark и stage acceptance | 12 |
| `julia/adapter/blackoil.jl`, `julia/verification/blackoil.jl` | Малый BO capability через тот же adapter | 13 |
| `configs/e01.yml`, `configs/e01_tolerances.yml`, `configs/e01_jobs.json` | Версионированные runtime, допуски и явный job list | 1, 3, 9–12 |
| `pyproject.toml`, `uv.lock`, `julia/Project.toml`, `julia/Manifest.toml` | Только зависимости с появившимся потребителем | 2, 3, 12 |
| `Makefile`, `README.md`, `julia/README.md`, `scripts/e01_gate.sh`, `reports/stages/E01.md` | Команды и фактическая приёмка E01 | 12 |
| `tests/unit/test_e01_versions.py`, `test_case_contracts.py`, `test_resources.py`, `test_worker.py`, `test_schedule.py`, `test_forward_results.py`, `test_forward_policy.py`, `test_physics_metrics.py`, `test_p1_worlds.py`, `test_e01_cli.py` | Независимые тесты Python contracts/control plane | 1–12 |
| `tests/integration/test_e01_worker.py`, `test_e01_physics.py`, `test_e01_restart.py`, `test_e01_p1.py`, `test_e01_blackoil.py` | Реальный закреплённый Julia/Jutul, без подмены physics mock-ом | 4–13 |

Не модифицировать `julia/smoke/smoke_case.jl`, старые frozen ожидания, исторический E00 report/plan и опубликованные source manifests для «устранения» расхождений E01.

## 3. Общий data contract для всех задач

### 3.1. Оси, единицы и идентичность

- Python/exchange: `cell_id` и connection indices с нуля, `cell_id = i + nx*(j + ny*k)`; Julia переводит в 1-based ровно при построении модели. HDF5 states имеют shape `(n_times, n_cells)`; в Julia допустима своя внутренняя раскладка, атрибут `axis_order` и асимметричный fixture доказывают round-trip.
- Координата z — глубина, положительная вниз; горизонтальный local Cartesian, `crs=null`. Gravity magnitude `9.80665 m/s²`. Нулевое g разрешено только явно помеченным analytical fixtures.
- Входные pressure — Pa, permeability — m², time — seconds; человеческие control rates — `m3_sc/day`, native — `m3_sc/s`. `1 day = 86400 s`; `1 mD = 9.869233e-16 m²`. Производство положительно в публичных таблицах; native отрицательный знак сохраняется отдельно в diagnostic ledger.
- Учебные surface reference: `p_sc=101325 Pa`, `T_sc=288.15 K`, `rho_w_sc=1000`, `rho_o_sc=800 kg/m³`; `c_w=4e-10`, `c_o=1e-9 Pa⁻¹`; `mu_w=0.001`, `mu_o=0.003 Pa·s`. OW: `B_alpha=rho_alpha_sc/rho_alpha(p)`, без растворённого газа, PV постоянен (`rock_compressibility=0`, явно). Это согласованная учебная слабосжимаемая модель, не полевые PVT.
- Corey: `Swc=0.2`, `Sorw=0.2`, exponents `(2,2)`, endpoints `(1,1)`, `Pc=0` для первых OW cases. Проверять область `[Swc, 1-Sorw]`, положительность вязкости/плотности и `Sw+So=1`; не исправлять невалидные входы clipping-ом.
- Artifact SHA — байты файла; `model_hash` — canonical JSON всех влияющих на F входов плюс хэши arrays/solver lock. `job_id` уникален на попытку; `case_id/world_id` стабилен. Время/RAM/hostname не входят в deterministic model hash.
- `F` materialized в E01: `simulate(case, output_request)`; физические поля уже внутри CaseBundle. E02 добавит отдельный `theta` renderer до вызова этого интерфейса. В E01 запрещено неявное чтение real CSV.

### 3.2. Минимальные typed records

`StrictModel` существующий, `extra='forbid'`, `frozen=True`; nested mappings/arrays не считаются физически immutable только из-за Pydantic frozen — запись/hash freeze обязательны.

| Record | Обязательные поля |
|---|---|
| `ArrayRef` | `path: str`, `dataset: str`, `sha256: str`, `shape: tuple[int, ...]`, `dtype: Literal['float64','int64','bool']`, `unit: str`, `axis_order: tuple[str,...]` |
| `GridSpec` | `shape: tuple[int,int,int]`, `extent_m: tuple[float,float,float]`, `cell_centers_m: ArrayRef`, `cell_volume_m3: ArrayRef`, `neighbors: ArrayRef`, `z_positive='down'`, `crs: None` |
| `RockSpec` | `porosity: ArrayRef`, `permeability_m2: ArrayRef` shape `(3,n_cells)`, `rock_compressibility_pa_inv=0.0` |
| `FluidSpec` | `kind='OW'`, `density_sc_kg_m3: tuple[float,float]`, `viscosity_pa_s: tuple[float,float]`, `compressibility_pa_inv: tuple[float,float]`, `p_sc_pa`, `t_sc_k`, `corey_exponents`, `residual_saturations`, `kr_endpoints`, `pc_model='zero'`, `educational=True`, `analytical_limit: bool=False`; численные значения §3.1 |
| `WellSpec` | `well_id: str`, `cells: tuple[int,...]`, `radius_m: float=0.1`, `reference_depth_m: float`, `model: Literal['simple','multisegment']='multisegment'` для mixed-layer/crossflow, `allow_crossflow: bool` |
| `ControlSegment` | `start_s: float`, `end_s: float`, `well_id: str`, `role: Literal['producer','injector','shut']`, `target: Literal['liquid_rate','water_rate','bhp','disabled']`, `value: float`, `bhp_limit_pa: float | None`, `connection_open: tuple[bool,...]` |
| `InitialStateSpec` | `kind: Literal['equilibrium','explicit','native_restart']`, `pressure_pa: ArrayRef | None`, `sw: ArrayRef | None`, `restart: RestartRef | None`, `meaning: Literal['synthetic_initial','developed_state']` |
| `BoundarySpec` | `kind: Literal['closed','pressure_water']`, `cells: tuple[int,...]`, `pressure_pa: float | None`, `trans_flow: float | None`, `fractional_flow: tuple[float,float]=(1.0,0.0)`; closed требует пустых cells и null pressure/trans_flow |
| `ObservationSpec` | `table_path: str | None`, `sha256: str | None`, `dynamic_channels: tuple[str,...]`, `pressure_available: bool`, `truth_access: Literal['forbidden']='forbidden'` |
| `CaseBundle` | `schema_version='case-1'`, `spec_version='4.0'`, `case_id`, `world_id`, `sector_id: str | None`, `information_mode='synthetic_forward'`, `start_date: str`, `cutoff: str`, `report_edges_s: tuple[float,...]`, grid/rock/fluids/wells/controls/initial/boundary/observations, `gravity_m_s2`, `renderer_version`, `units: dict[str,str]`, `seeds: dict[str,int]`, `source_hashes: dict[str,str]`, `model_hash: str` |
| `RestartRef` | `manifest_path`, `sha256`, `completed_report_step: int`, `completed_time_s`, `model_hash`, `schedule_prefix_hash`, `environment_lock_hash`, `native_format='Jutul-native'` |
| `OutputRequest` | `state_times_s: tuple[float,...]`, `keep_native_restart: bool`, `chunk_months: int=1`, `diagnostic_substeps: bool=False` |
| `JobDescriptor` | `schema_version='job-1'`, `job_id`, `case_path`, `case_sha256`, `model_hash`, `solver_config_path`, `solver_config_sha256`, `output_request: OutputRequest`, `seed: int`, `result_dir`, `attempt: int`, `resume_from: RestartRef | None` |
| `CostRecord` | `wall_s`, `cpu_s`, `peak_rss_bytes`, `output_bytes`, `accepted_steps`, `cut_steps`, `nonlinear_iterations`, `retry_count`, `measurement_method` |
| `ForwardResult` | `schema_version='forward-1'`, `job_id`, `case_sha256`, `model_hash`, `physics_class`, `status`, `reason: str | None`, `completed_time_s`, `times_s`, `states: dict[str,ArrayRef]`, `monthly_path`, `connections_path`, `balances_path`, `restart: RestartRef | None`, `solver_metadata: dict[str,str]`, `cost: CostRecord`, `parent_attempt_ids: tuple[str,...]` |

Nullable result paths допускаются только у неуспешного/неполного результата. `COMPLETE` требует всей запрошенной оси, проверенных output hashes, restart при запросе и прохождения shape/finite проверки. Хэш отсутствующего поля не заменять нулевой строкой: отсутствие обозначать `None` с причиной.

### 3.3. Статусы и policy

`ForwardResult.status`: `COMPLETE`, `INVALID_INPUT`, `PHYSICALLY_INVALID`, `CONTROL_INFEASIBLE`, `NUMERICAL_FAILURE`, `RESOURCE_FAILURE`, `TIMEOUT`, `PROTOCOL_FAILURE`, `INCOMPLETE_BUDGET`.

`RunRecord.status` остаётся `RUNNING/PASS/FAIL`. Для forward command только COMPLETE даёт PASS; partial/budget/timeout даёт FAIL и exit code 2 с отдельным algorithm status. Диагностическая verification command может PASS, когда ожидаемый отрицательный fixture корректно вернул CONTROL_INFEASIBLE; этот fixture не становится успешной forward realization. Stage status и physics gate хранятся отдельно; E01 не ставит `G_LOCAL`/`G_PHYS(field)` в PASS.

**Retry:** максимум две попытки на один physical model: исходная и одна зарегистрированная numerical retry с `max_timestep_days/2`, большим лимитом нелинейных итераций (15→25), неизменными convergence tolerances и физическими входами. Автоматически повторять только NUMERICAL_FAILURE. Не повторять INVALID_INPUT/PHYSICALLY_INVALID/CONTROL_INFEASIBLE/TIMEOUT/RESOURCE_FAILURE. Системный stop сохраняет завершённые chunks/jobs; продолжение — отдельная команда. Все попытки учитываются в forward budget, даже неуспешные; результат первой попытки не перезаписывается.

## 4. Tasks

### Task 1: E01.0 — согласовать версии config/run/source lineage

**Files:** Modify `src/so_recon/__init__.py`, `config/schema.py:ProjectConfig`, `config/load.py:resolved_config_dict`, `registry/run.py:RunContext.start`, `registry/source_manifest.py:build_source_manifest`, `cli.py:_manifest_body`; Create `configs/e01.yml`; Test `tests/unit/test_e01_versions.py`.

**Interfaces:** Consumes existing `load_project_config(Path) -> ProjectConfig`, `RunContext.start(...)`, `build_source_manifest(sources, paths, *, config_version)`. Produces support for 3.0/4.0; extended `build_source_manifest(..., spec_version: Literal['3.0','4.0']='3.0')`. Новые команды всегда передают версию явно.

- [ ] **1.1. Написать тест версии в обоих manifests.** Использовать существующий `tmp_project`, не private CSV:

```python
import pytest
from so_recon.config.load import load_project_config
from so_recon.paths import ProjectPaths
from so_recon.registry.run import RunContext
from so_recon.registry.source_manifest import build_source_manifest

@pytest.mark.parametrize('version', ['3.0', '4.0'])
def test_lineage_uses_validated_config(tmp_project, version):
    path = tmp_project / 'configs/project.yml'
    text = path.read_text().replace('"3.0"', f'"{version}"')
    path.write_text(text)
    cfg = load_project_config(path)
    paths = ProjectPaths.from_config(tmp_project, cfg.paths)
    ctx = RunContext.start(command='probe', argv=[], cfg=cfg, paths=paths)
    manifest = build_source_manifest(
        cfg.sources, paths, config_version=cfg.config_version,
        spec_version=cfg.spec_version,
    )
    assert ctx.record.spec_version == manifest.spec_version == version
```

- [ ] **1.2. Run RED:** `uv run pytest tests/unit/test_e01_versions.py -q`. Ожидается отказ 4.0/неизвестный keyword. Используемый `ProjectPaths.from_config(root, cfg.paths)` подтверждён в существующем `paths.py`.
- [ ] **1.3. Разрешить две версии и убрать глобальный runtime stamp.** В `__init__.py` добавить `SUPPORTED_SPEC_VERSIONS = ('3.0', '4.0')`, `LATEST_SPEC_VERSION = '4.0'`; старый `SPEC_VERSION='3.0'` оставить legacy alias до удаления его потребителей. В соответствующих существующих constructors:

```python
# ProjectConfig
spec_version: Literal['3.0', '4.0']

# RunContext.start -> RunRecord constructor
spec_version=cfg.spec_version if cfg is not None else UNAVAILABLE,

# build_source_manifest -> return
return SourceManifest(
    spec_version=spec_version, config_version=config_version, sources=entries,
)
```

В `_manifest_body` передавать `spec_version=cfg.spec_version`; при 4.0 публиковать `source_manifest-4.0.json`, при 3.0 прежний `source_manifest.json`. Чтение старых записей не меняет их содержимое. Схема source manifest остаётся 2: изменилось значение разрешённой версии, не структура.
- [ ] **1.4. Создать `configs/e01.yml`.** Скопировать только существующие paths/julia defaults; `spec_version: '4.0'`, `config_version: 'E01.1'`, `project_name: SO-RECON`, `sources: {files: []}`. Smoke остаётся на legacy `configs/project.yml`. Resource block добавит Task 3; до него новый config применим только к проверки совместимости.
- [ ] **1.5. Закрыть отрицательные случаи.** Добавить parameterized tests для неизвестной 5.0, malformed YAML, `cfg=None` → `spec_version='unavailable'` и FAIL через существующий runner; прочитать сохранённый старый RunRecord с 3.0. Зафиксировать SHA существующих resolved legacy config/source manifest до изменения и проверить отсутствие дрейфа. Добавленные optional E01 fields исключать из `resolved_config_dict` у 3.0 явно, а не глобальным `exclude_defaults`, который меняет старую сериализацию.

```python
def test_start_without_config_has_no_invented_spec(tmp_project):
    cfg = load_project_config(tmp_project / 'configs/project.yml')
    paths = ProjectPaths.from_config(tmp_project, cfg.paths)
    ctx = RunContext.start(command='invalid', argv=[], cfg=None, paths=paths)
    ctx.finish('FAIL', notes=['config unavailable'])
    assert ctx.record.spec_version == 'unavailable'
    assert ctx.record.status == 'FAIL'
```

- [ ] **1.6. Run GREEN:** `uv run pytest tests/unit/test_e01_versions.py tests/unit/test_config.py tests/unit/test_run_registry.py tests/unit/test_source_manifest.py tests/unit/test_runner.py tests/unit/test_cli.py -q`; затем `make smoke`. Ожидается PASS без `--freeze-expected` и без diff legacy artifacts. Commit checkpoint: `fix: preserve spec version through config and lineage`.

### Task 2: E01.0 — минимальные contracts и безопасный scientific exchange

**Files:** Create `src/so_recon/simulator/contracts.py`, `case_io.py`, `tests/unit/test_case_contracts.py`; Modify Python/Julia dependency manifests.

**Interfaces:** Produces records §3.2; `validate_case(case: CaseBundle, paths: ProjectPaths) -> ValidationReport`; `load_case(path: Path, paths: ProjectPaths) -> CaseBundle`; `write_case(case: CaseBundle, paths: ProjectPaths, ctx: RunContext) -> ArtifactRef`. `ValidationReport` содержит `valid: bool`, `errors: tuple[str,...]`; каждый error с полем и причиной. `ArrayRef` проверяется до запуска Julia.

- [ ] **2.1. Написать отрицательные проверки индексов/единиц/NaN.** В тесте создать HDF5 `(2,3)` с асимметричными значениями; проверить shape и `axis_order` после чтения. Путь с `..`, symlink наружу, неверный SHA, NaN permeability, control нефти у producer, несовпадение mask/well cells должны быть отвергнуты до subprocess.

```python
import numpy as np
import pytest
from so_recon.simulator.case_io import validate_numeric_array

@pytest.mark.parametrize('bad', [np.nan, np.inf, -1.0, 0.0])
def test_permeability_must_be_finite_positive(bad):
    with pytest.raises(ValueError, match='permeability'):
        validate_numeric_array('permeability', np.array([bad]), positive=True)

def test_porosity_is_not_silently_clipped():
    with pytest.raises(ValueError, match='porosity'):
        validate_numeric_array('porosity', np.array([1.2]), fraction=True)
```

- [ ] **2.2. Run RED:** `uv run pytest tests/unit/test_case_contracts.py -q` → отсутствующий модуль/functions.
- [ ] **2.3. Добавить зависимости у потребителя.** `uv add h5py`; добавить HDF5 в существующий Julia project через `Pkg.add("HDF5")`, не `Pkg.update()`. Проверить, что pinned Jutul/JutulDarcy/JSON версии прежние; выполнить `uv sync --frozen` и Julia instantiate. Schema и validation остаются Python strict models; общий JSON shape проверяется также Julia decoder.
- [ ] **2.4. Реализовать records таблицы §3.2 в указанном порядке.** Использовать `Literal`/`Field(gt=0)` и model validators для зависимых полей. Ядро array validation:

```python
import numpy as np
from numpy.typing import NDArray

def validate_numeric_array(
    name: str, values: NDArray[np.float64], *,
    positive: bool = False, fraction: bool = False,
) -> None:
    if not np.isfinite(values).all():
        raise ValueError(f'{name}: nonfinite values')
    if positive and not (values > 0).all():
        raise ValueError(f'{name}: must be positive')
    if fraction and not ((values >= 0) & (values <= 1)).all():
        raise ValueError(f'{name}: outside [0,1]')
```

Grid validator требует `nx*ny*nz` элементов rock/state, neighbors из допустимых cell IDs без self/duplicate edges, symmetric Cartesian topology, уникальные wells/connections, monotonically increasing report edges с первым 0. Pressure positive, saturation sum tolerance `1e-12`. Пустой observations table допустим; неизвестные единицы — error. В 4.0 input paths всегда project-relative и разрешаются через существующий `ProjectPaths.resolve` с containment после symlink.
- [ ] **2.5. Реализовать immutable publication.** HDF5 писать во временный соседний файл одним writer, flush/close, SHA, сравнение существующего destination, atomic rename; затем existing `register_artifact` и `ctx.add_output`. JSON manifest публикуется последним через `write_json_artifact`. Потеря процесса до manifest оставляет незавершённый staging, не валидный CaseBundle. Повтор разных bytes по тому же artifact path запрещён. Source hashes для synthetic обозначают generator/config/arrays; не выдумывать raw CSV hashes.
- [ ] **2.6. Проверить anti-transpose exchange.** Julia читает Python `(2,3)` array со значениями `[[1,2,4],[8,16,32]]` и пишет semantic cell/time pairs обратно; тест сравнивает каждую пару, не только shape. Native layout преобразуется явно по `axis_order`. JSON с тем же model hash и изменённым fluid/controls/array SHA должен отвергаться.
- [ ] **2.7. Run GREEN:** unit contracts + real HDF5 exchange + `make smoke`. Commit checkpoint: `feat: add versioned forward case and result contracts`.

### Task 3: E01.0 — ресурсные профили, probe и budget ledger

**Files:** Create `src/so_recon/config/resources.py`, `environment/resources.py`, `simulator/budget.py`, `tests/unit/test_resources.py`; Modify `config/schema.py`, `config/load.py`, `configs/e01.yml`, `pyproject.toml`, `uv.lock`.

**Interfaces:** `ResourceProfile` содержит profile ID, soft/hard/reserve/disk bytes, wall/job timeout seconds, forward limit, Julia/BLAS threads, poll interval, `max_swap_growth_bytes`. `ResourceSnapshot` содержит physical/available bytes, process-tree RSS/CPU, swap used, pressure status + method, disk free, monotonic timestamp. `probe_resources(pid: int | None, output_root: Path) -> ResourceSnapshot`. `BudgetLedger.reserve(job_id, estimated_s, estimated_bytes) -> None` проверяет до запуска и учитывает попытку; `complete(job_id, result)`, `stop(reason)` атомарно обновляют ledger. Policy rejection = `BudgetStop(status, reason)`.

- [ ] **3.1. Написать boundary tests с injected snapshots/time.** Проверить 8 GiB laptop (не отрицательный hard), занятую другими процессами память, исчерпание диска, swap growth, wall timeout, лимит attempts и резерв cleanup. Не выделять 16 GiB RAM для теста.

```python
from so_recon.simulator.budget import effective_hard_bytes

def test_memory_limit_accounts_for_other_apps():
    gib = 2**30
    assert effective_hard_bytes(
        configured=16*gib, total=24*gib,
        available=8*gib, project_rss=4*gib, reserve=6*gib,
    ) == 6*gib
```

- [ ] **3.2. Run RED:** `uv run pytest tests/unit/test_resources.py -q`.
- [ ] **3.3. Добавить `psutil` через uv и typed resource config.** В `ProjectConfig` добавить `resources: ResourceProfile | None = None`; отсутствие разрешено для 3.0 и 4.0 manifest/config-only checks, но любой E01 physical command отказывает без ресурcного профиля. Legacy resolved serialization удаляет только это новое поле. В `configs/e01.yml`:

```yaml
resources:
  profile: P0_VERIFY
  soft_bytes: 12884901888
  hard_bytes: 17179869184
  reserve_bytes: 6442450944
  disk_budget_bytes: 5368709120
  wall_budget_s: 600
  job_timeout_s: 300
  startup_timeout_s: 300
  max_new_forward: 64
  julia_workers: 1
  julia_threads: 4
  blas_threads: 1
  poll_interval_s: 0.25
  max_swap_growth_bytes: 536870912
```

P1 explicit override в CLI выбирает утверждённый preset: wall 3600, max forward 2000, job timeout 900, остальные пределы прежние. Cold startup входит в wall budget; исторический smoke timeout 1800 не меняется. Job timeout 300/900 и swap growth 512 MiB — консервативные решения E01, отдельно от нормативных session limits.
- [ ] **3.4. Реализовать cap и guards.**

```python
def effective_hard_bytes(
    *, configured: int, total: int, available: int,
    project_rss: int, reserve: int,
) -> int:
    return max(0, min(configured, total-reserve, project_rss+available-reserve))
```

Hard меньше soft → `effective_soft=min(configured_soft, 0.75*effective_hard)`; cap=0 запрещает запуск. Сумма RSS — консервативный индикатор, не точное unique resident memory. macOS pressure читать read-only native probe; сохранять raw categorical result/method. При недоступности OS pressure — `unknown`, available/swap guards продолжают работать, отчёт явно отмечает ограничение. Не подменять неизвестное измерение нулём. Hardware probe фиксирует arch, OS, total RAM, CPU counts, внешнее питание (или unknown), Julia executable/version; MPS/CUDA capability не требует установки PyTorch в E01.
- [ ] **3.5. Реализовать polling watchdog и ledger.** Soft → warning и release завершённых allocations/cache; hard/critical pressure/swap limit → запрет новых jobs и просьба закончить текущий chunk, при продолжающемся росте terminate process group. Disk guard учитывает размер текущей сессии, predicted next output и свободное место, с резервом 64 MiB для записей failure/ledger. Полный диск при невозможности записать run.json остаётся существующей `RunRecordUnavailableError`, не ложным PASS.
- [ ] **3.6. Тестировать resume accounting.** Ledger stores `session_id`, attempted/completed/pending IDs, cost, input hashes и checkpoint refs. Новая resume session получает свой wall budget и parent ledger hash; суммарная стоимость не обнуляется. Выполненные jobs пропускаются только после SHA verification; incomplete job не skip.
- [ ] **3.7. Run GREEN:** resources tests + config/runner regression. Commit checkpoint: `feat: enforce measured resource and session budgets`.

### Task 4: E01.0 — persistent worker и изоляция jobs

**Files:** Create `src/so_recon/simulator/worker.py`, `julia/worker/main.jl`, `tests/unit/test_worker.py`, `tests/integration/test_e01_worker.py`; reuse `simulator/julia_bridge.py:find_julia`.

**Interfaces:** `PersistentJuliaWorker(executable: Path, project: Path, session_dir: Path, profile: ResourceProfile)` — context manager; `.submit(job: JobDescriptor, ledger: BudgetLedger) -> ForwardResult`; `.close() -> None`; `.pid: int`. `main.jl` читает JSON lines с `op='run'|'shutdown'`, `job_path`, `job_id`; replies содержат `job_id/status/result_path/result_sha256`. `READY` содержит pid/versions/protocol version; stdout строго protocol, native logs перенаправлены в job stdout/stderr files.

- [ ] **4.1. Написать transport tests.** Fake executable в tests нужен только для hung process, malformed JSON, wrong ID/SHA, crash и flooded stderr. Проверять, что worker constructor не ждёт бесконечно READY, reader не блокируется на полном PIPE, deadline измеряется monotonic clock, shutdown уничтожает descendants. Не использовать этот fake для physics acceptance.

```python
from so_recon.simulator.worker import validate_reply
import pytest

def test_stale_reply_is_not_accepted():
    with pytest.raises(ValueError, match='job_id'):
        validate_reply({'job_id': 'previous', 'status': 'COMPLETE'}, 'current')
```

- [ ] **4.2. Run RED:** `uv run pytest tests/unit/test_worker.py -q`.
- [ ] **4.3. Реализовать Python process lifecycle.** `subprocess.Popen([...], start_new_session=True, stdin=PIPE, stdout=PIPE, stderr=logfile, text=True)`; отдельный bounded protocol reader + queue, ожидание через timeout ≤poll interval; никакого неограниченного `readline()` в control loop. `JULIA_NUM_THREADS`/BLAS выставляются из profile. `.close()` отправляет shutdown, после grace 5 s SIGTERM process group, ещё через 5 s SIGKILL, wait/reap. Исключения и Ctrl-C проходят через finally. Existing `SubprocessJuliaLauncher` не заменять.

```python
def validate_reply(reply: dict[str, object], job_id: str) -> None:
    if reply.get('job_id') != job_id:
        raise ValueError('job_id mismatch')
    if reply.get('status') not in {
        'COMPLETE', 'INVALID_INPUT', 'PHYSICALLY_INVALID',
        'CONTROL_INFEASIBLE', 'NUMERICAL_FAILURE', 'RESOURCE_FAILURE',
        'TIMEOUT', 'PROTOCOL_FAILURE', 'INCOMPLETE_BUDGET',
    }:
        raise ValueError('unknown worker status')
```

- [ ] **4.4. Реализовать Julia loop без model global.** Загружать Jutul/JutulDarcy/JSON/HDF5 один раз. Каждый op run загружает descriptor и проверяет bytes SHA всех inputs, строит новую локальную model/state в функции, пишет result в уникальный directory атомарно, возвращает только immutable paths. `try/catch/finally`, `GC.gc()` после освобождения locals; stdout logging redirect не должен захватывать protocol response. До Task 5 поддержать отдельный diagnostic `ping` c echo SHA, а physics `run` возвращает `INVALID_INPUT: adapter unavailable`; он не может COMPLETE без solver.
- [ ] **4.5. Реальный protocol тест.** Запустить Julia один раз, выполнить два `ping`, проверить один pid, exact job IDs/hashes и shutdown. `uv run pytest tests/integration/test_e01_worker.py -m julia -q`. Test skip допустим в обычном developer run без Julia, но stage gate требует реальные executed tests и отвергает skip.
- [ ] **4.6. Подключить resource watchdog.** Проверить injected hard/disk timeout while fake job running → classified result, FAILED run, сохранённые completed refs. Stop file `stop-after-chunk` является atomic control plane request и не меняет immutable job inputs. Task 8 реализует чтение на границе native chunks. Force-killed неизвестная ошибка не классифицируется OOM только по exit -9: RESOURCE_FAILURE требует guard/OS evidence; иначе PROTOCOL_FAILURE c exit/signal.
- [ ] **4.7. Run GREEN:** transport/real protocol + E00 bridge regression. Commit checkpoint: `feat: add isolated persistent Julia job transport`.

### Task 5: построить OW model на нативном JutulDarcy

**Files:** Create `julia/adapter/SOReconAdapter.jl`, `model.jl`, `julia/verification/fixtures.jl`; Extend `julia/worker/main.jl`; Create `tests/integration/test_e01_physics.py`.

**Interfaces:** Julia `build_ow(case::AbstractDict, arrays::AbstractDict) -> NamedTuple{(:model,:parameters,:state0)}`; `load_arrays(case, root)` читает только ArrayRefs с проверенными SHA; `verification_case(name::Symbol; nx=16, nz=1) -> (case, arrays)` возвращает маленький fully materialized fixture. В тестах `:closed_cell`, `:bl`, `:hydrostatic`, `:two_layer`, `:boundary`, `:five_spot`; имена/физика уточнены в Tasks 9–10. `run_job(job, root)` — exported worker entrypoint; COMPLETE разрешён только после integration Task 7.

- [ ] **5.1. Написать Julia test физического constructor.** `fixtures.jl` включается и из standalone verification scripts, и из worker; скрипт не запускает тесты при простом include. Создать `:closed_cell`: `(1,1,1)`, extent `(10,10,10) m`, phi 0.2, k 100 mD, p `1.5e7 Pa`, Sw 0.3, без wells, 1 day. Никаких external files/примеров с загрузкой SPE-данных.

```julia
using Test, Jutul, JutulDarcy
include("../adapter/SOReconAdapter.jl")
using .SOReconAdapter
case, arrays = verification_case(:closed_cell)
physical = build_ow(case, arrays)
@test all(physical.state0[:Reservoir][:Pressure] .> 0.0)
@test sum(physical.state0[:Reservoir][:Saturations][:, 1]) ≈ 1.0
@test sum(pore_volume(physical.model, physical.parameters)) ≈ 200.0
```

- [ ] **5.2. Run RED:** `julia --project=julia --startup-file=no julia/verification/fixtures.jl --test-model` → отсутствующий constructor. Запуск этого и следующих real verification scripts учитывается в явном P0 development session ledger через launcher; не запускать бесконтрольно десятки отдельных Julia процессов. Standalone команда предназначена для одной адресной диагностики; серийный gate Task 12 использует persistent worker.
- [ ] **5.3. Реализовать геометрию и фазовую модель.** `arrays` содержит нормализованные поля `porosity`, `permeability_m2`, `pressure_pa`, `sw` с подтверждённой раскладкой. Ядро `model.jl`:

```julia
function build_ow(case::AbstractDict, arrays::AbstractDict)
    dims = Tuple(Int.(case["grid"]["shape"]))
    extent = Tuple(Float64.(case["grid"]["extent_m"]))
    mesh = CartesianMesh(dims, extent)
    domain = reservoir_domain(mesh;
        permeability = arrays["permeability_m2"],
        porosity = arrays["porosity"])
    wells = [setup_well(domain, Int.(w["cells"]) .+ 1;
        name = Symbol(w["well_id"]), radius = Float64(w["radius_m"]),
        simple_well = w["model"] == "simple",
        reference_depth = Float64(w["reference_depth_m"]))
        for w in case["wells"]]
    fluid = case["fluids"]
    rhoS = Float64.(fluid["density_sc_kg_m3"])
    sys = ImmiscibleSystem((AqueousPhase(), LiquidPhase()); reference_densities = rhoS)
    model = setup_reservoir_model(domain, sys; wells = wells, extra_outputs = false)
    rho = ConstantCompressibilityDensities(
        p_ref = Float64(fluid["p_sc_pa"]), density_ref = rhoS,
        compressibility = Float64.(fluid["compressibility_pa_inv"]))
    kr = BrooksCoreyRelativePermeabilities(sys,
        Float64.(fluid["corey_exponents"]),
        Float64.(fluid["residual_saturations"]),
        Float64.(fluid["kr_endpoints"]))
    replace_variables!(model; PhaseMassDensities = rho, RelativePermeabilities = kr)
    parameters = setup_parameters(model)
    sw = arrays["sw"]
    state0 = setup_reservoir_state(model;
        Pressure = arrays["pressure_pa"], Saturations = vcat(sw', (1 .- sw)'))
    return (; model, parameters, state0)
end
```

Код показывает подтверждённый native constructor path. При переносе дополнить viscosity parameters одинаковыми значениями для reservoir и всех wells, gravity field по native domain API и constant PV — отдельными шагами ниже, до признания constructor готовым. FluidSpec использует имена JSON из этого блока: `density_sc_kg_m3`, `viscosity_pa_s`, `compressibility_pa_inv`, `p_sc_pa`, `t_sc_k`, `corey_exponents`, `residual_saturations`, `kr_endpoints`.
- [ ] **5.4. Закрепить viscosity/gravity на существующем parameter API.** После `setup_parameters` и до `setup_reservoir_state` применить:

```julia
mu = Float64.(fluid["viscosity_pa_s"])
for (name, submodel) in pairs(model.models)
    if name == :Reservoir || JutulDarcy.model_or_domain_is_well(submodel)
        parameters[name][:PhaseViscosities] .= reshape(mu, 2, 1)
    end
end
@assert Jutul.gravity_constant == 9.80665
@assert case["gravity_m_s2"] == Jutul.gravity_constant
```

Основной OW adapter принимает только native g=9.80665, не обещает произвольный gravity keyword. У JutulDarcy `TwoPointGravityDifference` строится из z-down centroids; проверить native values по `compute_face_gdz` и hydrostatic equilibrium. В горизонтальном BL нет перепада z и gravity contribution точно нулевой при том же native g. Отдельный zero-gravity diagnostic без wells может занулить только `parameters[:Reservoir][:TwoPointGravityDifference]`, с явной пометкой analytical limit; он не входит в основной dispatcher и не меняет global constant. Test сравнивает configured mu на всех submodels и направление gravity segregation. PV постоянен за счёт default pore-volume parameter без pressure-dependent replacement; подтвердить `pore_volume` до/после изменения pressure.
- [ ] **5.5. Проверить SI/PVT независимым числом.** Для pressure `(p_sc, 1.5e7)` проверить `rho=rho_sc*exp(c*(p-p_sc))`, `B=rho_sc/rho`, positivity и монотонность. При p_sc B=1; pressure growth увеличивает density. Показать в test, что `mu_o/mu_w=3` и phase order не переставлен. Для horizontal BL fluid допускается отдельный incompressible limit c=0, явно `analytical_limit=true`; validator разрешает его только registered analytic fixtures. Общий P1 остаётся c>0.
- [ ] **5.6. Связать worker с adapter.** `SOReconAdapter.jl` только imports/includes/exports, `model.jl` не держит singleton model. При jobs A→B→A rebuild state0/controls/parameters; immutable warm package state сохраняется, model state освобождается. Numerical engine calls только JutulDarcy.
- [ ] **5.7. Run GREEN:** constructor/PVT tests + `make smoke`. Commit checkpoint: `feat: construct explicit educational oil-water models`.

### Task 6: календарь, rate/BHP controls и события подключений

**Files:** Create `src/so_recon/simulator/schedule.py`, `julia/adapter/controls.jl`, `tests/unit/test_schedule.py`; Extend `julia/verification/fixtures.jl` и integration physics tests.

**Interfaces:** `month_edges(start: date, count: int) -> tuple[date,...]`; `compile_schedule(report_edges_s: tuple[float,...], segments: tuple[ControlSegment,...]) -> Schedule`; `Schedule` = frozen record `edges_s: tuple[float,...]`, `month_index: tuple[int,...]`, `controls_by_interval: tuple[tuple[ControlSegment,...],...]`. Julia `build_forces(model, controls, boundary) -> forces`; `controls` явно для одного interval, отсутствующие role/mask не наследуются.

- [ ] **6.1. Написать calendar tests.** Февраль високосного года, событие внутри месяца, zero-duration, gaps/overlaps, rate=0 без phase observations, role switch с двумя потоками в одном месяце.

```python
from datetime import date
from so_recon.simulator.schedule import month_edges

def test_leap_year_calendar():
    edges = month_edges(date(2020, 1, 1), 3)
    assert [(b-a).days for a, b in zip(edges, edges[1:])] == [31, 29, 31]
```

- [ ] **6.2. Run RED:** `uv run pytest tests/unit/test_schedule.py -q`.
- [ ] **6.3. Реализовать explicit календарь без среднего месяца 30 дней.**

```python
from datetime import date

def month_edges(start: date, count: int) -> tuple[date, ...]:
    if start.day != 1 or count < 1:
        raise ValueError('start must be month start and count positive')
    base = start.year*12 + start.month-1
    return tuple(date((base+i)//12, (base+i)%12+1, 1) for i in range(count+1))
```

`compile_schedule` строит sorted union report edges + всех event boundaries; каждый well имеет ровно один control на каждом interval, иначе INVALID_INPUT. Uptime synthetic задаётся известными open/shut intervals; интеграл положительного потока при нулевом uptime отвергается. Нет восстановления неизвестного полевого uptime в E01.
- [ ] **6.4. Реализовать controls через native targets.**

```julia
function native_control(c::AbstractDict, rho_water_sc::Float64)
    role, target = c["role"], c["target"]
    if role == "shut"
        target == "disabled" || error("shut requires disabled target")
        return DisabledControl()
    elseif target == "bhp"
        bhp = BottomHolePressureTarget(Float64(c["value"]))
        return role == "producer" ? ProducerControl(bhp) :
            InjectorControl(bhp, [1.0, 0.0]; density = rho_water_sc)
    elseif role == "producer" && target == "liquid_rate"
        return ProducerControl(SurfaceLiquidRateTarget(-Float64(c["value"])/86400.0))
    elseif role == "injector" && target == "water_rate"
        return InjectorControl(SurfaceWaterRateTarget(Float64(c["value"])/86400.0),
            [1.0, 0.0]; density = rho_water_sc)
    end
    error("unsupported role/target pair")
end
```

Target value nonnegative для rates и positive для BHP; валидатор запрещает одновременно отдельные oil/water prescribed production rates. Для rate controls limits словарь вида `Dict(:P => (bhp=minimum_pa,), :I => (bhp=maximum_pa,))`; отключить неявные native default limits (`set_default_limits=false`) и задавать только записанные case constraints. Limit branch фиксировать через actual operating target, requested/achieved rates и BHP; нельзя выдавать исполненный BHP-контроль за выполненный rate.
- [ ] **6.5. Реализовать completion masks отдельно от surface shutdown.**

```julia
mask = PerforationMask(Float64.(control_segment["connection_open"]))
well_id = Symbol(control_segment["well_id"])
forces[well_id] = setup_forces(model.models[well_id]; mask = mask)
```

Для каждого interval применять mask и role заново ко всем wells. При открытых двух слоях и выключенной поверхности сохранять native well coupling, если выбранная нативная control семантика допускает crossflow. Отдельный zero-net-surface/crossflow fixture использует активную native zero-net-rate постановку при необходимости; `DisabledControl` не объявлять эквивалентом этой постановки без теста. Полная isolation = все masks false; требовать нулевой connection mass flux. Если native model не реализует заданный allow_crossflow, INVALID_INPUT с причиной вместо скрытого изменения semantics.
- [ ] **6.6. Run GREEN:** calendar unit + реальный двухинтервальный rate/BHP fixture. Проверить native negative producer sign, `Vo+Vw = q_liquid*uptime` в достижимом случае; недостижимый q → CONTROL_INFEASIBLE с actual target. Commit checkpoint: `feat: map calendar controls and completion events to Jutul`.

### Task 7: accepted-substep интегралы и проверяемые результаты

**Files:** Create `julia/adapter/outputs.jl`, `src/so_recon/simulator/results.py`, `src/so_recon/validation/__init__.py`, `validation/balance.py`, `tests/unit/test_forward_results.py`.

**Interfaces:** Julia `extract_interval(result, model, forces) -> NamedTuple` возвращает accepted substep times, signed surface phase rates, connection component mass flux, pressure и component inventory. Python `integrate_monthly(steps: pa.Table, month_edges_s: tuple[float,...]) -> pa.Table`; `component_balance(inventory, net_source_integrals, floor) -> BalanceMetrics`; `load_forward_result(path: Path, paths: ProjectPaths) -> ForwardResult` checks hashes/fields. `BalanceMetrics`: `cumulative_relative`, `median_step_relative`, `max_step_relative`, `absolute_residual` per component.

- [ ] **7.1. Написать независимый интегральный test с меняющимся фазовым составом.** Table columns `well_id,start_s,end_s,oil_prod_m3_s,water_prod_m3_s,water_inj_m3_s`; все публичные rates неотрицательные, production/injection разделены. Один substep не пересекает event/month boundary. Нефть 1 m³/day первые 10 дней и 3 последние 20 → 70, не 90 m³. Нулевая жидкость → `fw=null, fw_valid=false`.

```python
import pyarrow as pa
import pytest
from so_recon.simulator.results import integrate_monthly

def test_monthly_volume_is_not_final_rate_times_month():
    steps = pa.Table.from_pylist([
        dict(well_id='P', start_s=0., end_s=864000.,
             oil_prod_m3_s=1/86400, water_prod_m3_s=0., water_inj_m3_s=0.),
        dict(well_id='P', start_s=864000., end_s=2592000.,
             oil_prod_m3_s=3/86400, water_prod_m3_s=0., water_inj_m3_s=0.),
    ])
    rows = integrate_monthly(steps, (0., 2592000.)).to_pylist()
    assert rows[0]['oil_prod_m3_sc'] == pytest.approx(70.)
```

- [ ] **7.2. Run RED:** `uv run pytest tests/unit/test_forward_results.py -q`.
- [ ] **7.3. Извлечь accepted ministeps штатным API.** На одном calendar chunk запускать с `output_substates=true`. Раскрывать `result.result`, а не high-level `result.states` (последний содержит только reservoir state). Зафиксировать этот выбор тестом с timestep cuts.

```julia
states, dt, report_index = Jutul.expand_to_ministeps(result.result)
step_forces = forces isa AbstractVector ? forces[report_index] : forces
oil = JutulDarcy.well_output(model, states, :Producer, step_forces, SurfaceOilRateTarget)
water = JutulDarcy.well_output(model, states, :Producer, step_forces, SurfaceWaterRateTarget)
oil_volume = -sum(oil .* dt)
water_volume = -sum(water .* dt)
```

В рабочем коде обходить реальные well IDs, не hardcode Producer. Проверять длины, положительные dt, sum(dt)=chunk duration, отсутствие failed/cut trial states. Последний scalar well rate годится только как diagnostic snapshot. Суммирование совместимо с backward-Euler flux quadrature решателя.
- [ ] **7.4. Сохранить actual flow branches и connection diagnostics.** В отдельных substep/connection таблицах: `(well_id,connection_id,cell_id,start_s,end_s,water_mass_kg_s,oil_mass_kg_s,total_mass_kg_s,connection_open,actual_target,bhp_pa)`. Потоки брать из нативного well/reservoir cross-term для фактического upwind state, не распределять surface дебит по kh. Для выбранного MultiSegmentWell использовать `JutulDarcy.multisegment_well_perforation_flux!(out, sys, state_res, state_well, rhoS, conn)` из `src/facility/wells/wells.jl` и фактические connection coefficients/mask из model/forces. Для вычисления native secondary fields восстановить их нативным update из substate, не реконструировать фазовый состав по surface fractions. Pin-specific extraction тестировать против conservation reservoir↔well и zero mask. Native well storage учитывается в полном component inventory; surface поток не обязан мгновенно равняться сумме connection flux при изменении well storage.
- [ ] **7.5. Реализовать monthly aggregator.** Разделить production/injection по фактическому знаку на каждом substep; при well role switch за месяц обе группы сохраняются. Накопить `V += rate*dt`, `fw=Vw/(Vo+Vw)` только при total>volume floor. Нельзя суммировать signed water так, чтобы закачка уничтожала добытую воду. Reject NaN, отрицательные публичные volumes, overlapping substeps и substep, пересекающий month boundary без явного split. Empty/shut well month сохранять как строку с null fw.
- [ ] **7.6. Реализовать независимый balance evaluator.** Определить `I_k` в standard m³ из native component masses / reference density, включая well inventory; boundary/component sources сохранять со знаком. Native cumulative sources проверять независимо по геометрии/state/PVT, не вычислять источник как разность inventory.

```python
import numpy as np

def relative_balance_errors(inventory, net_source_integrals, floor=1e-6):
    inventory = np.asarray(inventory, dtype=np.float64)
    flux = np.asarray(net_source_integrals, dtype=np.float64)
    if inventory.shape[0] != flux.shape[0]+1:
        raise ValueError('inventory must include initial state')
    residual = np.diff(inventory, axis=0)-flux
    scale = np.maximum(np.maximum(abs(inventory[:-1]), abs(flux)), floor)
    cumulative = inventory[-1]-inventory[0]-flux.sum(axis=0)
    cumulative_scale = np.maximum(
        np.maximum(abs(inventory[0]), abs(flux).sum(axis=0)), floor,
    )
    return abs(residual)/scale, abs(cumulative)/cumulative_scale
```

Функция `component_balance` оборачивает эти массивы в `BalanceMetrics`. `floor=1e-6 m³_sc`, дополнительно показывать residual/throughput и абсолютную невязку, чтобы большой initial inventory не скрывал ошибку малого отбора. Unit test: closed inventory constant → 0; намеренно потерять один interval injection → nonzero fail; mass/density conversion проверить ручным числом.
- [ ] **7.7. Записать minimal outputs.** HDF5 `states.h5`: time, cell_id, pressure, Sw, So, PV и B в requested dates; Parquet: monthly, connection integrals/pressure summaries, balances, accepted-step diagnostics. `.h5` — один writer/world. Model geometry/rock повторно не копировать на каждый timestep. Finalize manifest only after flush/hash всех файлов; большие full substates не помещать в `ForwardResult` JSON.
- [ ] **7.8. Run GREEN:** unit integral/balance/shape/corruption + real case с внутренними cuts и month event; compare summed standard phase volumes с native mass balance, а не с собственным aggregator второй раз. Commit checkpoint: `feat: extract conservative monthly forward outputs`.

### Task 8: native restart, bounded retries и отсутствие скрытого состояния

**Files:** Create `julia/adapter/restart.jl`, `src/so_recon/simulator/forward.py`, `tests/unit/test_forward_policy.py`, `tests/integration/test_e01_restart.py`; Extend worker/result contracts.

**Interfaces:** Python `simulate(case: CaseBundle, output_request: OutputRequest, *, worker: PersistentJuliaWorker, ctx: RunContext, ledger: BudgetLedger) -> ForwardResult`; `resume(case: CaseBundle, restart: RestartRef, future_policy: tuple[ControlSegment,...], *, worker, ctx, ledger) -> ForwardResult`. Julia `run_forward(job, case, arrays)` manages one-month chunks and native output; `verify_restart(ref, case, lock_hash)` checks all metadata before native read. `future_policy` в E01 может повторять исходный suffix либо изменять только интервалы после checkpoint; одинаковый prefix обязателен.

- [ ] **8.1. Написать policy tests до кода.**

```python
import pytest
from so_recon.simulator.forward import should_retry

@pytest.mark.parametrize('status,attempt,expected', [
    ('NUMERICAL_FAILURE', 0, True), ('NUMERICAL_FAILURE', 1, False),
    ('RESOURCE_FAILURE', 0, False), ('CONTROL_INFEASIBLE', 0, False),
    ('TIMEOUT', 0, False), ('INVALID_INPUT', 0, False),
])
def test_only_one_numerical_retry(status, attempt, expected):
    assert should_retry(status, attempt) is expected
```

- [ ] **8.2. Run RED:** `uv run pytest tests/unit/test_forward_policy.py -q`.
- [ ] **8.3. Реализовать retry policy и status mapping.**

```python
def should_retry(status: str, attempt: int) -> bool:
    return status == 'NUMERICAL_FAILURE' and attempt == 0
```

В descriptor retry хранить original model hash, отдельный solver-config hash, parent attempt ID. Wall/forward cost первой попытки остаётся в ledger. Перед retry заново проверить guards и remaining budget. Никакого `likelihood=0`, success-on-partial, изменения permeability/So или tolerance ради сходимости.
- [ ] **8.4. Включить native restart на границе report steps.** Использовать нативный `output_path` и `restart`; не собирать continuation только из HDF5 pressure/Sw. Проверенная семантика Jutul 0.4.31: `restart=k` начинает step k и читает native state `k-1`; continuation после step k требует `restart=k+1` либо `true` с проверенным последним сохранённым индексом. Сохранить reservoir, well и facility state, native report/config context.

```julia
# Native prefix: dt_prefix and forces_prefix are the exact first k report intervals.
prefix = simulate_reservoir(state0, model, dt_prefix;
    parameters = parameters, forces = forces_prefix,
    output_path = native_work_dir, output_substates = true, info_level = -1)
# Continuation uses the full unchanged schedule prefix plus the requested suffix.
continued = simulate_reservoir(state0, model, dt_full;
    parameters = parameters, forces = forces_full,
    output_path = native_work_dir, restart = k+1,
    output_substates = true, info_level = -1)
```

Immutable checkpoint snapshot и mutable native working directory различать: завершённый snapshot копируется/публикуется с checksum manifest, continuation пишет в новый job working directory. Запрещено менять родительские restart bytes. Manifest перечисляет каждый native file relative path/SHA, version/lock hash, actual step/time, state/controls prefix hash. Restart из другой версии/изменённого prefix/corrupted file → INVALID_INPUT; это не numerical retry.
- [ ] **8.5. Ограничить память по времени.** Выполнять chunks по одному месяцу; после extraction monthly arrays/selected states освободить full substates/reports текущего chunk. На следующем chunk использовать native continuation. High-level API может перечитывать предыдущие reports/states: выставить нативные output flags/ограничение in-memory reports, проверить фактическое поведение через source и RSS. Если high-level wrapper загружает всю историю, использовать `setup_reservoir_simulator`/`simulate!` с `output_states=false` и `read_results(range=...)` только текущего chunk. Эта замена должна проходить тот же round-trip test. `extra_outputs=false` само по себе не доказательство bounded memory.
- [ ] **8.6. Написать и выполнить настоящий continuous-vs-restart test.** 6 months, событие mask на month 3, role switch на month 4. A: continuous native run; B: завершить после month 3, остановить процесс, открыть **новый** worker и native restart, завершить suffix. Сравнить каждую общую дату So/Sw/p, monthly Vo/Vw/injection, component inventory и connection summaries. Assert initial snapshot не продублирован и prefix integrals не посчитаны дважды. Допуски — Task 9. Кроме штатного restart проверить chunked/default output path против continuous.
- [ ] **8.7. Проверить jobs A→B→A.** A/B имеют разные K, fluids, initial states и masks; оба A совпадают с A в свежем worker по научным допускам. Same worker pid у последовательных jobs; отсутствие роста retained memory после серии 5 одинаковых warm runs. Drift >max(256 MiB, 20% post-warm baseline) после GC даёт warning + worker recycle перед следующим job; повторяемый рост → RESOURCE_FAILURE, не бесконечный recycle. Threshold — диагностическое решение плана, memory peak guards действуют всегда.
- [ ] **8.8. Проверить graceful stop/resume.** После completed month нажать stop flag, получить INCOMPLETE_BUDGET + valid restart и ledger completed jobs; новая команда возобновляет missing suffix и не меняет ready outputs. Force kill mid-chunk оставляет последний завершённый checkpoint, незавершённый staging не принимает.
- [ ] **8.9. Run GREEN:** policy + real restart/A→B→A + E00 smoke. Commit checkpoint: `feat: resume native states and classify failed forward jobs`.

### Task 9: аналитическая OW verification и допуски до benchmark

**Files:** Create `configs/e01_tolerances.yml`, `julia/verification/analytic.jl`, `src/so_recon/validation/physics.py`, `tests/unit/test_physics_metrics.py`; Extend fixture and real physics tests.

**Interfaces:** `load_tolerances(Path) -> dict[str,float]` с exact required keys; `evaluate_physics(case_name: str, outputs: dict[str,Path], tolerances: dict[str,float]) -> PhysicsCheck`; `PhysicsCheck` = typed `name, status, metrics, thresholds, input_hashes, evidence_paths, reason`. Julia `run_analytic(name::Symbol, options::AbstractDict)` использует тот же `build_ow/build_forces/run_forward` и не содержит второго numerical solver. Python `bl_saturation(x: ndarray, t_pvi: float) -> ndarray` — аналитический reference в оговорённом пределе.

- [ ] **9.1. Зафиксировать tolerance config ДО первого scoring run.** Все числа ниже — начальные критерии плана; изменение после диагностического провала требует нового config hash, объяснения и повторения affected matrix, а не маскировки отдельного failure.

```yaml
schema_version: e01-tolerances-1
balance_cumulative_relative_max: 1.0e-3
balance_step_median_relative_max: 1.0e-5
balance_cumulative_target: 1.0e-5
balance_absolute_floor_m3_sc: 1.0e-6
saturation_sum_abs_max: 1.0e-10
saturation_bound_slack: 1.0e-8
closed_connection_mass_kg_s_max: 1.0e-10
closed_state_saturation_drift_max: 1.0e-8
closed_pressure_relative_drift_max: 1.0e-7
hydrostatic_gradient_relative_max: 1.0e-5
hydrostatic_saturation_drift_max: 1.0e-6
hydrostatic_pressure_relative_drift_max: 1.0e-6
restart_saturation_abs_max: 1.0e-6
restart_pressure_relative_max: 1.0e-6
restart_volume_relative_max: 1.0e-5
restart_inventory_relative_max: 1.0e-5
rate_control_relative_max: 1.0e-4
bl_pv_l1_max_at_128: 0.08
bl_refinement_ratio_max: 1.1
five_spot_symmetry_abs_max: 1.0e-4
refinement_so_pv_mae_max: 0.02
refinement_inventory_relative_max: 0.01
refinement_monthly_volume_relative_max: 0.02
```

Rate/volume denominator `max(abs(reference),1e-6)` в соответствующих SI/surface units; pressure denominator `max(abs(reference),1 Pa)`. Нулевой расход проверяется абсолютным допуском, не относительной ошибкой от нуля. Баланс threshold ≤1e-3 — минимальный gate; достижение цели ≤1e-5 сообщается отдельной метрикой.
- [ ] **9.2. Написать unit tests самого evaluator.** Не только pass fixture: подменить один inventory/connection/So value так, чтобы evaluator обязан был fail. BL reference проверяется при t=0, за shock и на injection boundary.

```python
import numpy as np
from so_recon.validation.physics import bl_saturation

def test_bl_front_has_known_shock_speed():
    # Corey n=2, equal viscosity, Swc=Sor=0, incompressible, horizontal.
    s = bl_saturation(np.array([0., 0.1, 0.9]), t_pvi=0.2)
    assert s[0] == 1.0
    assert s[1] > 1/np.sqrt(2)
    assert s[2] == 0.0
```

- [ ] **9.3. Run RED:** `uv run pytest tests/unit/test_physics_metrics.py -q`.
- [ ] **9.4. Реализовать BL analytic reference.**

```python
import numpy as np

def bl_saturation(x, t_pvi):
    x = np.asarray(x, dtype=np.float64)
    if t_pvi < 0 or (x < 0).any():
        raise ValueError('negative BL coordinate/time')
    if t_pvi == 0:
        return np.where(x == 0, 1.0, 0.0)
    shock_s = 1/np.sqrt(2)
    grid = np.linspace(shock_s, 1.0, 20001)
    denominator = grid**2+(1-grid)**2
    derivative = 2*grid*(1-grid)/denominator**2
    xi = x/t_pvi
    behind = np.interp(xi, derivative[::-1], grid[::-1])
    return np.where(xi <= derivative[0], behind, 0.0)
```

Reference усреднять quadrature по объёму каждой ячейки (64 равномерных подпункта); увеличить до 128 и показать negligible quadrature error относительно numerical tolerance. Не сравнивать квазиточечный reference с ячеечным average без учёта shock location.
- [ ] **9.5. Реальный closed/PVT fixture.** Провести один cell и закрытую 8-cell систему без sources, по 3 report steps; проверить component inventory, pressure и saturation drift. Density check из Task 5 независим от static conservation. Намеренно нарушенная PVT positivity должна завершиться PHYSICALLY_INVALID до solver, не численной несходимостью. Black-oil phase appearance остаётся Task 13.
- [ ] **9.6. Реальный BL fixture.** 1D `(32,1,1)`, `(64,1,1)`, `(128,1,1)`, одинаковые L=100 m, A=1 m², phi=0.2, k=100 mD. Суммарная закачка 0.2 PV за 1 day, downstream pressure/boundary, horizontal centers одной глубины, Pc=0, c=0, equal mu, `Swc=Sor=0`, n=2, initial Sw=0. Report steps 4/8, не >12; adaptive substeps ограничены исходным и половинным dt. Проверить breakthrough shape, PV L1≤0.08 на 128, ошибка 64→128 не возрастает >10%, mass balance. Отдельный analytic-limit config не переносится в P1 fluids.
- [ ] **9.7. Реальная гидростатика.** Вертикальная колонка `(1,1,2)`, высота 100 m, closed, water-only immiscible OW limit Sw=1. Инициализировать штатным equilibrium либо решением hydrostatic pressure для той же rho(p); это тест физики OW в однофазном пределе. Положительное dp/dz согласовано с z-down. До timestep проверить hydrostatic discrete face residual, после — отсутствие искусственного дрейфа. Отдельный gravity segregation case с тяжёлой водой сверху должен уменьшать её центр высоты/увеличивать среднюю глубину; он не обязан оставаться неподвижным.
- [ ] **9.8. Run GREEN:** evaluator unit, native analytic fixtures через один worker, отчёт всех metrics включая failed/unrun. Commit checkpoint: `test: verify oil-water balance and analytic limits`.

### Task 10: wells/crossflow/boundaries и grid/time convergence

**Files:** Create `julia/verification/operations.jl`, `refinement.jl`; Extend `fixtures.jl`, Python physics evaluator и `tests/integration/test_e01_physics.py`.

**Interfaces:** `run_operational(name::Symbol, options)` — fixture через тот же adapter. Python `aggregate_so(so: ndarray, pv: ndarray, zone_id: ndarray, n_zones: int) -> ndarray` с NaN при PV=0; `compare_refinement(coarse_result, fine_result, common_support) -> PhysicsCheck`. `common_support` содержит fixed physical zones и intersection PV; здесь вложенные Cartesian cells дают точное соответствие без GIS.

- [ ] **10.1. Написать независимый support test.**

```python
import numpy as np
from so_recon.validation.physics import aggregate_so

def test_support_average_uses_pore_volume():
    mean = aggregate_so(np.array([0.2, 0.8]), np.array([1., 3.]),
                        np.array([0, 0]), 1)
    np.testing.assert_allclose(mean, [0.65])
```

- [ ] **10.2. Run RED:** `uv run pytest tests/unit/test_physics_metrics.py -q`; новый test падает на отсутствующем aggregator.
- [ ] **10.3. Реализовать support aggregation.**

```python
def aggregate_so(so, pv, zone_id, n_zones):
    num = np.bincount(zone_id, weights=pv*so, minlength=n_zones)
    den = np.bincount(zone_id, weights=pv, minlength=n_zones)
    return np.divide(num, den, out=np.full(n_zones, np.nan), where=den > 0)
```

Входная shape/finite/index validation выполняется до вызова; missing collector не изображается So=0. Inventory агрегировать отдельно, а не как `mean So * mean B`.
- [ ] **10.4. Two-layer mixing + crossflow.** `(4,4,2)`, слои 5+5 m, k_h 200/50 mD, phi 0.25/0.15, один producer с двумя connections; инжекторы отдельно по слоям, всего ≤3 wells. Разные начальные Sw=0.25/0.65, заданная liquid rate. Проверить фазовый surface состав как результат native mixing и component balance. Второй case: противоположные pressure potentials по слоям и zero net surface с открытым стволом; обязательны ненулевые противоположно направленные connection flux, conservation с well storage. Третий case: та же поверхность, обе connections закрыты → нулевые потоки по каждой. Не объявлять shutdown и isolation одним тестом.
- [ ] **10.5. Открытие/изоляция и смена роли.** Один immutable case содержит intervals: producer→shut→injector; mask `[1,1]→[1,0]→[1,1]`. Событие посреди календарного месяца. Verify per-interval requested/actual role, закрытый connection flux ≤1e-10 kg/s, mass balance, раздельные positive production/injection month volumes. Дополнительно BHP-feasible и намеренно BHP-infeasible tests: предел достигнут, фактический q отличается, status CONTROL_INFEASIBLE с evidence; default target не подменён молча.
- [ ] **10.6. Граница/аквифер.** Сначала 2-cell closed benchmark, затем тот же сектор с `pressure_water` boundary на outer cell; записать граничный component influx в balance. Минимальная поддержка аквифера — явно бесконечный pressure-water boundary, а не полевой calibrated aquifer. Дополнительно finite storage case: присоединённая водонасыщенная buffer-cell с известным PV входит в computational inventory, геометрия по-прежнему ≤128 cells. Проверить падение pressure support finite buffer относительно fixed-pressure boundary, exchange sign и сохранение sector+buffer inventory; pressure-water model не объявлять finite aquifer.

```julia
bc = [FlowBoundaryCondition(cell_id+1, pressure_pa;
    fractional_flow = [1.0, 0.0], density = rho_water_sc,
    trans_flow = trans_flow)]
forces = setup_reservoir_forces(model;
    control = controls, limits = limits, set_default_limits = false, bc = bc)
```

- [ ] **10.7. Five-spot в P1 resource profile.** Homogeneous square `(16,16,1)=256`, extent `(400,400,10) m`, phi=0.2, k=100 mD, 36 calendar months, четыре symmetric injectors по 10 m³_sc/day и центральный producer 40 m³_sc/day. Начальное Sw=0.2, p=15 MPa; BHP limits 5/30 MPa. Injectors размещены симметрично в ячейках `(2,2),(2,13),(13,2),(13,13)` с zero-based indices. Central producer — native SimpleWell с четырьмя равноправными connections в `(7,7),(7,8),(8,7),(8,8)` на одной глубине: так pressure control/support имеет точную симметрию even grid. Не объявлять одну из четырёх central cells точным центром. Gravity присутствует, горизонтальные face differences нулевые; Pc=0. Проверить symmetry saturation относительно двух осей ≤1e-4 и balance. Cases с 5 wells не запускать под P0; SimpleWell тут не заменяет MultiSegmentWell crossflow tests.
- [ ] **10.8. Сгущение времени/сетки.** BL 64/128 с одинаковым continuous rock/controls и dt/dt2; five-spot 16²→32² с теми же физическими locations, PV-conservative cell mapping. Сохранять тот же непрерывный k/phi; не генерировать новую случайную геологию на fine сетке. Сравнивать So на fixed 8×8 support, inventory и каждый monthly phase integral по заранее записанным допускам. Если well WI меняется при refinement, native WI пересчитывается из той же radius/geometry, а не подгоняется к coarse результату. Показать как discretization sensitivity, без заявления fine posterior.
- [ ] **10.9. Run GREEN:** operational negative/positive cases, masks/roles/boundary, BL/five-spot refinement через registered sessions; никакого skip mandatory physics. Commit checkpoint: `test: verify well operations crossflow and refinement`.

### Task 11: детерминированный двухслойный P1 generator с отдельной truth

**Files:** Create `src/so_recon/synthetic/p1.py`, `world_io.py`, `tests/unit/test_p1_worlds.py`, `tests/integration/test_e01_p1.py`.

**Interfaces:** `P1Design` = StrictModel `shape=(16,16,2)`, `extent_m=(400,400,20)`, `n_months=36`, `start_date='2000-01-01'`, `design_id='p1-two-layer-v1'`, `family: Literal['base','low_vertical','high_contrast']`; `render_p1(seed: int, design: P1Design) -> RenderedWorld`. `RenderedWorld` содержит `parent_world_id`, `theta: dict[str,object]`, `arrays: dict[str,ndarray]`, `case_fields: dict[str,object]`, `static_observations: pa.Table`, `design`; `write_world(world, forward, paths, ctx) -> ArtifactRef`. `WorldManifest` содержит schema/spec/generator version, parent/design/view/split/seed/model IDs, case/obs/truth refs, provenance и forward status.

- [ ] **11.1. Написать tests до renderer.** Тот же seed/design даёт одинаковые scientific arrays/model hash; другой seed меняет geology; 2 слоя неоднородны; phi и k не генерируются независимо; controls не зависят от future/hidden So; input context file не содержит full-field K/So/pressure/latents/truth paths.

```python
import numpy as np
from so_recon.synthetic.p1 import P1Design, render_p1

def test_p1_reproducibility_and_heterogeneity():
    design = P1Design()
    a, b = render_p1(41, design), render_p1(41, design)
    np.testing.assert_array_equal(a.arrays['permeability_m2'], b.arrays['permeability_m2'])
    assert a.arrays['sw'].size == 512
    assert np.std(a.arrays['permeability_m2'][0]) > 0
    assert a.parent_world_id == b.parent_world_id
    assert a.design.n_months == 36
```

- [ ] **11.2. Run RED:** `uv run pytest tests/unit/test_p1_worlds.py -q`.
- [ ] **11.3. Реализовать низкоразмерную совместную geology.** 12 standard-normal coefficients и известный versioned cosine basis, по 6 мод на слой; NumPy PCG64 с сохранением seed, coefficients и generator version. Для E01 это fully specified joint forward generator; не заявлять наличие conditional latent density p(theta|G) из E02. Ядро для каждого layer:

```python
import numpy as np

def layer_fields(coefficients, nx, ny, layer, family):
    x = (np.arange(nx)+0.5)/nx
    y = (np.arange(ny)+0.5)/ny
    xx, yy = np.meshgrid(x, y, indexing='ij')
    modes = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2)]
    field = sum(
        a*np.cos(np.pi*i*xx)*np.cos(np.pi*j*yy)/(1+i+j)
        for a, (i, j) in zip(coefficients, modes, strict=True)
    )
    contrast = 1.0 if family != 'high_contrast' else 1.8
    log_k_md = np.log(120.0 if layer == 0 else 40.0)+contrast*field
    k_md = np.exp(log_k_md)
    phi = 0.12+0.18/(1+np.exp(-0.7*field))
    return k_md, phi
```

Flatten order `F` соответствует `i + nx*j`; слои concatenate в k order. `Kx=Ky=k`, `Kz/Kx=0.05` base, `0.001` low_vertical. Верхний/нижний contrasts заданы в design и записаны в theta; валидировать numerical overflow/nonfinite, не обрезать экстремальные tails тайно. Здесь 12 мод — определённая размерность генератора, не обнуление остаточных мод уже существующего high-dimensional prior. E02 обязан согласовать conditional density относительно sparse G перед inverse использованием.
- [ ] **11.4. Задать known initial state и четыре wells.** Две injection wells `(i,j)=(2,2),(2,13)` и две production `(13,2),(13,13)` в 0-based grid; открыты оба слоя, radius 0.1 m. Start Sw=Swc=0.2, oil-connected hydrostatic initial pressure datum `1.5e7 Pa` at z=0. Native equilibrium проверяется коротким closed preflight; initial fields сохраняются и считаются synthetic initial, не поздними полевыми Soi. Учебные oil/water/Pc/Corey из §3.1; no virgin reset между monthly chunks.
- [ ] **11.5. Задать независимую controls policy.** Каждый injector 20 m³_sc/day, каждый producer total liquid 20 m³_sc/day; injection modulation множитель 0.75 на months 13–24 и 1.25 на months 25–36; producers получают тот же множитель. Это заранее заданная policy по времени, не по hidden So. Limits producer min 5 MPa, injector max 30 MPa. На начале month 19 закрыть нижний connection producer P2, открыть на month 25; событие не меняет автоматически исторический q. CONTROL_INFEASIBLE сохраняется как outcome, seed не подменяется «удачным».
- [ ] **11.6. Сформировать sparse static observations и masks.** Разрешённый G: phi/log(k) на 8 well-layer supports (4 wells ×2 layers), noise independent stream (`SeedSequence.spawn` geology/static/control streams), sigma_phi=0.01, sigma_logk=0.2; initial So_log на этих supports, sigma=0.02, дата старта. Out-of-range noisy measurement сохранять с quality flag, не clipping; latent true saturation ограничена физикой. Наблюдение — support average, не well flux-weighted saturation. `pressure_available=false`, dynamic channels будущего inverse context = fw и заданные controls; истинное pressure сохраняется только для diagnostics. Ещё нет dynamic noise likelihood — noiseless forward outputs готовы для E02.
- [ ] **11.7. Разделить физический case и вход inverse.** CaseBundle для solver обязан содержать полные K/phi/initial; такой bundle **не** передаётся encoder. World хранит отдельные `context.json` (design, controls, sparse observations, masks, allowed dates) и `truth/` (theta, full geology, So/Sw/p, true noiseless outputs). Явный allowlist context writer:

```python
def context_payload(world_manifest):
    allowed = ('parent_world_id', 'design_id', 'derived_view_id', 'split',
               'observation_ref', 'control_ref', 'observation_masks', 'cutoff')
    return {key: world_manifest[key] for key in allowed}
```

`derived_view_id='full36-v1'`, `split='exploratory'`, parent ID от generator/design/seed, prefixes/noise variants inherit parent. Test truth-changing perturbation не меняет context кроме непосредственно разрешённых observations; прямых ссылок на truth нет. Production data loader E02 получает только context path, evaluator — отдельный truth ref.
- [ ] **11.8. Сохранить initial + selected true states.** Dates start/month12/month24/month36, полная monthly phase history и completion summaries, native final restart. Поддержать одинаковый seed/design repeat, checksum reuse только при идентичных arrays/config; run-specific machine/time stamps отдельно. Создать ровно 5 первых parents `41,42,43,44,45`: base 41–43, low_vertical 44, high_contrast 45. Все outcome rows, включая failures, входят в manifest. Никакого автоматического расширения до 128 worlds.
- [ ] **11.9. Run GREEN:** unit generator/leakage/immutability; real forward первых parents с BHP/balance/state finite checks. Даже при неудаче сложной family сохранить diagnostic result и отметить, что first P1 suite не принят полностью; не удалять failed row. Commit checkpoint: `feat: generate reproducible two-layer synthetic truth worlds`.

### Task 12: CLI, измеренный benchmark и фактическая приёмка E01

**Files:** Create `src/so_recon/simulator/commands.py`, `validation/e01_report.py`, `validation/plots.py`, `configs/e01_jobs.json`, `scripts/e01_gate.sh`, `tests/unit/test_e01_cli.py`, `reports/stages/E01.md`; Modify `src/so_recon/cli.py`, `pyproject.toml`, `uv.lock`, `Makefile`, `README.md`, `julia/README.md`.

**Interfaces:** `run_e01_suite(cfg, paths, suite: str, *, resume_ledger: Path | None) -> RunContext`; `build_e01_report(run_dirs: tuple[Path,...], paths: ProjectPaths) -> StageReport`; `render_e01_report(report: StageReport) -> str`. `StageReport` содержит spec/stages/config/lock versions, commit/dirty, source/code hashes, required/complete/failed/unrun checks, commands+exit status, OW gate/BO status, costs, artifact refs и acceptance notes. Existing `execute_run` — единственная точка создания run/exception→FAIL.

- [ ] **12.1. Написать CLI failure tests.** Wrong case hash, missing resources, unsupported profile, missing Julia, unknown suite, wall stop, reused result с неверным SHA. Ни одна команда не возвращает exit 0 только потому, что записала checkpoint/report. Wrapper tests используют fake runner только для mapping; реальная matrix остаётся integration evidence.

```python
from so_recon.simulator.commands import forward_exit_code

def test_incomplete_result_is_nonzero_exit():
    assert forward_exit_code('COMPLETE') == 0
    assert forward_exit_code('INCOMPLETE_BUDGET') == 2
    assert forward_exit_code('RESOURCE_FAILURE') == 2
```

- [ ] **12.2. Run RED:** `uv run pytest tests/unit/test_e01_cli.py -q`.
- [ ] **12.3. Добавить команды, не меняя глобальный argparse порядок.** `_parser` already has global `--root/--config`, затем subcommand. Новые names: `forward`, `forward-resume`, `verify-physics`, `synthetic-p1`, `benchmark-forward`, `e01-report`. Bodies вынести в `simulator/commands.py`; `cli.py` только parsing/dispatch. Регистрация результата/FAIL через existing runner.

```python
def forward_exit_code(status: str) -> int:
    return 0 if status == 'COMPLETE' else 2

# Insert in existing _parser after creating sub:
verify = sub.add_parser('verify-physics', help='run registered native physics checks')
verify.add_argument('--suite', choices=['p0', 'p1', 'bo'], required=True)
verify.add_argument('--resume-ledger', type=Path)
```

Для остальных commands: `forward --case PATH`; `forward-resume --case PATH --restart PATH`; `synthetic-p1 --seeds 41 42 43 44 45`; `benchmark-forward --case PATH --warm-runs 5`; `e01-report --runs PATH [PATH ...]`. Здесь PATH — аргумент пользователя/фактический путь output, не имя будущего артефакта. Все relative CLI paths после resolve проходят ProjectPaths containment. `verify-physics --suite p1` включает весь зарегистрированный P1 список (physics, первые worlds и benchmark), не требует отдельного автоматического повторного `synthetic-p1`/`benchmark-forward`.
- [ ] **12.4. Зафиксировать exact planned jobs.** `configs/e01_jobs.json` имеет schema version, ordered IDs, case kind, profile, expected outcome, output request, thread setting, cache bypass и solver tolerance hash. Начальный список ниже не включает development отладки; их реальные расходы также записываются в ledger. Каждый полный trajectory attempt считается новым forward evaluation; chunks того же trajectory отдельно считаются как `native_chunk_calls`, не 36 независимых новых worlds. Resume attempt считается консервативно новым evaluation с parent ID. Numeric retry добавляет evaluation, даже если закончился неудачей.

| Session | Planned job IDs | Количество без retry |
|---|---|---:|
| P0, 10 min | `closed_cell`, `closed_box`, `hydrostatic`, `gravity_segregation`, `bl32`, `bl64`, `bl128`, `bl64_halfdt`, `mixing`, `crossflow`, `isolated`, `events_roles`, `bhp_feasible`, `bhp_infeasible`, `boundary_pressure`, `aquifer_finite`, `restart_continuous`, `restart_prefix`, `restart_suffix_new_worker`, `isolation_a1`, `isolation_b`, `isolation_a2`, `isolation_a_fresh` | 23 |
| P1, 1 h | `five_spot_coarse`, `five_spot_fine`, `five_spot_halfdt`; `world41`…`world45`; `warm41_1`…`warm41_5`; `threads1_1`, `threads1_2`, `threads2_1`, `threads2_2` | 17 |
| BO capability, отдельная P0 session | `bo_closed`, `bo_depletion`, `bo_restart_prefix`, `bo_restart_suffix_new_worker` | 4 |

P0 maximum с одной retry на каждом job =46 (<64); P1 =34 (<2000); BO =8. Invalid-schema/transport tests не являются физическими forward, но их время/память учитываются тестовым launcher отдельно. Standalone адресная диагностика не оправдывает обход memory/watchdog.

`world41` — cold model specialization + first full P1, следующие пять warm41 runs **обходят result cache** и используют один pid; model hash сохраняется. Сравнение thread1/2 выполняется в собственных последовательно запущенных workers, warm-up/cold costs записываются отдельно; не более одного активного worker. Настройка threads4 остаётся default до измеренного основания смены. Не сравнивать первый cold run одного режима с warm другого.
- [ ] **12.5. Реализовать benchmark tables.** Таблица per-attempt: start/end, cold import/compile, first OW specialization, wall/cpu, native chunks, accepted/cut steps, nonlinear iterations, peak tree RSS, total/available RAM, pressure/swap, output bytes, restart read/write seconds, status/retry. p50/p90 warm по 5 runs явно exploratory; failed attempts показаны отдельно с потраченным временем, не удалены из denominator failure rate. Если coefficient of variation>0.25 либо выражен upward memory drift — отчёт рекомендует 10–20 повторов, но не запускает их без отдельного принятого budget.

```python
import numpy as np

def warm_summary(seconds: list[float]) -> dict[str, float | int | str]:
    if len(seconds) < 5:
        raise ValueError('at least five completed warm timings required')
    x = np.asarray(seconds, dtype=np.float64)
    return {'n': len(x), 'p50_s': float(np.quantile(x, 0.5)),
            'p90_s': float(np.quantile(x, 0.9)),
            'tail_status': 'exploratory'}
```

Counts для **следующего** E02 эксперимента ещё не утверждены этим планом. E01 BudgetReport содержит измеренные unit costs, точные 23/17/4 jobs текущей matrix, расходы restart/retry/output и доступный headroom. Не выдавать forward-only estimate за полный ML/SMC budget; поля обучения/SMC/adjoint записать `not_applicable_in_E01`, а next-stage total `not_estimated_without_E02_job_plan`.
- [ ] **12.6. Реализовать resume orchestration.** Read ledger/hash lock/config → проверка completed outputs → missing jobs в прежнем порядке. New session budget не означает разрешения менять physics или skip failed cases. Benchmark cache-bypass уникален на attempt; production повтор не выполняет solver при валидном immutable matching result, кроме явно requested replay. Кэш key включает case arrays/geometry/control/fluids/initial/boundary/solver version и численные настройки/output request. E01 не кэширует likelihood.
- [ ] **12.7. Создать scientific plots из файлов.** Добавить Matplotlib через uv. 2-layer true So на start/12/24/36, monthly oil/water/injection, component residuals и cold/warm/RAM; подпиcи «synthetic truth; exploratory; educational OW». Plot читается из опубликованных HDF5/Parquet, численные экстремумы/оси проверяются, PNG визуально просматривается. Нет posterior/uncertainty графика там, где posterior ещё не существует. Код figures сохраняется; AI image generation здесь не применяется.
- [ ] **12.8. Сохранить compatibility verification без автоматического rebaseline.** Новый `e01_gate.sh` начинает `uv sync --frozen`, regression `pytest tests/unit tests/test_no_absolute_paths.py`, Ruff/mypy, existing `make smoke` + existing Julia smoke integration. Перед/после фиксирует SHA legacy smoke expectations, config, source manifest. Новый environment report и lock hashes сохранять в run-scoped E01 artifacts; deterministic E00 environment report не перезаписывать без отдельной объяснённой миграции. Исторический `scripts/gate.sh` удаляет `.venv` и сравнивает lockfiles с HEAD: не использовать его как незаметную команду повторного пересоздания окружения в этом плане. После авторизованного commit dependency changes можно отдельно проверить полный foundation gate в согласованном окружении.
- [ ] **12.9. Отделить адресные native tests от массового запуска.** Новые physics integration tests получают marker `e01_physics`; pytest default не должен молча запускать 17 P1 trajectories при каждом unit cycle. Register marker и CLI/option `--run-e01-physics` в tests/conftest.py; без опции они явно SKIP, а `scripts/e01_gate.sh` выполняет registered suites и затем integration validators с опцией на их сохранённых artifacts. Validator читает actual native results, не запускает suite второй раз. Stage gate требует присутствия всех mandatory checks и нуля skipped/unrun в его собственной matrix; обычный pytest PASS с skips недостаточен.
- [ ] **12.10. Выполнить итоговые команды с заранее выбранными бюджетами.**

```bash
uv run pytest tests/unit tests/test_no_absolute_paths.py -q
uv run ruff check .
uv run ruff format --check .
uv run mypy
make smoke
uv run pytest tests/integration/test_julia_smoke.py -m julia -q
uv run so-recon --config configs/e01.yml verify-physics --suite p0
uv run so-recon --config configs/e01.yml verify-physics --suite p1
```

При превышении 10 min/1 h — partial отчёт с конкретными remaining jobs, exit 2; не увеличивать бюджет автоматически. Пройденные jobs остаются. `--resume-ledger` принимает путь ledger, напечатанный предыдущей командой; будущий исполнитель использует **фактический** путь результата, не выдуманный run ID. E01 report command получает эти же реальные run directories. Whole stage не обещает завершение всех native tests за один P0 запуск после cold compile.
- [ ] **12.11. Записать итоговый `reports/stages/E01.md` только по evidence.** До выполнения статуса PASS в файле нет; начальный header `NOT_RUN`. Итог содержит: проверенный commit/dirty; input/tolerance/lock hashes; команды/exit codes; пути manifests и PNG; matrix metrics/thresholds; failed/retry rows; physical class/educational fluids; legacy regression; restart round-trip; cold/warm/RAM/disk; budget incomplete list; отдельный BO status; разрешённый следующий этап. `PASS_WITH_LIMITATIONS` допустим для OW с BO NOT_RUN, но не для проваленного mandatory OW/restart/balance. E02 разрешён только при принятом OW scope; поле/PhysicsDecision/ML не заявляются готовыми.
- [ ] **12.12. Run GREEN:** e01 CLI/report tests, native acceptance matrix и artifact checksum verification. Commit checkpoint: `feat: expose and report bounded E01 forward verification`.

### Task 13: отдельный малый black-oil capability перед PhysicsDecision

**Files:** Create `julia/adapter/blackoil.jl`, `julia/verification/blackoil.jl`, `tests/integration/test_e01_blackoil.py`; Extend `FluidSpec`, `ForwardResult`, `SOReconAdapter.jl`, stage report.

**Interfaces:** `build_blackoil(case, arrays) -> (model, parameters, state0)` выбран dispatcher-ом того же adapter по `fluids.kind='BO'`. Добавить `BlackOilFluidSpec` с `pvt_source`, `pvt_table_hashes`, `rv=0.0`, gas reference density, three-phase relperm definition. Case schema `case-2` читает case-1 через явный legacy decoder; OW manifests не переписываются. BO outputs дополнительно `Sg`, `Rs`, `Bg`, component gas inventory; native restart хранит фазовое состояние/BlackOilUnknown, а не только Saturations.

- [ ] **13.1. Написать capability assertions.** Sw+So+Sg=1, density/viscosity/B positive, газ появляется при depletion ниже bubble point, component gas mass conserved с учётом dissolved term `Rs*So/Bo`. Ошибка JSON/unknown physics class — INVALID_INPUT. Test BO не отмечать passed на одном успешном constructor.

```python
import numpy as np

def assert_bo_state(sw, so, sg, rs, bo, bg):
    np.testing.assert_allclose(sw+so+sg, 1.0, atol=1e-10)
    assert np.all(sg >= -1e-8)
    assert np.all(rs >= 0)
    assert np.all(bo > 0) and np.all(bg > 0)
```

- [ ] **13.2. Run RED:** native BO test через suite bo → unsupported BO/model decoder, не false PASS.
- [ ] **13.3. Использовать встроенную учебную SPE1 PVT таблицу.** Закреплённый source содержит `JutulDarcy.blackoil_bench_pvt(:spe1)` и `setup_reservoir_model_from_blackoil_tables`; никаких remote artifacts/GLMakie/full SPE9 model. Экспортировать возвращённые численные PVT tables в versioned fixture artifact, hash вместе с JutulDarcy tree/version; отметить academic benchmark, не Romashka PVT. Ядро constructor:

```julia
setup = JutulDarcy.blackoil_bench_pvt(:spe1)
pvt = setup[:pvt]
model = JutulDarcy.setup_reservoir_model_from_blackoil_tables(domain;
    pvtw = pvt[1], pvto = pvt[2], pvdg = pvt[3],
    reference_densities = setup[:rhoS], wells = wells, extra_outputs = false)
parameters = setup_parameters(model)
state0 = setup_reservoir_state(model;
    Pressure = 200e5, ImmiscibleSaturation = 0.1,
    BlackOilUnknown = BlackOilX(50.0, JutulDarcy.OilOnly, false))
```

Перед run проверить, что initial pressure/Rs совместимы с именно экспортированным saturation table. Учебная трёхфазная relperm — нативные независимые Corey curves, exponents `(2,2,2)`, residual saturations `(0.1,0.1,0.0)`, endpoints `(1,1,1)`, без hysteresis; это explicit capability approximation, не выбранная полевая Stone/LET модель. До setup_parameters заменить `RelativePermeabilities` через `BrooksCoreyRelativePermeabilities(3, [2.0,2.0,2.0], [0.1,0.1,0.0], [1.0,1.0,1.0])`. Зарегистрировать эту definition и hash вместе с PVT. Проверить `Swc+Sorw+Sgc<1` и endpoints positivity. BO использует aqueous/oil/gas order. В Task 6 изменить signature на `native_control(c::AbstractDict, rho_water_sc::Float64, n_phases::Int=2)` и определить `water_mix = vcat(1.0, zeros(n_phases-1))`; оба вызова InjectorControl используют water_mix вместо `[1.0,0.0]`. Regression доказывает прежние OW результаты при n_phases=2.
- [ ] **13.4. Выполнить четыре registered jobs.** `bo_closed`: 1 cell без источников; `bo_depletion`: `(4,1,1)`, producer BHP по ступеням от выше bubble point к ниже на 6 интервалах, 1 well; `bo_restart_prefix/suffix`: тот же depletion с новым worker после шага появления газа. Сравнить continuous/restart по So/Sw/Sg/p, dissolved/free gas inventory и surface Vo/Vw/Vg. Units/source scales для газа не заимствовать из OW как жидкость. P0 guards действуют, early BO compilation входит в отдельную session.
- [ ] **13.5. Классифицировать outcome отдельно.** BO capability PASS доказывает поддержку/баланс/phase transition/restart на учебном case. Он не выбирает физику Ромашки и не является парным OW/BO sensitivity study с matched oil inventory — это E06. Failure/NOT_RUN сохраняется отдельным gate; OW deliverables не удаляются и не объявляются failed только из-за отсутствующего BO benchmark. До PhysicsDecision эта зависимость обязательна.
- [ ] **13.6. Run GREEN:** BO schema/controls regression + реальные четыре jobs + OW restart regression при изменении phase dispatch. Commit checkpoint: `test: record educational black-oil capability and restart`.

## 5. Проверенные точки native API и источники

Для исполнения сначала читать локальные исходники **закреплённого** пакета, а не копировать latest docs. Путь найти переносимой командой; personal depot paths в публикуемый report не вставлять:

```bash
julia --project=julia --startup-file=no -e 'using Jutul, JutulDarcy; println(pkgdir(Jutul)); println(pkgdir(JutulDarcy))'
```

При составлении плана проверены:

| Пакет/относительный файл | Что подтверждено |
|---|---|
| JutulDarcy `src/utils.jl` | `setup_reservoir_model`, `setup_reservoir_forces`, `well_output`, `simulate_reservoir`, BO tables constructor |
| JutulDarcy `examples/introduction/wells_intro.jl` | `ConstantCompressibilityDensities`, replacement variables, native sign conventions |
| JutulDarcy `test/multimodel.jl` | `PerforationMask`, well force `mask`, нулевой маскированный поток |
| JutulDarcy `src/forces/bc.jl` | `FlowBoundaryCondition`, phase fractions и density |
| JutulDarcy `src/multiphase.jl` | `TwoPointGravityDifference`, z/depth-derived face gravity |
| JutulDarcy `src/variables/viscosity.jl` | Native `PhaseViscosities` parameter |
| JutulDarcy `src/test_utils/setup_multimodel.jl` | Учебная `blackoil_bench_pvt(:spe1)` и начальный `BlackOilX` |
| Jutul `src/utils.jl` | `expand_to_ministeps` возвращает states, dt, report-step mapping |
| Jutul `src/simulator/config.jl` | `output_substates`, `output_path`, `in_memory_reports`, `output_states` |
| Jutul `src/simulator/simulator.jl` | Native restart step k читает k−1; substates только успешных шагов |

Дополнительная внешняя проверка API: [официальная high-level документация JutulDarcy](https://sintefmath.github.io/JutulDarcy.jl/v0.2.43/man/highlevel) описывает `output_path/restart` и `ReservoirSimResult`. Это более старая версия документации, поэтому контракты конкретного 0.3.11 определены прочитанными локальными исходниками. Наличие метода в исходниках не заменяет real capability test; приведённые kernel snippets ещё не исполненная реализация.

## 6. Критерии передачи в E02 и self-review плана

| Требование SPEC/STAGES/DECISIONS | Задачи и доказательство |
|---|---|
| 3.0/4.0 config/run/source stamping, failure before cfg, immutable legacy | 1; source bytes/versions + existing smoke |
| CaseBundle/ForwardResult/minimal job schema, array units/layout/hashes | 2; corruption/transpose/invalid units tests |
| Resource profiles, hardware, pressure/swap, disk/wall/attempt limits | 3–4; watchdog tests + реальные измерения 12 |
| Persistent process, isolated jobs, logs, failed process cleanup | 4, 8; real A→B→A/new-worker parity |
| OW explicit PVT/Corey/Pc/gravity/initial conditions | 5, 9; native model/PVT/hydro/segregation |
| Liquid production control, water injection, sign/standard conditions | 6–7; native target + independent integrals |
| Events, roles, BHP mismatch, masks, crossflow, surface shutdown distinction | 6, 10; positive/negative physical cases |
| Monthly integration over accepted substeps and real calendar | 6–7; leap month/event/cut-step tests |
| Closed-cell/BL/hydrostatics/five-spot | 9–10; аналитика и symmetry, полный matrix |
| Boundary/aquifer and two-layer mixing | 10; external/finite-buffer inventories |
| Grid/time convergence on common support | 9–10; same realization, conserved PV, states/volumes/inventory |
| Minimal outputs and native restart, no full-history RAM assertion | 7–8; measured chunked-vs-continuous trajectory |
| Numerical retry vs input/physics/resource failure | 3–4, 8; named outcomes + cost ledger |
| First heterogeneous P1 worlds, sparse static G, hidden So, reproducibility | 11; full model for solver, allowlisted context for inverse |
| Cold/warm, RAM, disk, first benchmark, stage report | 12; 23 P0 +17 P1 job manifest and evidence |
| Early BO capability separate from field PhysicsDecision | 13; separate status, due before E06 |
| E00 does not get restarted; inverse/ML/field/long-run scope boundary | §§1, 3.3, Tasks 1/12/13 |

**Итоговый OW acceptance checklist для исполнителя:**

- [ ] Legacy smoke PASS с неизменными frozen expectations и согласованными 3.0 records.
- [ ] 4.0 runs и source/case/result lineage действительно 4.0; unknown cfg не stamp-ится вымышленной версией.
- [ ] Все mandatory OW physical checks прошли на реальном solver; нет скрытых skipped/unrun.
- [ ] Surface volumes получены из accepted substeps, match component balance, closure/role tests выполнены.
- [ ] Continuous/native restart/new-worker результаты совпадают в объявленных допусках.
- [ ] A→B→A не наследует model state; cold/warm/RAM/pressure/swap/disk измерены.
- [ ] Все 5 первых P1 parents зарегистрированы со status, seeds/design/renderer, theta/controls/noiseless outputs и requested true states.
- [ ] Resource/time limit и incomplete не превратились в forward success/scientific PASS.
- [ ] Native artifacts/Parquet/HDF5 hashes проверены; test failures и retry costs не удалены.
- [ ] Отчёт E01 содержит limitations и отдельный BO status; разрешение E02 ограничено принятым OW scope.

**Self-review выполнен при составлении:** требования E01 и строки SPEC §23 сопоставлены задачам таблицы; зависимости records/interfaces согласованы; BO не блокирует первый OW цикл; first P1 suite не запускает training corpus. Численные критерии являются проектными решениями до benchmark, не результатами уже выполненных измерений. Верификацию самой реализации выполняет будущий исполнитель; наличие этого плана не меняет статус E01 на PASS.
