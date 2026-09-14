"""Typed records for the E02 inverse problem (plan §2), defined exactly once.

Thirteen later tasks import these records. They are written so that the failures which
would otherwise be SILENT are refused at construction:

* A latent vector shorter than the schema it names is a truncation, not a shorter vector.
  SPEC §7.5 keeps every remaining mode, so `DensitySchema.validate_theta` compares the
  declared lengths instead of slicing.
* A mixture whose weights do not sum to one is a distribution nobody declared.
  `validate_probability_weights` checks the sum to `PROBABILITY_SUM_TOLERANCE` and never
  renormalises: positive weights that do not sum to one are an error at their source.
* `-inf` is a legal log-density OUTSIDE the support and nowhere else; NaN and `+inf` are
  refused everywhere. In JSON `-inf` is written as `null` beside a companion
  `<field>_in_support` flag, never as the `-Infinity` literal that is not valid JSON —
  and the flag is what tells «outside the support» from «not recorded». Every record that
  holds an extended-real log value carries its flag, so a checkpoint written while a
  particle sits outside the support reloads to the same numbers it was written from.
* `beta < 1` is an intermediate tempering distribution and never a posterior (SPEC
  §13.4.2), so `PosteriorBundle` refuses the combination rather than trusting a label.

The exceptions at the bottom are refusals with a reason attached. None of them is a
physical result: a renderer that could not evaluate, a likelihood that overflowed, an
observation outside its support and a missing E01 dependency are process failures, and
converting any of them into a zero likelihood with a catch-all would turn a broken run into
a confident one. They deliberately have no public common base class.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Annotated, Any, Literal

import numpy as np
import numpy.typing as npt
from pydantic import (
    ConfigDict,
    Field,
    ValidationInfo,
    field_serializer,
    field_validator,
    model_validator,
)
from scipy.special import ndtr

from so_recon.config.schema import StrictModel
from so_recon.registry.artifact import ArtifactRef
from so_recon.simulator.contracts import Sha256

#: The two array types of the stage. CPU Float64 throughout (COMPUTE §5).
F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]

#: Probability weights are checked, never repaired. Plan §2 fixes the tolerance.
PROBABILITY_SUM_TOLERANCE = 1e-12

#: The latent measure E02 works in: counting measure on the family index times Lebesgue
#: measure on the latent coordinates. The renderer's Jacobian is NOT folded in here.
#: Annotated as `Literal` so the record default below stays the literal type it declares.
LATENT_MEASURE: Literal["counting_x_latent_lebesgue"] = "counting_x_latent_lebesgue"

#: Order of `v` for the P1 schema: eight whitened geology coordinates, then the three
#: nuisance coordinates of the noise law. A reduced or toy schema declares its own order
#: and does not use this slicing.
N_P1_GEOLOGY_IN_V = 8
NOISE_SIGMA_INDEX = 8
NOISE_RHO_INDEX = 9
LOG_BIAS_INDEX = 10

#: The declared noise law (plan §2). `sigma` and `log_bias` are in g-space.
NU_FIXED = 5.0
SIGMA_MIN = 0.015
SIGMA_SPAN = 0.065
RHO_MAX = 0.8
LOG_BIAS_SCALE = 0.03

#: What an observation is allowed to be used FOR. Static information that conditioned the
#: prior is not multiplied in a second time as an independent likelihood (SPEC §5), so the
#: role travels with the row rather than with the call site.
ObservationRole = Literal["condition_prior", "heldout_diagnostic", "likelihood"]
OBSERVATION_ROLES: tuple[ObservationRole, ...] = (
    "condition_prior",
    "heldout_diagnostic",
    "likelihood",
)

SmcPhase = Literal["initialize", "reweight", "resample", "rejuvenate", "finalize"]

#: SPEC §17.4.1 separates process status, algorithm status and scientific outcome. These
#: are the ALGORITHM statuses of SPEC §17.4.1 plus E02's own `EVALUATION_FAILURE`, which a
#: runtime or likelihood exception produces and which is never a science verdict.
AlgorithmStatus = Literal[
    "COMPLETE",
    "INCOMPLETE_BUDGET",
    "POSTERIOR_NOT_CONVERGED",
    "NO_TARGET_SUPPORT",
    "EVALUATION_FAILURE",
]

#: The scientific half, kept apart from the algorithm status on purpose: a checkpoint that
#: was written successfully does not make an unfinished algorithm a converged posterior.
ConvergenceStatus = Literal["NOT_ASSESSED", "CONVERGED", "NOT_CONVERGED"]

#: A saturation, a water cut and a mixture weight all live on the unit interval.
UnitInterval = Annotated[float, Field(ge=0.0, le=1.0)]


# --------------------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------------------


class _Reasoned(Exception):
    """Keeps the refusal's reason addressable instead of only printable.

    Private on purpose. A shared PUBLIC base would invite `except InferenceError: return
    -inf`, and plan §2 forbids exactly that: none of these becomes a physical zero
    likelihood through a catch-all.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class E01DependencyError(_Reasoned, RuntimeError):
    """The accepted E01 oil-water scope could not be proved from published evidence."""


class RendererNumericalError(_Reasoned, RuntimeError):
    """The latent-to-physical renderer could not produce a usable field."""


class LikelihoodNumericalError(_Reasoned, RuntimeError):
    """A likelihood term could not be evaluated in log space."""


class ObservationSupportMismatch(_Reasoned, ValueError):
    """An observation's support does not match the grid or schedule it was scored on."""


class UnsupportedObservationError(_Reasoned, ValueError):
    """An observation kind this stage does not model was offered as evidence."""


# --------------------------------------------------------------------------------------
# shared numeric guards
# --------------------------------------------------------------------------------------


def _finite(values: Sequence[float], *, label: str) -> tuple[float, ...]:
    out = tuple(float(x) for x in values)
    for index, value in enumerate(out):
        if not math.isfinite(value):
            raise ValueError(f"{label}[{index}] must be finite, got {value}")
    return out


def _log_value(value: float, *, label: str) -> float:
    """A log-density or log-weight: `-inf` outside the support, never NaN and never +inf."""
    if math.isnan(value) or value == math.inf:
        raise ValueError(
            f"{label} must be finite or -inf (which means outside the support); "
            f"NaN and +inf are not log-densities, got {value}"
        )
    return value


#: Suffix of the companion flag every extended-real log field carries. The pair is the
#: whole JSON contract of plan §2: `{"log_l": null, "log_l_in_support": false}` reloads to
#: `-inf`, and a field that was never recorded is simply absent instead of ambiguous.
IN_SUPPORT_SUFFIX = "_in_support"


def _in_support(value: object) -> bool:
    """False only for exactly `-inf`.

    A value that is not a number at all is reported as in support, so that the field's own
    type check is what the caller is told about rather than a confusing missing flag.
    """
    try:
        return float(value) != -math.inf  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return True


def _normalise_log_scalar(data: dict[str, Any], field: str) -> None:
    flag = f"{field}{IN_SUPPORT_SUFFIX}"
    if field not in data:
        return
    if data[field] is None:
        data[field] = -math.inf
    derived = _in_support(data[field])
    declared = data.get(flag)
    if declared is None:
        data[flag] = derived
    elif bool(declared) != derived:
        raise ValueError(
            f"{flag}={bool(declared)} contradicts {field}={data[field]}: the flag says "
            "whether the value is inside the support and is never set independently of it"
        )


def _normalise_log_array(data: dict[str, Any], field: str) -> None:
    flag = f"{field}{IN_SUPPORT_SUFFIX}"
    if field not in data or not isinstance(data[field], list | tuple):
        return
    values = tuple(-math.inf if item is None else item for item in data[field])
    data[field] = values
    derived = tuple(_in_support(item) for item in values)
    declared = data.get(flag)
    if declared is None:
        data[flag] = derived
    elif (
        not isinstance(declared, list | tuple) or tuple(bool(item) for item in declared) != derived
    ):
        raise ValueError(
            f"{flag}={declared!r} contradicts {field}={list(values)!r}: one flag per "
            "entry, saying whether that entry is inside the support"
        )


def _normalise_log_fields(
    data: Any, *, scalars: tuple[str, ...] = (), arrays: tuple[str, ...] = ()
) -> Any:
    """Accept `null` for `-inf` on the way in, and derive the companion flags.

    Runs `mode='before'`, so it sees the raw mapping from `model_validate_json` exactly as
    it sees the keyword arguments of a direct construction: `LoglikResult(value=-inf, ...)`
    and the JSON `{"value": null, "value_in_support": false}` produce the same record.
    """
    if not isinstance(data, dict):
        return data
    out: dict[str, Any] = dict(data)
    for name in scalars:
        _normalise_log_scalar(out, name)
    for name in arrays:
        _normalise_log_array(out, name)
    return out


def _json_log_scalar(value: float) -> float | None:
    return None if value == -math.inf else value


def _json_log_array(values: tuple[float, ...]) -> list[float | None]:
    return [_json_log_scalar(value) for value in values]


def validate_probability_weights(weights: Sequence[float], *, label: str) -> tuple[float, ...]:
    """Check a probability vector. Positive weights are never renormalised silently."""
    values = tuple(float(w) for w in weights)
    if not values:
        raise ValueError(f"{label} must carry at least one weight")
    for index, value in enumerate(values):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{label}[{index}] must be a positive finite probability, got {value}")
    total = math.fsum(values)
    if abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE:
        raise ValueError(
            f"{label} must sum to 1 within {PROBABILITY_SUM_TOLERANCE}, got {total!r}: "
            "a positive weight vector that does not sum to one is an error at its source, "
            "not something to renormalise here"
        )
    return values


# --------------------------------------------------------------------------------------
# the latent record and its schema
# --------------------------------------------------------------------------------------


class ThetaRecord(StrictModel):
    """One point of the latent space: a family index and two real vectors.

    `v` carries the coordinates the proposal moves and, for the P1 schema, its last three
    entries are the nuisance coordinates of the noise law. `z_perp` carries the residual
    coordinates that are kept rather than zeroed (SPEC §7.5). Neither is a log-density:
    `log_prior_parts` on `Particle` is a diagnostic and never an input to `log_prob`.
    """

    schema_id: str = Field(min_length=1)
    s: int = Field(ge=0)
    v: tuple[float, ...]
    z_perp: tuple[float, ...]
    basis_hash: Sha256

    @field_validator("v", "z_perp")
    @classmethod
    def _coordinates_are_finite(
        cls, value: tuple[float, ...], info: ValidationInfo
    ) -> tuple[float, ...]:
        return _finite(value, label=str(info.field_name))


class DensitySchema(StrictModel):
    """What a latent vector must look like to belong to one density.

    The schema is what makes a dimension mismatch an error instead of a slice. It also
    names the basis the coordinates are whitened in, so a theta produced against another
    basis cannot be scored against this one.
    """

    schema_id: str = Field(min_length=1)
    n_v: int = Field(ge=1)
    n_residual: int = Field(ge=0)
    families: tuple[int, ...]
    basis_hash: Sha256
    transform_version: str = Field(min_length=1)
    measure: Literal["counting_x_latent_lebesgue"] = LATENT_MEASURE

    @field_validator("families")
    @classmethod
    def _families_are_a_sorted_support(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value:
            raise ValueError("families must name at least one family index")
        if any(s < 0 for s in value):
            raise ValueError(f"family indices are non-negative, got {value}")
        if list(value) != sorted(set(value)):
            raise ValueError(f"families must be sorted and unique, got {value}")
        return value

    def validate_theta(self, theta: ThetaRecord) -> None:
        """Refuse a theta this schema does not describe. Nothing is coerced or padded."""
        if theta.schema_id != self.schema_id or theta.basis_hash != self.basis_hash:
            raise ValueError(
                f"schema/basis mismatch: theta names {theta.schema_id!r}/{theta.basis_hash} "
                f"and this schema is {self.schema_id!r}/{self.basis_hash}"
            )
        if theta.s not in self.families:
            raise ValueError(f"family outside support: s={theta.s} is not in {self.families}")
        if len(theta.v) != self.n_v or len(theta.z_perp) != self.n_residual:
            raise ValueError(
                f"latent dimension mismatch: expected {self.n_v} v and "
                f"{self.n_residual} z_perp coordinates, got {len(theta.v)} and "
                f"{len(theta.z_perp)}"
            )


class PriorContext(StrictModel):
    """The conditional Gaussian law of the geology coordinates, given the static data G.

    `mean`, `chol` and `rotation` are the conditioned mean, its Cholesky factor and the
    whitening rotation, all in the geology block. The field is called `density_schema`
    rather than `schema` because `schema` is an attribute of pydantic's own base model and
    shadowing it warns; the contents are the plan's `schema`.

    The arrays are copied and marked read-only on the way in. A frozen record whose arrays
    a later task can still overwrite in place would be frozen in name only.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    density_schema: DensitySchema
    n_geology: int = Field(ge=1)
    n_state_residual: int = Field(ge=0)
    mean: F64
    chol: F64
    rotation: F64
    design: dict[str, Any]
    g_hash: Sha256
    information_hash: Sha256

    @field_validator("mean", "chol", "rotation")
    @classmethod
    def _finite_and_read_only(cls, value: F64, info: ValidationInfo) -> F64:
        array = np.array(value, dtype=np.float64, copy=True)
        if not bool(np.all(np.isfinite(array))):
            raise ValueError(f"{info.field_name} must be finite everywhere")
        array.setflags(write=False)
        return array

    @model_validator(mode="after")
    def _shapes_match_the_declared_dimension(self) -> PriorContext:
        n = self.n_geology
        if self.mean.shape != (n,):
            raise ValueError(f"mean: expected shape {(n,)}, got {self.mean.shape}")
        for name in ("chol", "rotation"):
            array: F64 = getattr(self, name)
            if array.shape != (n, n):
                raise ValueError(f"{name}: expected shape {(n, n)}, got {array.shape}")
        diagonal = np.diag(self.chol)
        if not bool(np.all(diagonal > 0.0)):
            raise ValueError(
                "chol must have a strictly positive diagonal: a zero on it is a mode with "
                f"no spread left, which SPEC §7.5 forbids, got {diagonal.tolist()}"
            )
        total = self.density_schema.n_v + self.density_schema.n_residual
        if n + self.n_state_residual > total:
            raise ValueError(
                f"n_geology {n} plus n_state_residual {self.n_state_residual} exceeds the "
                f"{total} latent coordinates the schema declares"
            )
        return self


class NoiseTheta(StrictModel):
    """The observation noise law, in g-space (plan §2).

    `nu` is fixed at five: a Student-t tail is a declared modelling choice of this stage,
    not something fitted to the residuals it is meant to absorb.
    """

    sigma: float = Field(gt=0.0)
    rho: float = Field(ge=0.0, lt=1.0)
    nu: float = Field(gt=2.0)
    log_bias: float

    @field_validator("sigma", "rho", "nu", "log_bias")
    @classmethod
    def _finite(cls, value: float, info: ValidationInfo) -> float:
        if not math.isfinite(value):
            raise ValueError(f"{info.field_name} must be finite, got {value}")
        return value

    @classmethod
    def from_latent(cls, v: Sequence[float]) -> NoiseTheta:
        """The declared map from the nuisance coordinates of `v` (plan §2).

        `Phi` is the standard normal CDF, so sigma and rho stay inside their declared
        ranges for every real coordinate instead of being clipped after the fact.
        """
        if len(v) <= LOG_BIAS_INDEX:
            raise ValueError(
                f"the nuisance block occupies v[{NOISE_SIGMA_INDEX}:{LOG_BIAS_INDEX + 1}], "
                f"so a latent vector needs at least {LOG_BIAS_INDEX + 1} coordinates, "
                f"got {len(v)}"
            )
        coordinates = _finite(v, label="v")
        return cls(
            sigma=SIGMA_MIN + SIGMA_SPAN * float(ndtr(coordinates[NOISE_SIGMA_INDEX])),
            rho=RHO_MAX * float(ndtr(coordinates[NOISE_RHO_INDEX])),
            nu=NU_FIXED,
            log_bias=LOG_BIAS_SCALE * coordinates[LOG_BIAS_INDEX],
        )


#: The fixed law the noise-recovery tests hold constant while they vary something else.
FIXED_NOISE_THETA = NoiseTheta(sigma=0.03, rho=0.4, nu=NU_FIXED, log_bias=0.0)


# --------------------------------------------------------------------------------------
# observations
# --------------------------------------------------------------------------------------


class HistoryRow(StrictModel):
    """One well-month of production history.

    `month_index` is a calendar ordinal from the start of the schedule, so a well that
    reports nothing in a month still has the month. `sigma_multiplier` is set by OBSERVED
    quality — a workover month, a suspect meter — and never by a fitted residual.
    """

    well_id: str = Field(min_length=1)
    month_index: int = Field(ge=0)
    raw_value: float | None
    bin_index: int | None
    quality_group: str = Field(min_length=1)
    observed_valid: bool
    reset: bool
    sigma_multiplier: float = Field(default=1.0, gt=0.0)

    @property
    def key(self) -> tuple[str, int]:
        return (self.well_id, self.month_index)

    @model_validator(mode="after")
    def _a_valid_row_names_its_bin(self) -> HistoryRow:
        if self.raw_value is not None and not math.isfinite(self.raw_value):
            raise ValueError(f"{self.key}: raw_value must be finite, got {self.raw_value}")
        if not math.isfinite(self.sigma_multiplier):
            raise ValueError(f"{self.key}: sigma_multiplier must be finite")
        if self.observed_valid and self.bin_index is None:
            raise ValueError(f"{self.key}: an observed row states the bin it fell in")
        if not self.observed_valid and self.bin_index is not None:
            raise ValueError(
                f"{self.key}: a row that was not observed carries no bin_index, got "
                f"{self.bin_index}"
            )
        if self.bin_index is not None and self.bin_index < 0:
            raise ValueError(f"{self.key}: bin_index is non-negative, got {self.bin_index}")
        return self


class LogRow(StrictModel):
    """One log observation on a cell support, at a date that may itself be uncertain.

    Several intervals that share one unknown date share one `date_group` and are
    marginalised together once; multiplying independent mixtures instead would invent
    information. `use` records the role this row already played, so information that
    conditioned the prior cannot reappear as an independent likelihood (SPEC §5).
    """

    observation_id: str = Field(min_length=1)
    support_cell_ids: tuple[int, ...]
    support_weights: tuple[float, ...]
    bin_index: int | None
    date_times_s: tuple[float, ...]
    date_weights: tuple[float, ...]
    date_group: str = Field(min_length=1)
    bias_group: str = Field(min_length=1)
    valid: bool
    use: ObservationRole

    @field_validator("support_cell_ids")
    @classmethod
    def _support_cells_are_unique(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value:
            raise ValueError("support_cell_ids must name at least one cell")
        if any(cell < 0 for cell in value):
            raise ValueError(f"support_cell_ids are non-negative, got {value}")
        if len(set(value)) != len(value):
            raise ValueError(f"support_cell_ids must be unique, got {value}")
        return value

    @field_validator("date_times_s")
    @classmethod
    def _dates_are_increasing(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        times = _finite(value, label="date_times_s")
        if not times:
            raise ValueError("date_times_s must name at least one candidate date")
        if any(b <= a for a, b in zip(times, times[1:], strict=False)):
            raise ValueError(f"date_times_s must be strictly increasing, got {times}")
        return times

    @model_validator(mode="after")
    def _weights_describe_the_things_they_weigh(self) -> LogRow:
        if len(self.support_weights) != len(self.support_cell_ids):
            raise ValueError(
                f"{self.observation_id}: {len(self.support_weights)} support weights for "
                f"{len(self.support_cell_ids)} support cells"
            )
        if len(self.date_weights) != len(self.date_times_s):
            raise ValueError(
                f"{self.observation_id}: {len(self.date_weights)} date weights for "
                f"{len(self.date_times_s)} candidate dates"
            )
        validate_probability_weights(
            self.support_weights, label=f"{self.observation_id} support_weights"
        )
        validate_probability_weights(self.date_weights, label=f"{self.observation_id} date_weights")
        if self.valid and self.bin_index is None:
            raise ValueError(f"{self.observation_id}: a valid log row states its bin")
        if not self.valid and self.bin_index is not None:
            raise ValueError(f"{self.observation_id}: an invalid log row carries no bin_index")
        if self.bin_index is not None and self.bin_index < 0:
            raise ValueError(
                f"{self.observation_id}: bin_index is non-negative, got {self.bin_index}"
            )
        return self


class ObservationBundle(StrictModel):
    """Everything one inverse problem is allowed to score against, and its identity.

    `bin_edges_by_group` is keyed by the group whose rows use it: a history row by its
    `quality_group`, a log row by its `bias_group`. E02 carries ONE bias group per bundle,
    because a second one needs its own anchored latent in the noise schema rather than a
    silently shared bias.
    """

    history: tuple[HistoryRow, ...]
    logs: tuple[LogRow, ...]
    bin_edges_by_group: dict[str, tuple[float, ...]]
    cutoff_s: float = Field(gt=0.0)
    information_hash: Sha256
    observation_hash: Sha256

    @field_validator("bin_edges_by_group")
    @classmethod
    def _edges_are_increasing(
        cls, value: dict[str, tuple[float, ...]]
    ) -> dict[str, tuple[float, ...]]:
        for group, edges in value.items():
            finite = _finite(edges, label=f"bin_edges_by_group[{group!r}]")
            if len(finite) < 3:
                raise ValueError(
                    f"group {group!r}: a bounded-bin kernel needs at least two bins, so at "
                    f"least three edges, got {len(finite)}"
                )
            if any(b <= a for a, b in zip(finite, finite[1:], strict=False)):
                raise ValueError(f"group {group!r}: bin edges must be strictly increasing")
        return value

    @model_validator(mode="after")
    def _rows_resolve_against_the_declared_bins(self) -> ObservationBundle:
        if not math.isfinite(self.cutoff_s):
            raise ValueError(f"cutoff_s must be finite, got {self.cutoff_s}")
        keys = [row.key for row in self.history]
        if keys != sorted(set(keys)):
            raise ValueError(
                "history rows must be sorted and unique by (well_id, month_index); a "
                "repeated well-month would be counted twice"
            )
        for row in self.history:
            self._check_bin(f"history row {row.key}", row.quality_group, row.bin_index)
        bias_groups = {row.bias_group for row in self.logs}
        if len(bias_groups) > 1:
            raise ValueError(
                f"one bias_group per bundle, got {sorted(bias_groups)}: a second group "
                "needs its own anchored latent in the noise schema, not a shared bias"
            )
        for log in self.logs:
            self._check_bin(f"log row {log.observation_id!r}", log.bias_group, log.bin_index)
        return self

    def _check_bin(self, label: str, group: str, bin_index: int | None) -> None:
        edges = self.bin_edges_by_group.get(group)
        if edges is None:
            raise ValueError(
                f"{label}: group {group!r} has no bin edges in bin_edges_by_group "
                f"{sorted(self.bin_edges_by_group)}"
            )
        if bin_index is not None and bin_index >= len(edges) - 1:
            raise ValueError(
                f"{label}: bin_index {bin_index} is outside the {len(edges) - 1} bins of "
                f"group {group!r}"
            )


class ModelObservations(StrictModel):
    """What the forward model predicts at the places the data live.

    `fw` is keyed by `(well_id, month_index)` and is `None` where the model produces no
    prediction — a shut well, a month outside the schedule — which is not the same as a
    prediction of zero. `so_support` is keyed by `(observation_id, time_s)`.
    """

    fw: dict[tuple[str, int], UnitInterval | None]
    so_support: dict[tuple[str, float], UnitInterval]
    model_hash: Sha256

    @model_validator(mode="after")
    def _predictions_are_finite(self) -> ModelObservations:
        for key, value in self.fw.items():
            if value is not None and not math.isfinite(value):
                raise ValueError(f"fw{key} must be finite, got {value}")
        for support_key, so in self.so_support.items():
            if not math.isfinite(so):
                raise ValueError(f"so_support{support_key} must be finite, got {so}")
        return self


class LoglikResult(StrictModel):
    """One likelihood evaluation, with the per-observation terms it was summed from.

    An empty observation set gives `L=1`: `value=0`, no terms, nothing used. That is the
    case plan §1 relies on when it says the result at `r=p0` must reproduce the CONDITIONAL
    prior exactly, so it is stated in the type rather than assumed at the call site.

    `value` and each entry of `terms` may be `-inf` — an observation this theta cannot
    produce at all — and each carries its own support flag so the record survives JSON.
    """

    value: float
    value_in_support: bool
    terms: tuple[float, ...]
    terms_in_support: tuple[bool, ...]
    n_used: int = Field(ge=0)
    observation_hash: Sha256

    @model_validator(mode="before")
    @classmethod
    def _log_values_travel_with_their_support(cls, data: Any) -> Any:
        return _normalise_log_fields(data, scalars=("value",), arrays=("terms",))

    @field_serializer("value", when_used="json")
    def _value_as_json(self, value: float) -> float | None:
        return _json_log_scalar(value)

    @field_serializer("terms", when_used="json")
    def _terms_as_json(self, terms: tuple[float, ...]) -> list[float | None]:
        return _json_log_array(terms)

    @model_validator(mode="after")
    def _terms_account_for_what_was_used(self) -> LoglikResult:
        _log_value(self.value, label="value")
        for index, term in enumerate(self.terms):
            _log_value(term, label=f"terms[{index}]")
        if self.n_used != len(self.terms):
            raise ValueError(
                f"n_used {self.n_used} does not match the {len(self.terms)} terms recorded"
            )
        if len(self.terms_in_support) != len(self.terms):
            raise ValueError(
                f"terms_in_support has {len(self.terms_in_support)} flags for "
                f"{len(self.terms)} terms"
            )
        if self.n_used == 0 and self.value != 0.0:
            raise ValueError(f"no observations means L=1, so value must be 0.0, got {self.value}")
        return self


# --------------------------------------------------------------------------------------
# the target, its particles and the SMC state
# --------------------------------------------------------------------------------------


class TargetEvaluation(StrictModel):
    """One evaluation of the tempered target at a theta, with the forward it came from.

    `forward_ref` is `None` for a target that needed no physical realisation — a toy
    problem, or a theta refused before the solver was asked. `cache_key` identifies the
    evaluation so that a resumed run reuses a finished expensive likelihood instead of
    paying for it twice.
    """

    theta: ThetaRecord
    log_p0: float
    log_p0_in_support: bool
    log_l: float
    log_l_in_support: bool
    log_r: float
    log_r_in_support: bool
    forward_ref: ArtifactRef | None
    cache_key: str = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _log_values_travel_with_their_support(cls, data: Any) -> Any:
        return _normalise_log_fields(data, scalars=("log_p0", "log_l", "log_r"))

    @field_validator("log_p0", "log_l", "log_r")
    @classmethod
    def _log_space(cls, value: float, info: ValidationInfo) -> float:
        return _log_value(value, label=str(info.field_name))

    @field_serializer("log_p0", "log_l", "log_r", when_used="json")
    def _log_as_json(self, value: float) -> float | None:
        return _json_log_scalar(value)


class Particle(StrictModel):
    """One particle and its ancestor.

    `log_prior_parts` is DIAGNOSTIC: the parts a renderer reports for inspection, which
    plan §2 keeps checkable rather than trusted. The number the algorithm uses is
    `evaluation.log_p0`, and nothing here sums the parts into it.
    """

    particle_id: int = Field(ge=0)
    ancestor_id: int = Field(ge=0)
    evaluation: TargetEvaluation
    log_prior_parts: dict[str, float] = Field(default_factory=dict)


class SMCState(StrictModel):
    """Everything needed to continue the same sequence after an interruption.

    A pending Metropolis proposal is stored with the uniform draw it will be compared
    against: keeping one without the other would silently re-randomise the accept decision
    on resume, which is a different chain, not the same one continued.
    """

    particles: tuple[Particle, ...]
    log_weights: tuple[float, ...]
    log_weights_in_support: tuple[bool, ...]
    beta: float = Field(ge=0.0, le=1.0)
    level: int = Field(ge=0)
    log_evidence: float
    log_evidence_in_support: bool
    phase: SmcPhase
    cursor: int = Field(ge=0)
    rng_state: dict[str, Any]
    proposal_hash: Sha256
    target_hash: Sha256
    pending_proposal: ThetaRecord | None
    pending_log_u: float | None
    algorithm_status: AlgorithmStatus
    diagnostics: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _log_values_travel_with_their_support(cls, data: Any) -> Any:
        return _normalise_log_fields(data, scalars=("log_evidence",), arrays=("log_weights",))

    @field_serializer("log_evidence", when_used="json")
    def _log_evidence_as_json(self, value: float) -> float | None:
        return _json_log_scalar(value)

    @field_serializer("log_weights", when_used="json")
    def _log_weights_as_json(self, values: tuple[float, ...]) -> list[float | None]:
        return _json_log_array(values)

    @model_validator(mode="after")
    def _state_is_self_consistent(self) -> SMCState:
        if len(self.log_weights) != len(self.particles):
            raise ValueError(
                f"log_weights has {len(self.log_weights)} entries for "
                f"{len(self.particles)} particles"
            )
        if len(self.log_weights_in_support) != len(self.log_weights):
            raise ValueError(
                f"log_weights_in_support has {len(self.log_weights_in_support)} flags for "
                f"{len(self.log_weights)} weights"
            )
        for index, weight in enumerate(self.log_weights):
            _log_value(weight, label=f"log_weights[{index}]")
        _log_value(self.log_evidence, label="log_evidence")
        if (self.pending_proposal is None) != (self.pending_log_u is None):
            raise ValueError(
                "a pending proposal is stored with the pending log u it is compared "
                "against; one without the other re-randomises the accept decision"
            )
        if self.pending_log_u is not None and not math.isfinite(self.pending_log_u):
            raise ValueError(
                "pending_log_u is the log of a uniform draw and is finite; None means "
                f"there is no pending proposal, got {self.pending_log_u}"
            )
        return self


class PosteriorBundle(StrictModel):
    """The published result of one inverse run, and what it is allowed to be called.

    SPEC §13.4.2: if `beta < 1` this is an intermediate distribution and NOT a posterior,
    so `COMPLETE` and an unfinished tempering cannot appear together. The algorithm status
    and the convergence verdict stay separate: a run that reached `beta = 1` and failed its
    convergence checks is `POSTERIOR_NOT_CONVERGED`, not a converged posterior with a note.
    """

    state_ref: ArtifactRef
    particles_ref: ArtifactRef
    diagnostics_ref: ArtifactRef
    ledger_ref: ArtifactRef
    algorithm_status: AlgorithmStatus
    beta: float = Field(ge=0.0, le=1.0)
    convergence_status: ConvergenceStatus
    physical_state_refs: tuple[ArtifactRef, ...]
    parent_run_ids: tuple[str, ...]

    @model_validator(mode="after")
    def _status_matches_the_tempering(self) -> PosteriorBundle:
        if self.algorithm_status == "COMPLETE" and self.beta != 1.0:
            raise ValueError(
                f"algorithm_status COMPLETE with beta={self.beta}: SPEC 13.4.2 makes "
                "beta<1 an intermediate distribution, not a posterior"
            )
        if self.algorithm_status == "POSTERIOR_NOT_CONVERGED" and self.beta != 1.0:
            raise ValueError(
                f"POSTERIOR_NOT_CONVERGED describes a run that reached beta=1 and failed "
                f"its convergence checks, got beta={self.beta}"
            )
        if self.algorithm_status == "COMPLETE" and self.convergence_status == "NOT_CONVERGED":
            raise ValueError(
                "a run whose convergence verdict is NOT_CONVERGED has algorithm_status "
                "POSTERIOR_NOT_CONVERGED, not COMPLETE"
            )
        return self


__all__ = [
    "FIXED_NOISE_THETA",
    "F64",
    "I64",
    "IN_SUPPORT_SUFFIX",
    "LATENT_MEASURE",
    "LOG_BIAS_INDEX",
    "LOG_BIAS_SCALE",
    "NOISE_RHO_INDEX",
    "NOISE_SIGMA_INDEX",
    "NU_FIXED",
    "N_P1_GEOLOGY_IN_V",
    "OBSERVATION_ROLES",
    "PROBABILITY_SUM_TOLERANCE",
    "RHO_MAX",
    "SIGMA_MIN",
    "SIGMA_SPAN",
    "AlgorithmStatus",
    "ConvergenceStatus",
    "DensitySchema",
    "E01DependencyError",
    "HistoryRow",
    "LikelihoodNumericalError",
    "LogRow",
    "LoglikResult",
    "ModelObservations",
    "NoiseTheta",
    "ObservationBundle",
    "ObservationRole",
    "ObservationSupportMismatch",
    "Particle",
    "PosteriorBundle",
    "PriorContext",
    "RendererNumericalError",
    "SMCState",
    "SmcPhase",
    "TargetEvaluation",
    "ThetaRecord",
    "UnsupportedObservationError",
    "validate_probability_weights",
]
