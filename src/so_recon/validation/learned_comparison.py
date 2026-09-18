"""The E03 comparison skeleton: per world/method/seed rows over the state evaluator.

The scientific matrix (plan §10) is Task 11's; what this module fixes now is the
STRUCTURE the matrix fills, and the two label laws that make a filled matrix honest:

* **B0 is the static conditional prior only** (plan §4.4, §10.1). The selection reads
  prior-start initial draws whose recorded sampling density EQUALS p0
  (`log_r == log_p0` — the artifact-level provenance proof that the ensemble was
  sampled under the prior, §0.2 item 6) and demands their physical forwards. It never
  consults the learned proposal, so changing q cannot change the selected B0.
* **A raw q ensemble is never a posterior** (plan §4.4): its rows carry
  `ensemble_kind=raw_proposal` and any posterior claim is refused at construction.

Per-run rows flatten the evaluator's products and scores plus the run diagnostics
the artifacts already carry (beta path, pre-resampling ESS, per-kernel acceptance,
unique ancestors, distinct physical states, runtime/RSS/disk/failures). Fields the
engine does not record (residual movement) are `None` with a note, never invented.

World averaging (plan §5.5) seeds-then-worlds: the per-world mean over inference
seeds first, then equal weight per independent world. Cells, zones and rows are
never pooled into a pseudo-sample, and methods are never pooled either: the
average is per method, and rows of more than one `method_id` are refused.

Operational products build from the posterior bundle and checkpoint alone — the
loader has no truth parameter to misuse; truth is scored separately by
`so_recon.validation.ensemble_states.score_ensemble_states`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from so_recon.inference.contracts import F64
from so_recon.paths import ProjectPaths
from so_recon.registry.artifact import ArtifactRef
from so_recon.simulator.case_io import read_array
from so_recon.simulator.results import load_forward_result
from so_recon.validation.e03_protocol import (
    DIAGNOSTIC_PARTIAL_ENSEMBLE_KIND,
    POSTERIOR_ENSEMBLE_KIND,
    PRIOR_ENSEMBLE_KIND,
    RAW_PROPOSAL_ENSEMBLE_KIND,
)
from so_recon.validation.ensemble_states import (
    DEFAULT_QUANTILE_PROBABILITIES,
    EnsembleStateProducts,
    ParticleZones,
    ZoneSupport,
    aggregate_particle_zones,
    ensemble_state_products,
    zone_value_map,
)

#: The provenance statement a B0 selection carries on its face.
B0_PROVENANCE = "prior_start_initial_draws_log_r_equals_log_p0"

#: Why `residual_movement` is None in every row this version can build: the SMC
#: checkpoint records moves without the z_perp they moved between, so the honest row
#: is an explicit absence, not a number derived from something else.
RESIDUAL_MOVEMENT_NOTE = (
    "not recorded by the SMC engine: per-move z_perp is absent from the checkpoint; "
    "needs a versioned diagnostic before the number can exist (plan §10.3)"
)


# --------------------------------------------------------------------------------------
# run diagnostics from the artifacts a run already published
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunDiagnostics:
    """The §10.2 per-run diagnostics, extracted from published payloads only."""

    beta_path: tuple[float, ...]
    pre_resampling_ess: tuple[float, ...]
    n_unique_ancestors: int
    n_physical_states: int
    acceptance_by_kernel: dict[str, float]
    kernel_probabilities: dict[str, float] | None
    runtime_wall_s: float | None
    cpu_s: float | None
    peak_rss_bytes: int | None
    output_bytes: int | None
    failures: int
    residual_movement: None
    residual_movement_note: str
    n_particles: int


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def run_diagnostics_from_payloads(
    smc_payload: Mapping[str, Any],
    checkpoint_manifest: Mapping[str, Any],
    *,
    forward_costs: Iterable[Mapping[str, Any]] = (),
) -> RunDiagnostics:
    """Read one run's diagnostics from its `learned_smc.json`-shape payload and checkpoint."""
    state = checkpoint_manifest["state"]
    particles = list(state["particles"])
    diagnostics = state.get("diagnostics", {})
    moves: list[Mapping[str, Any]] = list(diagnostics.get("moves", []))
    by_kernel: dict[str, list[float]] = {}
    for move in moves:
        by_kernel.setdefault(str(move["kernel"]), []).append(float(bool(move["accepted"])))
    forwards = {
        str(particle["evaluation"]["forward_ref"]["artifact_id"])
        for particle in particles
        if particle["evaluation"].get("forward_ref") is not None
    }
    session = dict(smc_payload.get("budget", {}).get("session", {}))
    peak_rss = [int(cost["peak_rss_bytes"]) for cost in forward_costs if cost.get("peak_rss_bytes")]
    beta_path = tuple(float(value) for value in smc_payload.get("beta_history", []))
    return RunDiagnostics(
        beta_path=beta_path,
        pre_resampling_ess=tuple(
            float(entry["pre_ess"]) for entry in smc_payload.get("resampling", [])
        ),
        n_unique_ancestors=len({int(particle["ancestor_id"]) for particle in particles}),
        n_physical_states=len(forwards),
        acceptance_by_kernel={
            name: float(sum(accepted) / len(accepted))
            for name, accepted in sorted(by_kernel.items())
        },
        kernel_probabilities=(
            {
                str(name): float(value)
                for name, value in dict(diagnostics.get("kernel_probabilities", {})).items()
            }
            or None
        ),
        runtime_wall_s=_finite_or_none(session.get("wall_s")),
        cpu_s=_finite_or_none(session.get("cpu_s")),
        peak_rss_bytes=max(peak_rss) if peak_rss else None,
        output_bytes=(
            int(session["output_bytes"]) if session.get("output_bytes") is not None else None
        ),
        failures=len(smc_payload.get("failures", [])),
        residual_movement=None,
        residual_movement_note=RESIDUAL_MOVEMENT_NOTE,
        n_particles=len(particles),
    )


# --------------------------------------------------------------------------------------
# B0: the static conditional prior, provenance-proven and source-independent
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class B0Selection:
    """The selected p0-draw ensemble with the proof it is one."""

    evaluations: tuple[Mapping[str, Any], ...]
    provenance: str

    @property
    def n_draws(self) -> int:
        return len(self.evaluations)


def _same_log_density(left: Any, right: Any) -> bool:
    """Extended-real equality: JSON carries -inf as None (inference.contracts)."""
    if left is None or right is None:
        return (
            (left is None) == (right is None)
            or (left is None and right == -math.inf)
            or (right is None and left == -math.inf)
        )
    return float(left) == float(right)


def b0_from_prior_start_checkpoint(checkpoint_manifest: Mapping[str, Any]) -> B0Selection:
    """Select B0 from a prior-start checkpoint: initial draws proven to be p0 draws.

    The proof is artifact-level: every initial evaluation's SAMPLING density `log_r`
    equals its PRIOR density `log_p0`, which is true exactly when the initial ensemble
    was drawn under the prior itself (a defensive mixture with epsilon < 1 records
    `log_r != log_p0` and is refused, §0.2 item 6). Each draw must carry its physical
    forward — B0 is a STATE ensemble. The selection deliberately reads nothing about
    any learned proposal, so changing q cannot change the selected B0.
    """
    state = checkpoint_manifest["state"]
    initial = state.get("diagnostics", {}).get("initial_evaluations")
    if not isinstance(initial, list) or not initial:
        raise ValueError(
            "B0 needs the checkpoint's persisted initial evaluations: none are present"
        )
    for index, evaluation in enumerate(initial):
        if evaluation.get("forward_ref") is None:
            raise ValueError(
                f"B0 is a physical state ensemble: initial draw {index} carries no forward"
            )
        if not _same_log_density(evaluation.get("log_r"), evaluation.get("log_p0")):
            raise ValueError(
                f"initial draw {index} has log_r != log_p0: the initial ensemble was "
                "sampled under a law other than p0, so it is not proven to be B0"
            )
    return B0Selection(
        evaluations=tuple(initial),
        provenance=B0_PROVENANCE,
    )


# --------------------------------------------------------------------------------------
# comparison rows
# --------------------------------------------------------------------------------------


def comparison_row(
    *,
    parent_id: str,
    method_id: str,
    inference_seed: int,
    ensemble_kind: str,
    posterior_claim: bool,
    products: EnsembleStateProducts | None = None,
    scores: Any | None = None,
    run: RunDiagnostics | None = None,
) -> dict[str, Any]:
    """One world/method/seed row of the comparison matrix (plan §10.2).

    A raw-q row exists (its state products and even its truth-conditional scores are
    legitimate diagnostics) but never with a posterior claim. The same holds for B0's
    uncorrected prior draws: `prior_ensemble` is a real ensemble with state evidence,
    and it is still not a posterior.
    """
    allowed = {
        RAW_PROPOSAL_ENSEMBLE_KIND,
        PRIOR_ENSEMBLE_KIND,
        POSTERIOR_ENSEMBLE_KIND,
        DIAGNOSTIC_PARTIAL_ENSEMBLE_KIND,
    }
    if ensemble_kind not in allowed:
        raise ValueError(
            f"unknown ensemble_kind {ensemble_kind!r}; expected one of {sorted(allowed)}"
        )
    if posterior_claim and ensemble_kind != POSTERIOR_ENSEMBLE_KIND:
        raise ValueError(
            f"ensemble_kind {ensemble_kind!r} never carries a posterior claim (plan §4.4): "
            "only a completed beta=1 posterior ensemble does"
        )
    row: dict[str, Any] = {
        "parent_id": parent_id,
        "method_id": method_id,
        "inference_seed": inference_seed,
        "ensemble_kind": ensemble_kind,
        "posterior_claim": posterior_claim,
        "estimator": "posterior_mean",
    }
    if products is not None:
        row.update(
            {
                "zone_names": list(products.zone_names),
                "posterior_mean_so": [
                    None if not np.isfinite(v) else float(v) for v in products.posterior_mean_so
                ],
                "s_probabilities": dict(products.s_probabilities),
                "representative_particle_index": products.representative_particle_index,
                "representative_rule": products.representative_rule,
                "n_particles": products.n_particles,
            }
        )
    if scores is not None:
        row.update(
            {
                "mae": scores.mae,
                "rmse": scores.rmse,
                "crps": scores.crps,
                "coverage": scores.coverage,
                "mean_width": scores.mean_width,
            }
        )
    if run is not None:
        row.update(
            {
                "beta_path": list(run.beta_path),
                "pre_resampling_ess": list(run.pre_resampling_ess),
                "acceptance_by_kernel": dict(run.acceptance_by_kernel),
                "n_unique_ancestors": run.n_unique_ancestors,
                "n_physical_states": run.n_physical_states,
                "runtime_wall_s": run.runtime_wall_s,
                "cpu_s": run.cpu_s,
                "peak_rss_bytes": run.peak_rss_bytes,
                "output_bytes": run.output_bytes,
                "failures": run.failures,
                "residual_movement": run.residual_movement,
                "residual_movement_note": run.residual_movement_note,
            }
        )
    return row


def average_over_worlds(rows: Sequence[Mapping[str, Any]], metric: str) -> float:
    """Equal weight per independent world: seed-mean within a world, then world mean.

    Never a pooled mean over rows (and the rows themselves carry scalars, never
    pooled cells or zones), per plan §5.5. The average is PER METHOD: every row
    must carry `method_id`, and rows of more than one method are refused —
    averaging across methods is exactly the pseudo-pooling §5.5 bans, so it can
    never be produced silently. Average each method's rows in its own call.
    """
    method_ids: list[str] = []
    for row in rows:
        if "method_id" not in row:
            raise ValueError(
                f"row for parent {row.get('parent_id')!r} carries no method_id: "
                "world averaging is per method and cannot group an unlabelled row"
            )
        method_id = str(row["method_id"])
        if method_id not in method_ids:
            method_ids.append(method_id)
    if len(method_ids) > 1:
        named = ", ".join(repr(method_id) for method_id in sorted(method_ids))
        raise ValueError(
            f"average_over_worlds is per method: the rows mix {len(method_ids)} methods "
            f"({named}); averaging across methods pools them, which plan §5.5 forbids — "
            "average each method's rows in a separate call"
        )
    by_world: dict[str, list[float]] = {}
    for row in rows:
        if metric not in row or row[metric] is None:
            raise ValueError(
                f"row for parent {row.get('parent_id')!r} carries no metric {metric!r}"
            )
        by_world.setdefault(str(row["parent_id"]), []).append(float(row[metric]))
    if not by_world:
        raise ValueError(f"no rows carry the metric {metric!r}")
    world_means = [sum(values) / len(values) for values in by_world.values()]
    return sum(world_means) / len(world_means)


# --------------------------------------------------------------------------------------
# artifact loading: each particle reads ITS OWN forward; truth is not an input
# --------------------------------------------------------------------------------------


def _states_row(result: Any, dataset: str, paths: ProjectPaths, time_index: int) -> F64:
    values = np.asarray(read_array(result.states[dataset], paths), dtype=np.float64)
    row: F64 = np.asarray(values[time_index], dtype=np.float64)
    if row.ndim != 1 or not np.isfinite(row).all():
        raise ValueError(
            f"forward {result.job_id} state {dataset!r} has no finite row {time_index}"
        )
    return row


def load_particle_states(
    refs: Sequence[ArtifactRef], paths: ProjectPaths, time_index: int
) -> tuple[F64, F64, F64]:
    """Per-particle (so, pv, bo) rows at one report time, each from its own forward.

    Forwards shared by several particles (a resampled copy) are read once and the
    bytes re-verified on load (`load_forward_result` + `read_array` digests).
    """
    cache: dict[str, tuple[F64, F64, F64]] = {}
    so_rows: list[F64] = []
    pv_rows: list[F64] = []
    bo_rows: list[F64] = []
    for ref in refs:
        if ref.artifact_id not in cache:
            result = load_forward_result(paths.resolve(ref.path), paths)
            if "so" in result.states:
                so = _states_row(result, "so", paths, time_index)
            elif "sw" in result.states and result.physics_class == "OW":
                so = 1.0 - _states_row(result, "sw", paths, time_index)
            else:
                raise ValueError(f"forward {result.job_id} publishes no OW saturation")
            if "pore_volume_m3" not in result.states or "bo" not in result.states:
                raise ValueError(
                    f"forward {result.job_id} publishes no pore_volume_m3/bo state: the "
                    "per-particle aggregation needs each particle's own PV and Bo"
                )
            cache[ref.artifact_id] = (
                so,
                _states_row(result, "pore_volume_m3", paths, time_index),
                _states_row(result, "bo", paths, time_index),
            )
        so, pv, bo = cache[ref.artifact_id]
        so_rows.append(so)
        pv_rows.append(pv)
        bo_rows.append(bo)
    stacked: tuple[F64, F64, F64] = (
        np.asarray(np.stack(so_rows), dtype=np.float64),
        np.asarray(np.stack(pv_rows), dtype=np.float64),
        np.asarray(np.stack(bo_rows), dtype=np.float64),
    )
    return stacked


@dataclass(frozen=True)
class BundleEnsemble:
    """The particle evidence of ONE completed run: zones, weights and s labels.

    This is the single place a finished SMC run is turned into an ensemble, so the
    operational products and any truth-conditional score are computed from the SAME
    validated forwards and the SAME weight vector, read once.
    """

    zones: ParticleZones
    weights: F64
    s_labels: tuple[int, ...]


def bundle_ensemble(
    bundle: Mapping[str, Any],
    checkpoint_manifest: Mapping[str, Any],
    paths: ProjectPaths,
    *,
    support: ZoneSupport,
    time_index: int,
) -> BundleEnsemble:
    """Load one COMPLETE beta=1 run's particle evidence, validated against its bundle.

    The posterior bundle names the distinct physical forwards; the checkpoint maps
    every particle to ITS OWN forward and carries its weights. Every particle forward
    must be one the bundle published — a forward from outside the bundle is refused.
    """
    if bundle.get("algorithm_status") != "COMPLETE" or float(bundle.get("beta", 0.0)) != 1.0:
        raise ValueError(
            "operational products require a COMPLETE beta=1 posterior bundle; partial "
            "ensembles are diagnostics and never a posterior"
        )
    published = {str(ref["artifact_id"]) for ref in bundle.get("physical_state_refs", ())}
    state = checkpoint_manifest["state"]
    particles = list(state["particles"])
    if not particles:
        raise ValueError("the checkpoint carries no particles")
    refs: list[ArtifactRef] = []
    labels: list[int] = []
    for particle in particles:
        evaluation = particle["evaluation"]
        raw = evaluation.get("forward_ref")
        if raw is None:
            raise ValueError(
                f"particle {particle.get('particle_id')} carries no physical forward: an "
                "ensemble without physical states has no state products"
            )
        ref = ArtifactRef.model_validate(raw)
        if ref.artifact_id not in published:
            raise ValueError(
                f"particle forward {ref.artifact_id} is not among the bundle's "
                "physical_state_refs: the state evidence and the ensemble disagree"
            )
        refs.append(ref)
        labels.append(int(evaluation["theta"]["s"]))
    log_weights = np.asarray(state["log_weights"], dtype=np.float64)
    if log_weights.shape != (len(particles),):
        raise ValueError(f"{log_weights.shape} log weights for {len(particles)} particles")
    weights = np.exp(log_weights - log_weights.max())
    so, pv, bo = load_particle_states(refs, paths, time_index)
    zones = aggregate_particle_zones(so=so, pv=pv, bo=bo, support=support)
    return BundleEnsemble(
        zones=zones,
        weights=np.asarray(weights, dtype=np.float64),
        s_labels=tuple(labels),
    )


def operational_products_from_bundle(
    bundle: Mapping[str, Any],
    checkpoint_manifest: Mapping[str, Any],
    paths: ProjectPaths,
    *,
    support: ZoneSupport,
    time_index: int,
    admissible_s: Sequence[int] | None = None,
    probabilities: tuple[float, ...] = DEFAULT_QUANTILE_PROBABILITIES,
) -> EnsembleStateProducts:
    """Build the month-36 (or any report time) operational products WITHOUT truth."""
    ensemble = bundle_ensemble(
        bundle, checkpoint_manifest, paths, support=support, time_index=time_index
    )
    return ensemble_state_products(
        ensemble.zones,
        ensemble.weights,
        support=support,
        s_labels=ensemble.s_labels,
        admissible_s=admissible_s,
        probabilities=probabilities,
    )


# --------------------------------------------------------------------------------------
# state maps: truth, mean, median, width, error, one representative particle
# --------------------------------------------------------------------------------------


def _panel(
    axis: Axes,
    values: F64,
    title: str,
    extent: tuple[float, float, float, float],
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    image = axis.imshow(
        values,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    axis.set_title(title)
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.figure.colorbar(image, ax=axis, shrink=0.78)


def ensemble_state_figure(
    products: EnsembleStateProducts,
    support: ZoneSupport,
    *,
    shape: tuple[int, int, int],
    extent_m: tuple[float, float, float],
    layer: int,
    truth_zone_so: F64 | None = None,
) -> Figure:
    """One layer's state maps in PHYSICAL coordinates (metres on both axes).

    The truth and error panels exist only when truth is supplied: the figure a run
    publishes operationally (no truth opened) is the four mean/median/width/
    representative panels.
    """
    probabilities = products.quantile_probabilities
    median_index = min(range(len(probabilities)), key=lambda i: abs(probabilities[i] - 0.5))
    median = products.so_quantiles[:, median_index]
    width = products.so_quantiles[:, -1] - products.so_quantiles[:, 0]
    panels: list[tuple[F64, str, str, dict[str, Any]]] = [
        (products.posterior_mean_so, "Posterior mean So", "viridis", {"vmin": 0.0, "vmax": 1.0}),
        (median, "Median So (Q50)", "viridis", {"vmin": 0.0, "vmax": 1.0}),
        (width, "Width (Q95 - Q05)", "magma", {}),
        (
            products.representative_zone_so,
            "Representative particle So",
            "viridis",
            {"vmin": 0.0, "vmax": 1.0},
        ),
    ]
    if truth_zone_so is not None:
        truth = np.asarray(truth_zone_so, dtype=np.float64)
        error = products.posterior_mean_so - truth
        limit = max(float(np.nanmax(np.abs(error))), 1.0e-6)
        panels[0:0] = [
            (truth, "Truth So", "viridis", {"vmin": 0.0, "vmax": 1.0}),
            (error, "Signed error (mean - truth)", "coolwarm", {"vmin": -limit, "vmax": limit}),
        ]
    fx, fy, _fz = extent_m
    figure, axes = plt.subplots(
        1, len(panels), figsize=(3.2 * len(panels), 3.4), constrained_layout=True
    )
    for axis, (zone_values, title, cmap, kwargs) in zip(axes, panels, strict=True):
        _panel(
            axis,
            zone_value_map(support, zone_values, shape)[layer],
            title,
            (0.0, fx, 0.0, fy),
            cmap,
            **kwargs,
        )
    figure.suptitle(
        f"layer {layer} | {support.kind} | estimator {products.estimator} | "
        f"representative rule {products.representative_rule}"
    )
    return figure


def render_ensemble_state_maps(
    products: EnsembleStateProducts,
    support: ZoneSupport,
    *,
    shape: tuple[int, int, int],
    extent_m: tuple[float, float, float],
    output_dir: Path,
    truth_zone_so: F64 | None = None,
) -> tuple[Path, ...]:
    """Save one PNG per layer; returns the paths in layer order."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for layer in range(shape[2]):
        figure = ensemble_state_figure(
            products,
            support,
            shape=shape,
            extent_m=extent_m,
            layer=layer,
            truth_zone_so=truth_zone_so,
        )
        path = output_dir / f"ensemble_state_layer{layer}.png"
        figure.savefig(path, dpi=150, bbox_inches="tight", metadata={"Software": "so-recon"})
        plt.close(figure)
        paths.append(path)
    return tuple(paths)


__all__ = [
    "B0_PROVENANCE",
    "B0Selection",
    "BundleEnsemble",
    "RESIDUAL_MOVEMENT_NOTE",
    "RunDiagnostics",
    "average_over_worlds",
    "b0_from_prior_start_checkpoint",
    "bundle_ensemble",
    "comparison_row",
    "ensemble_state_figure",
    "load_particle_states",
    "operational_products_from_bundle",
    "render_ensemble_state_maps",
    "run_diagnostics_from_payloads",
]
