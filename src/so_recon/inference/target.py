"""Evaluate the E02 target through the one verified physical forward path."""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np

from so_recon.geology.density import Density
from so_recon.geology.renderer import build_inverse_case, render_theta
from so_recon.inference.cache import ArtifactCache, forward_key, likelihood_key
from so_recon.inference.contracts import (
    LoglikResult,
    ModelObservations,
    NoiseTheta,
    ObservationBundle,
    PriorContext,
    TargetEvaluation,
    ThetaRecord,
)
from so_recon.observation.bins import BinGrid
from so_recon.observation.history import history_loglik
from so_recon.observation.logs import logs_loglik
from so_recon.observation.predict import predict_observations
from so_recon.paths import ProjectPaths
from so_recon.registry.artifact import ArtifactRef, register_artifact, write_json_artifact
from so_recon.registry.hashing import sha256_file, sha256_json
from so_recon.registry.run import RunContext
from so_recon.simulator.budget import BudgetLedger
from so_recon.simulator.case_io import model_hash_payload
from so_recon.simulator.contracts import ForwardResult, OutputRequest
from so_recon.simulator.forward import (
    BASE_MAX_NONLINEAR_ITERATIONS,
    DEFAULT_MAX_TIMESTEP_DAYS,
    SolverConfig,
    simulate,
)
from so_recon.simulator.results import RESULT_FILENAME, load_forward_result, write_forward_result
from so_recon.simulator.worker import PersistentJuliaWorker

OBSERVATION_OPERATOR_VERSION = "e02-observation-operator-1"


class RunFactory(Protocol):
    def __call__(self, command: str, parent_run_ids: tuple[str, ...]) -> RunContext: ...


SimulateFunction = Callable[..., ForwardResult]
LoadForwardFunction = Callable[[Path, ProjectPaths], ForwardResult]


class ForwardEvaluationError(RuntimeError):
    """A physical attempt did not produce the complete result the target must score."""

    def __init__(self, status: str, reason: str | None, job_id: str) -> None:
        self.status = status
        self.reason = reason
        self.job_id = job_id
        super().__init__(f"forward {job_id} ended as {status}: {reason or 'no reason recorded'}")


def require_complete_forward(result: ForwardResult) -> ForwardResult:
    """Return a complete result; never map a runtime/resource failure to ``log L=-inf``."""
    if result.status != "COMPLETE":
        raise ForwardEvaluationError(result.status, result.reason, result.job_id)
    return result


def _grids(observations: ObservationBundle) -> dict[str, BinGrid]:
    """Reconstruct report centres from the contract's versioned edge partitions."""
    grids: dict[str, BinGrid] = {}
    for group, raw_edges in observations.bin_edges_by_group.items():
        edges = np.asarray(raw_edges, dtype=np.float64)
        centers = np.empty(edges.size - 1, dtype=np.float64)
        centers[0] = edges[0]
        for index in range(1, centers.size):
            centers[index] = 2.0 * edges[index] - centers[index - 1]
        if not math.isclose(centers[-1], edges[-1], rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"bin edges for group {group!r} do not encode reports from 0 to 1: "
                f"reconstructed final centre is {centers[-1]!r}"
            )
        grids[group] = BinGrid(centers=centers, edges=edges)
    return grids


def evaluate_loglik(
    prediction: ModelObservations,
    observations: ObservationBundle,
    noise: NoiseTheta,
) -> LoglikResult:
    """Combine the history factors and coupled log-date factors without hiding either."""
    grids = _grids(observations)
    history = history_loglik(
        observations.history,
        prediction.fw,
        noise,
        grids,
        observation_hash=observations.observation_hash,
    )
    logs = logs_loglik(
        observations.logs,
        prediction,
        noise,
        grids,
        observation_hash=observations.observation_hash,
    )
    terms = (*history.terms, *logs.terms)
    value = -math.inf if any(term == -math.inf for term in terms) else math.fsum(terms)
    return LoglikResult(
        value=value,
        value_in_support=value != -math.inf,
        terms=terms,
        terms_in_support=tuple(term != -math.inf for term in terms),
        n_used=len(terms),
        observation_hash=observations.observation_hash,
    )


def _source_hash(paths: ProjectPaths) -> str:
    """Hash the Python/Julia adapter sources that can change F without changing a case."""
    files = [
        paths.root / "src/so_recon/simulator/case_io.py",
        paths.root / "src/so_recon/simulator/forward.py",
        paths.root / "src/so_recon/simulator/results.py",
    ]
    files.extend(sorted((paths.julia / "adapter").glob("*.jl")))
    files.extend(sorted((paths.julia / "worker").glob("*.jl")))
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"forward source files are missing: {missing}")
    return sha256_json({paths.relative(path): sha256_file(path) for path in files})


def _output_request(observations: ObservationBundle) -> OutputRequest:
    requested = {0.0}
    requested.update(time for row in observations.logs for time in row.date_times_s)
    return OutputRequest(
        state_times_s=tuple(sorted(requested)),
        keep_native_restart=False,
        chunk_months=1,
    )


class PhysicalTarget:
    """``theta -> (log p0, log L, log r)`` with separate immutable F and L caches."""

    def __init__(
        self,
        prior: Density,
        proposal: Density,
        context: PriorContext,
        observations: ObservationBundle,
        worker: PersistentJuliaWorker,
        ledger: BudgetLedger,
        run_factory: RunFactory,
        *,
        parent_run_ids: tuple[str, ...] = (),
        forward_cache: ArtifactCache | None = None,
        likelihood_cache: ArtifactCache | None = None,
        simulate_fn: SimulateFunction = simulate,
        load_forward_fn: LoadForwardFunction = load_forward_result,
        adapter_hash: str | None = None,
    ) -> None:
        self.prior = prior
        self.proposal = proposal
        self.context = context
        self.observations = observations
        self.worker = worker
        self.ledger = ledger
        self.run_factory = run_factory
        self.parent_run_ids = parent_run_ids
        self.paths = worker.paths
        self.forward_cache = forward_cache or ArtifactCache(self.paths, "forward")
        self.likelihood_cache = likelihood_cache or ArtifactCache(self.paths, "likelihood")
        self.simulate_fn = simulate_fn
        self.load_forward_fn = load_forward_fn
        self.solver_config = SolverConfig(
            max_timestep_days=DEFAULT_MAX_TIMESTEP_DAYS,
            max_nonlinear_iterations=BASE_MAX_NONLINEAR_ITERATIONS,
        )
        self.solver_hash = sha256_json(self.solver_config.model_dump(mode="json"))
        self.adapter_hash = adapter_hash or _source_hash(self.paths)
        self.operator_hash = sha256_json(
            {
                "version": OBSERVATION_OPERATOR_VERSION,
                "history": "recursive-history-1",
                "logs": "dated-so-log-1",
            }
        )
        self.observation_semantic_hash = sha256_json(self.observations.model_dump(mode="json"))
        self.fingerprint = sha256_json(
            {
                "kind": "physical-target-1",
                "prior": prior.fingerprint,
                "proposal": proposal.fingerprint,
                "information_hash": context.information_hash,
                "observation_semantic_hash": self.observation_semantic_hash,
                "solver_hash": self.solver_hash,
                "adapter_hash": self.adapter_hash,
                "environment_lock_hash": worker.environment_lock_hash,
                "operator_hash": self.operator_hash,
            }
        )
        self.checkpoint_hashes = {
            "prior_hash": prior.fingerprint,
            "proposal_hash": proposal.fingerprint,
            "basis_hash": context.density_schema.basis_hash,
            "information_hash": context.information_hash,
            "observation_hash": self.observation_semantic_hash,
            "solver_hash": self.solver_hash,
            "adapter_hash": self.adapter_hash,
            "lock_hash": worker.environment_lock_hash,
            "operator_hash": self.operator_hash,
        }

    def _publish_forward(self, result: ForwardResult, ctx: RunContext) -> ArtifactRef:
        path = ctx.run_dir / RESULT_FILENAME
        write_forward_result(result, path)
        ref = register_artifact(
            path,
            self.paths,
            schema_version=result.schema_version,
            producer_run_id=ctx.run_id,
            media_type="application/json",
            now=datetime.now(UTC),
        )
        ctx.add_output("forward_result", ref)
        return ref

    def evaluate(self, theta: ThetaRecord) -> TargetEvaluation:
        """Evaluate once. Any renderer, forward or likelihood exception remains an error."""
        self.context.density_schema.validate_theta(theta)
        log_p0 = self.prior.log_prob(theta)
        log_r = self.proposal.log_prob(theta)
        if not self.observations.history and not self.observations.logs:
            return TargetEvaluation(
                theta=theta,
                log_p0=log_p0,
                log_p0_in_support=log_p0 != -math.inf,
                log_l=0.0,
                log_l_in_support=True,
                log_r=log_r,
                log_r_in_support=log_r != -math.inf,
                forward_ref=None,
                cache_key=sha256_json(
                    {
                        "target": self.fingerprint,
                        "theta": theta.model_dump(mode="json"),
                        "empty_observations": True,
                    }
                ),
            )
        rendered = render_theta(theta, self.context)
        ctx = self.run_factory("inverse-evaluation", self.parent_run_ids)
        try:
            case = build_inverse_case(rendered, self.context, self.paths, ctx)
            request = _output_request(self.observations)
            f_key = forward_key(
                physical={**model_hash_payload(case), "cutoff": case.cutoff},
                solver_hash=self.solver_hash,
                output_request=request.model_dump(mode="json"),
                adapter_hash=self.adapter_hash,
                environment_lock_hash=self.worker.environment_lock_hash,
            )
            forward_ref = self.forward_cache.get(f_key)
            if forward_ref is None:
                result = self.simulate_fn(
                    case,
                    request,
                    worker=self.worker,
                    ctx=ctx,
                    ledger=self.ledger,
                    solver_config=self.solver_config,
                )
                require_complete_forward(result)
                forward_ref = self._publish_forward(result, ctx)
                self.forward_cache.put(f_key, forward_ref)
            else:
                result = require_complete_forward(
                    self.load_forward_fn(self.paths.resolve(forward_ref.path), self.paths)
                )
                ctx.add_output("forward_result.cache_hit", forward_ref)

            noise_hash = sha256_json(rendered.noise.model_dump(mode="json"))
            l_key = likelihood_key(
                forward_hash=f_key,
                observation_hash=self.observation_semantic_hash,
                operator_hash=self.operator_hash,
                noise_hash=noise_hash,
            )
            likelihood_ref = self.likelihood_cache.get(l_key)
            if likelihood_ref is None:
                predicted = predict_observations(result, self.observations, self.paths)
                likelihood = evaluate_loglik(predicted, self.observations, rendered.noise)
                likelihood_ref = write_json_artifact(
                    ctx.run_dir / "likelihood.json",
                    likelihood.model_dump(mode="json"),
                    self.paths,
                    schema_version="e02-loglik-1",
                    producer_run_id=ctx.run_id,
                    parent_artifact_ids=(forward_ref.artifact_id,),
                    now=datetime.now(UTC),
                )
                ctx.add_output("likelihood", likelihood_ref)
                self.likelihood_cache.put(l_key, likelihood_ref)
            else:
                likelihood = LoglikResult.model_validate_json(
                    self.paths.resolve(likelihood_ref.path).read_text(encoding="utf-8")
                )
                ctx.add_output("likelihood.cache_hit", likelihood_ref)

            ctx.finish("PASS")
            return TargetEvaluation(
                theta=theta,
                log_p0=log_p0,
                log_p0_in_support=log_p0 != -math.inf,
                log_l=likelihood.value,
                log_l_in_support=likelihood.value != -math.inf,
                log_r=log_r,
                log_r_in_support=log_r != -math.inf,
                forward_ref=forward_ref,
                cache_key=l_key,
            )
        except BaseException as exc:
            if ctx.record.status == "RUNNING":
                ctx.finish("FAIL", notes=[f"{type(exc).__name__}: {exc}"])
            raise


__all__ = [
    "OBSERVATION_OPERATOR_VERSION",
    "ForwardEvaluationError",
    "PhysicalTarget",
    "evaluate_loglik",
    "require_complete_forward",
]
