"""E00 smoke scenario: deterministic fixture -> case.json -> JutulDarcy -> frozen expectations."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

from so_recon.config.schema import ProjectConfig, StrictModel
from so_recon.environment.report import check_locked_versions, read_locked_versions
from so_recon.paths import ProjectPaths
from so_recon.registry.artifact import register_artifact, write_artifact, write_json_artifact
from so_recon.registry.atomic import write_bytes_atomic
from so_recon.registry.hashing import sha256_bytes
from so_recon.registry.run import RunContext, RunStatus
from so_recon.runner import CommandBody, execute_run
from so_recon.simulator.julia_bridge import (
    JuliaLauncher,
    JuliaSmokeResult,
    run_julia_smoke,
)
from so_recon.synthetic.fixture import (
    CASE_SCHEMA_VERSION,
    FIXTURE_SCHEMA_VERSION,
    build_smoke_case,
    build_smoke_fixture,
    case_bytes,
    write_smoke_fixture,
)

EXPECTED_FILENAME = "smoke_expected.json"
EXPECTED_SCHEMA_VERSION = "2"
SCHEMA_VERSIONS = {
    "smoke_fixture": FIXTURE_SCHEMA_VERSION,
    "smoke_case": CASE_SCHEMA_VERSION,
    "julia_smoke_result": "1",
    "smoke_expected": EXPECTED_SCHEMA_VERSION,
    "run_record": "1",
}

LauncherFactory = Callable[[], JuliaLauncher]


class SmokeExpectation(StrictModel):
    """Frozen expectations. Committed, therefore deterministic: no timestamps (I5)."""

    schema_version: str = EXPECTED_SCHEMA_VERSION
    fixture_content_hash: str
    case_sha256: str
    nx: int
    n_steps: int
    cumulative_oil_m3: float
    cumulative_water_injected_m3: float
    mean_so_final: float
    julia_version: str
    jutul_version: str
    jutuldarcy_version: str


def _close(a: float, b: float, rel_tol: float) -> bool:
    return abs(a - b) <= rel_tol * max(1.0, abs(b))


def compare_with_expected(
    expected: SmokeExpectation,
    *,
    fixture_hash: str,
    case_sha256: str,
    result: JuliaSmokeResult,
    rel_tol: float,
) -> list[str]:
    mismatches: list[str] = []
    for label, got, exp in (
        ("fixture_content_hash", fixture_hash, expected.fixture_content_hash),
        ("case_sha256", case_sha256, expected.case_sha256),
        # A version bump changes the numbers it is compared against, so it is a failure,
        # not a note: reproducibility is only claimed against the locked stack.
        ("julia_version", result.julia_version, expected.julia_version),
        ("jutul_version", result.jutul_version, expected.jutul_version),
        ("jutuldarcy_version", result.jutuldarcy_version, expected.jutuldarcy_version),
    ):
        if got != exp:
            mismatches.append(f"{label}: got {got}, expected {exp}")
    for name in ("nx", "n_steps"):
        got_i, exp_i = getattr(result, name), getattr(expected, name)
        if got_i != exp_i:
            mismatches.append(f"{name}: got {got_i}, expected {exp_i}")
    for name in ("cumulative_oil_m3", "cumulative_water_injected_m3", "mean_so_final"):
        got_f, exp_f = getattr(result, name), getattr(expected, name)
        if not _close(got_f, exp_f, rel_tol):
            mismatches.append(f"{name}: got {got_f!r}, expected {exp_f!r} (rel_tol={rel_tol})")
    return mismatches


def smoke_body(
    *,
    cfg: ProjectConfig,
    paths: ProjectPaths,
    launcher_factory: LauncherFactory,
    freeze_expected: bool,
) -> CommandBody:
    def body(ctx: RunContext, log: logging.Logger) -> tuple[RunStatus, list[str]]:
        now = datetime.now(UTC)

        fixture = build_smoke_fixture(cfg.smoke)
        files = write_smoke_fixture(fixture, ctx.run_dir / "fixture")
        for key, file_path in files.items():
            ctx.add_output(
                f"fixture_{key}",
                register_artifact(
                    file_path,
                    paths,
                    schema_version=FIXTURE_SCHEMA_VERSION,
                    producer_run_id=ctx.run_id,
                    media_type=(
                        "application/vnd.apache.parquet"
                        if file_path.suffix == ".parquet"
                        else "application/json"
                    ),
                    now=now,
                ),
            )
        log.info("fixture built: seed=%d hash=%s", fixture.seed, fixture.content_hash)

        case = build_smoke_case(cfg.smoke, fixture)
        payload = case_bytes(case)
        case_sha = sha256_bytes(payload)
        case_ref = write_artifact(
            ctx.run_dir / "case.json",
            payload,
            paths,
            schema_version=CASE_SCHEMA_VERSION,
            producer_run_id=ctx.run_id,
            media_type="application/json",
            now=now,
        )
        ctx.add_output("case", case_ref)
        log.info("case written: sha256=%s", case_sha)

        # The factory runs here on purpose: a missing Julia must become a FAIL record.
        launcher = launcher_factory()

        julia_out = ctx.run_dir / "julia_smoke.json"
        result = run_julia_smoke(
            launcher,
            paths.resolve(cfg.julia.smoke_script),
            ctx.run_dir / "case.json",
            julia_out,
            expected_input_sha256=case_sha,
        )
        ctx.update(
            julia_version=result.julia_version,
            jutul_version=result.jutul_version,
            jutuldarcy_version=result.jutuldarcy_version,
        )
        ctx.add_output(
            "julia_smoke",
            write_artifact(
                julia_out,
                julia_out.read_bytes(),
                paths,
                schema_version="1",
                producer_run_id=ctx.run_id,
                media_type="application/json",
                parent_artifact_ids=[case_ref.artifact_id],
                now=now,
            ),
        )
        log.info(
            "julia smoke ok: oil=%.6g winj=%.6g mean_so=%.6f (%.1fs)",
            result.cumulative_oil_m3,
            result.cumulative_water_injected_m3,
            result.mean_so_final,
            result.wall_time_s,
        )

        # Drift from the locked Julia stack is a failure (amendment 9).
        lock_mismatches = check_locked_versions(
            read_locked_versions(paths),
            julia_version=result.julia_version,
            jutul_version=result.jutul_version,
            jutuldarcy_version=result.jutuldarcy_version,
        )
        if lock_mismatches:
            for m in lock_mismatches:
                log.error("locked version mismatch: %s", m)
            return "FAIL", lock_mismatches

        expected_path = paths.configs / EXPECTED_FILENAME
        if freeze_expected:
            expectation = SmokeExpectation(
                fixture_content_hash=fixture.content_hash,
                case_sha256=case_sha,
                nx=result.nx,
                n_steps=result.n_steps,
                cumulative_oil_m3=result.cumulative_oil_m3,
                cumulative_water_injected_m3=result.cumulative_water_injected_m3,
                mean_so_final=result.mean_so_final,
                julia_version=result.julia_version,
                jutul_version=result.jutul_version,
                jutuldarcy_version=result.jutuldarcy_version,
            )
            ref = write_json_artifact(
                ctx.run_dir / EXPECTED_FILENAME,
                expectation.model_dump(mode="json"),
                paths,
                schema_version=EXPECTED_SCHEMA_VERSION,
                producer_run_id=ctx.run_id,
                parent_artifact_ids=[case_ref.artifact_id],
                now=now,
            )
            write_bytes_atomic(expected_path, (ctx.run_dir / EXPECTED_FILENAME).read_bytes())
            ctx.add_output("smoke_expected", ref)
            return "PASS", [f"expected frozen to {paths.relative(expected_path)}"]

        if not expected_path.is_file():
            return "FAIL", [
                f"expected file missing: {paths.relative(expected_path)}; "
                "run with --freeze-expected"
            ]

        expected = SmokeExpectation.model_validate(
            json.loads(expected_path.read_text(encoding="utf-8"))
        )
        mismatches = compare_with_expected(
            expected,
            fixture_hash=fixture.content_hash,
            case_sha256=case_sha,
            result=result,
            rel_tol=cfg.smoke.rel_tol,
        )
        if mismatches:
            for m in mismatches:
                log.error("smoke mismatch: %s", m)
            return "FAIL", mismatches
        return "PASS", []

    return body


def run_smoke(
    *,
    cfg: ProjectConfig,
    paths: ProjectPaths,
    argv: Sequence[str],
    launcher_factory: LauncherFactory,
    freeze_expected: bool = False,
) -> RunContext:
    paths.ensure_dirs()
    return execute_run(
        command="smoke",
        argv=argv,
        cfg=cfg,
        paths=paths,
        schema_versions=SCHEMA_VERSIONS,
        body=smoke_body(
            cfg=cfg,
            paths=paths,
            launcher_factory=launcher_factory,
            freeze_expected=freeze_expected,
        ),
    )
