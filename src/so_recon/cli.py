"""so-recon command line: manifest | env-report | smoke | the six E01 commands.

This module parses and dispatches. Every command BODY lives elsewhere — `so_recon.smoke`
for the legacy smoke case and `so_recon.simulator.commands` for E01's `forward`,
`forward-resume`, `verify-physics`, `synthetic-p1`, `benchmark-forward` and `e01-report` —
so that adding a command adds a parser here and nothing else.

`--root` and `--config` are GLOBAL and come before the subcommand. That order is part of
the interface every existing invocation was written against and the new commands do not
move it.

Every subcommand runs inside execute_run, so any failure — including a configuration
error or a missing Julia executable — leaves a run.json with status FAIL (invariant I6).

Two cases leave no record, both of them before any run can exist: argparse rejecting the
command line (`so-recon nope`, `--help`), which exits with SystemExit before main() has a
root or a config; and a repository root that cannot be located, where there is nowhere to
write. Everything downstream of those two points is recorded.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from so_recon import SpecVersion
from so_recon.config.load import load_project_config
from so_recon.config.schema import JuliaConfig, ProjectConfig
from so_recon.environment.report import (
    ENVIRONMENT_SCHEMA_VERSION,
    build_environment_stamp,
    collect_environment,
    report_json_bytes,
    write_environment_report,
    write_environment_stamp,
)
from so_recon.inference.commands import run_e02, run_e02_report, run_inverse_budget
from so_recon.paths import ProjectPaths, RepoRootNotFoundError, find_repo_root
from so_recon.registry.hashing import sha256_bytes
from so_recon.registry.run import RUN_RECORD_SCHEMA_VERSION, RunContext, RunStatus
from so_recon.registry.source_manifest import (
    MANIFEST_SCHEMA_VERSION,
    SourceManifestStamp,
    build_source_manifest,
    write_manifest_stamp,
    write_source_manifest,
)
from so_recon.runner import RunRecordUnavailableError, execute_run
from so_recon.simulator.commands import (
    SuiteRunner,
    run_benchmark_forward,
    run_e01_report,
    run_e01_suite,
    run_forward,
    run_forward_resume,
    run_synthetic_p1,
    suite_exit_code,
)
from so_recon.simulator.julia_bridge import JuliaLauncher, default_launcher
from so_recon.smoke import run_smoke

LauncherFactory = Callable[[ProjectPaths, JuliaConfig, str | None], JuliaLauncher]

# SPEC 19.12 wants lineage recorded the same way by every command, so each one declares
# the schemas it actually produces. `manifest` used to leave this empty while `smoke`
# filled it in, which made run records incomparable across commands.
MANIFEST_SCHEMA_VERSIONS = {
    "source_manifest": MANIFEST_SCHEMA_VERSION,
    "run_record": RUN_RECORD_SCHEMA_VERSION,
}
ENV_REPORT_SCHEMA_VERSIONS = {
    "environment_report": ENVIRONMENT_SCHEMA_VERSION,
    "run_record": RUN_RECORD_SCHEMA_VERSION,
}

# One published file per spec version. A 4.0 run must not overwrite the 3.0 manifest E00
# published: the two describe the same sources under different specifications, and the
# historical one is frozen. The manifest schema itself is unchanged (still version 2).
PUBLISHED_MANIFEST_NAMES: dict[SpecVersion, str] = {
    "3.0": "source_manifest.json",
    "4.0": "source_manifest-4.0.json",
}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="so-recon", description="SO-RECON project CLI")
    p.add_argument("--root", type=Path, default=None, help="repository root (default: auto)")
    p.add_argument("--config", type=Path, default=None, help="path to project.yml")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("manifest", help="hash raw sources into reports/manifests/source_manifest.json")
    sub.add_parser("env-report", help="write reports/environment_report.md")
    smoke = sub.add_parser("smoke", help="run deterministic fixture + JutulDarcy smoke case")
    smoke.add_argument(
        "--freeze-expected",
        action="store_true",
        help="write configs/smoke_expected.json from this run",
    )
    smoke.add_argument("--julia", default=None, help="path to julia executable")

    # ---- E01 (plan 12.3). The global options above keep their place; each command below
    # takes only its own arguments, and every PATH is a real path the operator has or a real
    # path a previous command printed — never the name of an artifact that does not exist yet.
    forward = sub.add_parser("forward", help="run one published case to a verified result")
    forward.add_argument("--case", required=True, help="path to a published case manifest")
    forward.add_argument("--julia", default=None, help="path to julia executable")

    resume = sub.add_parser(
        "forward-resume", help="continue a case from a published native checkpoint"
    )
    resume.add_argument("--case", required=True, help="path to the case being continued")
    resume.add_argument("--restart", required=True, help="path to the checkpoint manifest")
    resume.add_argument("--julia", default=None, help="path to julia executable")

    verify = sub.add_parser("verify-physics", help="run registered native physics checks")
    verify.add_argument("--suite", choices=["p0", "p1", "bo"], required=True)
    verify.add_argument("--resume-ledger", type=Path)
    verify.add_argument(
        "--replay",
        action="store_true",
        help=(
            "re-run every group and every forward even when --resume-ledger names a session "
            "that completed them; plan 12.6's explicitly requested replay"
        ),
    )
    verify.add_argument("--julia", default=None, help="path to julia executable")

    synthetic = sub.add_parser("synthetic-p1", help="render, run and publish P1 parent worlds")
    synthetic.add_argument("--seeds", type=int, nargs="+", required=True)
    synthetic.add_argument("--julia", default=None, help="path to julia executable")

    bench = sub.add_parser("benchmark-forward", help="one cold run and N warm repeats")
    bench.add_argument("--case", required=True, help="path to a published case manifest")
    bench.add_argument("--warm-runs", type=int, default=5)
    bench.add_argument("--julia", default=None, help="path to julia executable")

    report = sub.add_parser("e01-report", help="build reports/stages/E01.md from real runs")
    report.add_argument("--runs", nargs="+", required=True, help="run directories to read")

    inverse = sub.add_parser("verify-inverse", help="run one bounded E02 verification suite")
    inverse.add_argument("--suite", choices=["math", "reduced"], required=True)
    inverse.add_argument("--experiment")

    budget = sub.add_parser("inverse-budget", help="forecast E02 physical call counts")
    budget.add_argument("--experiment", required=True)

    p1_inverse = sub.add_parser("inverse-p1", help="run exactly one requested P1 inverse")
    p1_inverse.add_argument("--experiment", required=True)
    p1_inverse.add_argument("--seed", type=int, required=True)
    p1_inverse.add_argument("--particles", type=int, choices=[32, 64], required=True)
    p1_inverse.add_argument("--inference-seed", type=int, required=True)

    inverse_resume = sub.add_parser("inverse-resume", help="resume one E02 SMC checkpoint")
    inverse_resume.add_argument("--checkpoint", type=Path, required=True)

    e02_report = sub.add_parser("e02-report", help="derive E02 status from actual run dirs")
    e02_report.add_argument("--runs", nargs="+", required=True, help="run directories to read")
    return p


def _manifest_body(
    cfg: ProjectConfig, paths: ProjectPaths
) -> Callable[[RunContext, logging.Logger], tuple[RunStatus, list[str]]]:
    def body(ctx: RunContext, log: logging.Logger) -> tuple[RunStatus, list[str]]:
        now = datetime.now(UTC)
        manifest = build_source_manifest(
            cfg.sources,
            paths,
            config_version=cfg.config_version,
            spec_version=cfg.spec_version,
        )
        ref, published = write_source_manifest(
            manifest,
            paths,
            run_dir=ctx.run_dir,
            published_path=paths.manifests / PUBLISHED_MANIFEST_NAMES[cfg.spec_version],
            producer_run_id=ctx.run_id,
            now=now,
        )
        write_manifest_stamp(
            SourceManifestStamp(
                manifest_version=MANIFEST_SCHEMA_VERSION,
                manifest_sha256=ref.sha256,
                run_id=ctx.run_id,
                created_at=now.isoformat(),
                git_commit=ctx.record.git_commit,
                git_dirty=ctx.record.git_dirty,
            ),
            ctx.run_dir / "source_manifest_stamp.json",
        )
        ctx.update(raw_input_hashes=manifest.hashes())
        ctx.add_output("source_manifest", ref)
        log.info("source manifest published to %s", paths.relative(published))
        return "PASS", []

    return body


def _env_report_body(
    paths: ProjectPaths,
) -> Callable[[RunContext, logging.Logger], tuple[RunStatus, list[str]]]:
    def body(ctx: RunContext, log: logging.Logger) -> tuple[RunStatus, list[str]]:
        now = datetime.now(UTC)
        report = collect_environment(paths)
        md_ref, json_ref = write_environment_report(
            report,
            paths,
            run_dir=ctx.run_dir,
            md_path=paths.reports / "environment_report.md",
            json_path=paths.manifests / "environment.json",
            producer_run_id=ctx.run_id,
            now=now,
        )
        write_environment_stamp(
            build_environment_stamp(
                paths,
                report_sha256=sha256_bytes(report_json_bytes(report)),
                run_id=ctx.run_id,
                now=now,
            ),
            ctx.run_dir / "environment_stamp.json",
        )
        ctx.update(
            julia_version=report.julia_manifest_version,
            jutul_version=report.julia_packages.get("Jutul"),
            jutuldarcy_version=report.julia_packages.get("JutulDarcy"),
        )
        ctx.add_output("environment_report", md_ref)
        ctx.add_output("environment_json", json_ref)
        log.info("environment report written")
        return "PASS", []

    return body


def _report(ctx: RunContext, paths: ProjectPaths) -> int:
    print(f"run_id={ctx.run_id} status={ctx.record.status} run_dir={paths.relative(ctx.run_dir)}")
    for note in ctx.record.notes:
        print(f"  {note}")
    return 0 if ctx.record.status == "PASS" else 1


def _record_startup_failure(root: Path, command: str, argv: Sequence[str], exc: Exception) -> int:
    """Configuration could not be loaded: still leave a FAIL record (invariant I6).

    The cause is printed BEFORE the record is opened, so the user learns why the run failed
    even when the artifacts tree is also unusable and `execute_run` raises
    `RunRecordUnavailableError` on its way out.
    """
    paths = ProjectPaths.default(root)
    print(f"config error: {exc}", file=sys.stderr)

    def body(ctx: RunContext, log: logging.Logger) -> tuple[RunStatus, list[str]]:
        return "FAIL", [f"config error: {exc}"]

    ctx = execute_run(command=command, argv=argv, cfg=None, paths=paths, body=body)
    return _report(ctx, paths)


def _e01(
    args: argparse.Namespace,
    cfg: ProjectConfig,
    paths: ProjectPaths,
    full_argv: list[str],
    suite_runner: SuiteRunner | None,
) -> int | None:
    """Dispatch an E01 command, or return None when this is not one.

    Each body returns its own exit code, because for these commands "the run was recorded"
    and "what was asked for happened" are different facts: a forward that published an
    INCOMPLETE_BUDGET result recorded itself perfectly and still did not do what was asked
    (plan 12.1).
    """
    if args.command == "verify-physics":
        ctx = run_e01_suite(
            cfg,
            paths,
            args.suite,
            resume_ledger=args.resume_ledger,
            replay=args.replay,
            argv=full_argv,
            julia=args.julia,
            runner=suite_runner,
        )
        _report(ctx, paths)
        return suite_exit_code(ctx.run_dir)
    outcome = None
    if args.command == "forward":
        outcome = run_forward(cfg, paths, case_path=args.case, argv=full_argv, julia=args.julia)
    elif args.command == "forward-resume":
        outcome = run_forward_resume(
            cfg,
            paths,
            case_path=args.case,
            restart_path=args.restart,
            argv=full_argv,
            julia=args.julia,
        )
    elif args.command == "synthetic-p1":
        outcome = run_synthetic_p1(cfg, paths, seeds=args.seeds, argv=full_argv, julia=args.julia)
    elif args.command == "benchmark-forward":
        outcome = run_benchmark_forward(
            cfg,
            paths,
            case_path=args.case,
            warm_runs=args.warm_runs,
            argv=full_argv,
            julia=args.julia,
        )
    elif args.command == "e01-report":
        outcome = run_e01_report(cfg, paths, run_dirs=args.runs, argv=full_argv)
    if outcome is None:
        return None
    _report(outcome.ctx, paths)
    # A FAIL record is never a zero, whatever the body computed for its own reasons.
    return outcome.exit_code or (0 if outcome.ctx.record.status == "PASS" else 1)


def _e02(
    args: argparse.Namespace,
    cfg: ProjectConfig,
    paths: ProjectPaths,
    full_argv: list[str],
) -> int | None:
    """Dispatch E02 commands while preserving technical and scientific exit boundaries."""
    if args.command == "verify-inverse":
        ctx = run_e02(
            cfg,
            paths,
            suite=args.suite,
            experiment=args.experiment,
            argv=full_argv,
        )
    elif args.command == "inverse-budget":
        ctx = run_inverse_budget(cfg, paths, experiment=args.experiment, argv=full_argv)
    elif args.command == "inverse-p1":
        ctx = run_e02(
            cfg,
            paths,
            suite="p1",
            experiment=args.experiment,
            argv=full_argv,
        )
    elif args.command == "inverse-resume":
        ctx = run_e02(
            cfg,
            paths,
            suite="resume",
            resume=args.checkpoint,
            argv=full_argv,
        )
    elif args.command == "e02-report":
        ctx = run_e02_report(cfg, paths, run_dirs=args.runs, argv=full_argv)
    else:
        return None
    return _report(ctx, paths)


def main(
    argv: Sequence[str] | None = None,
    *,
    launcher_factory: LauncherFactory | None = None,
    suite_runner: SuiteRunner | None = None,
) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(raw_argv)
    full_argv = ["so-recon", *raw_argv]

    try:
        root = args.root.resolve() if args.root else find_repo_root()
    except RepoRootNotFoundError as exc:
        # One of the two record-less paths (the other is argparse rejecting the command
        # line above, which exits before main() gets here). This one is record-less
        # because there is no repository to write the record into.
        print(f"cannot locate repository root: {exc}", file=sys.stderr)
        return 1

    # ONE handler for every path that can open a run, the startup-failure path included:
    # _record_startup_failure calls execute_run too, so guarding only the two dispatch
    # branches would leave "bad config + unusable artifacts tree" as a raw traceback.
    try:
        try:
            cfg = load_project_config(args.config or (root / "configs" / "project.yml"))
            paths = ProjectPaths.from_config(root, cfg.paths)
        # Deliberately broad: any configuration failure must still leave a FAIL record.
        except Exception as exc:
            return _record_startup_failure(root, args.command, full_argv, exc)

        e01 = _e01(args, cfg, paths, full_argv, suite_runner)
        if e01 is not None:
            return e01
        e02 = _e02(args, cfg, paths, full_argv)
        if e02 is not None:
            return e02

        if args.command == "smoke":
            factory = launcher_factory or default_launcher
            ctx = run_smoke(
                cfg=cfg,
                paths=paths,
                argv=full_argv,
                launcher_factory=lambda: factory(paths, cfg.julia, args.julia),
                freeze_expected=args.freeze_expected,
            )
            return _report(ctx, paths)

        manifest = args.command == "manifest"
        body = _manifest_body(cfg, paths) if manifest else _env_report_body(paths)
        schema_versions = MANIFEST_SCHEMA_VERSIONS if manifest else ENV_REPORT_SCHEMA_VERSIONS
        ctx = execute_run(
            command=args.command,
            argv=full_argv,
            cfg=cfg,
            paths=paths,
            body=body,
            schema_versions=schema_versions,
        )
        return _report(ctx, paths)
    except RunRecordUnavailableError as exc:
        print(f"cannot record this run: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
