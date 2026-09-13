"""One place that guarantees invariant I6: every command run leaves a run record.

Whatever the body raises — a missing source, a missing Julia executable, a bug — the
traceback goes to run.log, the message goes to run.json notes, and the record is closed
with status FAIL. execute_run never propagates an exception from the body. The only thing
it does raise is RunRecordUnavailableError, when the record itself cannot be opened or
closed — the named boundary of I6.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

from so_recon.config.schema import ProjectConfig
from so_recon.logging_setup import configure_logging
from so_recon.paths import ProjectPaths
from so_recon.registry.run import RunContext, RunStatus

CommandBody = Callable[[RunContext, logging.Logger], tuple[RunStatus, list[str]]]


class RunRecordUnavailableError(RuntimeError):
    """The run record could not be opened, or could not be closed, so it is not usable.

    This is the boundary of invariant I6. I6 promises a FAIL record for any failure that
    happens once a run exists; it cannot promise one when the artifacts tree itself is
    unusable (for example `artifacts/` present as a regular file, a read-only mount, or a
    disk that fills between run start and run finish). Raising a named error at both ends
    keeps those cases a clean, explained exit instead of a raw traceback.
    """


def _finish(ctx: RunContext, status: RunStatus, notes: Sequence[str], *, command: str) -> None:
    """Close the record, translating a write failure into the named boundary error.

    Closing is itself a write, so it can fail for the same reasons opening can (a full
    disk between start and finish, a mount that turned read-only). Left bare, that would
    escape `execute_run` uncaught and reach the user as a raw traceback — a narrow hole in
    invariant I6's promise that the caller always gets an explained exit.
    """
    try:
        ctx.finish(status, notes=list(notes))
    except Exception as exc:
        raise RunRecordUnavailableError(
            f"cannot close the run record for {command!r} at {ctx.run_dir / 'run.json'}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def execute_run(
    *,
    command: str,
    argv: Sequence[str],
    cfg: ProjectConfig | None,
    paths: ProjectPaths,
    body: CommandBody,
    schema_versions: dict[str, str] | None = None,
    parent_run_ids: Sequence[str] = (),
) -> RunContext:
    try:
        # Claim the validated record destination first. A broken reports/interim directory
        # does not prevent writing a FAIL record under an otherwise usable artifacts tree.
        ctx = RunContext.start(
            command=command,
            argv=argv,
            cfg=cfg,
            paths=paths,
            schema_versions=schema_versions,
            parent_run_ids=parent_run_ids,
        )
    except Exception as exc:
        raise RunRecordUnavailableError(
            f"cannot open a run record for {command!r} under {paths.runs}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    log = None
    try:
        log = configure_logging(ctx.run_id, ctx.run_dir / "run.log")
        paths.ensure_dirs()
        status, notes = body(ctx, log)
    # Deliberately broad: invariant I6 says EVERY failure becomes a FAIL record, so nothing
    # may escape here. (No suppression needed here: BLE001 is not in this project's ruleset.)
    except Exception as exc:
        if log is not None:
            log.exception("command %s failed", command)
        _finish(ctx, "FAIL", [f"exception: {type(exc).__name__}: {exc}"], command=command)
    else:
        _finish(ctx, status, notes, command=command)
    finally:
        for handler in logging.getLogger("so_recon").handlers:
            handler.flush()
    return ctx
