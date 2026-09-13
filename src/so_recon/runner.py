"""One place that guarantees invariant I6: every command run leaves a run record.

Whatever the body raises — a missing source, a missing Julia executable, a bug — the
traceback goes to run.log, the message goes to run.json notes, and the record is closed
with status FAIL. execute_run never propagates an exception from the body.
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
    """The run could not be opened at all, so no FAIL record can be written.

    This is the boundary of invariant I6. I6 promises a FAIL record for any failure that
    happens once a run exists; it cannot promise one when the artifacts tree itself is
    unusable (for example `reports/` present as a regular file, or a read-only mount).
    Raising a named error here keeps that case a clean, explained exit instead of a raw
    traceback.
    """


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
        # Owned here, not by callers: creating the runtime directories is itself a way the
        # run can fail before any record exists, and it must not escape as a raw traceback.
        paths.ensure_dirs()
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
    log = configure_logging(ctx.run_id, ctx.run_dir / "run.log")
    try:
        status, notes = body(ctx, log)
    # Deliberately broad: invariant I6 says EVERY failure becomes a FAIL record, so nothing
    # may escape here. (No suppression needed here: BLE001 is not in this project's ruleset.)
    except Exception as exc:
        log.exception("command %s failed", command)
        ctx.finish("FAIL", notes=[f"exception: {type(exc).__name__}: {exc}"])
    else:
        ctx.finish(status, notes=notes)
    finally:
        for handler in logging.getLogger("so_recon").handlers:
            handler.flush()
    return ctx
