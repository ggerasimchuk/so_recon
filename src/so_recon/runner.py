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
    ctx = RunContext.start(
        command=command,
        argv=argv,
        cfg=cfg,
        paths=paths,
        schema_versions=schema_versions,
        parent_run_ids=parent_run_ids,
    )
    log = configure_logging(ctx.run_id, ctx.run_dir / "run.log")
    try:
        status, notes = body(ctx, log)
    except Exception as exc:  # noqa: BLE001 - deliberate: every failure becomes a FAIL record
        log.exception("command %s failed", command)
        ctx.finish("FAIL", notes=[f"exception: {type(exc).__name__}: {exc}"])
    else:
        ctx.finish(status, notes=notes)
    finally:
        for handler in logging.getLogger("so_recon").handlers:
            handler.flush()
    return ctx
