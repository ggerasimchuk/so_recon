"""E01.12.7 — which published files the stage figures are drawn from.

Discovery, and nothing else: this module opens no figure and draws none. It lives beside
`plots.py` rather than inside `simulator/commands.py` because finding a published forward
is a question about ARTIFACTS, not about running one — and because keeping it here is part
of what lets `commands.py` import the report and the plots at module level instead of
deferring both inside a function to dodge a circular import.

The rule the functions share: a missing input is a figure that is not drawn, never one
drawn from a default.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from so_recon.paths import ProjectPaths
from so_recon.simulator.contracts import ForwardResult
from so_recon.simulator.results import RESULT_FILENAME, load_forward_result
from so_recon.simulator.suite_record import BENCHMARK_FILENAME
from so_recon.synthetic.world_io import SUITE_MANIFEST_FILENAME


def figure_inputs(run_dirs: Sequence[Path], paths: ProjectPaths) -> dict[str, Path | None]:
    """Locate the published files the stage figures are drawn from.

    Only files a real session published, and preferably the ones the CITED runs published:
    `reports/p1_suite_manifest.json` is rewritten by every P1 session, so a report about an
    earlier session must not be illustrated with a later session's world. The manifest is
    used when it names an accepted world; otherwise the cited run directories themselves are
    searched for the largest COMPLETE forward they hold. A missing input is a figure that is
    not drawn, never one drawn from a default.
    """
    found: dict[str, Path | None] = {
        "states": None,
        "monthly": None,
        "balances": None,
        "benchmark": None,
    }
    suite_manifest = paths.reports / SUITE_MANIFEST_FILENAME
    if suite_manifest.is_file():
        payload = json.loads(suite_manifest.read_text(encoding="utf-8"))
        for row in payload.get("rows", []):
            if not row.get("accepted") or not row.get("manifest_path"):
                continue
            manifest = json.loads(
                paths.resolve(str(row["manifest_path"])).read_text(encoding="utf-8")
            )
            outputs = manifest.get("forward_outputs", {})
            state = outputs.get("state.so")
            if state:
                found["states"] = paths.resolve(str(state).split("#")[0])
            for role in ("monthly", "balances"):
                if outputs.get(role):
                    found[role] = paths.resolve(str(outputs[role]))
            break
    if found["states"] is None:
        _biggest_published_forward(run_dirs, paths, found)
    for run_dir in run_dirs:
        candidate = run_dir / BENCHMARK_FILENAME
        if candidate.is_file():
            found["benchmark"] = candidate
    return found


def _with_child_runs(run_dirs: Sequence[Path], paths: ProjectPaths) -> list[Path]:
    """The cited runs plus every run that names one of them as a parent.

    A suite's forwards are recorded as their OWN runs — that is what keeps a continuation
    out of its parent's result directory — so the artifacts a suite produced live beside it
    rather than under it, and a search that only looked inside the cited directories would
    find none of them.
    """
    cited = {run_dir.name for run_dir in run_dirs}
    out = list(run_dirs)
    if not paths.runs.is_dir():
        return out
    for candidate in sorted(paths.runs.iterdir()):
        if candidate.name in cited or not (candidate / "run.json").is_file():
            continue
        try:
            record = json.loads((candidate / "run.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if cited.intersection(record.get("parent_run_ids", ())):
            out.append(candidate)
    return out


def _biggest_published_forward(
    run_dirs: Sequence[Path], paths: ProjectPaths, found: dict[str, Path | None]
) -> None:
    """The largest COMPLETE forward under the cited runs, as the figures' subject.

    Largest by published values — times times cells — because the figure that says most
    about the stage is the one drawn from the longest trajectory it really ran. The record
    is re-read through `load_forward_result`, so what is drawn is bytes that still verify.
    """
    best: tuple[int, ForwardResult] | None = None
    for run_dir in _with_child_runs(run_dirs, paths):
        for record_path in sorted(run_dir.rglob(RESULT_FILENAME)):
            try:
                result = load_forward_result(record_path, paths)
            except Exception:  # a record that no longer verifies is not drawn from
                continue
            state = result.states.get("so")
            if result.status != "COMPLETE" or state is None:
                continue
            size = int(state.shape[0]) * int(state.shape[1])
            if best is None or size > best[0]:
                best = (size, result)
    if best is None:
        return
    result = best[1]
    state = result.states["so"]
    found["states"] = paths.resolve(state.path)
    if result.monthly_path is not None:
        found["monthly"] = paths.resolve(result.monthly_path)
    if result.balances_path is not None:
        found["balances"] = paths.resolve(result.balances_path)


__all__ = ["figure_inputs"]
