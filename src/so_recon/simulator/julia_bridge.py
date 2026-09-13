"""The only module that knows how to launch Julia. E05 builds the simulator adapter on top."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Literal, Protocol

from so_recon.config.schema import JuliaConfig, StrictModel
from so_recon.paths import ProjectPaths

log = logging.getLogger(__name__)

JULIA_ENV_VAR = "SO_RECON_JULIA"


class JuliaNotFoundError(RuntimeError):
    """No usable julia executable was found."""


class JuliaRunError(RuntimeError):
    """Julia failed, timed out, or returned a result that does not match its input."""


def _failure_detail(out_path: Path, stderr: str) -> str:
    """Prefer Julia's own error message over the stderr tail.

    The smoke script catches its own exceptions, writes {"status":"error","message":...}
    to --out and exits 1, so stderr is typically EMPTY and the real cause lives only in
    that file. Reporting the stderr tail alone would hand the operator a FAIL record
    saying nothing at all.
    """
    try:
        payload = json.loads(out_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        payload = None
    if isinstance(payload, dict) and payload.get("message"):
        # Bounded like the stderr tail: the runner persists this string into run.json,
        # and an unbounded message would put arbitrary content into a lineage record.
        return f"julia reported: {str(payload['message'])[:4000]}"
    tail = stderr[-4000:].strip()
    return f"stderr tail:\n{tail}" if tail else "julia produced no stderr and no error message"


def _is_executable(p: Path) -> bool:
    return p.is_file() and os.access(p, os.X_OK)


def find_julia(explicit: str | None = None) -> Path:
    """Locate the julia executable. A named interpreter is authoritative, never a hint.

    `--julia <path>` and `$SO_RECON_JULIA` both name ONE specific interpreter, so a
    value that is not a usable executable is an error rather than a reason to search on.
    Falling through would let `--julia /opt/julia-1.11/bin/juli` (a typo) run the julia
    that happens to be on PATH and record ITS version in run.json while argv records the
    1.11 request — false provenance, exactly what E00 exists to prevent. The two are
    treated identically because they are the same act: an operator pointing the run at a
    chosen interpreter; a stale export is as dangerous as a mistyped flag, and more so,
    because nothing on the command line reveals it.

    Only the unnamed fallbacks (PATH, then juliaup) are a search, and there a missing
    candidate legitimately means "try the next one".
    """
    named = (("--julia", explicit), (f"${JULIA_ENV_VAR}", os.environ.get(JULIA_ENV_VAR)))
    for source, value in named:
        if value:
            candidate = Path(value)
            if not _is_executable(candidate):
                raise JuliaNotFoundError(
                    f"{source} names {value!r}, which is not an executable file; "
                    "refusing to fall back to another julia"
                )
            return candidate
    which = shutil.which("julia")
    if which and _is_executable(Path(which)):
        return Path(which)
    juliaup = Path.home() / ".juliaup" / "bin" / "julia"
    if _is_executable(juliaup):
        return juliaup
    raise JuliaNotFoundError(
        f"julia executable not found; install via juliaup or set {JULIA_ENV_VAR}"
    )


class JuliaLauncher(Protocol):
    def launch(self, script: Path, args: list[str], out_path: Path) -> None: ...


class SubprocessJuliaLauncher:
    def __init__(self, julia_exe: Path, project: Path, timeout_s: int) -> None:
        self.julia_exe = julia_exe
        self.project = project
        self.timeout_s = timeout_s

    def launch(self, script: Path, args: list[str], out_path: Path) -> None:
        cmd = [
            str(self.julia_exe),
            f"--project={self.project}",
            "--startup-file=no",
            str(script),
            *args,
            "--out",
            str(out_path),
        ]
        log.info("launching julia: %s", " ".join(cmd[:4]))
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, check=False, timeout=self.timeout_s
            )
        except subprocess.TimeoutExpired as exc:
            raise JuliaRunError(f"julia timed out after {self.timeout_s}s") from exc
        if proc.returncode != 0:
            raise JuliaRunError(
                f"julia exited with {proc.returncode}; {_failure_detail(out_path, proc.stderr)}"
            )


class JuliaSmokeResult(StrictModel):
    status: Literal["ok"]
    input_sha256: str
    case_schema_version: str
    julia_version: str
    jutuldarcy_version: str
    jutul_version: str
    nx: int
    n_steps: int
    cumulative_oil_m3: float
    cumulative_water_injected_m3: float
    mean_so_final: float
    wall_time_s: float


def run_julia_smoke(
    launcher: JuliaLauncher,
    script: Path,
    case_path: Path,
    out_path: Path,
    *,
    expected_input_sha256: str,
) -> JuliaSmokeResult:
    launcher.launch(script, ["--case", str(case_path)], out_path)
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    if payload.get("status") != "ok":
        raise JuliaRunError(f"julia smoke failed: {payload.get('message', payload)}")
    result = JuliaSmokeResult.model_validate(payload)
    if result.input_sha256 != expected_input_sha256:
        raise JuliaRunError(
            "julia read a different input: "
            f"input_sha256 {result.input_sha256} != expected {expected_input_sha256}"
        )
    return result


def default_launcher(
    paths: ProjectPaths, cfg: JuliaConfig, julia_exe: str | None = None
) -> SubprocessJuliaLauncher:
    return SubprocessJuliaLauncher(
        julia_exe=find_julia(julia_exe),
        project=paths.resolve(cfg.project),
        timeout_s=cfg.timeout_s,
    )
