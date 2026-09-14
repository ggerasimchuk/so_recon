"""Read-only probe of the machine: what is there, and what this run is using.

This module measures and never decides. It owns every OS-specific call in the resource
control plane, hands the result back as a frozen `ResourceSnapshot`, and leaves every
threshold to `so_recon.simulator.budget`, which takes snapshots as arguments. That split
is what lets an 8 GiB laptop, a full disk and a swapping session be tested exactly, with
no allocation and no sleeping.

Two honesty rules run through it:

* **A sum of RSS is a conservative indicator, not exact unique resident memory.** Pages
  shared between the parent and a Julia worker are counted in both, so a complete sum
  overstates rather than understates what the run occupies. Overstating is the safe
  direction for a guard; calling it "the" memory use of the tree would not be true. The
  claim holds only while the whole tree was read, which is why a tree that could not be
  read does not return a partial sum.
* **An unavailable measurement keeps its own name.** macOS publishes the kernel's memory
  pressure level through a read-only sysctl; other systems do not, and the sysctl can
  fail. That case is `unknown` with the method recorded, never a zero and never `normal`.
  `so_recon.simulator.budget` keeps its available-memory and swap guards working and says
  in its decision that the pressure signal was missing. A process tree that cannot be read
  follows the same rule — `None` with the method, never a zero, because a zero RSS would
  quietly disable both the hard cap and the soft warning while still looking measured.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import platform
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import psutil
from pydantic import Field

from so_recon.config.schema import StrictModel
from so_recon.environment.report import VersionProbe, subprocess_probe
from so_recon.simulator.julia_bridge import JuliaNotFoundError, find_julia

PressureStatus = Literal["normal", "warn", "critical", "unknown"]
ExternalPower = Literal["ac", "battery", "unknown"]

#: The OS boundary of the process-tree measurement, injected so that an unreadable tree
#: can be described in a test instead of needing one this run is forbidden to read.
TreeReader = Callable[[int], list[psutil.Process]]

#: macOS publishes the kernel's own memory-pressure level here. Reading a sysctl by name
#: with a null new-value pointer cannot change anything, which is what makes this probe
#: read-only by construction rather than by convention.
MACOS_PRESSURE_SYSCTL = "kern.memorystatus_vm_pressure_level"

#: The categorical levels the kernel reports. An unrecognised level stays `unknown` with
#: its raw value preserved, rather than being rounded down to "normal".
MACOS_PRESSURE_LEVELS: dict[int, PressureStatus] = {1: "normal", 2: "warn", 4: "critical"}

#: How the accelerator capability is established. E01 does not install PyTorch to find
#: out whether a GPU exists, so the answer is a capability of the host, not of a runtime.
ACCELERATOR_METHOD = "platform/arch for Metal (MPS), nvidia-smi on PATH for CUDA; no PyTorch import"


class MemoryPressure(StrictModel):
    """The OS memory-pressure signal: the categorical result and how it was obtained."""

    status: PressureStatus
    #: Exactly what the OS returned, as text. None when nothing was returned at all.
    raw: str | None
    #: The probe used, or the reason there was none. Always populated.
    method: str = Field(min_length=1)


class ProcessTreeUsage(StrictModel):
    """What this run's process tree occupies, or an explicit statement that it is unknown.

    `rss_bytes`/`cpu_s` are None when the tree could not be read. They are never zero for
    that reason: a zero here would tell the memory guard that the run occupies nothing.
    """

    rss_bytes: int | None = Field(ge=0)
    cpu_s: float | None = Field(ge=0.0)
    #: How many processes were actually read, so a partial view is visible as one.
    sampled: int = Field(ge=0)
    method: str = Field(min_length=1)


class ResourceSnapshot(StrictModel):
    """One instant of the machine, as measured. Carries no thresholds and no verdicts."""

    total_bytes: int = Field(ge=0)
    available_bytes: int = Field(ge=0)
    #: Summed RSS over this process and its descendants: conservative, not exact — and
    #: None when the tree could not be read at all. `process_measurement_method` says
    #: which, the same way `pressure.method` does for an unknown pressure level.
    process_rss_bytes: int | None = Field(ge=0)
    process_cpu_s: float | None = Field(ge=0.0)
    process_measurement_method: str = Field(min_length=1)
    swap_used_bytes: int = Field(ge=0)
    pressure: MemoryPressure
    disk_free_bytes: int = Field(ge=0)
    #: A monotonic clock, so a wall budget cannot be defeated by a system clock change.
    monotonic_s: float = Field(ge=0.0)


class HardwareProfile(StrictModel):
    """What the host is. Recorded per run; never part of a deterministic model hash."""

    arch: str
    os: str
    total_ram_bytes: int = Field(ge=0)
    cpu_physical: int | None
    cpu_logical: int | None
    external_power: ExternalPower
    julia_executable: str | None
    julia_version: str | None
    mps_capable: bool
    cuda_capable: bool
    accelerator_method: str = Field(min_length=1)


def _sysctl_int(name: str) -> int | None:
    """Read one integer sysctl by name, or None if it cannot be read.

    Read-only: `newp` is NULL and `newlen` is 0, so the call has no way to set anything.
    """
    libname = ctypes.util.find_library("c")
    if libname is None:
        return None
    try:
        libc = ctypes.CDLL(libname, use_errno=True)
    except OSError:
        return None
    libc.sysctlbyname.restype = ctypes.c_int
    libc.sysctlbyname.argtypes = [
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    value = ctypes.c_int32()
    size = ctypes.c_size_t(ctypes.sizeof(value))
    rc = libc.sysctlbyname(name.encode("ascii"), ctypes.byref(value), ctypes.byref(size), None, 0)
    return int(value.value) if rc == 0 else None


def classify_pressure(raw: int | None, *, method: str) -> MemoryPressure:
    """Turn a raw kernel level into the categorical result, keeping both.

    Nothing absent becomes a number and no number becomes a reassurance: a level that was
    not read at all is `unknown` with `raw=None`, and a level that was read but is not one
    this code knows is `unknown` with the raw value preserved — never rounded down to
    "normal" on the grounds that it is not one of the alarming ones.
    """
    if raw is None:
        return MemoryPressure(status="unknown", raw=None, method=f"{method} (read failed)")
    status = MACOS_PRESSURE_LEVELS.get(raw)
    if status is None:
        return MemoryPressure(status="unknown", raw=str(raw), method=f"{method} (level not known)")
    return MemoryPressure(status=status, raw=str(raw), method=method)


def read_memory_pressure() -> MemoryPressure:
    """The OS memory-pressure level, or an explicit `unknown` with the reason."""
    system = platform.system()
    if system != "Darwin":
        return MemoryPressure(
            status="unknown",
            raw=None,
            method=f"unavailable: no OS memory-pressure probe implemented for {system}",
        )
    method = f"sysctlbyname:{MACOS_PRESSURE_SYSCTL}"
    return classify_pressure(_sysctl_int(MACOS_PRESSURE_SYSCTL), method=method)


def read_process_tree(pid: int) -> list[psutil.Process]:
    """The process and its descendants. Raises whatever psutil raises."""
    root = psutil.Process(pid)
    return [root, *root.children(recursive=True)]


def process_tree_usage(pid: int, *, tree: TreeReader = read_process_tree) -> ProcessTreeUsage:
    """Summed RSS and CPU seconds over `pid` and its descendants, or an explicit unknown.

    Summing RSS double-counts shared pages, so a successful total is an upper bound on
    what the run occupies rather than its exact unique resident set. That claim only
    holds while every process in the tree was actually read, which is why this
    distinguishes two very different failures:

    * `NoSuchProcess` is the documented race — the process exited between being listed
      and being read, and something that has exited really is using nothing. It is
      skipped, and the total stays a measurement.
    * `AccessDenied`, `ZombieProcess` and anything else mean the tree could NOT be read.
      Returning the partial sum would understate it, and returning zero would tell the
      memory guard that this run occupies nothing — silently disabling both the hard cap
      and the soft warning while the snapshot still looked like a successful measurement.
      So the result is `None` with the method recording what failed, exactly as an
      unavailable memory-pressure level stays `unknown`.

    `ZombieProcess` is caught FIRST because psutil derives it from `NoSuchProcess`, and
    the two mean opposite things here: a zombie's pid still exists and its memory simply
    cannot be read, so catching it as "the process is gone" would put the exact zero this
    function exists to avoid back into the snapshot.

    `tree` is the injected OS boundary: it is what lets the unreadable-tree branch be
    tested without needing a process this test run is not allowed to read.
    """
    method = f"psutil process tree from pid {pid}"
    try:
        processes = tree(pid)
    except psutil.ZombieProcess as exc:
        return ProcessTreeUsage(
            rss_bytes=None,
            cpu_s=None,
            sampled=0,
            method=f"{method}: unreadable ({type(exc).__name__})",
        )
    except psutil.NoSuchProcess:
        # The whole tree is gone. Nothing is a real measurement here, not a substitution.
        return ProcessTreeUsage(
            rss_bytes=0, cpu_s=0.0, sampled=0, method=f"{method}: root process has exited"
        )
    except psutil.Error as exc:
        return ProcessTreeUsage(
            rss_bytes=None,
            cpu_s=None,
            sampled=0,
            method=f"{method}: unreadable ({type(exc).__name__})",
        )
    rss = 0
    cpu_s = 0.0
    sampled = 0
    for proc in processes:
        try:
            with proc.oneshot():
                rss += proc.memory_info().rss
                times = proc.cpu_times()
                cpu_s += times.user + times.system
            sampled += 1
        except psutil.ZombieProcess as exc:
            return ProcessTreeUsage(
                rss_bytes=None,
                cpu_s=None,
                sampled=sampled,
                method=f"{method}: unreadable at pid {proc.pid} ({type(exc).__name__})",
            )
        except psutil.NoSuchProcess:
            continue
        except psutil.Error as exc:
            return ProcessTreeUsage(
                rss_bytes=None,
                cpu_s=None,
                sampled=sampled,
                method=f"{method}: unreadable at pid {proc.pid} ({type(exc).__name__})",
            )
    return ProcessTreeUsage(
        rss_bytes=rss, cpu_s=cpu_s, sampled=sampled, method=f"{method}: {sampled} processes read"
    )


def free_disk_bytes(path: Path) -> int:
    """Free space on the filesystem holding `path`, or the nearest existing ancestor.

    A run directory does not exist until the run claims it, and a guard that refused to
    measure until then would be useless exactly when it matters — before anything is
    written. The last candidate is the filesystem root, so this resolves in practice; if
    even that fails, the OSError is raised rather than turned into a fabricated zero.
    """
    last: OSError | None = None
    for candidate in (path, *path.parents):
        try:
            return shutil.disk_usage(candidate).free
        except OSError as exc:
            last = exc
    raise last if last is not None else OSError(f"cannot measure free space at {path}")


def probe_resources(pid: int | None, output_root: Path) -> ResourceSnapshot:
    """Measure the machine and this run's process tree. Reads only; changes nothing."""
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    usage = process_tree_usage(pid if pid is not None else psutil.Process().pid)
    return ResourceSnapshot(
        total_bytes=memory.total,
        available_bytes=memory.available,
        process_rss_bytes=usage.rss_bytes,
        process_cpu_s=usage.cpu_s,
        process_measurement_method=usage.method,
        swap_used_bytes=swap.used,
        pressure=read_memory_pressure(),
        disk_free_bytes=free_disk_bytes(output_root),
        monotonic_s=time.monotonic(),
    )


def _external_power() -> ExternalPower:
    """Whether the host is on external power. A machine with no battery reports unknown."""
    try:
        battery = psutil.sensors_battery()
    except (AttributeError, NotImplementedError, OSError):
        return "unknown"
    if battery is None:
        return "unknown"
    return "ac" if battery.power_plugged else "battery"


def _julia_version(probe: VersionProbe, exe: Path) -> str | None:
    out = probe([str(exe), "--version"])  # "julia version 1.12.7"
    parts = (out or "").split()
    return parts[2] if len(parts) >= 3 else None


def probe_hardware(*, probe: VersionProbe = subprocess_probe) -> HardwareProfile:
    """Record what the host is: arch, OS, RAM, CPUs, power, Julia and GPU capability."""
    try:
        julia_exe: Path | None = find_julia()
    except JuliaNotFoundError:
        julia_exe = None
    return HardwareProfile(
        arch=platform.machine(),
        os=f"{platform.system()} {platform.release()}",
        total_ram_bytes=psutil.virtual_memory().total,
        cpu_physical=psutil.cpu_count(logical=False),
        cpu_logical=psutil.cpu_count(logical=True),
        external_power=_external_power(),
        julia_executable=str(julia_exe) if julia_exe is not None else None,
        julia_version=_julia_version(probe, julia_exe) if julia_exe is not None else None,
        mps_capable=platform.system() == "Darwin" and platform.machine() == "arm64",
        cuda_capable=shutil.which("nvidia-smi") is not None,
        accelerator_method=ACCELERATOR_METHOD,
    )
