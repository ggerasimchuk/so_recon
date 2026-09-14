"""E01.12.7 — the stage figures, drawn from published files and from nothing else.

Four figures, and each one opens the artifact a real forward wrote: `states.h5` for the
two-layer truth, `monthly.parquet` for the production and injection volumes,
`balances.parquet` for the component residuals, and the benchmark record for the cold/warm
and memory panel. Nothing here recomputes physics, re-runs a case or draws a curve from a
number typed into this file.

Three rules the module holds.

**Every caption says what the picture is.** `CAPTION` below is stamped on every figure:
synthetic truth, exploratory split, educational oil-water fluids. A saturation map with no
caption is indistinguishable from a field result, and this project publishes neither field
results nor pictures that could be mistaken for them.

**There is no uncertainty figure.** E01 computes no posterior, so it draws no credible
interval, no ensemble spread and no error bar around a truth — a band drawn where nothing
was inferred is a claim nobody made.

**The extremes are checked before the axes are drawn.** `_finite_range` refuses a field that
is not finite and returns the real minimum and maximum, which the colour scale is then fixed
to. A plot whose colour limits were auto-chosen hides exactly the excursion a reader opens it
to find.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
import pyarrow.parquet as pq
from numpy.typing import NDArray

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (the backend must be chosen before pyplot)

#: Stamped on every figure this module produces.
CAPTION = "synthetic truth; exploratory; educational OW"

#: Where the stage's figures are published, relative to the repository root.
FIGURES_RELDIR = "reports/figures"

SECONDS_PER_DAY = 86400.0


class PlotInputError(RuntimeError):
    """A figure was asked for from a file that cannot support it."""


def _finite_range(values: NDArray[np.float64], what: str) -> tuple[float, float]:
    """The real extremes, having proved there are any. Never an auto-scaled guess."""
    if values.size == 0:
        raise PlotInputError(f"{what}: nothing to plot")
    if not np.isfinite(values).all():
        raise PlotInputError(f"{what}: the published field is not finite")
    return float(values.min()), float(values.max())


def _caption(fig: Any, extra: str = "") -> None:
    """Stamp the caption BELOW the figure, where it cannot cover a panel.

    The text is placed just outside the figure and `_save` writes with a tight bounding
    box, which grows the saved area to include every artist. Placing it inside would put it
    over the bottom row of a multi-panel figure, and shrinking the layout to make room
    clipped the title instead.
    """
    text = CAPTION if not extra else f"{CAPTION} — {extra}"
    fig.text(0.5, -0.01, text, ha="center", va="top", fontsize=8, color="#444444")


def _save(fig: Any, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return out_path


def _read_states(path: Path) -> dict[str, NDArray[np.float64]]:
    with h5py.File(path, "r") as handle:
        return {
            name: np.asarray(handle[name][...], dtype=np.float64)
            for name in ("so", "sw", "pressure_pa", "pore_volume_m3")
            if name in handle
        }


#: The dataset `results.write_forward_outputs` publishes the time axis under.
TIME_DATASET = "time_s"


def _read_times(path: Path) -> NDArray[np.float64]:
    with h5py.File(path, "r") as handle:
        if TIME_DATASET in handle:
            return np.asarray(handle[TIME_DATASET][...], dtype=np.float64)
        raise PlotInputError(
            f"{path}: the states file carries no {TIME_DATASET!r} dataset, so its columns "
            "cannot be labelled with the times they are"
        )


def plot_two_layer_truth(
    states_path: Path, out_path: Path, *, shape: tuple[int, int, int], title: str
) -> Path:
    """True oil saturation of the two-layer box at every published report time.

    One row per layer, one column per published state. The colour scale is FIXED to the real
    extremes of the whole published field, so the four columns are comparable with each other
    and a reader can see the flood advance rather than four independently normalised maps.
    """
    states = _read_states(states_path)
    if "so" not in states:
        raise PlotInputError(f"{states_path}: no published oil saturation to plot")
    so = states["so"]
    nx, ny, nz = shape
    if so.shape[1] != nx * ny * nz:
        raise PlotInputError(
            f"{states_path}: {so.shape[1]} cells published, {nx * ny * nz} in the declared grid"
        )
    times = _read_times(states_path)
    low, high = _finite_range(so, "so")
    n_times = so.shape[0]
    fig, axes = plt.subplots(
        nz, n_times, figsize=(3.0 * n_times, 3.0 * nz), squeeze=False, constrained_layout=True
    )
    image = None
    for layer in range(nz):
        for index in range(n_times):
            field = so[index].reshape(nz, ny, nx)[layer]
            axis = axes[layer][index]
            image = axis.imshow(field, origin="lower", vmin=low, vmax=high, cmap="viridis")
            axis.set_xticks([])
            axis.set_yticks([])
            if layer == 0:
                axis.set_title(f"t = {times[index] / SECONDS_PER_DAY:.0f} d", fontsize=9)
            if index == 0:
                axis.set_ylabel(f"layer {layer}", fontsize=9)
    fig.suptitle(f"{title} — true oil saturation (So in [{low:.3f}, {high:.3f}])", fontsize=11)
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.8, label="So [-]")
    _caption(fig, "no posterior exists for this world; nothing here is an uncertainty band")
    return _save(fig, out_path)


def plot_monthly_volumes(monthly_path: Path, out_path: Path, *, title: str) -> Path:
    """Monthly oil production, water production and water injection, from the published table."""
    table = pq.read_table(monthly_path).to_pylist()
    if not table:
        raise PlotInputError(f"{monthly_path}: the monthly table is empty")
    months = sorted({int(row["month_index"]) for row in table})
    series: dict[str, NDArray[np.float64]] = {}
    for column, label in (
        ("oil_prod_m3_sc", "oil produced"),
        ("water_prod_m3_sc", "water produced"),
        ("water_inj_m3_sc", "water injected"),
    ):
        if column not in table[0]:
            raise PlotInputError(f"{monthly_path}: no column {column!r}")
        values = np.zeros(len(months), dtype=np.float64)
        for row in table:
            values[months.index(int(row["month_index"]))] += float(row[column])
        series[label] = values
    fig, axis = plt.subplots(figsize=(9.0, 4.0), constrained_layout=True)
    for label, values in series.items():
        _finite_range(values, label)
        axis.plot(months, values, marker="o", markersize=3, label=label)
    axis.set_xlabel("month index")
    axis.set_ylabel("standard volume [m3_sc]")
    axis.set_title(f"{title} — monthly volumes")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=9)
    _caption(fig)
    return _save(fig, out_path)


def plot_component_residuals(balances_path: Path, out_path: Path, *, title: str) -> Path:
    """Both published balance statements, per component, against the gate they were scored on.

    `balances.parquet` carries TWO balances — the whole model against the surface flux and
    the reservoir against the connection flux — and one row per (balance, component). Reading
    it as one row per component would score half the table, so the balance label is an axis
    here rather than something a reader has to assume.
    """
    rows = pq.read_table(balances_path).to_pylist()
    if not rows:
        raise PlotInputError(f"{balances_path}: the balance table is empty")
    labels = [f"{row['balance']}\n{row['component']}" for row in rows]
    cumulative = np.asarray([float(row["cumulative_relative"]) for row in rows])
    median = np.asarray([float(row["median_step_relative"]) for row in rows])
    _finite_range(cumulative, "cumulative_relative")
    _finite_range(median, "median_step_relative")
    x = np.arange(len(rows), dtype=np.float64)
    fig, axis = plt.subplots(figsize=(9.0, 4.2), constrained_layout=True)
    axis.bar(x - 0.2, np.maximum(cumulative, 1e-18), width=0.4, label="cumulative relative")
    axis.bar(x + 0.2, np.maximum(median, 1e-18), width=0.4, label="median step relative")
    axis.set_yscale("log")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, fontsize=8)
    axis.set_ylabel("relative residual [-]")
    axis.set_title(f"{title} — component balance residuals (both published statements)")
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend(fontsize=9)
    _caption(fig, f"worst cumulative {cumulative.max():.3g}, worst median step {median.max():.3g}")
    return _save(fig, out_path)


def plot_cost(attempts: Sequence[Mapping[str, Any]], out_path: Path, *, title: str) -> Path:
    """Cold against warm wall time, and the peak process-tree RSS beside it.

    Failed attempts keep their bar and their spent time (plan 12.5): removing them would
    make the warm distribution look better than the session was.
    """
    if not attempts:
        raise PlotInputError("no attempts to plot")
    index = np.arange(len(attempts), dtype=np.float64)
    wall = np.asarray([float(a["wall_s"]) for a in attempts])
    rss = np.asarray([float(a.get("peak_rss_bytes", 0)) / 1024**3 for a in attempts])
    colours = [
        "#1f77b4" if a.get("kind") == "warm" else "#d62728" if a.get("kind") == "cold" else "#999"
        for a in attempts
    ]
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(9.0, 5.5), sharex=True, constrained_layout=True
    )
    top.bar(index, wall, color=colours)
    top.set_ylabel("wall [s]")
    top.set_title(f"{title} — cold (red) against warm (blue) repeats")
    top.grid(True, axis="y", alpha=0.3)
    bottom.plot(index, rss, marker="o", color="#2ca02c")
    bottom.set_ylabel("peak tree RSS [GiB]")
    bottom.set_xlabel("attempt")
    bottom.grid(True, alpha=0.3)
    bottom.set_xticks(index)
    bottom.set_xticklabels(
        [f"{a.get('kind', '?')}\n{a.get('status', '?')}" for a in attempts], fontsize=7
    )
    _caption(fig, "five warm repeats are an exploratory tail, not a service level")
    return _save(fig, out_path)


def write_e01_figures(
    figures_dir: Path,
    *,
    states_path: Path | None = None,
    grid_shape: tuple[int, int, int] = (16, 16, 2),
    monthly_path: Path | None = None,
    balances_path: Path | None = None,
    benchmark_path: Path | None = None,
    title: str = "E01 P1 world",
) -> tuple[Path, ...]:
    """Draw whichever figures the published files support, and say which were drawn.

    A missing artifact is a figure that is not drawn, never a figure drawn from a default.
    """
    drawn: list[Path] = []
    if states_path is not None and states_path.is_file():
        drawn.append(
            plot_two_layer_truth(
                states_path,
                figures_dir / "e01_true_so.png",
                shape=grid_shape,
                title=title,
            )
        )
    if monthly_path is not None and monthly_path.is_file():
        drawn.append(
            plot_monthly_volumes(monthly_path, figures_dir / "e01_monthly.png", title=title)
        )
    if balances_path is not None and balances_path.is_file():
        drawn.append(
            plot_component_residuals(balances_path, figures_dir / "e01_balances.png", title=title)
        )
    if benchmark_path is not None and benchmark_path.is_file():
        payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
        attempts = payload.get("attempts")
        if isinstance(attempts, list) and attempts:
            drawn.append(plot_cost(attempts, figures_dir / "e01_cost.png", title=title))
    return tuple(drawn)
