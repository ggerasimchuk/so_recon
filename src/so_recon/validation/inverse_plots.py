"""Static scientific E02 figures rendered only from supplied numeric products."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from matplotlib.figure import Figure

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _array(products: dict[str, Any], key: str, *, ndim: int) -> np.ndarray:
    values = np.asarray(products[key], dtype=np.float64)
    if values.ndim != ndim or not np.isfinite(values).all():
        raise ValueError(f"products[{key!r}] must be a finite {ndim}D array, got {values.shape}")
    return values


def _save(fig: Figure, path: Path) -> Path:
    fig.savefig(path, dpi=150, bbox_inches="tight", metadata={"Software": "so-recon"})
    plt.close(fig)
    return path


def _saturation(products: dict[str, Any], output_dir: Path) -> Path:
    truth = _array(products, "true_so", ndim=2)
    median = _array(products, "median_so", ndim=2)
    q05 = _array(products, "q05_so", ndim=2)
    q95 = _array(products, "q95_so", ndim=2)
    prior_width = _array(products, "prior_width", ndim=2)
    if not all(array.shape == truth.shape for array in (median, q05, q95, prior_width)):
        raise ValueError("all saturation products must share the same report support")
    width = q95 - q05
    ratio = np.divide(
        prior_width,
        width,
        out=np.full_like(width, np.nan),
        where=width > 0.0,
    )
    error = median - truth
    error_limit = max(float(np.max(np.abs(error))), 1.0e-6)
    panels: list[tuple[np.ndarray, str, str, dict[str, float]]] = [
        (truth, "True So", "viridis", {"vmin": 0.0, "vmax": 1.0}),
        (median, "Weighted median So", "viridis", {"vmin": 0.0, "vmax": 1.0}),
        (error, "Signed error", "coolwarm", {"vmin": -error_limit, "vmax": error_limit}),
        (width, "Q95 - Q05", "magma", {}),
        (ratio, "Prior / final width", "cividis", {}),
    ]
    if "physical_realization_so" in products:
        realization = _array(products, "physical_realization_so", ndim=2)
        if realization.shape != truth.shape:
            raise ValueError("physical realization must share the fixed report support")
        panels.insert(
            2,
            (
                realization,
                "Physical realization nearest median",
                "viridis",
                {"vmin": 0.0, "vmax": 1.0},
            ),
        )
    fig, axes = plt.subplots(
        1, len(panels), figsize=(3.2 * len(panels), 3.4), constrained_layout=True
    )
    for axis, (values, title, cmap, kwargs) in zip(axes, panels, strict=True):
        image = axis.imshow(values, origin="lower", cmap=cmap, aspect="equal", **kwargs)
        axis.set_title(title)
        axis.set_xlabel("report x index")
        axis.set_ylabel("report y index")
        fig.colorbar(image, ax=axis, shrink=0.78)
    beta = float(products.get("beta_final", 1.0))
    state = "POSTERIOR" if beta == 1.0 else f"INTERMEDIATE beta={beta:.3g}"
    fig.suptitle(
        f"{state} | {products.get('layer_label', 'layer')} | "
        f"{products.get('date_label', 'date')} | saturation [1]"
    )
    return _save(fig, output_dir / "inverse_saturation.png")


def _watercut(products: dict[str, Any], output_dir: Path) -> Path:
    observed = _array(products, "watercut_observed", ndim=1)
    predictive = _array(products, "watercut_predictive", ndim=2)
    if predictive.shape[1] != observed.size:
        raise ValueError("predictive watercut time axis must match observed bins")
    months = np.arange(observed.size)
    fig, axis = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    for row in predictive:
        axis.plot(months, row, color="tab:blue", alpha=0.25, linewidth=1)
    axis.plot(months, observed, "o", color="black", label="observed rounded bin centre")
    axis.set(xlabel="calendar month", ylabel="producer watercut [1]", ylim=(-0.02, 1.02))
    axis.legend()
    axis.grid(alpha=0.2)
    return _save(fig, output_dir / "inverse_watercut.png")


def _diagnostics(products: dict[str, Any], output_dir: Path) -> Path:
    beta = _array(products, "beta", ndim=1)
    cess = _array(products, "cess", ndim=1)
    ess = _array(products, "ess", ndim=1)
    ancestors = _array(products, "ancestors", ndim=1)
    wall = _array(products, "cost_wall_s", ndim=1)
    ram = _array(products, "cost_ram_bytes", ndim=1)
    if not all(values.size == beta.size for values in (cess, ess, ancestors, wall, ram)):
        raise ValueError("SMC diagnostic products must share a phase axis")
    phase = np.arange(beta.size)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    axes[0].plot(phase, beta, label="beta")
    axes[0].plot(phase, cess / max(float(cess.max()), 1.0), label="CESS / max")
    axes[0].plot(phase, ess / max(float(ess.max()), 1.0), label="ESS / max")
    axes[0].plot(phase, ancestors / max(float(ancestors.max()), 1.0), label="ancestors / max")
    axes[0].set(xlabel="SMC phase", ylabel="dimensionless diagnostic")
    axes[0].legend()
    cost_axis = axes[1]
    cost_axis.plot(phase, wall, color="tab:green", label="cumulative wall [s]")
    cost_axis.set(xlabel="SMC phase", ylabel="cumulative wall [s]")
    ram_axis = cost_axis.twinx()
    ram_axis.plot(phase, ram / 1024**2, color="tab:red", label="RSS [MiB]")
    ram_axis.set_ylabel("RSS [MiB]")
    lines = [*cost_axis.lines, *ram_axis.lines]
    cost_axis.legend(lines, [line.get_label() for line in lines])
    fig.suptitle("Tempering, ancestry and measured resource diagnostics")
    return _save(fig, output_dir / "inverse_diagnostics.png")


def render_inverse_figures(products: dict[str, Any], output_dir: Path) -> tuple[Path, ...]:
    """Render only figure families whose complete numeric inputs are supplied."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[Path] = []
    saturation_keys = {"true_so", "median_so", "q05_so", "q95_so", "prior_width"}
    watercut_keys = {"watercut_observed", "watercut_predictive"}
    diagnostic_keys = {"beta", "cess", "ess", "ancestors", "cost_wall_s", "cost_ram_bytes"}
    if saturation_keys <= products.keys():
        rendered.append(_saturation(products, output_dir))
    if watercut_keys <= products.keys():
        rendered.append(_watercut(products, output_dir))
    if diagnostic_keys <= products.keys():
        rendered.append(_diagnostics(products, output_dir))
    if not rendered:
        raise ValueError("products contain no complete inverse figure family")
    return tuple(rendered)


__all__ = ["render_inverse_figures"]
