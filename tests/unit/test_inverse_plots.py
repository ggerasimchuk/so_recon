"""Scientific figures are rendered from the supplied numeric products."""

from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import numpy as np

from so_recon.validation.inverse_plots import render_inverse_figures


def test_real_supplied_arrays_generate_nonempty_png_assets(tmp_path: Path) -> None:
    products = {
        "date_label": "month 12",
        "layer_label": "layer 1",
        "true_so": np.array([[0.8, 0.7], [0.6, 0.5]]),
        "median_so": np.array([[0.75, 0.65], [0.55, 0.45]]),
        "q05_so": np.full((2, 2), 0.3),
        "q95_so": np.full((2, 2), 0.9),
        "prior_width": np.full((2, 2), 0.8),
        "watercut_observed": np.array([0.0, 0.1, 0.4]),
        "watercut_predictive": np.array([[0.0, 0.05, 0.3], [0.0, 0.15, 0.5]]),
        "beta": np.array([0.0, 0.4, 1.0]),
        "cess": np.array([64.0, 50.0, 45.0]),
        "ess": np.array([64.0, 48.0, 42.0]),
        "ancestors": np.array([64.0, 50.0, 35.0]),
        "cost_wall_s": np.array([0.0, 2.0, 5.0]),
        "cost_ram_bytes": np.array([1.0e8, 1.2e8, 1.1e8]),
    }
    paths = render_inverse_figures(products, tmp_path)
    assert {path.name for path in paths} == {
        "inverse_diagnostics.png",
        "inverse_saturation.png",
        "inverse_watercut.png",
    }
    for path in paths:
        assert path.stat().st_size > 10_000
        image = mpimg.imread(path)
        assert image.shape[0] > 300 and image.shape[1] > 500


def test_intermediate_beta_is_not_labelled_as_a_posterior(tmp_path: Path) -> None:
    products = {
        "beta_final": 0.7,
        "true_so": np.ones((2, 2)),
        "median_so": np.ones((2, 2)),
        "q05_so": np.zeros((2, 2)),
        "q95_so": np.ones((2, 2)),
        "prior_width": np.ones((2, 2)),
    }
    paths = render_inverse_figures(products, tmp_path)
    assert [path.name for path in paths] == ["inverse_saturation.png"]
