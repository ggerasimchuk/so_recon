"""Independent quadrature keeps its integration measure explicit."""

from __future__ import annotations

import math
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from scipy.stats import norm

from so_recon.paths import ProjectPaths
from so_recon.registry.run import RunContext
from so_recon.synthetic.reduced_inverse import ReducedDesign, build_reduced_case, oil_corey_exponent
from so_recon.validation.reference_inverse import (
    quadrature_reference,
    reference_status,
    trapezoid_weights,
)


class CaseContext:
    run_id = "reduced-unit"

    def __init__(self, root: Path) -> None:
        self.run_dir = root / "artifacts" / "runs" / self.run_id
        self.run_dir.mkdir(parents=True)
        self.outputs: dict[str, object] = {}

    def add_output(self, key: str, ref: object) -> None:
        self.outputs[key] = ref


def test_nonuniform_quadrature_weights_are_used() -> None:
    out = quadrature_reference(
        np.array([0.0, 1.0]), np.zeros(2), np.zeros(2), np.array([0.25, 0.75])
    )
    assert np.isclose(out["mean"], 0.75)
    np.testing.assert_allclose(out["weights"], [0.25, 0.75])


def test_gaussian_grid_recovers_normalization_and_moments() -> None:
    nodes = np.linspace(-7.0, 7.0, 2001)
    dx = nodes[1] - nodes[0]
    integration = np.full(nodes.size, dx)
    integration[[0, -1]] *= 0.5
    out = quadrature_reference(nodes, norm.logpdf(nodes), np.zeros(nodes.size), integration)
    assert out["mean"] == pytest.approx(0.0, abs=1e-13)
    assert out["variance"] == pytest.approx(1.0, abs=1e-9)
    assert out["logz"] == pytest.approx(math.log(norm.cdf(7) - norm.cdf(-7)), abs=1e-9)
    np.testing.assert_allclose(out["quantiles"], norm.ppf([0.05, 0.5, 0.95]), atol=0.008)


def test_nonuniform_trapezoid_weights_integrate_linear_function() -> None:
    nodes = np.array([0.0, 0.25, 1.0])
    widths = trapezoid_weights(nodes)
    assert widths.sum() == pytest.approx(1.0)
    assert widths @ nodes == pytest.approx(0.5)


def test_reference_status_fails_closed_on_any_unresolved_bound() -> None:
    passing = {
        "mean_change": 0.01,
        "max_quantile_change": 0.02,
        "relative_evidence_change": 0.01,
        "tail_to_evidence_lower": 1.0e-7,
    }
    assert reference_status(passing) == "CONVERGED"
    for key in passing:
        failed = dict(passing)
        failed[key] = 1.0
        assert reference_status(failed) == "UNRESOLVED_REFERENCE"


def test_likelihood_changes_posterior_without_renormalizing_the_grid_measure() -> None:
    nodes = np.array([-2.0, -0.5, 1.0, 3.0])
    widths = np.array([0.75, 1.5, 1.75, 1.0])
    prior = norm.logpdf(nodes)
    likelihood = norm.logpdf(0.7, nodes, 0.6)
    out = quadrature_reference(nodes, prior, likelihood, widths)
    expected_mass = np.exp(prior + likelihood) * widths
    expected = expected_mass / expected_mass.sum()
    np.testing.assert_allclose(out["weights"], expected)
    assert out["logz"] == pytest.approx(math.log(expected_mass.sum()))


@pytest.mark.parametrize(
    ("nodes", "prior", "likelihood", "widths", "match"),
    [
        ([0.0], [0.0], [0.0], [1.0], "at least two"),
        ([0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0], "increasing"),
        ([0.0, 1.0], [0.0], [0.0, 0.0], [1.0, 1.0], "shape"),
        ([0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0], "positive"),
        ([0.0, 1.0], [0.0, 0.0], [-math.inf, -math.inf], [1.0, 1.0], "support"),
    ],
)
def test_invalid_quadrature_measure_or_support_is_refused(
    nodes: list[float],
    prior: list[float],
    likelihood: list[float],
    widths: list[float],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        quadrature_reference(
            np.asarray(nodes), np.asarray(prior), np.asarray(likelihood), np.asarray(widths)
        )


def test_reduced_latent_changes_oil_corey_physics_not_the_rock(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    paths = ProjectPaths.default(root)
    context = cast(RunContext, CaseContext(root))
    low = build_reduced_case(-1.0, ReducedDesign(), paths, context)
    high = build_reduced_case(1.0, ReducedDesign(), paths, context)
    assert low.fluids.kind == high.fluids.kind == "OW"
    assert low.fluids.corey_exponents[0] == high.fluids.corey_exponents[0] == 2.0
    assert low.fluids.corey_exponents[1] == pytest.approx(oil_corey_exponent(-1.0))
    assert high.fluids.corey_exponents[1] == pytest.approx(oil_corey_exponent(1.0))
    assert low.rock.porosity.sha256 == high.rock.porosity.sha256
    assert low.rock.permeability_m2.sha256 == high.rock.permeability_m2.sha256
    assert low.model_hash != high.model_hash
    assert low.grid.shape == (16, 1, 1)
    assert len(low.report_edges_s) == 13
    assert "/truth/" not in str(low.model_dump(mode="json"))


def test_reduced_v2_records_the_informative_control_change_under_a_new_identity() -> None:
    design = ReducedDesign()
    assert design.design_id == "e02-reduced-corey-v2"
    assert design.rate_m3_sc_day == 4.0
