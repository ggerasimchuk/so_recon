"""E01.11: the deterministic two-layer P1 generator, and the truth/context boundary.

Two claims are made here and nothing else is.

**The generator is a known low-dimensional forward map.** Twelve standard-normal
coefficients and a versioned cosine basis determine both layers' permeability AND porosity,
so a world is reproducible from `(generator version, design, seed)` alone and the two rock
fields are not drawn independently. The tests below pin the basis, the coefficients, the
reproducibility of the scientific arrays and the model hash, and the refusal of a geology
that overflowed instead of the silent clipping of its tail.

**The inverse problem's input is separated from the truth STRUCTURALLY.** `context_payload`
is an allowlist: a key that is not in its tuple cannot reach `context.json`, whatever a
manifest happens to carry. The leakage tests prove the boundary by perturbing the truth: a
change off the observed supports must leave the context bytes IDENTICAL, and a change on a
support must move exactly one field of them — the observation reference, which is the one
thing the observations are allowed to say. Everything else about the truth reaches the
solver's `CaseBundle` and the world's `truth/` directory, and neither is ever handed to an
encoder.

Nothing here runs Julia. The forward itself is `tests/integration/test_e01_p1.py`.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from so_recon.paths import ProjectPaths
from so_recon.registry.hashing import sha256_file
from so_recon.registry.run import RunContext
from so_recon.simulator.case_io import load_case, write_case
from so_recon.simulator.contracts import (
    MILLIDARCY_M2,
    SECONDS_PER_DAY,
    STANDARD_GRAVITY_M_S2,
    ControlSegment,
    CostRecord,
    ForwardResult,
    WellSpec,
)
from so_recon.synthetic.p1 import (
    DESIGN_ID,
    GENERATOR_VERSION,
    MODES,
    P1_PARENTS,
    SIGMA_LOG_PERMEABILITY,
    SIGMA_OIL_SATURATION,
    SIGMA_POROSITY,
    P1Design,
    RenderedWorld,
    layer_fields,
    layer_geology,
    oil_connected_hydrostatic_pa,
    output_request,
    render_p1,
    static_observations,
    streams,
    support_mean,
    support_truth,
)
from so_recon.synthetic.world_io import (
    CONTEXT_ALLOWLIST,
    CONTEXT_FILENAME,
    DERIVED_VIEW_ID,
    SPLIT,
    build_p1_case,
    context_payload,
    world_locations,
    world_row,
    write_world,
)

# --------------------------------------------------------------------------------------
# the brief's own test, first and unaltered
# --------------------------------------------------------------------------------------


def test_p1_reproducibility_and_heterogeneity() -> None:
    design = P1Design()
    a, b = render_p1(41, design), render_p1(41, design)
    np.testing.assert_array_equal(a.arrays["permeability_m2"], b.arrays["permeability_m2"])
    assert a.arrays["sw"].size == 512
    assert np.std(a.arrays["permeability_m2"][0]) > 0
    assert a.parent_world_id == b.parent_world_id
    assert a.design.n_months == 36


# --------------------------------------------------------------------------------------
# 11.3 the low-dimensional joint geology
# --------------------------------------------------------------------------------------


def _layer_view(values: NDArray[np.float64], design: P1Design, layer: int) -> NDArray[np.float64]:
    """One layer of a cell-indexed array, back in `(nx, ny)` with `cell = i + nx*j`."""
    nx, ny, _ = design.shape
    return values[layer * nx * ny : (layer + 1) * nx * ny].reshape((nx, ny), order="F")


def test_every_scientific_array_and_the_model_hash_repeat_for_the_same_seed(
    tmp_project: Path,
) -> None:
    paths = ProjectPaths.default(tmp_project)
    paths.ensure_dirs()
    design = P1Design()
    a, b = render_p1(42, design), render_p1(42, design)
    assert sorted(a.arrays) == sorted(b.arrays)
    for name, values in a.arrays.items():
        np.testing.assert_array_equal(values, b.arrays[name], err_msg=name)
    assert a.theta == b.theta
    ctx = RunContext.start(command="forward", argv=[], cfg=None, paths=paths)
    case_a = build_p1_case(a, paths, ctx)
    case_b = build_p1_case(b, paths, ctx)
    assert case_a.model_hash == case_b.model_hash


def test_a_different_seed_moves_the_geology_and_the_identity() -> None:
    design = P1Design()
    a, b = render_p1(41, design), render_p1(42, design)
    assert a.parent_world_id != b.parent_world_id
    assert not np.array_equal(a.arrays["permeability_m2"], b.arrays["permeability_m2"])
    assert not np.array_equal(a.arrays["porosity"], b.arrays["porosity"])
    assert a.theta["coefficients"] != b.theta["coefficients"]


def test_both_layers_are_heterogeneous_and_are_not_each_other() -> None:
    world = render_p1(41, P1Design())
    kx = world.arrays["permeability_m2"][0]
    phi = world.arrays["porosity"]
    top_k, bottom_k = _layer_view(kx, world.design, 0), _layer_view(kx, world.design, 1)
    top_phi, bottom_phi = _layer_view(phi, world.design, 0), _layer_view(phi, world.design, 1)
    for field in (top_k, bottom_k, top_phi, bottom_phi):
        assert float(np.std(field)) > 0.0
    assert not np.array_equal(top_k, bottom_k)
    assert not np.array_equal(top_phi, bottom_phi)
    # The design's own layer contrast: the upper layer is the more permeable one.
    assert float(np.median(top_k)) > float(np.median(bottom_k))


def test_porosity_and_permeability_come_from_the_same_latent_field() -> None:
    """Not independent draws: both are monotone functions of one cosine field per layer."""
    world = render_p1(43, P1Design())
    for layer in range(2):
        k = _layer_view(world.arrays["permeability_m2"][0], world.design, layer).ravel()
        phi = _layer_view(world.arrays["porosity"], world.design, layer).ravel()
        # Rank correlation is EXACTLY one: log k and phi are strictly increasing functions
        # of the same field, so a pair drawn from two streams could not produce this.
        np.testing.assert_array_equal(np.argsort(np.argsort(k)), np.argsort(np.argsort(phi)))


def test_the_basis_is_the_versioned_six_mode_cosine_basis() -> None:
    assert MODES == ((0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2))
    design = P1Design()
    nx, ny, _ = design.shape
    # A single active mode reproduces its own analytic shape, weighted by 1/(1+i+j).
    for index, (i, j) in enumerate(MODES):
        coefficients = np.zeros(len(MODES))
        coefficients[index] = 1.0
        k_md, _ = layer_fields(coefficients, nx, ny, 0, "base")
        x = (np.arange(nx) + 0.5) / nx
        y = (np.arange(ny) + 0.5) / ny
        xx, yy = np.meshgrid(x, y, indexing="ij")
        expected = np.cos(np.pi * i * xx) * np.cos(np.pi * j * yy) / (1 + i + j)
        np.testing.assert_allclose(np.log(k_md) - np.log(120.0), expected, rtol=0, atol=1e-12)


def test_theta_records_the_generator_the_seed_and_the_twelve_coefficients() -> None:
    world = render_p1(41, P1Design())
    theta = world.theta
    assert theta["generator_version"] == GENERATOR_VERSION
    assert theta["design_id"] == DESIGN_ID
    assert theta["seed"] == 41
    assert theta["bit_generator"] == "PCG64"
    assert theta["n_coefficients"] == 12
    coefficients = theta["coefficients"]
    assert isinstance(coefficients, list) and len(coefficients) == 2
    assert all(len(layer) == 6 for layer in coefficients)
    assert theta["modes"] == [list(mode) for mode in MODES]
    assert theta["streams"] == ["geology", "static_observation", "control"]
    assert theta["reserved_streams"] == ["control"]
    # The generator's dimension is DEFINED as twelve; it is not a truncated prior, and E01
    # claims no conditional latent density. The artifact has to say so itself.
    claim = str(theta["dimensionality_claim"])
    assert "p(theta|G)" in claim and "E02" in claim
    assert "not a ThetaRecord" in str(theta["record_kind"])


def test_a_geology_that_overflowed_is_refused_rather_than_clipped() -> None:
    design = P1Design()
    with pytest.raises(ValueError, match="nonfinite"):
        layer_geology(np.full(6, 1.0e6), design, 0)
    with pytest.raises(ValueError, match="nonfinite"):
        layer_geology(np.array([np.nan, 0.0, 0.0, 0.0, 0.0, 0.0]), design, 0)
    # A finite but absurd tail is refused too, with the value it reached.
    with pytest.raises(ValueError, match="refused rather than truncated"):
        layer_geology(np.array([12.0, 0.0, 0.0, 0.0, 0.0, 0.0]), design, 0)


def test_the_kernel_constants_are_the_ones_the_design_declares() -> None:
    """The verbatim kernel's literals, pinned against the design that reports them."""
    design = P1Design()
    nx, ny, _ = design.shape
    coefficients = np.array([0.3, -0.2, 0.1, 0.4, -0.5, 0.25])
    for family in ("base", "low_vertical", "high_contrast"):
        variant = P1Design(family=family)
        for layer in range(2):
            k_md, phi = layer_fields(coefficients, nx, ny, layer, family)
            base = np.log(variant.layer_base_permeability_md[layer])
            field = (np.log(k_md) - base) / variant.contrast
            np.testing.assert_allclose(
                phi, 0.12 + 0.18 / (1.0 + np.exp(-0.7 * field)), rtol=0, atol=1e-12
            )
    assert P1Design(family="high_contrast").contrast == 1.8
    assert P1Design(family="base").contrast == 1.0
    assert P1Design(family="base").kz_over_kx == 0.05
    assert P1Design(family="low_vertical").kz_over_kx == 0.001
    assert P1Design(family="high_contrast").kz_over_kx == 0.05
    assert design.layer_base_permeability_md == (120.0, 40.0)


def test_the_families_change_the_rock_as_the_design_says() -> None:
    base = render_p1(41, P1Design(family="base"))
    contrast = render_p1(41, P1Design(family="high_contrast"))
    vertical = render_p1(41, P1Design(family="low_vertical"))
    spread = float(np.std(np.log(base.arrays["permeability_m2"][0])))
    assert float(np.std(np.log(contrast.arrays["permeability_m2"][0]))) > spread
    # Only the vertical direction moves for low_vertical: kx and ky are untouched.
    np.testing.assert_array_equal(
        base.arrays["permeability_m2"][:2], vertical.arrays["permeability_m2"][:2]
    )
    ratio = vertical.arrays["permeability_m2"][2] / vertical.arrays["permeability_m2"][0]
    np.testing.assert_allclose(ratio, 0.001, rtol=1e-12)
    np.testing.assert_allclose(
        base.arrays["permeability_m2"][2] / base.arrays["permeability_m2"][0], 0.05, rtol=1e-12
    )
    assert len({base.parent_world_id, vertical.parent_world_id, contrast.parent_world_id}) == 3


def test_permeability_is_stated_in_square_metres() -> None:
    world = render_p1(41, P1Design())
    kx_md = world.arrays["permeability_m2"][0] / MILLIDARCY_M2
    assert float(np.min(kx_md)) > 1.0
    assert float(np.max(kx_md)) < 1.0e5
    np.testing.assert_allclose(
        world.arrays["log_permeability_m2"], np.log(world.arrays["permeability_m2"][0]), rtol=1e-15
    )


def test_the_three_streams_are_independent_and_reproducible() -> None:
    drawn = [np.random.default_rng(s).standard_normal(4) for s in streams(41)]
    for a, b in ((0, 1), (0, 2), (1, 2)):
        assert not np.array_equal(drawn[a], drawn[b])
    # The static-observation stream is what the noise comes from; it depends on the world's
    # seed and on nothing else, which is what the leakage tests below rest on.
    np.testing.assert_array_equal(
        np.random.default_rng(streams(41).static_observation).standard_normal(4), drawn[1]
    )
    assert not np.array_equal(
        np.random.default_rng(streams(42).static_observation).standard_normal(4), drawn[1]
    )


# --------------------------------------------------------------------------------------
# 11.4 the known initial state, the grid and the four wells
# --------------------------------------------------------------------------------------


def test_the_grid_is_the_p1_loop_box() -> None:
    design = P1Design()
    assert design.shape == (16, 16, 2)
    assert design.extent_m == (100.0, 100.0, 20.0)
    assert design.n_cells == 512
    world = render_p1(41, design)
    centers = world.arrays["cell_centers_m"]
    assert centers.shape == (512, 3)
    # z is depth, positive down, and the two layer centres are the only depths there are.
    assert sorted(set(centers[:, 2].tolist())) == [5.0, 15.0]
    np.testing.assert_allclose(world.arrays["cell_volume_m3"], 6.25 * 6.25 * 10.0)


def test_the_initial_state_is_oil_connected_hydrostatic_from_the_datum() -> None:
    design = P1Design()
    world = render_p1(41, design)
    sw = world.arrays["sw"]
    pressure = world.arrays["pressure_pa"]
    np.testing.assert_array_equal(sw, np.full(512, 0.2))
    # The datum itself: 1.5e7 Pa at z = 0.
    np.testing.assert_allclose(
        oil_connected_hydrostatic_pa(np.array([0.0]), design=design), [1.5e7], rtol=0, atol=1e-6
    )
    depth = world.arrays["cell_centers_m"][:, 2]
    np.testing.assert_allclose(pressure, oil_connected_hydrostatic_pa(depth, design=design))
    # dp/dz is the OIL density at the local pressure times g. The DISCRETE statement the
    # solver balances is the two-point one JutulDarcy forms across the single vertical face,
    # `p[lower] - p[upper] = g * dz * (rho[upper] + rho[lower]) / 2`; the column is
    # initialised from the CONTINUOUS solution, so what is left is the trapezoid's own
    # quadrature error over 10 m and nothing physical. Below a milli-pascal it is five
    # orders under `hydrostatic_gradient_relative_max`, and the closed preflight of
    # `tests/integration/test_e01_p1.py` measures the same balance on the real model.
    rho_o = 800.0 * np.exp(1e-9 * (pressure - 101325.0))
    top = float(pressure[depth == 5.0][0])
    bottom = float(pressure[depth == 15.0][0])
    head = (
        0.5 * float(rho_o[depth == 5.0][0] + rho_o[depth == 15.0][0]) * STANDARD_GRAVITY_M_S2 * 10.0
    )
    residual_pa = abs((bottom - top) - head)
    assert residual_pa < 1.0e-3
    assert residual_pa / head < 1.0e-8
    # A single pressure per layer: the column is a function of depth only, so every
    # horizontal face starts with exactly zero potential difference.
    assert len(set(pressure.tolist())) == 2


def test_four_wells_open_both_layers_as_multisegment_completions() -> None:
    world = render_p1(41, P1Design())
    wells: tuple[WellSpec, ...] = world.case_fields["wells"]
    assert [w.well_id for w in wells] == ["I1", "I2", "P1", "P2"]
    nx, ny, _ = world.design.shape
    expected = {"I1": (2, 2), "I2": (2, 13), "P1": (13, 2), "P2": (13, 13)}
    for well in wells:
        i, j = expected[well.well_id]
        assert well.cells == (i + nx * j, i + nx * (j + ny))
        assert well.radius_m == 0.1
        assert well.model == "multisegment"
        assert well.allow_crossflow is True
        assert well.reference_depth_m == 0.0


# --------------------------------------------------------------------------------------
# 11.5 the independent controls policy
# --------------------------------------------------------------------------------------


def _controls(world: RenderedWorld) -> tuple[ControlSegment, ...]:
    segments: tuple[ControlSegment, ...] = world.case_fields["controls"]
    return segments


def test_the_controls_do_not_depend_on_the_seed_the_family_or_the_truth() -> None:
    """A policy fixed in TIME, never read off a hidden saturation."""
    a = _controls(render_p1(41, P1Design()))
    b = _controls(render_p1(45, P1Design(family="high_contrast")))
    c = _controls(render_p1(44, P1Design(family="low_vertical")))
    assert a == b == c


def test_the_rates_are_the_spec_9_1_targets_with_the_time_modulation() -> None:
    world = render_p1(41, P1Design())
    edges = world.case_fields["report_edges_s"]
    by_well: dict[str, list[ControlSegment]] = {}
    for segment in _controls(world):
        by_well.setdefault(segment.well_id, []).append(segment)
    assert sorted(by_well) == ["I1", "I2", "P1", "P2"]
    for well_id, segments in by_well.items():
        segments.sort(key=lambda s: s.start_s)
        for segment in segments:
            if well_id.startswith("I"):
                assert segment.role == "injector"
                assert segment.target == "water_rate"
                assert segment.bhp_limit_pa == 3.0e7
            else:
                assert segment.role == "producer"
                # SPEC 9.1: a TOTAL standard liquid rate, never two phase rates.
                assert segment.target == "liquid_rate"
                assert segment.bhp_limit_pa == 5.0e6
        # 20 m3_sc/day, then 0.75x on months 13-24 and 1.25x on months 25-36.
        for first, last, value in ((0, 12, 20.0), (12, 24, 15.0), (24, 36, 25.0)):
            covering = [s for s in segments if s.start_s >= edges[first] and s.end_s <= edges[last]]
            assert covering, (well_id, first)
            assert {s.value for s in covering} == {value}
        # Every month of the horizon is covered exactly once.
        assert segments[0].start_s == edges[0]
        assert segments[-1].end_s == edges[36]
        for left, right in zip(segments, segments[1:], strict=False):
            assert left.end_s == right.start_s


def test_the_completion_event_isolates_p2_lower_connection_for_months_19_to_24() -> None:
    world = render_p1(41, P1Design())
    edges = world.case_fields["report_edges_s"]
    shut = [s for s in _controls(world) if s.connection_open == (True, False)]
    assert len(shut) == 1
    assert shut[0].well_id == "P2"
    assert shut[0].start_s == edges[18]  # the start of month 19
    assert shut[0].end_s == edges[24]  # reopened at the start of month 25
    # The event does not rewrite the rate the policy had already fixed for those months.
    assert shut[0].value == 15.0


def test_the_calendar_is_thirty_six_real_months() -> None:
    world = render_p1(41, P1Design())
    edges = world.case_fields["report_edges_s"]
    assert len(edges) == 37
    assert world.case_fields["start_date"] == "2000-01-01"
    assert world.case_fields["cutoff"] == "2003-01-01"
    # Real calendar months: January 2000 is 31 days and February 2000 is 29.
    assert edges[1] == 31 * SECONDS_PER_DAY
    assert edges[2] - edges[1] == 29 * SECONDS_PER_DAY
    request = output_request(world.design)
    assert request.state_times_s == (edges[0], edges[12], edges[24], edges[36])
    assert request.keep_native_restart is True


# --------------------------------------------------------------------------------------
# 11.6 the sparse static observations
# --------------------------------------------------------------------------------------


def test_support_mean_is_an_average_over_the_declared_support() -> None:
    values = np.array([1.0, 2.0, 5.0, 11.0], dtype=np.float64)
    assert support_mean(values, (0,)) == 1.0
    assert support_mean(values, (1, 2)) == 3.5
    assert support_mean(values, (0, 2, 3)) == pytest.approx(17.0 / 3.0)
    with pytest.raises(ValueError, match="support"):
        support_mean(values, ())


def test_the_static_observations_are_eight_supports_and_three_quantities() -> None:
    world = render_p1(41, P1Design())
    rows = world.static_observations.to_pylist()
    assert len(rows) == 24
    assert sorted({(r["well_id"], r["layer_index"]) for r in rows}) == [
        (well, layer) for well in ("I1", "I2", "P1", "P2") for layer in (0, 1)
    ]
    assert sorted({r["quantity"] for r in rows}) == [
        "log_permeability_m2",
        "oil_saturation",
        "porosity",
    ]
    sigmas = {r["quantity"]: r["sigma"] for r in rows}
    assert sigmas["porosity"] == SIGMA_POROSITY
    assert sigmas["log_permeability_m2"] == SIGMA_LOG_PERMEABILITY
    assert sigmas["oil_saturation"] == SIGMA_OIL_SATURATION
    assert {r["date"] for r in rows} == {"2000-01-01"}
    assert all(len(r["support_cell_ids"]) == 1 for r in rows)
    # The published table carries NO latent column: it is what the context references.
    assert "latent_value" not in world.static_observations.column_names


def test_every_measurement_really_carries_its_noise() -> None:
    world = render_p1(41, P1Design())
    latent = {
        r["observation_id"]: r["latent_value"]
        for r in support_truth(world.arrays, world.design).to_pylist()
    }
    for row in world.static_observations.to_pylist():
        assert row["value"] != latent[row["observation_id"]]
        assert abs(row["value"] - latent[row["observation_id"]]) < 6.0 * row["sigma"]


def test_an_out_of_range_measurement_is_kept_with_a_flag_and_never_clipped() -> None:
    flagged = 0
    for seed, family in P1_PARENTS:
        world = render_p1(seed, P1Design(family=family))
        truth = {
            r["observation_id"]: r["latent_value"]
            for r in support_truth(world.arrays, world.design).to_pylist()
        }
        for row in world.static_observations.to_pylist():
            if row["valid_range_low"] is None:
                # An unbounded quantity: every value of it is a value it can take.
                assert row["quantity"] == "log_permeability_m2"
                assert row["quality_flag"] == "ok"
                continue
            inside = row["valid_range_low"] <= row["value"] <= row["valid_range_high"]
            assert (row["quality_flag"] == "ok") == inside
            flagged += 0 if inside else 1
            # The latent truth stays physical whatever the noise did to the measurement.
            assert row["valid_range_low"] <= truth[row["observation_id"]] <= row["valid_range_high"]
    # The initial oil saturation sits exactly on the edge of the mobile range, so the flag
    # is a path this suite really exercises rather than a column nobody fills.
    assert flagged > 0


def test_the_observations_are_the_support_average_of_the_truth_plus_that_stream() -> None:
    design = P1Design()
    world = render_p1(41, design)
    repeat = static_observations(world.arrays, design, streams(41).static_observation)
    assert repeat.to_pylist() == world.static_observations.to_pylist()
    # The latent value is read off the truth at the support, not off a well flux.
    for row in support_truth(world.arrays, design).to_pylist():
        cells = list(row["support_cell_ids"])
        if row["quantity"] == "porosity":
            expected = support_mean(world.arrays["porosity"], cells)
        elif row["quantity"] == "log_permeability_m2":
            expected = support_mean(world.arrays["log_permeability_m2"], cells)
        else:
            expected = support_mean(1.0 - world.arrays["sw"], cells)
        assert row["latent_value"] == pytest.approx(expected)


def test_pressure_is_not_an_available_channel() -> None:
    world = render_p1(41, P1Design())
    assert world.case_fields["pressure_available"] is False
    assert world.case_fields["dynamic_channels"] == ("fw", "controls")
    assert "pressure" not in {r["quantity"] for r in world.static_observations.to_pylist()}


# --------------------------------------------------------------------------------------
# 11.7 the truth/context boundary
# --------------------------------------------------------------------------------------


def test_context_payload_is_exactly_the_allowlist() -> None:
    assert CONTEXT_ALLOWLIST == (
        "parent_world_id",
        "design_id",
        "derived_view_id",
        "split",
        "observation_ref",
        "control_ref",
        "observation_masks",
        "cutoff",
    )
    manifest: dict[str, Any] = dict.fromkeys(CONTEXT_ALLOWLIST, "x")
    manifest["truth_ref"] = {"path": "artifacts/worlds/w/truth/theta.json"}
    manifest["model_hash"] = "a" * 64
    manifest["seed"] = 41
    payload = context_payload(manifest)
    assert tuple(payload) == CONTEXT_ALLOWLIST
    assert "truth_ref" not in payload and "seed" not in payload and "model_hash" not in payload


def test_a_manifest_missing_an_allowed_key_is_refused() -> None:
    with pytest.raises(KeyError):
        context_payload(dict.fromkeys(CONTEXT_ALLOWLIST[:-1], "x"))


def _second_project(root: Path) -> ProjectPaths:
    """A second, independent repository root, so two versions of ONE world can be written.

    The parent world id is a function of the generator, the design and the seed, so a
    perturbed truth claims the same path as the original — and the content-addressed writers
    refuse it, correctly. Writing the perturbed world into its own root is what lets the two
    context files be compared byte for byte.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text("[project]\nname='x'\n")
    (root / "src" / "so_recon").mkdir(parents=True, exist_ok=True)
    paths = ProjectPaths.default(root)
    paths.ensure_dirs()
    return paths


def _write_context(world: RenderedWorld, paths: ProjectPaths) -> Path:
    ctx = RunContext.start(command="forward", argv=[], cfg=None, paths=paths)
    write_world(world, None, paths, ctx)
    return world_locations(paths, world.parent_world_id).context_file


def _tainted(world: RenderedWorld, cell: int, delta: float) -> RenderedWorld:
    """The same world with ONE cell's rock and saturation moved. Nothing else changes."""
    arrays = {name: values.copy() for name, values in world.arrays.items()}
    arrays["porosity"][cell] += delta
    arrays["permeability_m2"][:, cell] *= 1.0 + delta
    arrays["log_permeability_m2"][cell] = float(np.log(arrays["permeability_m2"][0][cell]))
    arrays["sw"][cell] += delta
    observations = static_observations(arrays, world.design, streams(world.seed).static_observation)
    return dataclasses.replace(world, arrays=arrays, static_observations=observations)


def test_a_truth_change_off_the_supports_leaves_the_context_bytes_identical(
    tmp_project: Path,
) -> None:
    paths = ProjectPaths.default(tmp_project)
    paths.ensure_dirs()
    world = render_p1(41, P1Design())
    supported = {
        cell for row in world.static_observations.to_pylist() for cell in row["support_cell_ids"]
    }
    off_support = next(c for c in range(512) if c not in supported)

    before = _write_context(world, paths).read_bytes()
    tainted = _tainted(world, off_support, 0.05)
    assert not np.array_equal(tainted.arrays["porosity"], world.arrays["porosity"])
    assert not np.array_equal(tainted.arrays["sw"], world.arrays["sw"])

    after = _write_context(tainted, _second_project(tmp_project / "second")).read_bytes()
    assert after == before


def test_a_truth_change_on_a_support_moves_only_the_observation_reference(
    tmp_project: Path,
) -> None:
    paths = ProjectPaths.default(tmp_project)
    paths.ensure_dirs()
    world = render_p1(41, P1Design())
    on_support = int(world.static_observations.to_pylist()[0]["support_cell_ids"][0])

    before = json.loads(_write_context(world, paths).read_text(encoding="utf-8"))
    tainted = _tainted(world, on_support, 0.05)
    after = json.loads(
        _write_context(tainted, _second_project(tmp_project / "second")).read_text(encoding="utf-8")
    )

    assert set(before) == set(after) == set(CONTEXT_ALLOWLIST)
    assert sorted(key for key in before if before[key] != after[key]) == ["observation_ref"]
    assert before["observation_ref"]["sha256"] != after["observation_ref"]["sha256"]
    assert before["observation_ref"]["path"] == after["observation_ref"]["path"]


def _longest_list(payload: Any) -> int:
    if isinstance(payload, dict):
        return max((_longest_list(v) for v in payload.values()), default=0)
    if isinstance(payload, list):
        return max(len(payload), *(_longest_list(v) for v in payload), 0)
    return 0


def test_the_context_file_names_no_truth_and_carries_no_full_field(tmp_project: Path) -> None:
    paths = ProjectPaths.default(tmp_project)
    paths.ensure_dirs()
    world = render_p1(41, P1Design())
    context_path = _write_context(world, paths)
    text = context_path.read_text(encoding="utf-8")
    payload = json.loads(text)
    assert set(payload) == set(CONTEXT_ALLOWLIST)
    # No truth directory, no generator parameters, no full-field array.
    for forbidden in ("truth", "theta", "geology.h5", "initial.h5", "model_hash", "seed"):
        assert forbidden not in text, forbidden
    assert _longest_list(payload) <= 16, "a 512-cell field cannot fit in a sparse context"
    assert payload["derived_view_id"] == DERIVED_VIEW_ID
    assert payload["split"] == SPLIT
    assert payload["cutoff"] == "2003-01-01"
    assert payload["observation_masks"]["pressure_available"] is False
    assert payload["observation_masks"]["dynamic_channels"] == ["fw", "controls"]
    assert len(payload["observation_masks"]["static_supports"]) == 8
    # Both referenced files exist and hash to what the context says.
    for ref in (payload["observation_ref"], payload["control_ref"]):
        target = paths.resolve(ref["path"])
        assert sha256_file(target) == ref["sha256"]
        assert "/truth/" not in ref["path"]


def test_the_solver_case_carries_the_whole_truth_the_context_does_not(
    tmp_project: Path,
) -> None:
    paths = ProjectPaths.default(tmp_project)
    paths.ensure_dirs()
    ctx = RunContext.start(command="forward", argv=[], cfg=None, paths=paths)
    world = render_p1(41, P1Design())
    case = build_p1_case(world, paths, ctx)
    assert case.rock.porosity.shape == (512,)
    assert case.rock.permeability_m2.shape == (3, 512)
    assert case.initial.kind == "explicit"
    assert case.initial.meaning == "synthetic_initial"
    assert case.initial.pressure_pa is not None and case.initial.sw is not None
    assert case.boundary.kind == "closed"
    # And the boundary of the inverse problem is declared on the case itself.
    assert case.observations.truth_access == "forbidden"
    assert case.observations.pressure_available is False
    assert case.observations.dynamic_channels == ("fw", "controls")
    assert case.observations.table_path is not None
    assert "/truth/" not in case.observations.table_path


def test_the_deterministic_files_of_a_world_are_written_once(tmp_project: Path) -> None:
    paths = ProjectPaths.default(tmp_project)
    paths.ensure_dirs()
    world = render_p1(41, P1Design())
    loc = world_locations(paths, world.parent_world_id)
    deterministic = (
        loc.grid_file,
        loc.geology_file,
        loc.initial_file,
        loc.case_file,
        loc.theta_file,
        loc.support_truth_file,
        loc.observations_file,
        loc.controls_file,
        loc.context_file,
    )
    write_world(world, None, paths, RunContext.start(command="f", argv=[], cfg=None, paths=paths))
    digests = {path: sha256_file(path) for path in deterministic}
    # A second run of the SAME seed and design rewrites none of them.
    write_world(world, None, paths, RunContext.start(command="f", argv=[], cfg=None, paths=paths))
    assert {path: sha256_file(path) for path in deterministic} == digests


def test_the_world_manifest_carries_the_identity_and_the_truth_reference(
    tmp_project: Path,
) -> None:
    paths = ProjectPaths.default(tmp_project)
    paths.ensure_dirs()
    world = render_p1(44, P1Design(family="low_vertical"))
    ctx = RunContext.start(command="f", argv=[], cfg=None, paths=paths)
    ref = write_world(world, None, paths, ctx)
    manifest = json.loads(paths.resolve(ref.path).read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "world-1"
    assert manifest["generator_version"] == GENERATOR_VERSION
    assert manifest["design_id"] == DESIGN_ID
    assert manifest["derived_view_id"] == DERIVED_VIEW_ID
    assert manifest["split"] == SPLIT
    assert manifest["seed"] == 44
    assert manifest["family"] == "low_vertical"
    assert manifest["parent_world_id"] == world.parent_world_id
    assert len(manifest["model_hash"]) == 64
    assert manifest["forward_status"] is None
    assert manifest["accepted"] is False
    assert "/truth/" in manifest["truth_ref"]["path"]
    # Machine and time stamps live OUTSIDE the deterministic identity (plan 3.1).
    assert set(manifest["provenance"]) >= {"run_id", "created_at", "numpy_version"}
    assert "created_at" not in manifest
    # The truth manifest names the full fields and warns what it is.
    truth = json.loads(paths.resolve(manifest["truth_ref"]["path"]).read_text(encoding="utf-8"))
    assert sorted(truth["arrays"]) == [
        "cell_centers_m",
        "initial_pressure_pa",
        "initial_sw",
        "permeability_m2",
        "porosity",
    ]
    assert "never handed to an encoder" in truth["warning"]


# --------------------------------------------------------------------------------------
# the published links: world -> case, world -> context, and the forward that belongs to it
# --------------------------------------------------------------------------------------


def _forward_case_digest(world: RenderedWorld, paths: ProjectPaths) -> str:
    """The `case_sha256` a real forward of this world would record.

    `simulate` publishes the case into ITS OWN run directory and puts `sha256_file` of that
    manifest on the result. This reproduces exactly that, through the same publication
    routine and in a run of its own, so a test can hold the digest a forward would carry
    without running one.
    """
    ctx = RunContext.start(command="forward", argv=[], cfg=None, paths=paths)
    return write_case(build_p1_case(world, paths, ctx), paths, ctx).sha256


def test_the_world_manifest_links_the_case_the_forward_ran(tmp_project: Path) -> None:
    """The world -> case link is published, and it is byte-for-byte the solver's own case.

    A `RunContext` that has not itself run a forward is the shape the suite publishes with,
    and it is the shape under which `case_ref` silently went null for every parent.
    """
    paths = ProjectPaths.default(tmp_project)
    paths.ensure_dirs()
    world = render_p1(41, P1Design())
    forward_digest = _forward_case_digest(world, paths)

    publisher = RunContext.start(command="forward", argv=[], cfg=None, paths=paths)
    ref = write_world(world, None, paths, publisher)
    manifest = json.loads(paths.resolve(ref.path).read_text(encoding="utf-8"))

    case_ref = manifest["case_ref"]
    assert case_ref is not None, "the world -> case link went null again"
    target = paths.resolve(case_ref["path"])
    assert sha256_file(target) == case_ref["sha256"]
    assert target == world_locations(paths, world.parent_world_id).case_file
    # A case carries the full rock and the full initial state, so it lives on the side of
    # the boundary that is never handed to an encoder.
    assert "/truth/" in case_ref["path"]
    # The case a forward published in ANOTHER run has the same bytes. That is what makes
    # `ForwardResult.case_sha256` comparable with this reference at all.
    assert case_ref["sha256"] == forward_digest
    assert load_case(target, paths).model_hash == manifest["model_hash"]


def test_a_suite_publisher_may_write_every_world_from_one_run(tmp_project: Path) -> None:
    """The case belongs to the WORLD, so one run can publish a parent set without collision.

    `case_io.write_case` puts one case in a run directory and refuses a second, which is the
    right rule for a run that executes a forward. A publisher is not that run.
    """
    paths = ProjectPaths.default(tmp_project)
    paths.ensure_dirs()
    ctx = RunContext.start(command="forward", argv=[], cfg=None, paths=paths)
    refs = [
        write_world(render_p1(seed, P1Design(family=family)), None, paths, ctx)
        for seed, family in P1_PARENTS[:3]
    ]
    cases = [
        json.loads(paths.resolve(ref.path).read_text(encoding="utf-8"))["case_ref"] for ref in refs
    ]
    assert len({case["path"] for case in cases}) == 3
    assert len({case["sha256"] for case in cases}) == 3


def test_the_suite_row_hands_a_loader_the_context_and_not_the_observations(
    tmp_project: Path,
) -> None:
    """Plan 11.7: an E02 data loader receives ONLY the context path, so the row must name it.

    The observations table alone is the sparse G without the masks, the controls or the
    cutoff — a loader that followed it would be reading half an input and would not know it.
    """
    paths = ProjectPaths.default(tmp_project)
    paths.ensure_dirs()
    world = render_p1(41, P1Design())
    ref = write_world(
        world, None, paths, RunContext.start(command="f", argv=[], cfg=None, paths=paths)
    )
    manifest = json.loads(paths.resolve(ref.path).read_text(encoding="utf-8"))
    row = world_row(world, None, manifest)

    assert row["context_ref"] == manifest["context_ref"]["path"]
    assert row["context_ref"].endswith(f"{DERIVED_VIEW_ID}/{CONTEXT_FILENAME}")
    assert row["context_ref"] != manifest["observation_ref"]["path"]
    target = paths.resolve(row["context_ref"])
    assert sha256_file(target) == manifest["context_ref"]["sha256"]
    assert set(json.loads(target.read_text(encoding="utf-8"))) == set(CONTEXT_ALLOWLIST)


def test_a_forward_of_another_case_is_refused(tmp_project: Path) -> None:
    """A result carrying a different model may not be published as this world's outcome."""
    paths = ProjectPaths.default(tmp_project)
    paths.ensure_dirs()
    world = render_p1(41, P1Design())
    other = render_p1(42, P1Design())
    ctx = RunContext.start(command="forward", argv=[], cfg=None, paths=paths)
    other_hash = build_p1_case(other, paths, ctx).model_hash
    foreign = _failed_result(other_hash, _forward_case_digest(world, paths))
    assert other_hash != build_p1_case(world, paths, ctx).model_hash
    with pytest.raises(ValueError, match="model_hash"):
        write_world(
            world, foreign, paths, RunContext.start(command="f", argv=[], cfg=None, paths=paths)
        )


def test_a_forward_of_different_case_bytes_is_refused(tmp_project: Path) -> None:
    """The model hash is the physics; the case digest is the bytes the solver actually read."""
    paths = ProjectPaths.default(tmp_project)
    paths.ensure_dirs()
    world = render_p1(41, P1Design())
    ctx = RunContext.start(command="forward", argv=[], cfg=None, paths=paths)
    case = build_p1_case(world, paths, ctx)
    stale = _failed_result(case.model_hash, "b" * 64)
    with pytest.raises(ValueError, match="case_sha256"):
        write_world(
            world, stale, paths, RunContext.start(command="f", argv=[], cfg=None, paths=paths)
        )


# --------------------------------------------------------------------------------------
# 11.8 / 11.9 the parent set and the outcome rows
# --------------------------------------------------------------------------------------


def test_the_first_parent_set_is_exactly_the_five_the_plan_names() -> None:
    assert P1_PARENTS == (
        (41, "base"),
        (42, "base"),
        (43, "base"),
        (44, "low_vertical"),
        (45, "high_contrast"),
    )


def _failed_result(model_hash: str, case_sha256: str) -> ForwardResult:
    return ForwardResult(
        job_id="job-x-a1",
        case_sha256=case_sha256,
        model_hash=model_hash,
        physics_class="OW",
        status="CONTROL_INFEASIBLE",
        reason="P2 could not hold its liquid rate above the 5 MPa floor",
        completed_time_s=0.0,
        times_s=(),
        states={},
        solver_metadata={},
        cost=CostRecord(
            wall_s=12.0,
            cpu_s=30.0,
            peak_rss_bytes=1024,
            output_bytes=0,
            accepted_steps=4,
            cut_steps=2,
            nonlinear_iterations=40,
            retry_count=0,
            measurement_method="test",
        ),
        parent_attempt_ids=(),
    )


def test_a_failed_forward_is_kept_as_an_outcome_row(tmp_project: Path) -> None:
    paths = ProjectPaths.default(tmp_project)
    paths.ensure_dirs()
    ctx = RunContext.start(command="forward", argv=[], cfg=None, paths=paths)
    world = render_p1(45, P1Design(family="high_contrast"))
    case = build_p1_case(world, paths, ctx)
    failed = _failed_result(case.model_hash, write_case(case, paths, ctx).sha256)
    ref = write_world(world, failed, paths, ctx)
    manifest = json.loads(paths.resolve(ref.path).read_text(encoding="utf-8"))
    assert manifest["forward_status"] == "CONTROL_INFEASIBLE"
    assert "5 MPa floor" in manifest["forward_reason"]
    assert manifest["accepted"] is False
    assert manifest["cost"]["wall_s"] == 12.0
    # A world that failed still publishes the case it failed ON (plan 3.2: nothing that is
    # present is nulled, and the absence of a RESULT is not the absence of an input).
    assert manifest["case_ref"]["sha256"] == failed.case_sha256
    row = world_row(world, failed, manifest)
    assert row["status"] == "CONTROL_INFEASIBLE"
    assert row["accepted"] is False
    assert row["seed"] == 45 and row["family"] == "high_contrast"
    assert row["cost"]["wall_s"] == 12.0
    assert row["gates"] is None


def test_complete_without_physics_evidence_is_not_accepted(tmp_project: Path) -> None:
    paths = ProjectPaths.default(tmp_project)
    paths.ensure_dirs()
    ctx = RunContext.start(command="forward", argv=[], cfg=None, paths=paths)
    world = render_p1(41, P1Design())
    case = build_p1_case(world, paths, ctx)
    complete = _failed_result(case.model_hash, write_case(case, paths, ctx).sha256).model_copy(
        update={"status": "COMPLETE", "reason": None}
    )
    ref = write_world(world, complete, paths, ctx)
    assert json.loads(paths.resolve(ref.path).read_text())["accepted"] is False


def test_versioned_p1_geometry_preserves_historical_design() -> None:
    assert P1Design().design_id == "p1-two-layer-v2"
    assert P1Design().extent_m == (100.0, 100.0, 20.0)
    assert P1Design(design_id="p1-two-layer-v1").extent_m == (400.0, 400.0, 20.0)


def _passing_physics_evidence() -> tuple[dict[str, Any], dict[str, float]]:
    from so_recon.synthetic.acceptance import SCREEN_VERSION
    from so_recon.validation.physics import load_tolerances

    tolerances = load_tolerances(Path(__file__).resolve().parents[2] / "configs/e01_tolerances.yml")
    metrics: dict[str, Any] = {
        "status": "COMPLETE",
        "balance_cumulative_relative": 0.0,
        "balance_step_median_relative": 0.0,
        "saturation_sum_abs": 0.0,
        "saturation_bound_violation": 0.0,
        "rate_control_relative": 0.0,
        "pressure_min_pa": 1e7,
        "producer_bhp_min_pa": 1e7,
        "injector_bhp_max_pa": 2e7,
        "equilibrium_preflight": {
            "status": "COMPLETE",
            "pressure_relative_drift": 0.0,
            "state_saturation_drift": 0.0,
            "connection_mass_kg_s": 0.0,
            "balance_cumulative_relative": 0.0,
            "balance_step_median_relative": 0.0,
        },
        "p2_lower_completion": {
            "shut": list(range(18, 24)),
            "open": [m for m in range(36) if not 18 <= m < 24],
        },
        "restart": {"completed_report_step": 36, "native_format": "Jutul-native"},
        "watercut_signal": {
            "version": SCREEN_VERSION,
            "by_producer": {
                well: {"valid": True, "maximum": 0.4, "range": 0.2} for well in ("P1", "P2")
            },
        },
    }
    return metrics, tolerances


@pytest.mark.parametrize("fault", ["missing", "balance", "nan", "signal", "preflight", "restart"])
def test_physics_acceptance_fails_closed(fault: str) -> None:
    from so_recon.synthetic.acceptance import evaluate_p1_acceptance

    gates, tolerances = _passing_physics_evidence()
    assert evaluate_p1_acceptance(gates, tolerances)["passed"] is True
    if fault == "missing":
        del gates["producer_bhp_min_pa"]
    elif fault == "balance":
        gates["balance_cumulative_relative"] = 1.0
    elif fault == "nan":
        gates["pressure_min_pa"] = float("nan")
    elif fault == "signal":
        for producer in gates["watercut_signal"]["by_producer"].values():
            producer["maximum"] = 0.0003
    elif fault == "preflight":
        gates["equilibrium_preflight"]["status"] = "FAILED"
    else:
        gates["restart"] = {}
    assert evaluate_p1_acceptance(gates, tolerances)["passed"] is False


@pytest.mark.parametrize("fault", ["none", "missing", "duplicate", "wrong_family", "failed_gate"])
def test_suite_requires_exact_parent_set_and_physics(tmp_project: Path, fault: str) -> None:
    from so_recon.synthetic.acceptance import evaluate_p1_acceptance
    from so_recon.synthetic.p1 import parent_world_id
    from so_recon.synthetic.world_io import write_suite_manifest

    gates, tolerances = _passing_physics_evidence()
    rows = [
        {
            "seed": seed,
            "family": family,
            "design_id": DESIGN_ID,
            "parent_world_id": parent_world_id(seed, P1Design(family=family)),
            "accepted": True,
            "status": "COMPLETE",
            "physical_acceptance": evaluate_p1_acceptance(gates, tolerances),
        }
        for seed, family in P1_PARENTS
    ]
    for row in rows:
        row["model_hash"] = "a" * 64
        row["physical_acceptance"]["checks"]["evidence_binding"] = True
        row["physical_acceptance"]["metrics"]["evidence_identity"] = {
            "parent_world_id": row["parent_world_id"],
            "forward": {"model_hash": "a" * 64},
        }
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows[-1] = rows[0]
    elif fault == "wrong_family":
        rows[-1]["family"] = "base"
    elif fault == "failed_gate":
        rows[-1]["physical_acceptance"] = None
    paths = ProjectPaths.default(tmp_project)
    paths.ensure_dirs()
    ref = write_suite_manifest(
        rows,
        paths.reports / "suite.json",
        paths,
        RunContext.start(command="test", argv=[], cfg=None, paths=paths),
    )
    suite = json.loads(paths.resolve(ref.path).read_text())
    assert suite["fully_accepted"] is (fault == "none")
    assert len(suite["rows"]) == len(rows)


def test_historical_v1_parent_ids_are_unchanged() -> None:
    from so_recon.synthetic.p1 import parent_world_id

    expected = [
        "p1-base-s0041-d45d0dc01dd5",
        "p1-base-s0042-67da57847444",
        "p1-base-s0043-45814954de89",
        "p1-low_vertical-s0044-212d729812e6",
        "p1-high_contrast-s0045-1f19d2e2680f",
    ]
    assert [
        parent_world_id(seed, P1Design(design_id="p1-two-layer-v1", family=family))
        for seed, family in P1_PARENTS
    ] == expected


@pytest.mark.parametrize("fault", ["none", "invalid", "missing", "duplicate", "startup_only"])
def test_late_signal_requires_valid_complete_dynamic_observations(
    monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    from so_recon.synthetic import acceptance

    rows = [
        {"well_id": well, "month_index": month, "fw": 0.1 + 0.01 * month, "fw_valid": True}
        for well in ("P1", "P2")
        for month in range(36)
    ]
    if fault == "invalid":
        rows[-1]["fw_valid"] = False
    elif fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows[-1] = rows[-2]
    elif fault == "startup_only":
        for row in rows:
            row["fw"] = 0.2 if row["month_index"] < 12 else 0.0
    monkeypatch.setattr(acceptance, "_rows", lambda *args: rows)
    result = _failed_result("a" * 64, "b" * 64)
    screen = acceptance.watercut_signal(result, ProjectPaths.default(Path("/tmp")))
    assert screen["passed"] is (fault == "none")


@pytest.mark.parametrize("fault", ["null_preflight", "list_preflight", "nan_producer"])
def test_malformed_nested_physical_evidence_fails_closed(fault: str) -> None:
    from so_recon.synthetic.acceptance import evaluate_p1_acceptance

    gates, tolerances = _passing_physics_evidence()
    if fault == "null_preflight":
        gates["equilibrium_preflight"] = None
    elif fault == "list_preflight":
        gates["equilibrium_preflight"] = []
    else:
        gates["watercut_signal"]["by_producer"]["P1"]["maximum"] = float("nan")
    assert evaluate_p1_acceptance(gates, tolerances)["passed"] is False


def test_arbitrary_passing_gates_cannot_accept_a_complete_world(tmp_project: Path) -> None:
    paths = ProjectPaths.default(tmp_project)
    paths.ensure_dirs()
    ctx = RunContext.start(command="forward", argv=[], cfg=None, paths=paths)
    world = render_p1(41, P1Design())
    case = build_p1_case(world, paths, ctx)
    complete = _failed_result(case.model_hash, write_case(case, paths, ctx).sha256).model_copy(
        update={"status": "COMPLETE", "reason": None}
    )
    gates, tolerances = _passing_physics_evidence()
    ref = write_world(world, complete, paths, ctx, gates=gates, tolerances=tolerances)
    assert json.loads(paths.resolve(ref.path).read_text())["accepted"] is False


def test_evidence_binding_rejects_foreign_attempt_preflight_and_changed_bytes(
    tmp_project: Path,
) -> None:
    from so_recon.synthetic.acceptance import evidence_matches_world, result_evidence_identity
    from so_recon.synthetic.p1 import closed_preflight_world

    paths = ProjectPaths.default(tmp_project)
    paths.ensure_dirs()
    ctx = RunContext.start(command="forward", argv=[], cfg=None, paths=paths)
    world = render_p1(41, P1Design())
    case = build_p1_case(world, paths, ctx)
    complete = _failed_result(case.model_hash, write_case(case, paths, ctx).sha256).model_copy(
        update={"status": "COMPLETE", "reason": None}
    )
    pre_ctx = RunContext.start(command="preflight", argv=[], cfg=None, paths=paths)
    pre_case = build_p1_case(closed_preflight_world(world), paths, pre_ctx)
    pre = _failed_result(pre_case.model_hash, write_case(pre_case, paths, pre_ctx).sha256)
    output = paths.artifacts / "binding-test.bin"
    output.write_bytes(b"scored output")
    complete = complete.model_copy(update={"monthly_path": "artifacts/binding-test.bin"})
    gates = {
        "evidence_identity": {
            "parent_world_id": world.parent_world_id,
            "forward": result_evidence_identity(complete, paths),
            "preflight": result_evidence_identity(pre, paths),
            "preflight_result": pre.model_dump(mode="json"),
        }
    }
    assert evidence_matches_world(world, complete, gates, paths, case)
    assert not evidence_matches_world(
        world, complete.model_copy(update={"job_id": "other-attempt"}), gates, paths, case
    )
    assert not evidence_matches_world(render_p1(42, P1Design()), complete, gates, paths, case)
    foreign_pre = pre.model_copy(update={"model_hash": "f" * 64})
    foreign_gates = {
        "evidence_identity": {
            **gates["evidence_identity"],
            "preflight": result_evidence_identity(foreign_pre, paths),
            "preflight_result": foreign_pre.model_dump(mode="json"),
        }
    }
    assert not evidence_matches_world(world, complete, foreign_gates, paths, case)
    output.write_bytes(b"changed after scoring")
    assert not evidence_matches_world(world, complete, gates, paths, case)
