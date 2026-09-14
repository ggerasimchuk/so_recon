"""E01.0: typed forward-case records and the safe Python->HDF5 exchange.

Every check here runs entirely in Python, before any Julia subprocess exists: a case that
cannot be trusted must be refused while it is still cheap to refuse it. The positive
assertions are the other half of the same contract — an asymmetric `(2, 3)` fixture whose
`axis_order` survives a write/read round trip, so a silent transpose cannot pass.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError

from so_recon.paths import ProjectPaths
from so_recon.registry.artifact import ArtifactImmutabilityError
from so_recon.registry.run import RunContext
from so_recon.simulator.case_io import (
    CASE_MANIFEST_FILENAME,
    CaseIntegrityError,
    cartesian_neighbors,
    case_array_refs,
    compute_model_hash,
    load_case,
    read_array,
    register_array,
    validate_case,
    validate_numeric_array,
    validate_saturations,
    write_array,
    write_case,
)
from so_recon.simulator.contracts import (
    CELL_AXES,
    CONTROL_RATE_UNIT,
    CONTROL_RATE_UNIT_KEY,
    FACE_AXES,
    MILLIDARCY_M2,
    SECONDS_PER_DAY,
    TIME_CELL_AXES,
    ArrayRef,
    BoundarySpec,
    CaseBundle,
    ControlSegment,
    CostRecord,
    FluidSpec,
    ForwardResult,
    InitialStateSpec,
    JobDescriptor,
    ObservationSpec,
    OutputRequest,
    RestartRef,
    WellSpec,
)
from tests.forward_case import (
    ASYMMETRIC,
    N_CELLS,
    SHAPE,
    build_case,
    control_segments,
    write_case_arrays,
)

NOW = datetime(2026, 9, 13, 12, 0, tzinfo=UTC)


def _paths(root: Path) -> ProjectPaths:
    return ProjectPaths.default(root)


@pytest.fixture
def case_on_disk(tmp_project: Path) -> tuple[ProjectPaths, dict[str, ArrayRef], CaseBundle]:
    paths = _paths(tmp_project)
    refs = write_case_arrays(paths)
    return paths, refs, build_case(refs)


# --------------------------------------------------------------------------------------
# 2.1 core array validation (the two checks spelled out in the brief)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1.0, 0.0])
def test_permeability_must_be_finite_positive(bad: float) -> None:
    with pytest.raises(ValueError, match="permeability"):
        validate_numeric_array("permeability", np.array([bad]), positive=True)


def test_porosity_is_not_silently_clipped() -> None:
    with pytest.raises(ValueError, match="porosity"):
        validate_numeric_array("porosity", np.array([1.2]), fraction=True)


def test_valid_arrays_pass_every_flag() -> None:
    validate_numeric_array("porosity", np.array([0.0, 0.5, 1.0]), fraction=True)
    validate_numeric_array("permeability", np.array([1e-15]), positive=True)


def test_saturations_must_sum_to_one_within_tolerance() -> None:
    sw = np.array([0.3, 0.7])
    validate_saturations(sw, 1.0 - sw)
    with pytest.raises(ValueError, match=r"saturation.*sum"):
        validate_saturations(sw, 1.0 - sw + 2e-12)


# --------------------------------------------------------------------------------------
# 2.1 / 2.6 the asymmetric HDF5 round trip: shape AND axis_order survive
# --------------------------------------------------------------------------------------


def test_asymmetric_array_round_trips_with_its_axis_order(tmp_project: Path) -> None:
    paths = _paths(tmp_project)
    ref = write_array(
        paths.artifacts / "arrays" / "states.h5",
        "sw",
        ASYMMETRIC,
        unit="1",
        axis_order=TIME_CELL_AXES,
        paths=paths,
    )
    assert ref.shape == (2, 3)
    assert ref.axis_order == TIME_CELL_AXES
    assert ref.dtype == "float64"
    assert ref.path == "artifacts/arrays/states.h5"

    back = read_array(ref, paths)
    assert back.shape == (2, 3)
    # Value by value, not just shape: a transpose of a (2,3) array is a (3,2) array, but a
    # reader that reshapes instead of transposing would keep the shape and move the values.
    assert back.tolist() == ASYMMETRIC.tolist()
    assert back[0, 2] == 4.0 and back[1, 0] == 8.0


def test_read_array_refuses_a_declared_axis_order_that_is_not_on_disk(tmp_project: Path) -> None:
    paths = _paths(tmp_project)
    ref = write_array(
        paths.artifacts / "arrays" / "states.h5",
        "sw",
        ASYMMETRIC,
        unit="1",
        axis_order=TIME_CELL_AXES,
        paths=paths,
    )
    lying = ref.model_copy(update={"axis_order": ("cell", "time")})
    with pytest.raises(ValueError, match="axis_order"):
        read_array(lying, paths)


def test_read_array_refuses_a_declared_shape_that_is_not_on_disk(tmp_project: Path) -> None:
    paths = _paths(tmp_project)
    ref = write_array(
        paths.artifacts / "arrays" / "states.h5",
        "sw",
        ASYMMETRIC,
        unit="1",
        axis_order=TIME_CELL_AXES,
        paths=paths,
    )
    with pytest.raises(ValueError, match="shape"):
        read_array(ref.model_copy(update={"shape": (3, 2)}), paths)


# --------------------------------------------------------------------------------------
# 2.1 paths, symlinks and hashes are refused before any subprocess
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_path",
    ["../outside.h5", "artifacts/../../outside.h5", "/etc/passwd", "artifacts/./a.h5"],
)
def test_array_ref_rejects_paths_that_are_not_plainly_project_relative(bad_path: str) -> None:
    with pytest.raises(ValidationError):
        ArrayRef(
            path=bad_path,
            dataset="values",
            sha256="a" * 64,
            shape=(2,),
            dtype="float64",
            unit="1",
            axis_order=CELL_AXES,
        )


def test_symlink_escaping_the_repository_is_refused(tmp_project: Path) -> None:
    paths = _paths(tmp_project)
    outside = tmp_project.parent / "outside.h5"
    real = write_array(
        paths.artifacts / "arrays" / "real.h5",
        "values",
        np.array([1.0, 2.0]),
        unit="1",
        axis_order=CELL_AXES,
        paths=paths,
    )
    outside.write_bytes((tmp_project / real.path).read_bytes())
    link = paths.artifacts / "arrays" / "linked.h5"
    link.symlink_to(outside)

    escaping = real.model_copy(update={"path": "artifacts/arrays/linked.h5"})
    with pytest.raises(ValueError, match="outside"):
        read_array(escaping, paths)


def test_wrong_sha256_is_refused(tmp_project: Path) -> None:
    paths = _paths(tmp_project)
    ref = write_array(
        paths.artifacts / "arrays" / "a.h5",
        "values",
        np.array([1.0, 2.0]),
        unit="1",
        axis_order=CELL_AXES,
        paths=paths,
    )
    with pytest.raises(ValueError, match="sha256"):
        read_array(ref.model_copy(update={"sha256": "f" * 64}), paths)


@pytest.mark.parametrize("bad", ["", "abc", "A" * 64, "g" * 64, "0" * 63])
def test_array_ref_rejects_a_malformed_digest(bad: str) -> None:
    with pytest.raises(ValidationError):
        ArrayRef(
            path="artifacts/a.h5",
            dataset="values",
            sha256=bad,
            shape=(2,),
            dtype="float64",
            unit="1",
            axis_order=CELL_AXES,
        )


def test_a_single_cell_grid_may_declare_its_empty_face_list() -> None:
    """`shape = (1, 1, 1)` has no faces, and the contract has to be able to say so.

    P0_VERIFY's own bounds start at one cell and `GridSpec.n_faces` already computes 0 for
    that grid, so an `ArrayRef` that refused the empty `(0, 2)` face list would make a case
    the plan allows inexpressible. A NEGATIVE extent is still not a shape.
    """
    ref = ArrayRef(
        path="artifacts/neighbors.h5",
        dataset="values",
        sha256="a" * 64,
        shape=(0, 2),
        dtype="int64",
        unit="1",
        axis_order=FACE_AXES,
    )
    assert ref.shape == (0, 2)
    assert cartesian_neighbors((1, 1, 1)).shape == (0, 2)
    with pytest.raises(ValidationError, match="non-negative"):
        ArrayRef.model_validate(ref.model_dump() | {"shape": (-1, 2)})


def test_unknown_unit_is_an_error() -> None:
    with pytest.raises(ValidationError, match="unit"):
        ArrayRef(
            path="artifacts/a.h5",
            dataset="values",
            sha256="a" * 64,
            shape=(2,),
            dtype="float64",
            unit="millidarcy",
            axis_order=CELL_AXES,
        )


def test_unknown_unit_in_the_case_units_table_is_an_error(
    case_on_disk: tuple[ProjectPaths, dict[str, ArrayRef], CaseBundle],
) -> None:
    _, refs, _ = case_on_disk
    with pytest.raises(ValidationError, match="units"):
        build_case(refs, units={"pressure": "bar"})


@pytest.mark.parametrize("bad", ["not-a-hash", "0" * 64, "A" * 64])
def test_a_source_hash_must_be_a_real_digest(
    case_on_disk: tuple[ProjectPaths, dict[str, ArrayRef], CaseBundle], bad: str
) -> None:
    """Synthetic source hashes name the generator and its arrays; none may be invented."""
    _, refs, _ = case_on_disk
    with pytest.raises(ValidationError, match="source_hashes"):
        build_case(refs, source_hashes={"generator": bad})


# --------------------------------------------------------------------------------------
# 2.1 physical content: NaN permeability and friends are caught by validate_case
# --------------------------------------------------------------------------------------


def test_a_complete_case_validates(
    case_on_disk: tuple[ProjectPaths, dict[str, ArrayRef], CaseBundle],
) -> None:
    paths, _, case = case_on_disk
    report = validate_case(case, paths)
    assert report.valid, report.errors
    assert report.errors == ()


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1.0, 0.0])
def test_nan_permeability_is_rejected_by_validate_case(tmp_project: Path, bad: float) -> None:
    paths = _paths(tmp_project)
    perm = np.full((3, N_CELLS), 100.0 * MILLIDARCY_M2)
    perm[1, 2] = bad
    refs = write_case_arrays(paths, permeability_m2=perm)
    report = validate_case(build_case(refs), paths)
    assert not report.valid
    assert any("rock.permeability_m2" in e for e in report.errors), report.errors


def test_porosity_outside_the_unit_interval_is_rejected(tmp_project: Path) -> None:
    paths = _paths(tmp_project)
    poro = np.full(N_CELLS, 0.25)
    poro[0] = 1.4
    report = validate_case(build_case(write_case_arrays(paths, porosity=poro)), paths)
    assert not report.valid
    assert any("rock.porosity" in e and "[0,1]" in e for e in report.errors), report.errors


def test_non_positive_initial_pressure_is_rejected(tmp_project: Path) -> None:
    paths = _paths(tmp_project)
    pressure = np.full(N_CELLS, 2.5e7)
    pressure[5] = -1.0
    report = validate_case(build_case(write_case_arrays(paths, pressure_pa=pressure)), paths)
    assert not report.valid
    assert any("initial.pressure_pa" in e for e in report.errors), report.errors


def test_initial_water_saturation_outside_the_mobile_range_is_rejected(tmp_project: Path) -> None:
    """Swc=0.2, Sorw=0.2, so Sw must stay in [0.2, 0.8]: no silent clipping (SPEC 3.1)."""
    paths = _paths(tmp_project)
    sw = np.full(N_CELLS, 0.3)
    sw[1] = 0.95
    report = validate_case(build_case(write_case_arrays(paths, sw=sw)), paths)
    assert not report.valid
    assert any("initial.sw" in e for e in report.errors), report.errors


def test_a_missing_array_file_is_reported_not_raised(
    case_on_disk: tuple[ProjectPaths, dict[str, ArrayRef], CaseBundle],
) -> None:
    paths, _, case = case_on_disk
    (paths.root / case.rock.porosity.path).unlink()
    report = validate_case(case, paths)
    assert not report.valid
    assert any("rock.porosity" in e for e in report.errors), report.errors


def test_validate_case_reports_a_symlink_escape_rather_than_crashing(tmp_project: Path) -> None:
    paths = _paths(tmp_project)
    refs = write_case_arrays(paths)
    outside = tmp_project.parent / "escaped.h5"
    outside.write_bytes((tmp_project / refs["porosity"].path).read_bytes())
    link = paths.artifacts / "arrays" / "escaped.h5"
    link.symlink_to(outside)
    escaped = refs["porosity"].model_copy(update={"path": "artifacts/arrays/escaped.h5"})
    report = validate_case(build_case({**refs, "porosity": escaped}), paths)
    assert not report.valid
    assert any("rock.porosity" in e and "outside" in e for e in report.errors), report.errors


# --------------------------------------------------------------------------------------
# 2.1 / 2.4 structural record validators
# --------------------------------------------------------------------------------------


def test_grid_requires_nx_ny_nz_elements_in_rock(tmp_project: Path) -> None:
    paths = _paths(tmp_project)
    refs = write_case_arrays(paths)
    short = write_array(
        paths.artifacts / "arrays" / "porosity_short.h5",
        "values",
        np.full(N_CELLS - 1, 0.25),
        unit="1",
        axis_order=CELL_AXES,
        paths=paths,
    )
    with pytest.raises(ValidationError, match="porosity"):
        build_case({**refs, "porosity": short})


def test_permeability_must_carry_three_directions(tmp_project: Path) -> None:
    paths = _paths(tmp_project)
    refs = write_case_arrays(paths)
    flat = write_array(
        paths.artifacts / "arrays" / "perm_flat.h5",
        "values",
        np.full(N_CELLS, 1e-13),
        unit="m2",
        axis_order=CELL_AXES,
        paths=paths,
    )
    with pytest.raises(ValidationError, match="permeability_m2"):
        build_case({**refs, "permeability_m2": flat})


@pytest.mark.parametrize(
    ("edges", "reason"),
    [
        ((0.0, SECONDS_PER_DAY, SECONDS_PER_DAY), "strictly increasing"),
        ((0.0, 2 * SECONDS_PER_DAY, SECONDS_PER_DAY), "strictly increasing"),
        ((SECONDS_PER_DAY, 2 * SECONDS_PER_DAY), "must start at 0"),
        ((0.0,), "at least one report interval"),
    ],
)
def test_report_edges_must_start_at_zero_and_increase(
    case_on_disk: tuple[ProjectPaths, dict[str, ArrayRef], CaseBundle],
    edges: tuple[float, ...],
    reason: str,
) -> None:
    _, refs, _ = case_on_disk
    with pytest.raises(ValidationError, match=reason):
        build_case(refs, report_edges_s=edges)


def test_duplicate_well_ids_are_rejected(
    case_on_disk: tuple[ProjectPaths, dict[str, ArrayRef], CaseBundle],
) -> None:
    _, refs, _ = case_on_disk
    wells = (
        WellSpec(well_id="INJ1", cells=(0, 4), reference_depth_m=1000.0, allow_crossflow=False),
        WellSpec(well_id="INJ1", cells=(3, 7), reference_depth_m=1000.0, allow_crossflow=False),
    )
    with pytest.raises(ValidationError, match="duplicate well"):
        build_case(refs, wells=wells)


def test_two_wells_may_not_share_a_connection(
    case_on_disk: tuple[ProjectPaths, dict[str, ArrayRef], CaseBundle],
) -> None:
    _, refs, _ = case_on_disk
    wells = (
        WellSpec(well_id="INJ1", cells=(0, 4), reference_depth_m=1000.0, allow_crossflow=False),
        WellSpec(well_id="PRO1", cells=(4, 7), reference_depth_m=1000.0, allow_crossflow=False),
    )
    with pytest.raises(ValidationError, match="claimed by two wells"):
        build_case(refs, wells=wells)


def test_a_well_may_not_repeat_a_cell() -> None:
    with pytest.raises(ValidationError, match="duplicate connection cells"):
        WellSpec(well_id="W", cells=(3, 3), reference_depth_m=1000.0, allow_crossflow=False)


def test_well_cells_must_exist_in_the_grid(
    case_on_disk: tuple[ProjectPaths, dict[str, ArrayRef], CaseBundle],
) -> None:
    _, refs, _ = case_on_disk
    wells = (
        WellSpec(well_id="INJ1", cells=(0, 4), reference_depth_m=1000.0, allow_crossflow=False),
        WellSpec(
            well_id="PRO1", cells=(3, N_CELLS + 1), reference_depth_m=1000.0, allow_crossflow=False
        ),
    )
    with pytest.raises(ValidationError, match="outside the grid"):
        build_case(refs, wells=wells)


def test_neighbors_must_be_the_symmetric_cartesian_topology(tmp_project: Path) -> None:
    paths = _paths(tmp_project)
    refs = write_case_arrays(paths)
    faces = cartesian_neighbors(SHAPE)
    broken = write_array(
        paths.artifacts / "arrays" / "neighbors_broken.h5",
        "values",
        faces[:-1],
        unit="1",
        axis_order=FACE_AXES,
        paths=paths,
    )
    report = validate_case(build_case({**refs, "neighbors": broken}), paths)
    assert not report.valid
    assert any("grid.neighbors" in e for e in report.errors), report.errors


@pytest.mark.parametrize(
    ("faces", "reason"),
    [
        (np.array([[0, 0]], dtype=np.int64), "self"),
        (np.array([[0, 1], [1, 0]], dtype=np.int64), "duplicate"),
        (np.array([[0, N_CELLS]], dtype=np.int64), "cell id"),
        (np.array([[0, -1]], dtype=np.int64), "cell id"),
    ],
)
def test_neighbor_edges_must_be_well_formed(
    tmp_project: Path, faces: NDArray[np.int64], reason: str
) -> None:
    paths = _paths(tmp_project)
    refs = write_case_arrays(paths)
    broken = write_array(
        paths.artifacts / "arrays" / "neighbors_bad.h5",
        "values",
        faces,
        unit="1",
        axis_order=FACE_AXES,
        paths=paths,
    )
    report = validate_case(build_case({**refs, "neighbors": broken}), paths)
    assert not report.valid
    assert any("grid.neighbors" in e and reason in e for e in report.errors), report.errors


def test_zero_gravity_needs_an_explicitly_marked_analytical_fixture(
    case_on_disk: tuple[ProjectPaths, dict[str, ArrayRef], CaseBundle],
) -> None:
    _, refs, _ = case_on_disk
    with pytest.raises(ValidationError, match="gravity"):
        build_case(refs, gravity_m_s2=0.0)
    marked = build_case(refs, gravity_m_s2=0.0, fluids=FluidSpec(analytical_limit=True))
    assert marked.gravity_m_s2 == 0.0


def test_gravity_defaults_to_the_standard_magnitude(
    case_on_disk: tuple[ProjectPaths, dict[str, ArrayRef], CaseBundle],
) -> None:
    _, _, case = case_on_disk
    assert case.grid.z_positive == "down"
    assert case.grid.crs is None
    assert case.gravity_m_s2 == 9.80665
    assert case.spec_version == "4.0"
    assert case.schema_version == "case-1"
    assert case.information_mode == "synthetic_forward"


def test_fluid_defaults_are_the_educational_reference_values() -> None:
    f = FluidSpec()
    assert f.kind == "OW"
    assert f.density_sc_kg_m3 == (1000.0, 800.0)
    assert f.viscosity_pa_s == (0.001, 0.003)
    assert f.compressibility_pa_inv == (4e-10, 1e-9)
    assert (f.p_sc_pa, f.t_sc_k) == (101325.0, 288.15)
    assert f.corey_exponents == (2.0, 2.0)
    assert f.residual_saturations == (0.2, 0.2)
    assert f.kr_endpoints == (1.0, 1.0)
    assert f.pc_model == "zero" and f.educational is True and f.analytical_limit is False


def test_residual_saturations_must_leave_a_mobile_range() -> None:
    with pytest.raises(ValidationError, match="residual_saturations"):
        FluidSpec(residual_saturations=(0.6, 0.5))


# --------------------------------------------------------------------------------------
# 2.1 controls: a producer is never given an oil (phase) rate
# --------------------------------------------------------------------------------------


def test_a_producer_cannot_be_controlled_on_an_oil_rate() -> None:
    """SPEC 9.1: a producer gets a total standard liquid rate, not a per-phase rate."""
    with pytest.raises(ValidationError):
        ControlSegment(
            start_s=0.0,
            end_s=SECONDS_PER_DAY,
            well_id="PRO1",
            role="producer",
            target="oil_rate",  # type: ignore[arg-type]
            value=10.0,
            bhp_limit_pa=1.0e7,
            connection_open=(True,),
        )


def test_a_producer_cannot_be_controlled_on_a_water_rate() -> None:
    with pytest.raises(ValidationError, match="producer"):
        ControlSegment(
            start_s=0.0,
            end_s=SECONDS_PER_DAY,
            well_id="PRO1",
            role="producer",
            target="water_rate",
            value=10.0,
            bhp_limit_pa=1.0e7,
            connection_open=(True,),
        )


def test_an_injector_cannot_be_controlled_on_a_liquid_rate() -> None:
    with pytest.raises(ValidationError, match="injector"):
        ControlSegment(
            start_s=0.0,
            end_s=SECONDS_PER_DAY,
            well_id="INJ1",
            role="injector",
            target="liquid_rate",
            value=10.0,
            bhp_limit_pa=None,
            connection_open=(True,),
        )


def test_a_shut_well_must_carry_the_disabled_target() -> None:
    with pytest.raises(ValidationError, match="shut"):
        ControlSegment(
            start_s=0.0,
            end_s=SECONDS_PER_DAY,
            well_id="PRO1",
            role="shut",
            target="bhp",
            value=1.0e7,
            bhp_limit_pa=None,
            connection_open=(True,),
        )
    shut = ControlSegment(
        start_s=0.0,
        end_s=SECONDS_PER_DAY,
        well_id="PRO1",
        role="shut",
        target="disabled",
        value=0.0,
        bhp_limit_pa=None,
        connection_open=(True,),
    )
    assert shut.value == 0.0


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_a_rate_target_needs_a_positive_magnitude(value: float) -> None:
    with pytest.raises(ValidationError, match="value"):
        ControlSegment(
            start_s=0.0,
            end_s=SECONDS_PER_DAY,
            well_id="INJ1",
            role="injector",
            target="water_rate",
            value=value,
            bhp_limit_pa=None,
            connection_open=(True,),
        )


def test_a_segment_must_span_a_positive_interval() -> None:
    with pytest.raises(ValidationError, match="end_s"):
        ControlSegment(
            start_s=SECONDS_PER_DAY,
            end_s=SECONDS_PER_DAY,
            well_id="INJ1",
            role="injector",
            target="water_rate",
            value=1.0,
            bhp_limit_pa=None,
            connection_open=(True,),
        )


def test_connection_mask_must_match_the_well_cells(
    case_on_disk: tuple[ProjectPaths, dict[str, ArrayRef], CaseBundle],
) -> None:
    _, refs, _ = case_on_disk
    inj, pro = control_segments()
    mismatched = (inj, pro.model_copy(update={"connection_open": (True, True, True)}))
    with pytest.raises(ValidationError, match="connection_open"):
        build_case(refs, controls=mismatched)


def test_control_rates_cross_the_exchange_as_human_day_rates(
    case_on_disk: tuple[ProjectPaths, dict[str, ArrayRef], CaseBundle],
) -> None:
    """Pins the unit on both sides of the 86400 that Julia applies (plan 3.1, SPEC 9.1).

    `ControlSegment.value` is m3_sc/day for a rate target, and Julia divides by
    SECONDS_PER_DAY when it builds the model. If either side ever moves to native
    m3_sc/s, this test and the declared unit below break together instead of a forward
    run quietly being wrong by a factor of 86400.
    """
    _, _, case = case_on_disk
    assert CONTROL_RATE_UNIT == "m3_sc/day"
    assert case.units[CONTROL_RATE_UNIT_KEY] == CONTROL_RATE_UNIT

    injector = next(c for c in case.controls if c.role == "injector")
    producer = next(c for c in case.controls if c.role == "producer")
    assert (injector.target, injector.value) == ("water_rate", 50.0)
    assert (producer.target, producer.value) == ("liquid_rate", 40.0)
    # The native rates Julia will construct from those values.
    assert injector.value / SECONDS_PER_DAY == pytest.approx(5.7870370370e-4, rel=1e-9)
    assert producer.value / SECONDS_PER_DAY == pytest.approx(4.6296296296e-4, rel=1e-9)
    # A bhp target is Pa, not a rate, so it is not touched by that conversion.
    assert case.units["pressure"] == "Pa"


@pytest.mark.parametrize(
    "units",
    [
        {"pressure": "Pa"},  # the convention is simply not stated
        {"control_rate": "m3_sc/s"},  # stated, but as the native unit Julia produces
        {"control_rate": "m3"},  # stated, but not a rate at all
    ],
    ids=["absent", "native", "not-a-rate"],
)
def test_a_case_must_declare_the_unit_its_control_rates_are_written_in(
    case_on_disk: tuple[ProjectPaths, dict[str, ArrayRef], CaseBundle],
    units: dict[str, str],
) -> None:
    _, refs, _ = case_on_disk
    with pytest.raises(ValidationError, match=CONTROL_RATE_UNIT_KEY):
        build_case(refs, units=units)


def test_a_control_segment_must_name_a_declared_well(
    case_on_disk: tuple[ProjectPaths, dict[str, ArrayRef], CaseBundle],
) -> None:
    _, refs, _ = case_on_disk
    inj, pro = control_segments()
    with pytest.raises(ValidationError, match="unknown well"):
        build_case(refs, controls=(inj, pro.model_copy(update={"well_id": "GHOST"})))


def test_control_segments_of_one_well_may_not_overlap(
    case_on_disk: tuple[ProjectPaths, dict[str, ArrayRef], CaseBundle],
) -> None:
    _, refs, _ = case_on_disk
    inj, pro = control_segments()
    overlap = pro.model_copy(
        update={"start_s": 10.0 * SECONDS_PER_DAY, "end_s": 20.0 * SECONDS_PER_DAY}
    )
    with pytest.raises(ValidationError, match="overlap"):
        build_case(refs, controls=(inj, pro, overlap))


def test_control_segments_may_not_run_past_the_last_report_edge(
    case_on_disk: tuple[ProjectPaths, dict[str, ArrayRef], CaseBundle],
) -> None:
    _, refs, _ = case_on_disk
    inj, pro = control_segments()
    late = pro.model_copy(update={"end_s": 90.0 * SECONDS_PER_DAY})
    with pytest.raises(ValidationError, match="report"):
        build_case(refs, controls=(inj, late))


# --------------------------------------------------------------------------------------
# 2.4 remaining records: initial state, boundary, observations, job, result
# --------------------------------------------------------------------------------------


def test_explicit_initial_state_needs_pressure_and_saturation(
    case_on_disk: tuple[ProjectPaths, dict[str, ArrayRef], CaseBundle],
) -> None:
    _, refs, _ = case_on_disk
    with pytest.raises(ValidationError, match="explicit"):
        InitialStateSpec(
            kind="explicit",
            pressure_pa=refs["pressure_pa"],
            sw=None,
            meaning="synthetic_initial",
        )


def test_equilibrium_initial_state_carries_no_arrays(
    case_on_disk: tuple[ProjectPaths, dict[str, ArrayRef], CaseBundle],
) -> None:
    _, refs, _ = case_on_disk
    equilibrium = InitialStateSpec(kind="equilibrium", meaning="synthetic_initial")
    assert equilibrium.pressure_pa is None
    with pytest.raises(ValidationError, match="equilibrium"):
        InitialStateSpec(
            kind="equilibrium",
            pressure_pa=refs["pressure_pa"],
            meaning="synthetic_initial",
        )


def _restart() -> RestartRef:
    return RestartRef(
        manifest_path="artifacts/runs/r1/restart.json",
        sha256="c" * 64,
        completed_report_step=2,
        completed_time_s=30.0 * SECONDS_PER_DAY,
        model_hash="d" * 64,
        schedule_prefix_hash="e" * 64,
        environment_lock_hash="f" * 64,
    )


def test_native_restart_initial_state_needs_a_restart_reference() -> None:
    with pytest.raises(ValidationError, match="native_restart"):
        InitialStateSpec(kind="native_restart", meaning="developed_state")
    state = InitialStateSpec(kind="native_restart", restart=_restart(), meaning="developed_state")
    assert state.restart is not None
    assert state.restart.native_format == "Jutul-native"


def test_closed_boundary_carries_no_cells_and_no_pressure() -> None:
    assert BoundarySpec(kind="closed", cells=()).cells == ()
    with pytest.raises(ValidationError, match="closed"):
        BoundarySpec(kind="closed", cells=(1,))
    with pytest.raises(ValidationError, match="closed"):
        BoundarySpec(kind="closed", cells=(), pressure_pa=2.5e7)


def test_pressure_boundary_needs_cells_pressure_and_transmissibility() -> None:
    with pytest.raises(ValidationError, match="pressure_water"):
        BoundarySpec(kind="pressure_water", cells=(1,), pressure_pa=2.5e7)
    spec = BoundarySpec(kind="pressure_water", cells=(1,), pressure_pa=2.5e7, trans_flow=1e-12)
    assert spec.fractional_flow == (1.0, 0.0)


def test_boundary_fractional_flow_must_sum_to_one() -> None:
    with pytest.raises(ValidationError, match="fractional_flow"):
        BoundarySpec(
            kind="pressure_water",
            cells=(1,),
            pressure_pa=2.5e7,
            trans_flow=1e-12,
            fractional_flow=(0.9, 0.2),
        )


def test_an_empty_observations_table_is_allowed() -> None:
    obs = ObservationSpec(dynamic_channels=(), pressure_available=False)
    assert obs.table_path is None and obs.sha256 is None
    assert obs.truth_access == "forbidden"


def test_an_observations_table_needs_both_a_path_and_a_digest() -> None:
    with pytest.raises(ValidationError, match="sha256"):
        ObservationSpec(
            table_path="data/interim/obs.parquet",
            sha256=None,
            dynamic_channels=("liquid_rate",),
            pressure_available=False,
        )


def test_output_request_times_must_increase() -> None:
    assert (
        OutputRequest(state_times_s=(0.0, SECONDS_PER_DAY), keep_native_restart=False).chunk_months
        == 1
    )
    with pytest.raises(ValidationError, match="increasing"):
        OutputRequest(state_times_s=(SECONDS_PER_DAY, 0.0), keep_native_restart=False)


def _job(**overrides: Any) -> JobDescriptor:
    fields: dict[str, Any] = {
        "job_id": "job-0001",
        "case_path": "artifacts/runs/r1/case.json",
        "case_sha256": "a" * 64,
        "model_hash": "b" * 64,
        "solver_config_path": "configs/solver.json",
        "solver_config_sha256": "c" * 64,
        "output_request": OutputRequest(
            state_times_s=(0.0, SECONDS_PER_DAY), keep_native_restart=False
        ),
        "seed": 1,
        "result_dir": "artifacts/runs/r1/forward",
        "attempt": 1,
        "resume_from": None,
    }
    fields.update(overrides)
    return JobDescriptor(**fields)


def test_job_descriptor_allows_at_most_one_registered_retry() -> None:
    assert _job().schema_version == "job-1"
    assert _job(attempt=2, parent_job_id="job-0000").attempt == 2
    with pytest.raises(ValidationError, match="attempt"):
        _job(attempt=3, parent_job_id="job-0000")
    with pytest.raises(ValidationError, match="attempt"):
        _job(attempt=0)


def test_a_retry_is_registered_against_the_attempt_it_repeats() -> None:
    """SPEC 3.3 allows one REGISTERED retry; an anonymous second attempt is not one."""
    assert _job().parent_job_id is None
    with pytest.raises(ValidationError, match="parent_job_id"):
        _job(attempt=2)
    with pytest.raises(ValidationError, match="no parent"):
        _job(attempt=1, parent_job_id="job-0000")
    with pytest.raises(ValidationError, match="its own parent"):
        _job(attempt=2, parent_job_id="job-0001")


def _cost() -> CostRecord:
    return CostRecord(
        wall_s=1.5,
        cpu_s=5.0,
        peak_rss_bytes=1 << 30,
        output_bytes=4096,
        accepted_steps=12,
        cut_steps=1,
        nonlinear_iterations=48,
        retry_count=0,
        measurement_method="resource.getrusage",
    )


def _result(**overrides: Any) -> ForwardResult:
    fields: dict[str, Any] = {
        "job_id": "job-0001",
        "case_sha256": "a" * 64,
        "model_hash": "b" * 64,
        "physics_class": "OW_immiscible",
        "status": "COMPLETE",
        "reason": None,
        "completed_time_s": 30.0 * SECONDS_PER_DAY,
        "times_s": (0.0, 15.0 * SECONDS_PER_DAY, 30.0 * SECONDS_PER_DAY),
        "states": {
            "sw": ArrayRef(
                path="artifacts/runs/r1/forward/states.h5",
                dataset="sw",
                sha256="d" * 64,
                shape=(3, N_CELLS),
                dtype="float64",
                unit="1",
                axis_order=TIME_CELL_AXES,
            )
        },
        "monthly_path": "artifacts/runs/r1/forward/monthly.parquet",
        "connections_path": "artifacts/runs/r1/forward/connections.parquet",
        "balances_path": "artifacts/runs/r1/forward/balances.parquet",
        "restart": None,
        "solver_metadata": {"linear_solver": "cpr"},
        "cost": _cost(),
        "parent_attempt_ids": (),
    }
    fields.update(overrides)
    return ForwardResult(**fields)


def test_a_complete_result_carries_every_path_and_no_reason() -> None:
    res = _result()
    assert res.schema_version == "forward-1"
    assert res.reason is None
    assert res.states["sw"].shape == (3, N_CELLS)


@pytest.mark.parametrize("field", ["monthly_path", "connections_path", "balances_path"])
def test_a_complete_result_may_not_drop_a_result_path(field: str) -> None:
    with pytest.raises(ValidationError, match="COMPLETE"):
        _result(**{field: None})


def test_a_complete_result_needs_states_covering_the_requested_axis() -> None:
    with pytest.raises(ValidationError, match="COMPLETE"):
        _result(states={})
    with pytest.raises(ValidationError, match="times_s"):
        _result(
            states={
                "sw": _result().states["sw"].model_copy(update={"shape": (2, N_CELLS)}),
            }
        )


def test_an_unsuccessful_result_may_drop_paths_but_must_give_a_reason() -> None:
    failed = _result(
        status="NUMERICAL_FAILURE",
        reason="timestep cut below the minimum at t=1.3e6 s",
        monthly_path=None,
        connections_path=None,
        balances_path=None,
        states={},
        times_s=(),
        completed_time_s=0.0,
    )
    assert failed.monthly_path is None
    with pytest.raises(ValidationError, match="reason"):
        _result(status="NUMERICAL_FAILURE", reason=None, states={}, times_s=())


def test_a_missing_hash_is_never_a_zero_string() -> None:
    """An absent digest is `None` plus a reason, never an all-zero placeholder (SPEC 3.2)."""
    with pytest.raises(ValidationError, match="zero"):
        _result(case_sha256="0" * 64)
    with pytest.raises(ValidationError, match="zero"):
        RestartRef.model_validate({**_restart().model_dump(), "sha256": "0" * 64})


# --------------------------------------------------------------------------------------
# 2.5 immutable publication
# --------------------------------------------------------------------------------------


def test_writing_the_same_array_twice_is_idempotent(tmp_project: Path) -> None:
    paths = _paths(tmp_project)
    target = paths.artifacts / "arrays" / "a.h5"
    first = write_array(
        target, "values", ASYMMETRIC, unit="1", axis_order=TIME_CELL_AXES, paths=paths
    )
    before = target.read_bytes()
    second = write_array(
        target, "values", ASYMMETRIC, unit="1", axis_order=TIME_CELL_AXES, paths=paths
    )
    assert first == second
    assert target.read_bytes() == before


def test_rewriting_an_array_path_with_different_bytes_is_refused(tmp_project: Path) -> None:
    paths = _paths(tmp_project)
    target = paths.artifacts / "arrays" / "a.h5"
    write_array(target, "values", ASYMMETRIC, unit="1", axis_order=TIME_CELL_AXES, paths=paths)
    before = target.read_bytes()
    with pytest.raises(ArtifactImmutabilityError):
        write_array(
            target, "values", ASYMMETRIC * 2.0, unit="1", axis_order=TIME_CELL_AXES, paths=paths
        )
    assert target.read_bytes() == before


def test_a_failed_array_write_leaves_no_staging_file_behind(tmp_project: Path) -> None:
    paths = _paths(tmp_project)
    target = paths.artifacts / "arrays" / "a.h5"
    write_array(target, "values", ASYMMETRIC, unit="1", axis_order=TIME_CELL_AXES, paths=paths)
    with pytest.raises(ArtifactImmutabilityError):
        write_array(
            target, "values", ASYMMETRIC * 2.0, unit="1", axis_order=TIME_CELL_AXES, paths=paths
        )
    assert sorted(p.name for p in target.parent.iterdir()) == ["a.h5"]


def test_write_case_registers_every_array_then_publishes_the_manifest(tmp_project: Path) -> None:
    """The whole of step 2.5 in one call: arrays into the registry, manifest last."""
    paths = _paths(tmp_project)
    refs = write_case_arrays(paths)
    case = build_case(refs)
    ctx = RunContext.start(command="forward", argv=[], cfg=None, paths=paths)

    manifest = write_case(case, paths, ctx, now=NOW)
    assert manifest.path == f"{paths.relative(ctx.run_dir)}/{CASE_MANIFEST_FILENAME}"
    assert manifest.schema_version == "case-1"
    assert ctx.record.outputs["case"].sha256 == manifest.sha256

    # Every array the case depends on is a recorded output, keyed by its place in the
    # record -- not merely a digest quoted by a manifest that nothing else knows about.
    labels = set(case_array_refs(case))
    assert labels <= set(ctx.record.outputs)
    for label, ref in case_array_refs(case).items():
        recorded = ctx.record.outputs[label]
        assert recorded.sha256 == ref.sha256
        assert recorded.path == ref.path
        assert recorded.media_type == "application/x-hdf5"
        assert recorded.producer_run_id == ctx.run_id

    # The lineage the manifest claims resolves to artifacts the run actually recorded.
    assert sorted(manifest.parent_artifact_ids) == sorted(
        {ctx.record.outputs[label].artifact_id for label in labels}
    )
    assert load_case(paths.root / manifest.path, paths) == case


def test_register_array_refuses_a_reference_that_does_not_match_its_file(
    tmp_project: Path,
) -> None:
    paths = _paths(tmp_project)
    refs = write_case_arrays(paths)
    ctx = RunContext.start(command="forward", argv=[], cfg=None, paths=paths)
    lying = refs["porosity"].model_copy(update={"sha256": "9" * 64})
    with pytest.raises(ValueError, match="hashes to"):
        register_array(lying, paths, ctx, key="rock.porosity", now=NOW)


def test_write_case_refuses_an_invalid_case_and_publishes_nothing(tmp_project: Path) -> None:
    paths = _paths(tmp_project)
    poro = np.full(N_CELLS, 0.25)
    poro[0] = np.nan
    case = build_case(write_case_arrays(paths, porosity=poro))
    ctx = RunContext.start(command="forward", argv=[], cfg=None, paths=paths)
    with pytest.raises(ValueError, match="rock.porosity"):
        write_case(case, paths, ctx, now=NOW)
    assert not (ctx.run_dir / CASE_MANIFEST_FILENAME).exists()
    # Validation runs before the first registry write, so nothing at all was published.
    assert ctx.record.outputs == {}


def test_republishing_the_same_case_is_idempotent(tmp_project: Path) -> None:
    paths = _paths(tmp_project)
    case = build_case(write_case_arrays(paths))
    ctx = RunContext.start(command="forward", argv=[], cfg=None, paths=paths)
    first = write_case(case, paths, ctx, now=NOW)
    second = write_case(case, paths, ctx, now=NOW)
    assert first.sha256 == second.sha256


# --------------------------------------------------------------------------------------
# 2.6 the model hash is the thing that makes a tampered case JSON detectable
# --------------------------------------------------------------------------------------


def test_model_hash_ignores_wall_time_ram_and_hostname_shaped_fields(tmp_project: Path) -> None:
    """Identity and provenance labels are not inputs to F, so they must not move the hash."""
    paths = _paths(tmp_project)
    refs = write_case_arrays(paths)
    base = build_case(refs)
    relabelled = base.model_copy(
        update={
            "case_id": "case-other",
            "world_id": "world-other",
            "source_hashes": {"generator": "9" * 64},
        }
    )
    assert compute_model_hash(relabelled) == compute_model_hash(base)


def test_model_hash_moves_with_every_input_that_changes_f(tmp_project: Path) -> None:
    paths = _paths(tmp_project)
    refs = write_case_arrays(paths)
    base = build_case(refs)
    thicker = build_case(refs, fluids=FluidSpec(viscosity_pa_s=(0.001, 0.004)))
    assert compute_model_hash(thicker) != compute_model_hash(base)
    inj, pro = control_segments()
    changed = (inj.model_copy(update={"value": 99.0 / SECONDS_PER_DAY}), pro)
    assert compute_model_hash(build_case(refs, controls=changed)) != compute_model_hash(base)
    # The array digests reach the hash through the ArrayRefs the case carries.
    other = write_case_arrays(paths, subdir="alt", porosity=np.full(N_CELLS, 0.3))
    assert compute_model_hash(build_case(other)) != compute_model_hash(base)


@pytest.mark.parametrize(
    "mutation",
    [
        {"fluids": {"viscosity_pa_s": [0.001, 0.009]}},
        {"controls": "value"},
        {"rock": {"permeability_m2": {"sha256": "9" * 64}}},
    ],
    ids=["fluid", "controls", "array-sha"],
)
def test_a_tampered_case_json_keeping_its_model_hash_is_refused(
    tmp_project: Path, mutation: dict[str, Any]
) -> None:
    paths = _paths(tmp_project)
    case = build_case(write_case_arrays(paths))
    ctx = RunContext.start(command="forward", argv=[], cfg=None, paths=paths)
    ref = write_case(case, paths, ctx, now=NOW)
    payload = json.loads((paths.root / ref.path).read_text(encoding="utf-8"))

    if "fluids" in mutation:
        payload["fluids"]["viscosity_pa_s"] = [0.001, 0.009]
    elif "controls" in mutation:
        payload["controls"][0]["value"] = payload["controls"][0]["value"] * 2.0
    else:
        payload["rock"]["permeability_m2"]["sha256"] = "9" * 64

    tampered = paths.artifacts / "tampered.json"
    tampered.parent.mkdir(parents=True, exist_ok=True)
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CaseIntegrityError, match="model_hash"):
        load_case(tampered, paths)


def test_load_case_refuses_a_manifest_outside_the_repository(tmp_project: Path) -> None:
    paths = _paths(tmp_project)
    outside = tmp_project.parent / "case.json"
    outside.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="outside"):
        load_case(outside, paths)
