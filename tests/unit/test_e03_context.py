"""Context builder, encoder and scaler contracts (plan E03 Task 04, §9 tests).

The one fixture is a synthetic canonical payload shaped exactly like the corpus context
stream; no physics runs here. What these tests pin is what plan §7.1 demands: typed
allowlist behaviour, zeros-vs-missing, prefix invariance, BHP-vs-rate encoding, and
permutation invariance of the three encoders at Float64.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pytest
import torch

from so_recon.config.learning import EncoderParams
from so_recon.ml.context import (
    BHP_MAX_PA,
    BHP_MIN_PA,
    ForbiddenContextInput,
    build_context_batch,
    default_context_spec,
    permute_batch,
)
from so_recon.ml.contracts import ContextBatch
from so_recon.ml.encoders import GraphEncoder, SummaryEncoder, TemporalSetEncoder, build_encoder
from so_recon.ml.normalization import FeatureScaler
from so_recon.simulator.schedule import month_edges_s
from so_recon.synthetic.inverse_corpus import WELL_TIME_FEATURES


def _control(well: str, target: str, value: float, *, upper: bool = True, lower: bool = True):
    return {
        "well_id": well,
        "start_s": 0.0,
        "end_s": 95_000_000.0,
        "role": "producer" if well.startswith("P") else "injector",
        "target": target,
        "value": value,
        "bhp_limit_pa": None,
        "connection_open": (upper, lower),
    }


def _payload(*, bhp: bool = False) -> dict[str, Any]:
    wells = [("I1", 2, 2, 0), ("I2", 2, 13, 0), ("P1", 13, 2, 1), ("P2", 13, 13, 1)]
    design = {
        "design_id": "e02-t2-v2",
        "shape": [16, 16, 2],
        "n_months": 36,
        "start_date": "2000-01-01",
        "well_columns": [
            {"well_id": name, "column": [i, j], "role": "producer" if role else "injector"}
            for name, i, j, role in wells
        ],
    }
    months = 36
    edges = month_edges_s(date(2000, 1, 1), months)
    # per-month control segments on the declared calendar (one segment per month edge)
    controls = []
    for name, _i, _j, role in wells:
        for month in range(months):
            controls.append(
                {
                    **_control(
                        name,
                        "bhp" if bhp else ("water_rate" if role == 0 else "liquid_rate"),
                        1.0e7 if bhp else 20.0,
                    ),
                    "start_s": edges[month],
                    "end_s": edges[month + 1],
                }
            )
    history = []
    for name, _i, _j, role in wells:
        if role == 0:
            continue
        for month in range(months):
            observed = month >= 0
            history.append(
                {
                    "well_id": name,
                    "month_index": month,
                    "raw_value": 0.31 if observed else None,
                    "bin_index": 31 if observed else None,
                    "quality_group": "watercut-0.01",
                    "observed_valid": observed,
                    "reset": False,
                    "sigma_multiplier": 1.0,
                }
            )
    grid_edges = [round(value, 4) for value in np.linspace(0.0, 1.0, 101)]
    observations = {
        "history": history,
        "logs": [],
        "bin_edges_by_group": {"watercut-0.01": grid_edges},
        "cutoff_s": edges[-1],
        "information_hash": "a" * 64,
        "observation_hash": "b" * 64,
    }
    return {
        "schema_version": "e03-parent-context-1",
        "parent_id": "fixture-0000",
        "design_id": "e02-t2-v2",
        "split": "train",
        "cutoff_s": edges[-1],
        "well_ids": ["P1", "P2"],
        "inference_input": {
            "context": {
                "design": {"inverse_design": design},
                "density_schema": {
                    "schema_id": "e02-t2-symmetric-12",
                    "n_v": 11,
                    "n_residual": 4,
                    "families": [0],
                    "basis_hash": "c" * 64,
                    "transform_version": "e02-t2-v2-conditional-1",
                    "measure": "counting_x_latent_lebesgue",
                },
            },
            "G": [
                {
                    "observation_id": f"{name}-L0-log_permeability_m2",
                    "support_cell_ids": [index],
                    "value": -13.0 + 0.1 * index,
                    "sigma": 0.2,
                }
                for index, (name, _i, _j, _r) in enumerate(wells)
            ],
            "U": controls,
            "observations": observations,
            "density_schema": observations and {},
            "basis": {"basis_hash": "c" * 64, "transform_version": "e02-t2-v2-conditional-1"},
        },
    }


@pytest.fixture(scope="module")
def spec():
    return default_context_spec(cutoff_s=9.5e7, max_wells=8, max_months=36)


def test_builds_the_declared_tensor_shapes(spec) -> None:
    batch = build_context_batch(_payload(), spec=spec)
    assert batch.well_time.shape == (4, 36, len(WELL_TIME_FEATURES))
    assert batch.well_mask.all()
    assert batch.static.shape == (4, len(spec.static_features))
    assert batch.edge_attr.shape[1] == len(spec.edge_features)
    assert batch.well_ids == ("I1", "I2", "P1", "P2")


def test_bhp_protocol_is_encoded_as_bhp_not_as_a_rate(spec) -> None:
    rate = build_context_batch(_payload(bhp=False), spec=spec)
    bhp = build_context_batch(_payload(bhp=True), spec=spec)
    kinds = {name: index for index, name in enumerate(spec.well_time_features)}
    # a rate protocol never lights the BHP kind, and vice versa: the two protocols are
    # different channels, never one dimensionless column (plan §7.1)
    np.testing.assert_array_equal(rate.well_time[..., kinds["control_kind_bhp"]], 0.0)
    assert rate.well_time[..., kinds["control_kind_water_rate"]].max() == 1.0
    assert bhp.well_time[..., kinds["control_kind_bhp"]].max() == 1.0
    assert bhp.well_time[..., kinds["control_kind_liquid_rate"]].max() == 0.0
    # BHP values normalize by the declared pressure range, not the rate scale
    value = bhp.well_time[0, 0, kinds["control_value_normalized"]]
    assert value == pytest.approx((1.0e7 - BHP_MIN_PA) / (BHP_MAX_PA - BHP_MIN_PA))


def test_observed_zero_is_not_missing(spec) -> None:
    payload = _payload()
    for row in payload["inference_input"]["observations"]["history"]:
        row["bin_index"] = 0
        row["raw_value"] = 0.0
    zero = build_context_batch(payload, spec=spec)
    kinds = {name: index for index, name in enumerate(spec.well_time_features)}
    producers = [index for index, well in enumerate(zero.well_ids) if well.startswith("P")]
    for w in producers:
        assert zero.well_time[w, 0, kinds["observed_valid"]] == 1.0
        assert zero.well_time[w, 0, kinds["observed_bin_center"]] == 0.0
    payload_missing = _payload()
    for row in payload_missing["inference_input"]["observations"]["history"]:
        row["observed_valid"] = False
        row["bin_index"] = None
        row["raw_value"] = None
    missing = build_context_batch(payload_missing, spec=spec)
    for w in producers:
        assert missing.well_time[w, 0, kinds["observed_valid"]] == 0.0
        assert missing.well_time[w, 0, kinds["observed_bin_center"]] == 0.0
        # the validity channel is what separates an observed zero from a missing month
        assert (
            zero.well_time[w, 0, kinds["observed_valid"]]
            != missing.well_time[w, 0, kinds["observed_valid"]]
        )


def test_future_tail_mutation_cannot_change_a_prefix(spec) -> None:
    full = build_context_batch(_payload(), spec=spec)
    prefix = build_context_batch(_payload(), spec=spec, prefix_months=12)
    np.testing.assert_array_equal(prefix.well_time, full.well_time[:, :12, :])
    np.testing.assert_array_equal(prefix.well_mask, full.well_mask[:, :12])
    mutated = _payload()
    for row in mutated["inference_input"]["observations"]["history"]:
        if row["month_index"] >= 12:
            row["bin_index"] = 99
            row["raw_value"] = 0.99
    again = build_context_batch(mutated, spec=spec, prefix_months=12)
    np.testing.assert_array_equal(again.well_time, prefix.well_time)


def test_prefix_bounds_are_validated(spec) -> None:
    with pytest.raises(ValueError, match="outside"):
        build_context_batch(_payload(), spec=spec, prefix_months=99)


@pytest.mark.parametrize(
    "key",
    (
        "truth",
        "theta",
        "seed",
        "s",
        "full_permeability",
        "full_porosity",
        "full_so",
        "full_pressure",
    ),
)
def test_forbidden_inputs_are_refused(spec, key) -> None:
    payload = _payload()
    payload["inference_input"][key] = {"leak": True}
    with pytest.raises(ForbiddenContextInput, match=repr(key)):
        build_context_batch(payload, spec=spec)


def test_wells_missing_from_the_declared_geometry_are_refused(spec) -> None:
    payload = _payload()
    payload["inference_input"]["observations"]["history"].append(
        {
            "well_id": "GHOST",
            "month_index": 0,
            "raw_value": 0.5,
            "bin_index": 50,
            "quality_group": "watercut-0.01",
            "observed_valid": True,
            "reset": False,
            "sigma_multiplier": 1.0,
        }
    )
    with pytest.raises(ForbiddenContextInput, match="GHOST"):
        build_context_batch(payload, spec=spec)


def test_p1_corpus_design_builds_from_declared_constants(spec) -> None:
    """The P1 corpus publishes no `well_columns` and BLOCK controls, not per-month ones.

    The real thin-slice corpus (e03-corpus-thin-1) is exactly this shape: its design is
    the frozen P1 constants, and its controls change three times in 36 months. The
    builder must read geometry from the declared design and months from the declared
    calendar — never guess months from control starts.
    """
    payload = _payload()
    payload["inference_input"]["context"]["design"] = {
        "p1_design": {
            "design_id": "p1-two-layer-v2",
            "family": "base",
            "shape": [16, 16, 2],
            "extent_m": [100.0, 100.0, 20.0],
            "n_months": 36,
            "start_date": "2000-01-01",
        }
    }
    edges = month_edges_s(date(2000, 1, 1), 36)
    blocks = []
    for name, _i, _j, role in [("I1", 0, 0, 0), ("I2", 0, 0, 0), ("P1", 0, 0, 1), ("P2", 0, 0, 1)]:
        for first, last, factor in ((0, 12, 1.0), (12, 24, 0.75), (24, 36, 1.25)):
            blocks.append(
                {
                    **_control(name, "water_rate" if role == 0 else "liquid_rate", 20.0 * factor),
                    "start_s": edges[first],
                    "end_s": edges[last],
                }
            )
    payload["inference_input"]["U"] = blocks
    batch = build_context_batch(payload, spec=spec)
    assert batch.well_ids == ("I1", "I2", "P1", "P2")
    kinds = {name: index for index, name in enumerate(spec.well_time_features)}
    i1 = batch.well_ids.index("I1")
    assert batch.well_time[i1, 5, kinds["control_value_normalized"]] == pytest.approx(1.0)
    assert batch.well_time[i1, 15, kinds["control_value_normalized"]] == pytest.approx(0.75)
    assert batch.well_time[i1, 30, kinds["control_value_normalized"]] == pytest.approx(1.25)
    # P1 geometry comes from the declared constants: P1 sits at column (13, 2)
    p1 = batch.well_ids.index("P1")
    assert batch.static[p1, spec.static_features.index("well_i_normalized")] == pytest.approx(
        13.0 / 16.0
    )
    assert batch.static[p1, spec.static_features.index("well_j_normalized")] == pytest.approx(
        2.0 / 16.0
    )


def test_a_calendar_that_disagrees_with_the_history_is_refused(spec) -> None:
    payload = _payload()
    payload["inference_input"]["context"]["design"]["inverse_design"]["n_months"] = 24
    with pytest.raises(ForbiddenContextInput, match="calendar"):
        build_context_batch(payload, spec=spec)


def test_shut_connection_is_its_own_channel(spec) -> None:
    payload = _payload()
    for segment in payload["inference_input"]["U"]:
        if segment["well_id"] == "P1" and segment["start_s"] > 0:
            segment["connection_open"] = (True, False)
    batch = build_context_batch(payload, spec=spec)
    kinds = {name: index for index, name in enumerate(spec.well_time_features)}
    p1 = batch.well_ids.index("P1")
    assert batch.well_time[p1, 0, kinds["lower_connection_open"]] == 1.0
    assert batch.well_time[p1, 20, kinds["lower_connection_open"]] == 0.0
    assert batch.well_time[p1, 20, kinds["upper_connection_open"]] == 1.0


@pytest.fixture(scope="module")
def batches(spec):
    payloads = [_payload()]
    other = _payload()
    for row in other["inference_input"]["observations"]["history"]:
        row["bin_index"] = 45
    payloads.append(other)
    return [build_context_batch(payload, spec=spec) for payload in payloads]


@pytest.mark.parametrize("encoder_cls", [GraphEncoder, TemporalSetEncoder, SummaryEncoder])
def test_encoders_are_well_permutation_invariant_at_float64(encoder_cls, batches) -> None:
    params = EncoderParams(variant="graph", width=32, heads=2, context_dim=24)
    if encoder_cls is GraphEncoder:
        encoder: torch.nn.Module = encoder_cls(params, len(WELL_TIME_FEATURES), 5)
    else:
        encoder = encoder_cls(params, len(WELL_TIME_FEATURES))
    encoder = encoder.to(torch.float64)
    encoder.eval()
    batch = batches[0]
    order = np.asarray([2, 0, 3, 1])
    permuted = permute_batch(batch, order)
    with torch.no_grad():
        base = encoder(batch)
        moved = encoder(permuted)
    delta = float((base - moved).abs().max())
    assert delta <= 1e-8, f"{encoder_cls.__name__} learned the well order: {delta}"


@pytest.mark.parametrize("encoder_cls", [GraphEncoder, TemporalSetEncoder, SummaryEncoder])
def test_masked_values_do_not_enter_the_pooled_context(encoder_cls, batches) -> None:
    params = EncoderParams(width=32, heads=2, context_dim=24)
    if encoder_cls is GraphEncoder:
        encoder: torch.nn.Module = encoder_cls(params, len(WELL_TIME_FEATURES), 5)
    else:
        encoder = encoder_cls(params, len(WELL_TIME_FEATURES))
    encoder = encoder.to(torch.float64)
    encoder.eval()
    batch = batches[0]

    # With a FIXED mask, the values at masked positions are irrelevant: two worlds
    # whose masked months hold different garbage produce the same context exactly
    # (masked months never enter the pooled representation, plan §7.1)
    def _padded(garbage: float) -> ContextBatch:
        well_time = np.array(batch.well_time)
        well_mask = np.array(batch.well_mask)
        well_time[0, 30:, :] = garbage
        well_mask[0, 30:] = False
        return ContextBatch(
            parent_id=batch.parent_id,
            design_id=batch.design_id,
            split=batch.split,
            spec_hash=batch.spec_hash,
            well_time=well_time,
            well_mask=well_mask,
            static=batch.static,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            well_ids=batch.well_ids,
        )

    with torch.no_grad():
        first = encoder(_padded(7.0))
        second = encoder(_padded(-123.5))
    delta = float((first - second).abs().max())
    assert delta == 0.0, f"{encoder_cls.__name__} read a masked month: {delta}"


def test_scaler_moves_only_the_declared_channels(batches, spec) -> None:
    scaler = FeatureScaler.fit(batches, spec)
    original = batches[0]
    scaled = scaler.transform(original)
    kinds = {name: index for index, name in enumerate(spec.well_time_features)}
    moved = "control_value_normalized"
    assert not np.allclose(
        scaled.well_time[..., kinds[moved]], original.well_time[..., kinds[moved]]
    )
    untouched = "observed_valid"
    np.testing.assert_array_equal(
        scaled.well_time[..., kinds[untouched]], original.well_time[..., kinds[untouched]]
    )
    # round-trip through the payload keeps the law
    clone = FeatureScaler.from_payload(scaler.payload(), spec)
    np.testing.assert_allclose(
        clone.transform(original).well_time, scaled.well_time, rtol=0.0, atol=0.0
    )


def test_scaler_refuses_another_spec(batches, spec) -> None:
    scaler = FeatureScaler.fit(batches, spec)
    other = default_context_spec(cutoff_s=1.0, max_wells=4, max_months=12)
    with pytest.raises(ValueError, match="another context spec"):
        FeatureScaler.from_payload(scaler.payload(), other)


def test_scaler_refuses_non_train_batches(spec) -> None:
    dev_payload = _payload()
    dev_payload["split"] = "development"
    train = build_context_batch(_payload(), spec=spec)
    dev = build_context_batch(dev_payload, spec=spec)
    with pytest.raises(ValueError, match="train"):
        FeatureScaler.fit([train, dev], spec)
    FeatureScaler.fit([train], spec)


def test_causal_blocks_see_only_the_past() -> None:
    from so_recon.ml.encoders import CausalTemporalBlock

    block = CausalTemporalBlock(4).to(torch.float64)
    block.eval()
    x = torch.zeros(1, 2, 6, 4, dtype=torch.float64)
    x[0, 0, 3, 0] = 5.0
    with torch.no_grad():
        out = block(x)
    # a change at t=3 must not alter t=0..2
    y = x.clone()
    y[0, 0, 5, 1] = 7.0
    with torch.no_grad():
        out2 = block(y)
    torch.testing.assert_close(out[0, :, :3], out2[0, :, :3])


def test_build_encoder_dispatch() -> None:
    params = EncoderParams(width=16, heads=2, context_dim=8)
    for variant in ("graph", "temporal_set", "summary"):
        assert build_encoder(variant, params, len(WELL_TIME_FEATURES), 5) is not None
    with pytest.raises(ValueError, match="unknown encoder variant"):
        build_encoder("mega-gnn", params, 10, 5)
