"""Train-only feature scaling for the E03 encoders (plan §6.3, §7.1).

The scaler is fit on TRAIN parents exclusively; validation and evaluation batches pass
through `transform` unchanged in structure. Only the declared continuous channels move —
binary flags, masks and one-hot kinds are passed through untouched, so a scaled tensor
still means what the spec says it means.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from so_recon.ml.contracts import ContextBatch, ContextSpec

#: The channels the scaler may move, by feature name. Everything else passes through.
SCALED_WELL_TIME_FEATURES: tuple[str, ...] = (
    "control_value_normalized",
    "observed_bin_center",
)
SCALED_STATIC_FEATURES: tuple[str, ...] = ("log_k_support_value",)

SCALER_VERSION = "e03-scaler-1"


class FeatureScaler:
    """Mean/std scaling of the declared continuous channels, fit on train only."""

    def __init__(
        self,
        *,
        spec: ContextSpec,
        well_time_means: dict[str, float],
        well_time_stds: dict[str, float],
        static_means: dict[str, float],
        static_stds: dict[str, float],
    ) -> None:
        self._spec = spec
        self._wt_mean = dict(well_time_means)
        self._wt_std = dict(well_time_stds)
        self._st_mean = dict(static_means)
        self._st_std = dict(static_stds)

    @classmethod
    def fit(cls, batches: Sequence[ContextBatch], spec: ContextSpec) -> FeatureScaler:
        if not batches:
            raise ValueError("a scaler needs at least one train batch")
        not_train = sorted({batch.split for batch in batches if batch.split != "train"})
        if not_train:
            raise ValueError(
                f"a scaler is fit on train batches only, these carry split={not_train}"
            )
        well_time = np.concatenate(
            [batch.well_time.reshape(-1, batch.well_time.shape[-1]) for batch in batches]
        )
        mask = np.concatenate([batch.well_mask.reshape(-1) for batch in batches])
        static = np.concatenate([batch.static for batch in batches])
        well_masked = well_time[mask]
        wt: dict[str, dict[str, float]] = {"mean": {}, "std": {}}
        for name in SCALED_WELL_TIME_FEATURES:
            index = spec.well_time_features.index(name)
            column = well_masked[:, index]
            wt["mean"][name] = float(column.mean())
            std = float(column.std())
            wt["std"][name] = std if std > 1e-8 else 1.0
        st: dict[str, dict[str, float]] = {"mean": {}, "std": {}}
        for name in SCALED_STATIC_FEATURES:
            index = spec.static_features.index(name)
            column = static[:, index]
            st["mean"][name] = float(column.mean())
            std = float(column.std())
            st["std"][name] = std if std > 1e-8 else 1.0
        return cls(
            spec=spec,
            well_time_means=wt["mean"],
            well_time_stds=wt["std"],
            static_means=st["mean"],
            static_stds=st["std"],
        )

    def transform(self, batch: ContextBatch) -> ContextBatch:
        well_time = np.array(batch.well_time, dtype=np.float64, copy=True)
        for name in SCALED_WELL_TIME_FEATURES:
            index = self._spec.well_time_features.index(name)
            well_time[..., index] = (well_time[..., index] - self._wt_mean[name]) / self._wt_std[
                name
            ]
        static = np.array(batch.static, dtype=np.float64, copy=True)
        for name in SCALED_STATIC_FEATURES:
            index = self._spec.static_features.index(name)
            static[:, index] = (static[:, index] - self._st_mean[name]) / self._st_std[name]
        return ContextBatch(
            parent_id=batch.parent_id,
            design_id=batch.design_id,
            split=batch.split,
            spec_hash=batch.spec_hash,
            well_time=well_time,
            well_mask=batch.well_mask,
            static=static,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            well_ids=batch.well_ids,
        )

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": SCALER_VERSION,
            "spec_hash": self._spec.spec_hash,
            "well_time_means": self._wt_mean,
            "well_time_stds": self._wt_std,
            "static_means": self._st_mean,
            "static_stds": self._st_std,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, object], spec: ContextSpec) -> FeatureScaler:
        if payload.get("schema_version") != SCALER_VERSION:
            raise ValueError(f"unknown scaler schema {payload.get('schema_version')!r}")
        if payload.get("spec_hash") != spec.spec_hash:
            raise ValueError(
                "the scaler was fit against another context spec: the feature order it "
                "learned is not the one this model reads"
            )
        return cls(
            spec=spec,
            well_time_means=dict(payload["well_time_means"]),  # type: ignore[arg-type]
            well_time_stds=dict(payload["well_time_stds"]),  # type: ignore[arg-type]
            static_means=dict(payload["static_means"]),  # type: ignore[arg-type]
            static_stds=dict(payload["static_stds"]),  # type: ignore[arg-type]
        )


__all__ = [
    "SCALED_STATIC_FEATURES",
    "SCALED_WELL_TIME_FEATURES",
    "SCALER_VERSION",
    "FeatureScaler",
]
