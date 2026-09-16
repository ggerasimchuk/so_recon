"""Minibatch loading of a published E03 corpus (plan E03 Task 03).

The dataset never talks to Julia: a corpus is its manifest plus the published shards,
and this module is the only reader the training loop needs. Labels, per-parent context
payloads and split membership come from the manifest — the training stream — while the
`truth/` files stay unread here; they belong to the evaluator.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pyarrow.parquet as pq

from so_recon.config.learning import SplitName
from so_recon.ml.contracts import CorpusManifest, ParentRow
from so_recon.paths import ProjectPaths
from so_recon.registry.artifact import ArtifactRef


@dataclass(frozen=True)
class LabelledParent:
    """One training example: the inference-side payload plus its theta label."""

    parent: ParentRow
    context_payload: dict[str, Any]

    @property
    def parent_id(self) -> str:
        return self.parent.parent_id


class CorpusDataset:
    """Read-only view of one published corpus."""

    def __init__(self, manifest: CorpusManifest, manifest_ref: ArtifactRef, paths: ProjectPaths):
        self._manifest = manifest
        self._ref = manifest_ref
        self._paths = paths
        self._root = paths.resolve(manifest_ref.path).parent
        self._labels: dict[str, dict[str, Any]] | None = None

    @classmethod
    def open(cls, manifest_ref: ArtifactRef, paths: ProjectPaths) -> CorpusDataset:
        from so_recon.synthetic.inverse_corpus import load_corpus_manifest

        manifest = load_corpus_manifest(manifest_ref, paths)
        return cls(manifest, manifest_ref, paths)

    @property
    def manifest(self) -> CorpusManifest:
        return self._manifest

    @property
    def manifest_ref(self) -> ArtifactRef:
        return self._ref

    def parents(self, split: SplitName | None = None) -> tuple[ParentRow, ...]:
        rows = self._manifest.parents
        if split is None:
            return rows
        return tuple(row for row in rows if row.split == split)

    def labels(self) -> dict[str, dict[str, Any]]:
        """The label table keyed by parent id, loaded once from `labels.parquet`."""
        if self._labels is None:
            path = self._paths.resolve(self._manifest.labels_ref.path)
            table = pq.read_table(path).to_pylist()
            self._labels = {str(row["parent_id"]): row for row in table}
            declared = {row.parent_id for row in self._manifest.parents}
            if set(self._labels) != declared:
                missing = sorted(declared - set(self._labels))
                extra = sorted(set(self._labels) - declared)
                raise ValueError(
                    f"labels.parquet disagrees with the manifest: missing={missing[:4]} "
                    f"extra={extra[:4]} — the labels and the manifest are published "
                    "together or not at all"
                )
        return self._labels

    def load_context(self, parent: ParentRow) -> dict[str, Any]:
        """The inference input of one parent, read from the context stream."""
        path = self._context_path(parent.parent_id)
        index = json.loads((self._root / "context" / "index.json").read_text(encoding="utf-8"))
        import hashlib

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if index["files"].get(path.name) != digest:
            raise ValueError(
                f"context file {path.name} does not match the stream index: the corpus "
                "context stream was modified after publication"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("parent_id") != parent.parent_id:
            raise ValueError(
                f"context file {path.name} names parent {payload.get('parent_id')!r}, "
                f"expected {parent.parent_id!r}"
            )
        if payload.get("split") != parent.split:
            raise ValueError(
                f"context file {path.name} declares split {payload.get('split')!r} but "
                f"the manifest says {parent.split!r}: a child view cannot change split"
            )
        return payload

    def _context_path(self, parent_id: str) -> Path:
        return self._root / "context" / f"{parent_id}.json"

    def labelled(self, split: SplitName | None = None) -> tuple[LabelledParent, ...]:
        return tuple(
            LabelledParent(parent=row, context_payload=self.load_context(row))
            for row in self.parents(split)
        )

    def batches(
        self,
        *,
        batch_size: int,
        split: SplitName,
        rng: np.random.Generator,
        epochs: int = 1,
    ) -> Iterator[tuple[LabelledParent, ...]]:
        """Deterministic minibatches: same rng state, same batches, in parent units.

        Batches are cut on PARENT boundaries — never inside one parent's views — so a
        world is never split across minibatches, and view weights keep their meaning.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        rows = list(self.parents(split))
        for _epoch in range(epochs):
            order = rng.permutation(len(rows))
            for start in range(0, len(rows), batch_size):
                chunk = [rows[int(i)] for i in order[start : start + batch_size]]
                yield tuple(
                    LabelledParent(parent=row, context_payload=self.load_context(row))
                    for row in chunk
                )

    def design_ids(self) -> tuple[str, ...]:
        return tuple(sorted({row.design_id for row in self._manifest.parents}))


def theta_arrays(rows: Sequence[ParentRow]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack labels as `(s, v, z_perp)` arrays in manifest order."""
    s = np.asarray([row.theta.s for row in rows], dtype=np.int64)
    v = np.asarray([row.theta.v for row in rows], dtype=np.float64)
    z = np.asarray([row.theta.z_perp for row in rows], dtype=np.float64)
    if v.ndim != 2 or z.ndim != 2 or len(s) != len(v) or len(s) != len(z):
        raise ValueError("corpus parents carry inconsistent latent dimensions")
    return s, v, z


__all__ = ["CorpusDataset", "LabelledParent", "theta_arrays"]
