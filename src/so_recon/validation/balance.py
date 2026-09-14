"""E01.7 — the independent component balance of a forward result.

Conservation is checked here and nowhere else, and the whole value of the check rests on
one property: **the source is not the inventory difference**. If the net source of a step
were computed as `I[k+1] - I[k]`, the residual would be identically zero and this module
would be an expensive way of printing zeros. What it is given instead is a source integral
the extractor computed from the native geometry (well indices, perforation gravity), the
native state (pressure, saturation) and the native PVT (phase densities and mobilities) —
quantities that never pass through an inventory — so a lost interval, a mis-signed
injector, a substep counted twice or a `dt` that does not add up all show up as a residual.

The two numbers SPEC §23.1 names are `cumulative_relative` and `median_step_relative`:

    «На smooth verification cases начальные допуски: cumulative component-balance
     relative error ≤1e−3 и median per-step ≤1e−5»

and the stricter §8.3 target is reported beside them rather than folded into them, as that
clause requires. `absolute_residual` and `throughput_relative` are reported for the reason
the plan gives: a large standing inventory divided into a small residual looks excellent
while a month's whole offtake is missing, so the absolute number in m³_sc and the residual
against what actually flowed are both on the record.

Signs. `net_source_integrals[k, c]` is the NET source of component `c` INTO the system over
step `k`, in standard m³ — negative for production, positive for injection — so that the
balance is the plain statement `I[k+1] - I[k] = source[k]`. The system whose inventory is
passed in has to be the same one the source is measured across: with the well storage
included in `I`, the source is the surface flux; with only the reservoir in `I`, it is the
reservoir-well connection flux. Mixing the two is a modelling error this module cannot see,
which is why the caller states the components it is balancing by name.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import model_validator

from so_recon.config.schema import StrictModel

#: `floor=1e-6 m³_sc` (plan 7.6). It is the scale below which a residual is not a physical
#: error but the arithmetic's own zero, and it exists so that dividing by a component that
#: is not present at all cannot manufacture a relative error of infinity.
BALANCE_FLOOR_M3_SC = 1e-6

#: SPEC 23.1, the initial tolerances on a smooth verification case.
CUMULATIVE_RELATIVE_TOLERANCE = 1e-3
MEDIAN_STEP_RELATIVE_TOLERANCE = 1e-5

#: SPEC 8.3's stricter per-step target. Reported separately, never used to relax or to
#: replace the 23.1 tolerances above: a result that misses it is still acceptable under
#: 23.1, and saying so requires both numbers to be visible.
STRICT_STEP_RELATIVE_TARGET = 1e-6


def relative_balance_errors(
    inventory: NDArray[np.float64] | list[list[float]],
    net_source_integrals: NDArray[np.float64] | list[list[float]],
    floor: float = BALANCE_FLOOR_M3_SC,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-step and cumulative relative balance error, per component (plan 7.6).

    `inventory` has one more row than `net_source_integrals`, because it has to include the
    state the first step started from; a caller that hands over only the states it ended
    steps at is silently balancing the wrong intervals, so the shape is checked rather than
    broadcast.

    Returns `(per_step, cumulative)`: `per_step[k, c]` is `|ΔI - source| / scale` over step
    `k`, and `cumulative[c]` is the same quantity over the whole horizon. Both are scaled by
    the larger of the inventory the step started from, the source that crossed it and
    `floor`, so neither a component with nothing in it nor a step with nothing flowing can
    divide by zero.
    """
    inventory = np.asarray(inventory, dtype=np.float64)
    flux = np.asarray(net_source_integrals, dtype=np.float64)
    if inventory.ndim != 2 or flux.ndim != 2:
        raise ValueError(
            f"inventory and net_source_integrals are (steps, components) tables, got shapes "
            f"{inventory.shape} and {flux.shape}"
        )
    if inventory.shape[0] != flux.shape[0] + 1:
        raise ValueError("inventory must include initial state")
    if inventory.shape[1] != flux.shape[1]:
        raise ValueError(
            f"inventory describes {inventory.shape[1]} components and net_source_integrals "
            f"{flux.shape[1]}; a balance is per component"
        )
    if not floor > 0.0:
        raise ValueError(f"floor must be positive, got {floor}")
    if not np.isfinite(inventory).all():
        raise ValueError("inventory holds nonfinite values")
    if not np.isfinite(flux).all():
        raise ValueError("net_source_integrals holds nonfinite values")
    residual = np.diff(inventory, axis=0) - flux
    scale = np.maximum(np.maximum(abs(inventory[:-1]), abs(flux)), floor)
    cumulative = inventory[-1] - inventory[0] - flux.sum(axis=0)
    cumulative_scale = np.maximum(
        np.maximum(abs(inventory[0]), abs(flux).sum(axis=0)),
        floor,
    )
    return abs(residual) / scale, abs(cumulative) / cumulative_scale


class BalanceMetrics(StrictModel):
    """What the balance evaluator concluded, one entry per component.

    Every tuple is indexed by `components`, which names what is being balanced rather than
    leaving the caller to remember that water comes first.
    """

    components: tuple[str, ...]
    cumulative_relative: tuple[float, ...]
    median_step_relative: tuple[float, ...]
    max_step_relative: tuple[float, ...]
    #: |I_end - I_start - Σ source| in standard m³: the number a large standing inventory
    #: cannot hide, unlike the relative one beside it.
    absolute_residual: tuple[float, ...]
    max_step_absolute: tuple[float, ...]
    #: The same absolute residual against what actually flowed (Σ|source|), rather than
    #: against what was merely standing there.
    throughput_relative: tuple[float, ...]
    initial_inventory_m3_sc: tuple[float, ...]
    final_inventory_m3_sc: tuple[float, ...]
    net_source_m3_sc: tuple[float, ...]
    floor_m3_sc: float
    n_steps: int

    @model_validator(mode="after")
    def _one_entry_per_component(self) -> BalanceMetrics:
        if not self.components:
            raise ValueError("a balance names at least one component")
        if len(set(self.components)) != len(self.components):
            raise ValueError(f"component names must be unique, got {self.components}")
        n = len(self.components)
        for name in (
            "cumulative_relative",
            "median_step_relative",
            "max_step_relative",
            "absolute_residual",
            "max_step_absolute",
            "throughput_relative",
            "initial_inventory_m3_sc",
            "final_inventory_m3_sc",
            "net_source_m3_sc",
        ):
            value: tuple[float, ...] = getattr(self, name)
            if len(value) != n:
                raise ValueError(
                    f"{name} has {len(value)} entries for {n} components {self.components}"
                )
        if self.n_steps < 1:
            raise ValueError(f"a balance covers at least one step, got {self.n_steps}")
        if not self.floor_m3_sc > 0.0:
            raise ValueError(f"floor_m3_sc must be positive, got {self.floor_m3_sc}")
        return self

    @property
    def within_spec_tolerance(self) -> bool:
        """SPEC 23.1: cumulative ≤1e−3 AND median per-step ≤1e−5, for every component."""
        return all(c <= CUMULATIVE_RELATIVE_TOLERANCE for c in self.cumulative_relative) and all(
            m <= MEDIAN_STEP_RELATIVE_TOLERANCE for m in self.median_step_relative
        )

    @property
    def meets_strict_target(self) -> bool:
        """SPEC 8.3's stricter per-step target, reported beside 23.1 and never instead."""
        return all(m <= STRICT_STEP_RELATIVE_TARGET for m in self.median_step_relative)

    def failures(self) -> tuple[str, ...]:
        """Every component that misses SPEC 23.1, named with the number it missed by."""
        out: list[str] = []
        for index, name in enumerate(self.components):
            if self.cumulative_relative[index] > CUMULATIVE_RELATIVE_TOLERANCE:
                out.append(
                    f"{name}: cumulative relative balance error "
                    f"{self.cumulative_relative[index]:.3e} exceeds "
                    f"{CUMULATIVE_RELATIVE_TOLERANCE:g} (SPEC 23.1); absolute residual "
                    f"{self.absolute_residual[index]:.6g} m3_sc"
                )
            if self.median_step_relative[index] > MEDIAN_STEP_RELATIVE_TOLERANCE:
                out.append(
                    f"{name}: median per-step relative balance error "
                    f"{self.median_step_relative[index]:.3e} exceeds "
                    f"{MEDIAN_STEP_RELATIVE_TOLERANCE:g} (SPEC 23.1)"
                )
        return tuple(out)


def component_balance(
    inventory: NDArray[np.float64] | list[list[float]],
    net_source_integrals: NDArray[np.float64] | list[list[float]],
    floor: float = BALANCE_FLOOR_M3_SC,
    *,
    components: tuple[str, ...],
) -> BalanceMetrics:
    """Wrap `relative_balance_errors` into the record a result publishes (plan 7.6)."""
    per_step, cumulative = relative_balance_errors(inventory, net_source_integrals, floor)
    inv = np.asarray(inventory, dtype=np.float64)
    flux = np.asarray(net_source_integrals, dtype=np.float64)
    if len(components) != inv.shape[1]:
        raise ValueError(
            f"components {components} name {len(components)} quantities for "
            f"{inv.shape[1]} columns of inventory"
        )
    step_absolute = abs(np.diff(inv, axis=0) - flux)
    absolute = abs(inv[-1] - inv[0] - flux.sum(axis=0))
    throughput = np.maximum(abs(flux).sum(axis=0), floor)
    return BalanceMetrics(
        components=components,
        cumulative_relative=tuple(float(v) for v in cumulative),
        median_step_relative=tuple(float(v) for v in np.median(per_step, axis=0)),
        max_step_relative=tuple(float(v) for v in per_step.max(axis=0)),
        absolute_residual=tuple(float(v) for v in absolute),
        max_step_absolute=tuple(float(v) for v in step_absolute.max(axis=0)),
        throughput_relative=tuple(float(v) for v in absolute / throughput),
        initial_inventory_m3_sc=tuple(float(v) for v in inv[0]),
        final_inventory_m3_sc=tuple(float(v) for v in inv[-1]),
        net_source_m3_sc=tuple(float(v) for v in flux.sum(axis=0)),
        floor_m3_sc=float(floor),
        n_steps=int(flux.shape[0]),
    )
