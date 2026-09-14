"""Checks a forward result has to pass before anyone is allowed to believe it.

This package holds evaluations that are deliberately INDEPENDENT of the code that produced
the numbers they judge. `balance` is the first of them: it compares an inventory against a
source integral that was computed from geometry, state and PVT rather than from the
inventory itself, so the residual it reports is a real one and can be nonzero.

`physics` is the second: it scores a forward's PUBLISHED artifacts against the tolerance
block fixed in `configs/e01_tolerances.yml` before any of those numbers existed, and against
an analytic Buckley-Leverett reference that is a formula rather than a second solver. That
config is the single source of truth for what is scored; `balance`'s own constants remain
the defaults of its low-level helper, and `tests/unit/test_physics_metrics.py` pins the two
to the same numbers so neither can move alone.
"""

from __future__ import annotations

from so_recon.validation.balance import (
    BALANCE_FLOOR_M3_SC,
    CUMULATIVE_RELATIVE_TOLERANCE,
    MEDIAN_STEP_RELATIVE_TOLERANCE,
    STRICT_STEP_RELATIVE_TARGET,
    BalanceMetrics,
    component_balance,
    relative_balance_errors,
)
from so_recon.validation.physics import (
    DEFAULT_TOLERANCES_RELPATH,
    TOLERANCE_SCHEMA_VERSION,
    PhysicsCheck,
    bl_cell_average,
    bl_front_position,
    bl_saturation,
    evaluate_physics,
    load_tolerances,
)

__all__ = [
    "BALANCE_FLOOR_M3_SC",
    "DEFAULT_TOLERANCES_RELPATH",
    "TOLERANCE_SCHEMA_VERSION",
    "CUMULATIVE_RELATIVE_TOLERANCE",
    "MEDIAN_STEP_RELATIVE_TOLERANCE",
    "STRICT_STEP_RELATIVE_TARGET",
    "BalanceMetrics",
    "PhysicsCheck",
    "bl_cell_average",
    "bl_front_position",
    "bl_saturation",
    "component_balance",
    "evaluate_physics",
    "load_tolerances",
    "relative_balance_errors",
]
