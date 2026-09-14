"""Checks a forward result has to pass before anyone is allowed to believe it.

This package holds evaluations that are deliberately INDEPENDENT of the code that produced
the numbers they judge. `balance` is the first of them: it compares an inventory against a
source integral that was computed from geometry, state and PVT rather than from the
inventory itself, so the residual it reports is a real one and can be nonzero.
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

__all__ = [
    "BALANCE_FLOOR_M3_SC",
    "CUMULATIVE_RELATIVE_TOLERANCE",
    "MEDIAN_STEP_RELATIVE_TOLERANCE",
    "STRICT_STEP_RELATIVE_TARGET",
    "BalanceMetrics",
    "component_balance",
    "relative_balance_errors",
]
