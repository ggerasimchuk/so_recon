"""Where the julia-marked suites of Tasks 9 and 10 reach the fixture publisher.

The publisher itself moved to `so_recon.validation.fixtures` in Task 12, because
`so-recon verify-physics` runs the same diagnostics from the command line and publishes the
same results: one implementation, exercised by the integration suites and by the production
command alike. This module keeps the import path those suites were written against.
"""

from __future__ import annotations

from so_recon.validation.fixtures import (
    FIXTURE_SEED,
    build_case,
    expected_cell_centers,
    publish_fixture,
    solver_cost,
)

__all__ = [
    "FIXTURE_SEED",
    "build_case",
    "expected_cell_centers",
    "publish_fixture",
    "solver_cost",
]
