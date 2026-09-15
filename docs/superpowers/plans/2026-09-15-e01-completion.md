# E01 completion and integration amendment

This completion run follows the explicit user request to finish E01 and merge its remaining work into master while preserving E02. It supersedes only the reference-grid selection of Task 10.8; the existing physical tolerance file is unchanged.

## Fixed scope before the next native run

The original 16/48 grid pair genuinely fails the 1 ml per-phase absolute criterion. Reservoir and well Newton tightening did not cure it. The measured 48/80 and 80/112 errors decreased to 1.580391 and 1.078436 ml. Compressible connate-water production is not a zero reference and is not discarded as floating point noise.

Keep the 16-grid symmetry/balance fixture and the 16/48 comparison as a published `five_spot_coarse_sensitivity` diagnostic, including its actual FAIL. Add exactly two registered forwards at 112 and 144 to determine whether this continuous problem can meet the UNCHANGED criterion at higher resolution. The acceptance pair is fixed at 112/144 before the new run; no automated search for a passing pair. All four grids keep the same well coordinates, number/length/radius of connections, rock, fluids, controls, calendar, support and five-day cap. Native well indices are recomputed. This adds two explicitly accounted launcher forwards under the existing 900 s launcher / one-hour P1 session bounds; production P1 worlds stay at 512 cells.

A passing reference pair certifies only this pair and this homogeneous pre-breakthrough problem. It does not certify the 16-grid monthly water prediction, heterogeneous P1 grids, E02 inversion accuracy or a field model. Stage status must retain this resolution limitation even if all required reference checks pass. Both the failed coarse comparison and the reference comparison are reported with grid sizes and actual metrics. Historical failures in the prior amendment remain unchanged.

## Implementation repairs

- Resume must verify source/input/environment identity and all published evidence before reusing a completed group; missing, changed or legacy-unbound evidence reruns. Cache-bypass jobs execute again.
- Black-oil production construction must verify the supported PVT source and recomputed canonical table/relperm hashes before simulation, including negative native tests.
- Validate positive finite native BO density and viscosity, in addition to formation-volume factors.

## Verification and integration

Run unit/regression, Ruff format/lint and mypy; run targeted native regressions for these defects. Commit implementation with no stage success claim, then run the bounded registered P0, P1 and BO suites from that clean code revision, regenerate stage/figure evidence, and commit those artifacts separately. Merge into master only after inspecting all outcomes. Keep E02 and the user's IDE changes intact. No push is requested.

## Targeted verification before the registered gate

The bounded native refinement regression passed in 626.68 seconds. The historical 16/48 pair remains FAIL at 3.030419 ml. The fixed 112/144 pair passed: near-zero phase discrepancy 0.816230 ml (limit 1 ml), common-support So PV-MAE 0.00116234 (limit 0.02), inventory relative discrepancy 5.45e-11 (limit 0.01), significant-phase relative discrepancy 3.87e-6 (limit 0.02). This establishes pairwise agreement under these criteria, not continuum error or accuracy of coarse-grid inverse calculations.

Other targeted evidence: 794 unit/regression tests passed; Ruff lint/format and mypy passed; native BO provenance/property regressions passed; native BO diagnostic 75/75 assertions passed. Independent source review findings were corrected and 94 physics/report tests rerun. The final registered gate and its code revision are recorded in `reports/stages/E01.md` after the clean-tree run.
