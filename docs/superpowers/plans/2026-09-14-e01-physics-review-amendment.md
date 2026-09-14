# E01 physical review corrections

This amendment records the corrections requested after review of `af0464e` and
the subsequent `2d66d50` world-publication fix. The original E01 plan remains the
historical design; this document supersedes the specific choices below. E00
evidence and field-physics claims are unchanged.

## Decisions fixed before corrected native scoring

1. **One physical well geometry on both grids.** The five-spot convergence study
   uses nested 16 by 16 and 48 by 48 grids. Odd refinement maps a coarse cell
   centre to a fine cell centre exactly. Each injector retains one vertical
   full-height connection and the central producer retains its four symmetric
   connections. The native well index is recalculated from the unchanged radius
   and completed height. Expanding a completion to all horizontal child cells
   is prohibited: that increases total completed length.
2. **Each phase keeps its own denominator.** Restore the monthly-volume error
   denominator `max(abs(reference phase volume), 1e-6)` in standard cubic metres.
   Production plus injection throughput is not an admissible substitute. Any
   absolute-volume criterion is explicitly versioned as `e01-tolerances-2`:
   `abs(fine-coarse) <= max(1e-6 m3_sc, 0.02*abs(coarse))`. References up to
   `5e-5 m3_sc` use the absolute criterion (one millilitre); larger references
   use the phase-relative criterion. Raw relative errors remain visible.
   This threshold was fixed before the corrected rerun, not fitted to the
   observed maximum: the first strict-reference run failed on a maximum
   absolute water difference of about 3.575 millilitres. Native nonlinear
   tolerances were first tightened to `tol_cnv=1e-8`, `tol_mb=1e-10`;
   the final diagnostic uses `tol_cnv=1e-10`, `tol_mb=1e-12`,
   `tol_cnv_well=1e-12` and a common five-day maximum timestep.
3. **A water boundary supplies water.** `pressure_water` must have fractional
   composition `(1, 0)` in both Python and Julia. Mixed-fluid boundary support is
   outside this contract.
4. **Physical acceptance requires evidence.** Solver completion is necessary but
   insufficient. Missing or failed physical checks prevent acceptance. Full
   suite acceptance additionally requires all five declared unique parents;
   publishing a subset is permitted but cannot imply full-suite acceptance.
5. **Version the informative P1 experiment.** Preserve explicit historical
   `p1-two-layer-v1` rendering and identities. A new default v2 uses a
   100 by 100 by 20 metre box, with the same grid dimensions, five seeds,
   geology family definitions, rates, calendar and BHP limits. Its smaller
   pore volume increases injected pore volumes without increasing requested
   well rates or selecting seeds using hidden states.

The v2 observability screen is declared before its native scoring: over months
13 through 36, at least one producer must reach water cut 0.05 and at least one
producer must have a temporal water-cut range of 0.02. The startup well-storage
transient is excluded. Every mandatory physical check must also pass. Failed
seeds remain in the suite; thresholds are not adjusted after seeing outcomes.
This is a screen for a nontrivial observable response, not an identifiability
proof, a noise model, an inverse validation, or field calibration. E02 still
needs sensitivity, ambiguity and noise-aware recovery tests.

## Verification required

- Negative Python and native boundary-composition tests.
- A regression where a large water-volume error previously passed because of
  dilution by oil production and injection.
- Native same-location and same-completed-length well checks, followed by the
  five-spot and BL grid/time comparisons on fixed physical support.
- Manifest regressions for missing, failed, incomplete and duplicate evidence.
- Historical v1 identity checks and a fresh v2 five-parent native run, including
  closed preflights, balance, BHP, rate, completion and signal checks.
- CLI consumers must use the same physical evidence collector and acceptance
  rules as integration tests. Reports must describe the implemented denominator.

Implementation and measured outcomes are reported separately; this amendment
does not mark E01 accepted merely because the changes were requested.

## Follow-up numerical findings

The 16 to 48 comparison did not meet the new one-millilitre near-zero phase
criterion. Tightening reservoir Newton tolerances alone did not resolve it:
the maximum discrepancy remained approximately 3 ml. With explicit well CNV
tolerance and a common five-day cap the discrepancy was 3.03042 ml.
The subsequent 48 to 80 comparison decreased it to 1.580391 ml but still failed.
The final 80 to 112 comparison gave 1.078436 ml and also failed the unchanged
1 ml criterion. Its saturation MAE on common support was 0.001997812,
component inventory relative difference 6.60e-11, and significant-phase volume
relative difference 4.96e-6, all below their limits. No tested pair passed
all refinement criteria. The mandatory numerical-acceptance assertion remains
failing; no failed comparison was relabelled as accepted.
These failures are retained; neither comparison proves the coarse resolution
meets the criterion. Small water production can arise from compressible connate
water under drawdown, in addition to initial well-fluid displacement. It cannot
be dismissed solely as floating-point noise.

Larger grids are supplemental spatial-refinement diagnostics of the P4 type,
not an expansion of the normal P1 world generator (which stays at 512 cells).
They use the existing P1 runtime caps: one native worker, 900 seconds per job,
one hour per session and the unchanged memory/disk guards. A final 112 by 112
diagnostic is bounded by those caps; no automatic extension beyond it is part
of this correction. These spatial diagnostics remain pre-breakthrough cases
and do not prove post-breakthrough or heterogeneous P1 saturation accuracy.

Physical acceptance also binds every metric set to the world, result and
preflight identities and the hashes of the output files actually scored.
Reusing another attempt's metrics, changing an output after scoring, or
supplying malformed/nonfinite evidence prevents acceptance. The collector is
read-only. It does not claim protection against deliberate fabrication of
otherwise well-formed metric values.

Diagnostic solver provenance now records the actual exported native options
in a content-addressed record. Older diagnostics without exported options
explicitly record them as unavailable; the publisher no longer invents a
one-day timestep or thirty-iteration solver configuration.

## Final P1 rerun on the corrected publisher and collector

The native P1 suite, native command acceptance test, and native water-boundary
regression passed together (3 tests, 100.15 seconds). All five P1 world records
had `accepted=true`, passing physical checks and verified evidence binding.

| Seed | Highest late water cut | Largest producer temporal range | Cumulative relative balance |
|---|---:|---:|---:|
| 41 | 0.906061 | 0.290250 | 1.759e-6 |
| 42 | 0.912902 | 0.146437 | 9.035e-6 |
| 43 | 0.896105 | 0.272801 | 7.136e-6 |
| 44 | 0.878303 | 0.331958 | 6.975e-7 |
| 45 | 0.920280 | 0.269151 | 1.813e-6 |

Reproduction commands, run from the E01 worktree:

```sh
uv run pytest tests/unit tests/test_no_absolute_paths.py -q
uv run pytest tests/integration/test_e01_p1.py tests/integration/test_e01_p1_command_acceptance.py tests/integration/test_e01_water_boundary.py -q
uv run pytest tests/integration/test_e01_analytic.py tests/integration/test_e01_physics.py -k 'analytic_fixtures or operational_fixtures' -q
uv run pytest tests/integration/test_e01_physics.py -k survives_refinement -q
```

The last command remains a **failing numerical acceptance test**. The first
three passing groups do not override it. E01 stage acceptance and permission
to treat the forward as numerically accepted for E02 are not established.
