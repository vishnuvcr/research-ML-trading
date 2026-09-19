# Detailed Research Execution Plan

## Phase 00 — Protocol and governance
Branch: research/phase-00-protocol

Steps:
1. Check repository instructions.
2. Check protocol and protocol version.
3. Check phase/status/error/conversation/cost files.
4. Validate that the primary rule is frozen before outcome analysis.
5. Record the phase decision.
6. Publish current status to GitHub Pages.

Outputs: protocol, protocol amendment log, status register, run/error/conversation logs.

Gate: no data analysis until this phase passes.

## Phase 01 — Data
Branch: research/phase-01-data

Steps:
1. Run mandatory preflight.
2. Inspect cache and manifest.
3. Reuse valid cached datasets.
4. If absent/stale, acquire NIFTY 50 and NIFTYBEES daily OHLCV from the configured source.
5. Record source and retrieval metadata.
6. Validate chronology, duplicates, OHLC relationships, gaps, timezone and suspicious observations.
7. Write canonical normalized datasets.
8. Compute checksums.
9. Write/update data manifest and data dictionary.
10. Record every warning/failure.
11. Update phase status.
12. Publish structured status/data to Pages.

Outputs: data/cache/*, data/processed/raw_normalized/*, data/manifests/*, data dictionary/checksums.

Gate: all required datasets validated and reproducible.

## Phase 02 — Features
Branch: research/phase-02-features

Steps:
1. Run preflight.
2. Load only validated Phase 01 data.
3. Compute P, BC, TC and CPR width.
4. Confirm signal availability only after reference-session close.
5. Construct primary signal Close_t > TC_t.
6. Align signal to next-session NIFTYBEES execution.
7. Generate secondary/exploratory features.
8. Generate baseline trend/volatility features.
9. Run explicit leakage tests.
10. Freeze feature-version metadata.
11. Update status/logs/pages.

Gate: feature table contains no future information and is deterministic.

## Phase 03 — Backtest
Branch: research/phase-03-backtest

Steps:
1. Run preflight.
2. Validate cost-model readiness.
3. Load frozen signal and validated data.
4. Simulate next-open-to-close NIFTYBEES trades.
5. Apply brokerage, statutory/exchange costs, GST, stamp duty and other applicable charges.
6. Apply configured slippage and spread.
7. Record intended and realised fills.
8. Generate complete trade ledger.
9. Calculate gross and net performance.
10. Compare against cash/no-signal and buy-and-hold benchmarks.
11. Preserve untouched final test data.
12. Update logs/status/pages.

Gate: net backtest reproducible and cost model validated.

## Phase 04 — Robustness
Branch: research/phase-04-robustness

Steps:
1. Preflight.
2. Stress slippage at 1.0x, 1.5x and 2.0x.
3. Stress execution delay.
4. Stress plausible transaction-cost ranges.
5. Evaluate walk-forward windows.
6. Evaluate volatility/trend/CPR-width regimes.
7. Evaluate meaningful historical subperiods.
8. Run parameter perturbation only for secondary/exploratory variants.
9. Preserve all failed candidates.
10. Update status/logs/pages.

Gate: robustness matrix complete with no hidden final-test tuning.

## Phase 05 — Statistical inference
Branch: research/phase-05-statistics

Steps:
1. Preflight.
2. Verify final test remains untouched until this phase.
3. Calculate primary effect size.
4. Generate block-bootstrap confidence intervals.
5. Use HAC/Newey-West where applicable.
6. Run forecast comparison where relevant.
7. Apply multiplicity control to defined hypothesis families.
8. Run strategy-family/data-snooping controls when needed.
9. Separate statistical and economic significance.
10. Write reproducible statistical tables.
11. Update logs/status/pages.

Gate: every inference traces to data, code, protocol and configuration.

## Phase 06 — Manuscript
Branch: research/phase-06-manuscript

Steps:
1. Preflight.
2. Pull verified outputs only.
3. Generate manuscript text from structured results.
4. Generate figures and tables.
5. Generate appendices and supplements.
6. Insert limitations and failure audit.
7. Insert exact cost-model assumptions.
8. Cross-check every number against machine-readable result files.
9. Generate reproducibility statement.
10. Update status/logs/pages.

Gate: no manuscript number exists without a machine-readable source artifact.

## Phase 07 — Final audit
Branch: research/phase-07-audit

Steps:
1. Preflight.
2. Check phase ordering and statuses.
3. Check protocol version consistency.
4. Check data checksums.
5. Check no look-ahead/leakage failures remain unresolved.
6. Check cost-accounting completeness.
7. Check every phase has a run log.
8. Check every error has a disposition.
9. Check conversation capture.
10. Re-run final result generation from a clean environment.
11. Cross-check manuscript against results.
12. Produce final audit report.
13. Publish final Pages package.

Release criteria: all mandatory checks pass and no open scientific-validity blocker remains.

## Automatic orchestration
The research initiator should execute phases in order and stop automatically on the first blocking failure. Each phase may also be run manually from its own GitHub Actions workflow.

A completed phase cannot authorize a later phase unless its acceptance gate passes.

## Slippage/cost rule
The primary endpoint is always net of configured costs. Gross results are diagnostic only.

## Data refresh rule
Historical data are not re-downloaded when a valid cache/manifest exists. Any refresh creates a new data version and records why the previous version was superseded.

## Intervention rule
Only serious scientific/data/cost/licensing issues stop automation. Routine null results, losing periods, model failures and failed robustness tests are logged and carried forward rather than hidden.
