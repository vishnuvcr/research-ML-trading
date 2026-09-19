# Research Workflow — Mandatory Repository Instructions

Status: ACTIVE
Protocol baseline date: 2026-09-19
Repository: vishnuvcr/research-ML-trading

These instructions govern every research program implemented in this repository.

## Required outputs
Every program must define research questions, aim, objectives, hypotheses, scientific methodology, data provenance/cache, signal definitions, trading/execution assumptions, statistical analysis, results, inferences, limitations, conclusion, figures/tables/charts, appendices/supplements and a final manuscript.

Results, inferences and conclusions MUST come only from executed analyses and may never be fabricated or back-filled.

## Protocol/change control
A detailed protocol must exist before analysis. Routine progress must not rewrite it. Material changes to hypotheses, primary endpoint, data universe, test window, split policy, cost model or material statistical method require a version increment, reason, affected phase and preserved Git history.

## Phase branches
- research/phase-00-protocol
- research/phase-01-data
- research/phase-02-features
- research/phase-03-backtest
- research/phase-04-robustness
- research/phase-05-statistics
- research/phase-06-manuscript
- research/phase-07-audit

Each phase has status, logs, inputs/outputs and a manual workflow_dispatch workflow.

## Preflight and logging
Before every phase step, validate protocol, current/prior status, error log, conversation log, data manifest, cost model, branch/ref and repository integrity. Missing/contradictory prerequisites fail closed.

Every step is logged regardless of outcome with timestamp, phase, step ID, inputs, code/version, command, result, outputs, checksums, error and decision. Phase status is updated after every step, including skipped/failed steps.

Every technical error, scientific warning, data anomaly, workflow failure and protocol deviation is written to the repository error log.

## Conversation capture
Research-relevant chats are archived with date/time, title/identifier, user instruction, assistant decisions, protocol/phase impact and repository changes. Exact transcript and reconstructed summary must be distinguished.

## Data/cache policy
Important data must be cached/versioned in the repo when licensing permits. Otherwise retain source metadata, retrieval parameters, checksums and permitted derived data. Identical historical data must not be downloaded on every run.

## Trading realism
All trading research must account for brokerage, STT, exchange fees, SEBI fees, GST, stamp duty, DP charges where applicable, funding/MTF interest where applicable, spread, slippage, latency, lot-size/position limits and material execution costs.

Paytm Money costs are versioned and sourced from official material. Conflicting official sources are a blocker until resolved or sensitivity-tested.

## Statistical integrity
Explicitly address exploratory vs confirmatory analysis, time-ordered train/validation/test, leakage/look-ahead bias, survivorship bias, overlapping observations, multiple testing, data snooping and regime dependence. Statistical significance is not economic significance.

## Reproducibility
Every result must trace to repository commit, protocol version, data checksum/manifest, environment, configuration and seed where relevant. Python is primary; Rust may be used for performance-sensitive code. Both run in GitHub Actions.

## GitHub Actions / Pages
Every phase has a manual run button. A master workflow may orchestrate phases automatically after initiation. Every run persists status, logs, errors, data metadata and artifacts. docs/ is the GitHub Pages research surface.

## Final manuscript
A completed program must produce a complete manuscript with title, abstract, introduction, questions/hypotheses, methods, data, signal/strategy definition, statistical analysis, results, tables, figures, robustness, economic significance, discussion, limitations, conclusion, references, appendices, supplements and reproducibility statement.

## Serious intervention
Automation stops only for unresolved scientific ambiguity, unavailable/contradictory data, material cost/regulatory ambiguity, missing credentials/licences, destructive overwrite risk or required protocol amendment. Log the issue and exact decision required.

---
