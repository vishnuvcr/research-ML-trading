# Protocol Amendments

## Version 1.1.0 — 2026-09-19
Reason: freeze the executable primary rule before any backtest.

Change:
- Primary signal: close_t > TC_t.
- Position: long NIFTYBEES when the signal is true; otherwise cash.
- Execution: next eligible session open to same-session close.
- Primary benchmark: cash/no-signal on the same period.

Scientific rationale:
The broad protocol required a fully pre-specified rule before outcome analysis. Freezing the executable rule now prevents selecting a CPR configuration after observing performance.

Impact:
Phase 03 backtest and all downstream analyses must use the frozen rule. Secondary/exploratory variants must be separately labelled.
