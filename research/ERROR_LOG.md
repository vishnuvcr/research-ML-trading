# Research Error / Deviation Log

## 2026-09-19 — Paytm Money cost-model source discrepancy

Severity: SCIENTIFIC-VALIDITY BLOCKER for affected F&O conclusions.

Paytm Money public material is not internally consistent across pages. A 2025 pricing update states flat ₹20 alignment across segments, while another F&O FAQ currently exposes ₹10 per executed order.

Action: do not hard-code a single F&O brokerage value without a dated authoritative applicable-rate profile. Keep the model fail-closed and use sensitivity analysis.

Status: OPEN

## Logging rule
Every future warning, error, failed test, data anomaly, workflow failure and protocol deviation must be appended with date/time, phase, severity, symptom, likely cause, scientific impact, action, status and resolution commit/run.
