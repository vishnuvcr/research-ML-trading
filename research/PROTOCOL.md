# Research Protocol

Protocol version: 1.0.0
Protocol date: 2026-09-19
Status: FROZEN FOR IMPLEMENTATION

## 1. Title
Effectiveness of Central Pivot Range (CPR) as a systematic market-timing signal in the NIFTY 50: a time-series, out-of-sample and transaction-cost-adjusted study

## 2. Primary research question
Does a pre-specified CPR-based trading rule produce statistically and economically meaningful out-of-sample performance in NIFTY 50 after realistic transaction costs and slippage?

## 3. Secondary research questions
1. Do CPR features add predictive information about next-period NIFTY 50 direction/returns beyond baseline technical features?
2. Does CPR width category alter conditional return distributions?
3. Do breakout/rejection/position features differ across pre-specified volatility and trend regimes?
4. Does performance persist across walk-forward windows rather than only one historical interval?
5. How sensitive are conclusions to slippage, transaction costs and execution delay?
6. Does ML augmentation improve out-of-sample prediction and net trading performance over the CPR-only rule?
7. Do observed gains survive multiple-testing and data-snooping controls?

## 4. Aim
Determine the predictive and trading effectiveness of pre-specified CPR features for NIFTY 50 while controlling for leakage, model selection, execution costs and out-of-sample validity.

## 5. Objectives
Primary: estimate out-of-sample net performance of the primary CPR strategy versus its pre-specified benchmark after modelled costs.

Secondary:
- quantify CPR/return relationships;
- characterize performance by regime;
- compare CPR-only, baseline-only and CPR+baseline models;
- quantify turnover and cost burden;
- estimate uncertainty around key metrics;
- stress test execution assumptions;
- document null/negative/unstable findings;
- produce a reproducible manuscript.

## 6. Hypotheses
H0: The CPR strategy has no positive incremental out-of-sample net performance relative to the pre-specified benchmark after costs.
H1: The CPR strategy has positive incremental out-of-sample net performance relative to the benchmark after costs.

Secondary hypotheses cover incremental predictive value, regime dependence, walk-forward stability and ML augmentation.

## 7. Scope and unit of analysis
Primary signal source: NIFTY 50 index.
Signal timing: only information available at the signal timestamp may be used. Prior completed session data are used for next-session CPR.
Tradable implementation is a distinct study using an explicitly declared proxy; the proxy is never silently substituted for the index.
Intraday and options implementations require separate execution/cost models.

## 8. CPR definitions
For a completed reference session:
- P = (High + Low + Close) / 3
- BC = (High + Low) / 2
- TC = P + (P - BC)
- CPR width = TC - BC
- normalized width = (TC - BC) / P

Candidate pre-specified features:
- position relative to BC/P/TC;
- open relative to prior CPR;
- breakout above TC / breakdown below BC;
- rejection after excursion;
- CPR width category learned only inside the training window;
- prior-session/context features.

Every derived value must be lagged to prevent look-ahead.

## 9. Baselines
At minimum:
- buy-and-hold NIFTY 50;
- cash/no-signal baseline;
- simple momentum/trend baseline;
- simple volatility baseline.

## 10. Data sources and cache
Preferred hierarchy: exchange/issuer primary source where accessible and permitted; then stable licensed source; then secondary providers such as yfinance with validation.
Every dataset requires source, retrieval time, period, symbol, exchange/timezone, frequency, adjustment state, schema version and checksum.
Identical historical datasets should be reused from cache.

## 11. Data validation
Test duplicates, chronology, gaps, OHLC validity, impossible values, timezone consistency, suspicious jumps and cache/source mismatches. Unresolved anomalies block the phase.

## 12. Train/validation/test
Preserve temporal order. Default initial split:
- earliest 60% development/training;
- next 20% validation;
- latest 20% untouched final test.

Primary evaluation: expanding-window walk-forward. Parameters are estimated only from information available at each training point.

## 13. Model families
Rule-based CPR-only.
Statistical: logistic/linear/regularized models where appropriate.
ML: tree ensembles, gradient boosting and SVM where justified.
ML is secondary and cannot redefine the primary endpoint retrospectively.

## 14. Leakage controls
Prohibit future OHLC, random shuffling, future normalization/quantiles, future regime labels, final-test tuning and accidental future joins. Include explicit leakage unit tests.

## 15. Trading simulation
Every trade records decision time, order time, intended price, fill, quantity, side, gross P&L, slippage, brokerage, statutory/exchange fees, total cost, net P&L, holding time and exit reason.
Model delay, slippage, spread, lot size, position limits and partial fills when applicable.

## 16. Paytm Money cost policy
Use research/COST_MODEL.yaml. Pricing is versioned by effective date. If official sources conflict, fail closed for the affected analysis and document the discrepancy. Apply historical schedules by trade date when known; otherwise run explicit cost sensitivity.

## 17. Primary endpoint
Difference in cumulative net return between the primary CPR strategy and its benchmark during the untouched out-of-sample period.

Secondary endpoints:
annualized net return/CAGR, Sharpe, Sortino, maximum drawdown, Calmar, volatility, turnover, expectancy, profit factor, hit rate, tail loss, net alpha and forecasting metrics when applicable.

## 18. Statistical analysis
See research/ANALYSIS_PLAN.md. Required components: descriptive statistics, confidence intervals, HAC/Newey-West where relevant, block bootstrap, forecast comparison, multiplicity/data-snooping control and economic significance.

## 19. Robustness
Stress slippage, costs, execution delay, reasonable parameter perturbations, walk-forward windows, regimes and subperiods/stress periods.

## 20. Results/inference/conclusion policy
No result may be populated before computation is executed and logged. Null and negative results are valid.
Inference must state period, population, effect size, uncertainty and design limitations.
Conclusion must answer the primary question without exceeding the design.

## 21. Phase acceptance
00 protocol ready
01 validated/cached data
02 leakage-tested features
03 gross/net backtests
04 robustness complete
05 statistical inference complete
06 manuscript package complete
07 audit passed and reproducible

## 22. Amendment policy
Changes to hypotheses, primary endpoint, data universe, test window, cost model, split policy or material statistical method require a protocol version increment and written amendment.
