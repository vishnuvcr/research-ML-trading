# Statistical Analysis Plan
Version: 1.0.0

## Descriptive
Report N sessions, N signals/trades, central tendency, dispersion, quantiles, turnover, holding duration and gross/net costs.

## Predictive
For direction: accuracy, balanced accuracy, precision, recall, specificity, F1, ROC-AUC, PR-AUC and Brier/calibration when probabilistic.
For returns: MAE, RMSE, correlation/rank correlation and out-of-sample R2 where appropriate.

## Regression
Use logistic/linear/robust models as appropriate. Use Newey-West/HAC standard errors where serial dependence/heteroskedasticity warrants. Report effect sizes and 95% CIs.

## Strategy performance
Cumulative net return, CAGR/annualized return, volatility, Sharpe, Sortino, maximum drawdown, Calmar, profit factor, win rate, expectancy, turnover, total costs, exposure and tail loss.

## Primary comparison
Compare paired strategy/benchmark returns using time-series-aware inference. Use moving-block or stationary bootstrap for serial dependence. Report point estimate, CI, method and block definition.

## Forecast comparison
Use Diebold-Mariano or a justified alternative for competing forecasts where appropriate.

## Multiplicity/data snooping
Keep the final test untouched. Record all candidate variants including failures. Use a pre-specified Holm/FDR procedure for hypothesis families where appropriate and consider White Reality Check / Hansen SPA-style controls for large strategy families.

## Regimes
Use pre-specified volatility, trend and CPR-width regimes. Thresholds are learned only from training data.

## Sensitivity
Base, 1.5x and 2x slippage; zero-slippage diagnostic; alternate execution delays; plausible transaction-cost range. Zero-cost performance is never the primary endpoint.

## Economic significance
Discuss separately from statistical significance.

## Missingness/exclusions
Record all exclusions and missing data. Never remove observations because they make performance worse.

## Reproducibility
Every statistical output references commit, protocol, data manifest, configuration and seed.
