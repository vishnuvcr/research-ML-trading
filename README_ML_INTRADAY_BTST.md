# ML Intraday + BTST engine

This branch adds a signal-first ML pipeline for Indian equities, designed to run in GitHub Actions.

## Pipeline

**Data → features → purged walk-forward validation → ensemble → regime filter → liquidity/participation sizing → ATR/% bracket → backtest → ranked signals**

### Models
- LightGBM
- Random Forest
- Logistic Regression with RobustScaler
- Ensemble weights derived from out-of-fold AUC

### Intraday target
15-minute bars with an 8-bar triple-barrier horizon. The barriers use the configurable ATR stop and ATR target.

### BTST target
Daily bars with the next-session OHLC as the evaluation window. The same ATR stop/target framework is used.

### Features
Multi-horizon returns, intraday/open-close/high-low/gap features, ATR/NATR, RSI, EMA/SMA gaps, MACD, stochastic, Williams %R, volume ratio/z-score, range z-score, drawdown, VWAP gap, relative market returns and time-of-day terms.

### Risk/execution controls
- ATR or fixed-% stop loss
- ATR or fixed-% profit target
- Risk-per-trade sizing
- Maximum gross exposure
- Maximum participation as % of ADV
- NIFTY volatility-regime size multiplier
- Slippage and transaction-cost assumptions in backtesting

All key controls are in `config/trading.yaml`.

## Outputs
- `docs/data.json`: current intraday + BTST signals and metrics
- `docs/signals.csv`: sortable signal table
- `docs/backtest.json`: summary metrics
- `docs/history.json`: intraday backtest trade returns

## Important
GitHub Actions is a scheduled batch environment and should not be treated as an exchange-grade execution engine. The current implementation is **signal-first** and does not submit broker orders. For live trading, replace `DataClient` with a broker/market-data adapter and keep hard risk controls/kill switches at the broker or execution layer.

The research sources supplied with this project emphasize multi-timeframe feature engineering, walk-forward testing, realistic transaction costs, regime detection, risk management and feature-stability monitoring. The 2023 study specifically used technical indicators including SMA, MACD, stochastic oscillator, RSI and Williams %R and a dynamic in-sample/out-of-sample procedure; the 2025 implementation study likewise emphasizes robust features, walk-forward training, risk management and realistic execution considerations.
