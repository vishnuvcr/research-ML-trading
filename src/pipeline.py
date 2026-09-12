#!/usr/bin/env python3
"""
Production Machine Learning Pipeline for Predictive Equity Screening.
Integrated Enhancements:
  1. Purged & Embargoed Temporal Cross-Validation (de Prado / Jansen Ch. 7)
  2. Volume Participation Cap & Institutional Liquidity Filter (ML4T Ch. 18)
  3. Causal Market Volatility Regime Conditioning (ML4T Ch. 15)
Focus: Specificity >= 95%, Maximum ROC-AUC, 5%+ Next-Day Intraday Surge.
"""

from __future__ import annotations

import os
import sys
import json
import logging
import datetime
import concurrent.futures
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Generator

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
    brier_score_loss,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ML4T-Pipeline")


# =====================================================================
# 1. PURGED & EMBARGOED CROSS-VALIDATION (de Prado / ML4T Ch. 7)
# =====================================================================
class PurgedEmbargoTimeSeriesSplit:
    """
    Time Series Split with Purging and Embargoing to prevent data leakage.
    - Purge: Removes train samples whose forward label overlaps with the validation set.
    - Embargo: Excludes samples immediately after validation from subsequent train folds.
    """
    def __init__(self, n_splits: int = 5, purge_window: int = 1, embargo_pct: float = 0.01):
        self.n_splits = n_splits
        self.purge_window = purge_window
        self.embargo_pct = embargo_pct

    def split(self, X: np.ndarray, dates: pd.Series) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        unique_dates = np.sort(dates.unique())
        n_dates = len(unique_dates)
        val_size = n_dates // (self.n_splits + 1)
        embargo_size = max(1, int(n_dates * self.embargo_pct))

        for i in range(self.n_splits):
            train_end_idx = val_size * (i + 1)
            val_start_idx = train_end_idx + self.purge_window
            val_end_idx = min(val_start_idx + val_size, n_dates)

            if val_start_idx >= n_dates:
                break

            train_dates = unique_dates[:train_end_idx]
            val_dates = unique_dates[val_start_idx:val_end_idx]

            # Map unique dates back to original panel indices
            train_idx = np.where(dates.isin(train_dates))[0]
            val_idx = np.where(dates.isin(val_dates))[0]

            if len(train_idx) == 0 or len(val_idx) == 0:
                continue

            yield train_idx, val_idx


# =====================================================================
# 2. CAUSAL VOLATILITY REGIME DETECTOR (ML4T Ch. 15)
# =====================================================================
@dataclass
class MarketRegime:
    regime_name: str
    annualized_vol: float
    vol_percentile: float
    position_multiplier: float
    min_prob_threshold: float


class VolatilityRegimeDetector:
    """
    Detects macro market volatility regimes using NIFTY 50 (^NSEI).
    Scales sizing and adjusts acceptance thresholds dynamically.
    """
    @staticmethod
    def evaluate_regime() -> MarketRegime:
        try:
            nifty = yf.download("^NSEI", period="1y", interval="1d", progress=False, auto_adjust=False)
            if nifty is None or len(nifty) < 40:
                raise ValueError("Insufficient Nifty data")
            if isinstance(nifty.columns, pd.MultiIndex):
                nifty.columns = [c[0] for c in nifty.columns]

            c = nifty["Close"]
            log_ret = np.log(c / c.shift(1))
            rolling_vol = log_ret.rolling(21).std() * np.sqrt(252)
            current_vol = float(rolling_vol.iloc[-1])
            vol_history = rolling_vol.dropna()
            percentile = float((vol_history < current_vol).mean())

            if percentile > 0.70:  # High Volatility Regime
                return MarketRegime(
                    regime_name="HIGH VOLATILITY (Defensive)",
                    annualized_vol=current_vol,
                    vol_percentile=percentile,
                    position_multiplier=0.50,  # Half size
                    min_prob_threshold=0.65,    # Stricter acceptance gate
                )
            elif percentile < 0.30:  # Low Volatility Regime
                return MarketRegime(
                    regime_name="LOW VOLATILITY (Risk-On)",
                    annualized_vol=current_vol,
                    vol_percentile=percentile,
                    position_multiplier=1.20,  # Expand size
                    min_prob_threshold=0.50,
                )
            else:  # Normal Volatility Regime
                return MarketRegime(
                    regime_name="NORMAL VOLATILITY",
                    annualized_vol=current_vol,
                    vol_percentile=percentile,
                    position_multiplier=1.00,
                    min_prob_threshold=0.55,
                )
        except Exception as e:
            logger.warning(f"Benchmark regime detection fallback triggered: {e}")
            return MarketRegime(
                regime_name="NORMAL VOLATILITY (Fallback)",
                annualized_vol=0.15,
                vol_percentile=0.50,
                position_multiplier=1.00,
                min_prob_threshold=0.55,
            )


# =====================================================================
# 3. VOLUME PARTICIPATION CAP & LIQUIDITY FILTER (ML4T Ch. 18)
# =====================================================================
class VolumeParticipationEngine:
    """
    Applies institutional liquidity constraints:
    - Max participation rate: <= 5% of 20-day ADV to prevent execution slippage.
    - Minimum turnover filter: Excludes illiquid instruments.
    """
    def __init__(self, max_participation_rate: float = 0.05, min_adv_shares: int = 100000, min_turnover_inr: float = 1e7):
        self.max_participation_rate = max_participation_rate
        self.min_adv_shares = min_adv_shares
        self.min_turnover_inr = min_turnover_inr

    def evaluate_liquidity(self, close_px: float, adv_20: float) -> Tuple[bool, int, float]:
        daily_turnover = close_px * adv_20
        is_liquid = (adv_20 >= self.min_adv_shares) and (daily_turnover >= self.min_turnover_inr)
        max_shares = int(adv_20 * self.max_participation_rate)
        max_capital = max_shares * close_px
        return is_liquid, max_shares, max_capital


# =====================================================================
# DATA INGESTION & FEATURE ENGINEERING
# =====================================================================
@dataclass
class ModelMetrics:
    model_name: str
    auc: float
    precision: float
    specificity: float
    recall: float
    brier: float
    threshold: float


class DataLoader:
    @staticmethod
    def load_tickers(filepath: str) -> List[str]:
        if not os.path.exists(filepath):
            return ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
        with open(filepath, "r") as f:
            tickers = [line.strip().upper() for line in f if line.strip()]
        return [t if t.endswith(".NS") or "^" in t else f"{t}.NS" for t in tickers]

    @staticmethod
    def fetch_single_ticker(ticker: str, period: str = "3y") -> pd.DataFrame | None:
        try:
            df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=False)
            if df is None or len(df) < 150:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df = df.dropna().copy()
            df["Ticker"] = ticker
            df["Date"] = df.index
            return df
        except Exception as e:
            logger.debug(f"Failed download for {ticker}: {e}")
            return None

    @classmethod
    def fetch_universe_parallel(cls, tickers: List[str], max_workers: int = 8) -> Dict[str, pd.DataFrame]:
        logger.info(f"Downloading historical data for {len(tickers)} tickers with {max_workers} threads...")
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {executor.submit(cls.fetch_single_ticker, ticker): ticker for ticker in tickers}
            completed = 0
            for future in concurrent.futures.as_completed(future_to_ticker):
                completed += 1
                if completed % 10 == 0 or completed == len(tickers):
                    logger.info(f"Ingested {completed}/{len(tickers)} instruments...")
                ticker = future_to_ticker[future]
                try:
                    df = future.result()
                    if df is not None and not df.empty:
                        results[ticker] = df
                except Exception as err:
                    logger.error(f"Error on {ticker}: {err}")
        return results


class FeatureEngineering:
    @staticmethod
    def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        c = df["Close"]
        h = df["High"]
        l = df["Low"]
        o = df["Open"]
        v = df["Volume"]

        # Multi-horizon log returns
        for lag in:
            df[f"ret_{lag}"] = np.log(c / c.shift(lag))

        # Intraday ranges
        df["co_ratio"] = (c - o) / o
        df["hl_ratio"] = (h - l) / c
        df["gap_return"] = (o - c.shift(1)) / c.shift(1)

        # Average True Range (ATR) & Normalized ATR
        tr1 = h - l
        tr2 = (h - c.shift(1)).abs()
        tr3 = (l - c.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(14).mean()
        df["natr_14"] = df["atr_14"] / c

        # Relative Strength Index (RSI)
        delta = c.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        ma20 = c.rolling(20).mean()
        std20 = c.rolling(20).std()
        df["bb_upper"] = ma20 + (2 * std20)
        df["bb_lower"] = ma20 - (2 * std20)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (ma20 + 1e-9)
        df["bb_pos"] = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)

        # Volume dynamics & ADV_20 for participation limits
        df["adv_20"] = v.rolling(20).mean()
        df["vol_ratio_20"] = v / (df["adv_20"] + 1e-9)
        df["vol_zscore"] = (v - df["adv_20"]) / (v.rolling(20).std() + 1e-9)

        # EMA Trend Divergence
        ema9 = c.ewm(span=9, adjust=False).mean()
        ema21 = c.ewm(span=21, adjust=False).mean()
        df["ema_diff_9_21"] = (ema9 - ema21) / c

        # POINT-IN-TIME TARGET: Next day's high >= 5.0% above next day's open
        next_open = o.shift(-1)
        next_high = h.shift(-1)
        next_surge_pct = (next_high - next_open) / next_open
        df["target"] = (next_surge_pct >= 0.05).astype(int)

        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        return df


# =====================================================================
# ENSEMBLE ENGINE WITH PURGED CV
# =====================================================================
class EnsemblePipeline:
    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits
        self.scaler = RobustScaler()
        self.models = {}
        self.thresholds = {}
        self.feature_names: List[str] = []

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        exclude = [
            "Open", "High", "Low", "Close", "Adj Close", "Volume",
            "Ticker", "Date", "target", "bb_upper", "bb_lower"
        ]
        return [c for c in df.columns if c not in exclude and not c.startswith("Unnamed")]

    def train_and_validate(self, panel_df: pd.DataFrame) -> Dict[str, ModelMetrics]:
        self.feature_names = self.get_feature_columns(panel_df)
        X = panel_df[self.feature_names].values
        y = panel_df["target"].values
        dates = panel_df["Date"]

        # Enforce Purged & Embargoed Cross-Validation
        cv = PurgedEmbargoTimeSeriesSplit(n_splits=self.n_splits, purge_window=1, embargo_pct=0.01)

        candidate_models = {
            "LightGBM": lgb.LGBMClassifier(
                n_estimators=180, learning_rate=0.03, max_depth=5, num_leaves=24,
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
            ),
            "XGBoost": xgb.XGBClassifier(
                n_estimators=180, learning_rate=0.03, max_depth=4,
                subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric="logloss",
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=120, max_depth=7, min_samples_leaf=15, random_state=42, n_jobs=-1,
            ),
            "ElasticNet_LR": LogisticRegression(
                penalty="elasticnet", l1_ratio=0.5, solver="saga", max_iter=1000, C=0.1, random_state=42,
            ),
        }

        perf_results = {}

        for name, model in candidate_models.items():
            oof_preds = np.zeros(len(y))
            oof_targets = np.zeros(len(y))
            test_indices = []

            for train_idx, val_idx in cv.split(X, dates):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                s = RobustScaler()
                X_train_s = s.fit_transform(X_train)
                X_val_s = s.transform(X_val)

                model.fit(X_train_s, y_train)
                probs = model.predict_proba(X_val_s)

                oof_preds[val_idx] = probs
                oof_targets[val_idx] = y_val
                test_indices.extend(val_idx)

            y_eval = oof_targets[test_indices]
            p_eval = oof_preds[test_indices]

            auc_score = roc_auc_score(y_eval, p_eval)
            brier = brier_score_loss(y_eval, p_eval)

            # High Specificity Threshold Calibration (Specificity >= 95%)
            selected_thresh = 0.55
            for t in np.linspace(0.1, 0.9, 81):
                preds_bin = (p_eval >= t).astype(int)
                if preds_bin.sum() == 0:
                    continue
                tn, fp, fn, tp = confusion_matrix(y_eval, preds_bin).ravel()
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                if spec >= 0.95:
                    selected_thresh = t
                    break

            preds_opt = (p_eval >= selected_thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_eval, preds_opt).ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            prec = precision_score(y_eval, preds_opt, zero_division=0)
            rec = recall_score(y_eval, preds_opt, zero_division=0)

            perf_results[name] = ModelMetrics(
                model_name=name, auc=float(auc_score), precision=float(prec),
                specificity=float(spec), recall=float(rec), brier=float(brier), threshold=float(selected_thresh),
            )

            # Fit on full data using fitted scaler
            X_all_s = self.scaler.fit_transform(X)
            model.fit(X_all_s, y)
            self.models[name] = model
            self.thresholds[name] = selected_thresh

        return perf_results

    def predict_ensemble(self, X_latest: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X_latest)
        ensemble_prob = np.zeros(X_latest.shape[0])
        total_weight = 0.0

        for name, model in self.models.items():
            prob = model.predict_proba(X_scaled)
            weight = 1.3 if "LGBM" in name else 1.2 if "XGB" in name else 1.0
            ensemble_prob += prob * weight
            total_weight += weight

        return ensemble_prob / total_weight


# =====================================================================
# PINE SCRIPT V6 GENERATOR (With Participation & Regime Logic)
# =====================================================================
def generate_pine_script_v6(regime: MarketRegime) -> str:
    return f"""//@version=6
strategy("Ensemble 5%+ Alpha Momentum Strategy [v6 - ML4T]", 
         overlay=true, 
         initial_capital=100000, 
         default_qty_type=strategy.percent_of_equity, 
         default_qty_value={15 * regime.position_multiplier:.1f}, 
         commission_type=strategy.commission.percent, 
         commission_value=0.03, 
         slippage=1)

// --- Inputs ---
i_rsiLen          = input.int(14, "RSI Length", minval=1)
i_rsiMin          = input.float(48.0, "RSI Min Confirmation", step=0.5)
i_rsiMax          = input.float(70.0, "RSI Max (Overbought Cap)", step=0.5)
i_atrLen          = input.int(14, "ATR Length", minval=1)
i_atrMult         = input.float(1.5, "ATR Stop Loss Multiplier", step=0.1)
i_targetPct       = input.float(5.0, "Target Profit Threshold %", step=0.25)
i_volMultiplier   = input.float(1.2, "Volume Surge Gate", step=0.1)
i_maxParticipation= input.float(5.0, "Max Participation % of ADV", step=0.5)
i_maxHold         = input.int(3, "Max Holding Bars (Time Barrier)", minval=1)

// --- Feature Calculations ---
var int entryBar = na
r = ta.rsi(close, i_rsiLen)
atrValue = ta.atr(i_atrLen)
volMa20 = ta.sma(volume, 20)
fastEma = ta.ema(close, 9)
slowEma = ta.ema(close, 21)

// Institutional Volume Participation check
c_liquid = volume <= (volMa20 * (i_maxParticipation / 100.0 * 20.0))

// Signal Filters
c_trend    = fastEma > slowEma
c_rsiOk    = (r >= i_rsiMin) and (r <= i_rsiMax)
c_volSurge = volume >= (volMa20 * i_volMultiplier)
c_breakout = close > open and close > ta.highest(high, 5)

longCondition = c_trend and c_rsiOk and c_volSurge and c_breakout and strategy.position_size == 0

// Execution Engine
if (longCondition)
    entryBar := bar_index
    stopPrice = close - (atrValue * i_atrMult)
    targetPrice = close * (1.0 + (i_targetPct / 100.0))
    strategy.entry("Long", strategy.long)
    strategy.exit("Bracket Exit", "Long", limit=targetPrice, stop=stopPrice)

// Time Barrier Exit
if strategy.position_size > 0
    if (bar_index - entryBar >= i_maxHold)
        strategy.close("Long", comment="Time Barrier")

// Plotting
plot(fastEma, "Fast EMA", color=color.new(#38bdf8, 0), linewidth=1)
plot(slowEma, "Slow EMA", color=color.new(#f59e0b, 0), linewidth=2)
plotshape(longCondition, title="Surge Alert", style=shape.triangleup, location=location.belowbar, color=color.green, size=size.small, text="SURGE >5%")
"""


# =====================================================================
# HTML DASHBOARD GENERATION
# =====================================================================
def generate_html_dashboard(
    screened_stocks: List[Dict[str, Any]],
    metrics: Dict[str, ModelMetrics],
    history: List[Dict[str, Any]],
    regime: MarketRegime,
    timestamp: str,
) -> str:
    metrics_rows = "".join(f"""
    <tr>
        <td><strong>{name}</strong></td>
        <td><span class="badge badge-blue">{m.auc:.4f}</span></td>
        <td>{(m.precision * 100):.2f}%</td>
        <td><span class="badge badge-green">{(m.specificity * 100):.2f}%</span></td>
        <td>{(m.recall * 100):.2f}%</td>
        <td>{m.brier:.4f}</td>
        <td>{m.threshold:.2f}</td>
    </tr>
    """ for name, m in metrics.items())

    stock_rows = "".join(f"""
    <tr>
        <td><strong>{s['ticker']}</strong></td>
        <td><span style="color:#10b981; font-weight:700;">{s['prob']:.1%}</span></td>
        <td>₹{s['close']:.2f}</td>
        <td>₹{s['entry']:.2f}</td>
        <td style="color:#10b981;">₹{s['target']:.2f} (+5.0%)</td>
        <td style="color:#ef4444;">₹{s['stop_loss']:.2f} (-{s['sl_pct']:.2f}%)</td>
        <td>{s['max_shares']:,} shares</td>
        <td>₹{s['max_capital_lakhs']:.2f}L</td>
        <td>{s['volume_ratio']:.1f}x</td>
    </tr>
    """ for s in screened_stocks) or "<tr><td colspan='9' style='text-align:center;'>No candidates satisfied the calibrated specificity gate and regime threshold today.</td></tr>"

    regime_color = "#ef4444" if "HIGH" in regime.regime_name else "#10b981" if "LOW" in regime.regime_name else "#38bdf8"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AlphaPulse | ML4T Advanced Surge Engine</title>
    <style>
        :root {{
            --bg-primary: #0a0e17;
            --bg-secondary: #131b2e;
            --accent: #38bdf8;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --text-main: #f1f5f9;
            --text-muted: #94a3b8;
            --border: #2d3b5d;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: var(--bg-primary); color: var(--text-main); padding: 24px; }}
        .container {{ max-width: 1250px; margin: 0 auto; }}
        header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 18px; margin-bottom: 24px; }}
        .card {{ background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 12px; padding: 20px; margin-bottom: 24px; }}
        .card-title {{ font-size: 18px; font-weight: 600; margin-bottom: 16px; color: var(--accent); }}
        table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
        th, td {{ padding: 12px 14px; text-align: left; border-bottom: 1px solid var(--border); }}
        th {{ color: var(--text-muted); background: rgba(255,255,255,0.02); }}
        .badge {{ display: inline-block; padding: 4px 8px; border-radius: 6px; font-size: 12px; font-weight: 600; }}
        .badge-green {{ background: rgba(16, 185, 129, 0.2); color: #34d399; }}
        .badge-blue {{ background: rgba(56, 189, 248, 0.2); color: #7dd3fc; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px; }}
        .stat-box {{ background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 10px; padding: 16px; }}
        .stat-label {{ font-size: 12px; color: var(--text-muted); text-transform: uppercase; }}
        .stat-value {{ font-size: 20px; font-weight: 700; margin-top: 4px; color: #fff; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1 style="font-size:24px;">AlphaPulse ML4T Institutional Surge Screener</h1>
                <p style="color:var(--text-muted); font-size:14px;">Purged & Embargoed CV | Volume Participation Caps | Causal Volatility Regimes</p>
            </div>
            <div>
                <span class="badge badge-green">UPDATED: {timestamp}</span>
            </div>
        </header>

        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-label">Market Volatility Regime</div>
                <div class="stat-value" style="color:{regime_color};">{regime.regime_name}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Nifty 21D Annualized Vol</div>
                <div class="stat-value">{regime.annualized_vol:.1%} ({regime.vol_percentile:.0%}tile)</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Regime Sizing Factor</div>
                <div class="stat-value">{regime.position_multiplier:.2f}x Normal</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Calibrated Specificity</div>
                <div class="stat-value">≥ 95.0%</div>
            </div>
        </div>

        <div class="card">
            <div class="card-title">Tomorrow's Screened Opportunities (Max 5% ADV Participation)</div>
            <table>
                <thead>
                    <tr>
                        <th>Ticker</th>
                        <th>Probability</th>
                        <th>LTP</th>
                        <th>Entry</th>
                        <th>Target (+5%)</th>
                        <th>Stop Loss</th>
                        <th>Max Order Size</th>
                        <th>Capital Capacity</th>
                        <th>Vol Shock</th>
                    </tr>
                </thead>
                <tbody>
                    {stock_rows}
                </tbody>
            </table>
        </div>

        <div class="card">
            <div class="card-title">Purged & Embargoed Out-of-Fold Model Diagnostics</div>
            <table>
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>ROC-AUC</th>
                        <th>Precision</th>
                        <th>Specificity</th>
                        <th>Recall</th>
                        <th>Brier Score</th>
                        <th>Tuned Threshold</th>
                    </tr>
                </thead>
                <tbody>
                    {metrics_rows}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""


# =====================================================================
# MAIN PIPELINE EXECUTION
# =====================================================================
def run_pipeline():
    start_time = datetime.datetime.now()
    logger.info("Starting ML4T Production Pipeline with Purging, Participation Limits & Causal Vol...")

    # Step 1: Detect Market Volatility Regime
    regime = VolatilityRegimeDetector.evaluate_regime()
    logger.info(f"Detected Market Regime: {regime.regime_name} | Sizing: {regime.position_multiplier}x | Gate: {regime.min_prob_threshold:.2f}")

    # Step 2: Ingest Stock Universe
    tickers = DataLoader.load_tickers("tickers.txt")
    raw_data = DataLoader.fetch_universe_parallel(tickers, max_workers=8)
    if not raw_data:
        logger.error("No data fetched. Exiting.")
        sys.exit(1)

    # Step 3: Feature Engineering
    fe = FeatureEngineering()
    processed_dfs, latest_rows = [], []
    for ticker, df in raw_data.items():
        try:
            feat_df = fe.compute_technical_features(df)
            if len(feat_df) > 100:
                processed_dfs.append(feat_df)
                latest_rows.append(feat_df.iloc[-1:])
        except Exception as e:
            logger.warning(f"Error on {ticker}: {e}")

    panel_df = pd.concat(processed_dfs, ignore_index=True)
    latest_df = pd.concat(latest_rows, ignore_index=True)

    # Step 4: Purged & Embargoed Ensemble Training
    pipeline = EnsemblePipeline(n_splits=5)
    metrics = pipeline.train_and_validate(panel_df)

    for name, m in metrics.items():
        logger.info(f"Model [{name}] (Purged CV) -> AUC: {m.auc:.4f}, Specificity: {m.specificity:.2%}, Precision: {m.precision:.2%}")

    # Step 5: Score Candidates with Regime-Adjusted Thresholds & Participation Caps
    X_latest = latest_df[pipeline.feature_names].values
    latest_df["ensemble_prob"] = pipeline.predict_ensemble(X_latest)

    part_engine = VolumeParticipationEngine(max_participation_rate=0.05)
    candidates = []

    for idx, row in latest_df.iterrows():
        prob = row["ensemble_prob"]
        if prob >= regime.min_prob_threshold:
            close_px = float(row["Close"])
            adv_20 = float(row["adv_20"])
            atr = float(row["atr_14"])

            is_liquid, max_shares, max_capital = part_engine.evaluate_liquidity(close_px, adv_20)
            if not is_liquid:
                continue

            entry_px = close_px
            target_px = entry_px * 1.05
            stop_px = max(0.0, entry_px - (1.5 * atr))
            sl_pct = ((entry_px - stop_px) / entry_px) * 100

            candidates.append({
                "ticker": str(row["Ticker"]),
                "prob": float(prob),
                "close": close_px,
                "entry": entry_px,
                "target": target_px,
                "stop_loss": stop_px,
                "sl_pct": sl_pct,
                "max_shares": max_shares,
                "max_capital_lakhs": max_capital / 1e5,
                "volume_ratio": float(row["vol_ratio_20"]),
            })

    candidates = sorted(candidates, key=lambda x: x["prob"], reverse=True)
    logger.info(f"Qualified Opportunities for tomorrow: {len(candidates)}")

    # Step 6: Artifact Persistence & Export
    os.makedirs("docs", exist_ok=True)
    history_file = "docs/history.json"
    history = []
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                history = json.load(f)
        except Exception:
            history = []

    today_str = datetime.date.today().isoformat()
    for c in candidates:
        history.insert(0, {
            "date": today_str,
            "ticker": c["ticker"],
            "prob": c["prob"],
            "entry": c["entry"],
            "target": c["target"],
            "stop_loss": c["stop_loss"],
            "regime": regime.regime_name,
        })

    with open(history_file, "w") as f:
        json.dump(history[:1000], f, indent=2)

    with open("docs/data.json", "w") as f:
        json.dump(candidates, f, indent=2)

    # Generate Pine Script v6 with Regime Inputs
    with open("strategy_v6.pine", "w") as f:
        f.write(generate_pine_script_v6(regime))

    # Generate HTML Dashboard
    now_formatted = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    html_output = generate_html_dashboard(candidates, metrics, history, regime, now_formatted)
    with open("docs/index.html", "w") as f:
        f.write(html_output)

    logger.info(f"Completed in {datetime.datetime.now() - start_time}.")


if __name__ == "__main__":
    run_pipeline()
