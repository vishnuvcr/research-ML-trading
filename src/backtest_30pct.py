#!/usr/bin/env python3
"""Walk-forward ML portfolio research engine.

Purpose: test whether a realistic long-only Indian-equity strategy can approach
30% monthly returns without look-ahead. 30% is a research target, never a
promise. The script separates train/validation/test periods, tunes execution
parameters only on validation, and reports untouched test performance.

Execution convention:
- Features are known at today's close.
- Signal is generated after today's close.
- Entry is next day's OPEN (with configurable slippage).
- During the entry day, OHLC is used to determine whether SL/TP was touched.
  If both are touched on the same daily bar, STOP is assumed first (conservative).
- Positions are held for at most max_hold sessions.
- Capital is equally allocated among the selected names, capped by risk budget.
"""
from __future__ import annotations
import argparse, json, os, warnings
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
warnings.filterwarnings("ignore")

FEATURES = [
    "ret_1","ret_2","ret_3","ret_5","ret_10","ret_21",
    "co","hl","gap","natr","rsi","bb_width","bb_pos",
    "vol_ratio","vol_z","ema_diff","breakout_20","breakout_60",
    "atr_pct","trend_strength"
]

@dataclass(frozen=True)
class Params:
    probability: float
    stop_atr: float
    target_pct: float
    max_hold: int
    top_n: int
    risk_per_trade: float


def features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy().sort_index()
    c, h, l, o, v = x.Close, x.High, x.Low, x.Open, x.Volume
    for n in (1,2,3,5,10,21): x[f"ret_{n}"] = np.log(c/c.shift(n))
    x["co"]=(c-o)/o; x["hl"]=(h-l)/c; x["gap"]=(o-c.shift(1))/c.shift(1)
    tr=pd.concat([(h-l),(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
    atr=tr.rolling(14).mean(); x["natr"]=atr/c; x["atr_pct"]=atr/c
    d=c.diff(); gain=d.clip(lower=0).rolling(14).mean(); loss=(-d.clip(upper=0)).rolling(14).mean()
    x["rsi"]=100-100/(1+gain/(loss+1e-9))
    ma=c.rolling(20).mean(); sd=c.rolling(20).std(); x["bb_width"]=(4*sd)/(ma+1e-9)
    x["bb_pos"]=(c-(ma-2*sd))/(4*sd+1e-9)
    adv=v.rolling(20).mean(); x["vol_ratio"]=v/(adv+1e-9); x["vol_z"]=(v-adv)/(v.rolling(20).std()+1e-9)
    e9=c.ewm(span=9,adjust=False).mean(); e21=c.ewm(span=21,adjust=False).mean(); x["ema_diff"]=(e9-e21)/c
    x["breakout_20"]=c/c.rolling(20).max().shift(1)-1
    x["breakout_60"]=c/c.rolling(60).max().shift(1)-1
    x["trend_strength"]=(e9-e21)/(atr+1e-9)
    # Point-in-time target: next session's high reaches +5% from next open.
    x["target"]=((h.shift(-1)-o.shift(-1))/o.shift(-1)>=0.05).astype(int)
    return x.replace([np.inf,-np.inf],np.nan).dropna(subset=FEATURES+["target"])


def load_one(ticker: str, period: str) -> tuple[str,pd.DataFrame|None]:
    try:
        d=yf.download(ticker,period=period,interval="1d",auto_adjust=False,progress=False,threads=False)
        if d is None or len(d)<250: return ticker,None
        if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
        d=d[["Open","High","Low","Close","Volume"]].dropna()
        d["Ticker"]=ticker
        return ticker,features(d)
    except Exception: return ticker,None


def load_universe(path: str, period: str, workers: int) -> pd.DataFrame:
    with open(path) as f: raw=[z.strip().upper() for z in f if z.strip()]
    tickers=[z if z.endswith(".NS") else z+".NS" for z in raw]
    out=[]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs=[ex.submit(load_one,t,period) for t in tickers]
        for i,fut in enumerate(as_completed(futs),1):
            _,d=fut.result()
            if d is not None and len(d)>200: out.append(d)
            if i%50==0: print(f"Downloaded {i}/{len(tickers)}")
    if not out: raise RuntimeError("No usable market data returned")
    return pd.concat(out).reset_index(names="Date").sort_values(["Date","Ticker"])


def fit_predict(train: pd.DataFrame, pred: pd.DataFrame) -> tuple[np.ndarray,float]:
    model=LGBMClassifier(n_estimators=350,learning_rate=.025,num_leaves=31,max_depth=6,
                         min_child_samples=40,subsample=.85,colsample_bytree=.85,
                         reg_alpha=.2,reg_lambda=1.0,random_state=42,verbosity=-1)
    model.fit(train[FEATURES],train.target)
    p=model.predict_proba(pred[FEATURES])[:,1]
    auc=roc_auc_score(pred.target,p) if pred.target.nunique()>1 else float("nan")
    return p,float(auc)


def simulate(df: pd.DataFrame, probs: pd.Series, params: Params, capital: float=1_000_000,
             commission_bps: float=3.0, slippage_bps: float=5.0) -> tuple[pd.DataFrame,pd.DataFrame]:
    # Signals are timestamped by signal date; trade occurs on next day's open.
    z=df.copy(); z["prob"]=probs
    dates=np.sort(z.Date.unique()); equity=capital; curve=[]; trades=[]
    open_positions=[]
    for i,date in enumerate(dates[:-1]):
        day=z[z.Date==date]
        next_date=dates[i+1]; nxt=z[z.Date==next_date].set_index("Ticker")
        # Force exits for positions whose holding period has expired.
        newpos=[]
        for p in open_positions:
            if i-p["entry_i"]>=p["max_hold"]:
                row=nxt.loc[p["ticker"]] if p["ticker"] in nxt.index else None
                exit_px=float(row.Open) if row is not None else p["last_close"]
                ret=exit_px/p["entry_px"]-1
                equity*=1+p["weight"]*ret
                trades.append({**p,"exit_date":next_date,"exit_px":exit_px,"return":ret,"reason":"TIME"})
            else: newpos.append(p)
        open_positions=newpos
        # Only enter when flat for that ticker. Select highest probabilities.
        held={p["ticker"] for p in open_positions}
        cand=day[(day.prob>=params.probability)&(~day.Ticker.isin(held))].nlargest(params.top_n,"prob")
        for _,sig in cand.iterrows():
            t=sig.Ticker
            if t not in nxt.index: continue
            row=nxt.loc[t]; entry=float(row.Open)*(1+slippage_bps/10000)
            atr=float(sig.natr)*float(sig.Close)
            stop=entry-params.stop_atr*atr; target=entry*(1+params.target_pct/100)
            risk_frac=min(params.risk_per_trade, params.risk_per_trade*0.05/(float(sig.natr)*params.stop_atr+1e-9))
            weight=max(0.0,min(1.0/params.top_n,risk_frac))
            # Daily OHLC execution model. Both hit => stop first.
            lo,hi=float(row.Low),float(row.High)
            if lo<=stop and hi>=target: exit_px=stop; reason="STOP_AND_TARGET_CONSERVATIVE"
            elif lo<=stop: exit_px=stop; reason="STOP"
            elif hi>=target: exit_px=target; reason="TARGET"
            else:
                exit_px=float(row.Close); reason="CLOSE"
                open_positions.append({"ticker":t,"entry_i":i,"entry_date":next_date,"entry_px":entry,
                                       "weight":weight,"max_hold":params.max_hold,"last_close":exit_px})
                continue
            ret=exit_px/entry-1-commission_bps/10000
            equity*=1+weight*ret
            trades.append({"ticker":t,"signal_date":date,"entry_date":next_date,"entry_px":entry,
                           "exit_date":next_date,"exit_px":exit_px,"weight":weight,"return":ret,
                           "reason":reason,"prob":float(sig.prob)})
        curve.append((next_date,equity))
    # Mark any remaining positions at final close.
    if dates.size:
        last=dates[-1]; lastdf=z[z.Date==last].set_index("Ticker")
        for p in open_positions:
            if p["ticker"] in lastdf.index:
                px=float(lastdf.loc[p["ticker"],"Close"]); ret=px/p["entry_px"]-1
                equity*=1+p["weight"]*ret
                trades.append({**p,"exit_date":last,"exit_px":px,"return":ret,"reason":"FINAL_MARK"})
        curve.append((last,equity))
    eq=pd.DataFrame(curve,columns=["Date","Equity"]).drop_duplicates("Date")
    tr=pd.DataFrame(trades)
    return eq,tr


def metrics(eq: pd.DataFrame,tr: pd.DataFrame,initial: float) -> dict:
    if eq.empty: return {"total_return_pct":0,"cagr_pct":0,"max_drawdown_pct":0,"sharpe":0,"win_rate_pct":0,"profit_factor":0,"trades":0}
    e=eq.Equity; r=e.pct_change().dropna(); years=max((eq.Date.iloc[-1]-eq.Date.iloc[0]).days/365.25,1/365.25)
    peak=e.cummax(); dd=e/peak-1
    monthly=e.resample("ME",on="Date").last().Equity.pct_change().dropna() if isinstance(eq.Date.dtype,pd.DatetimeTZDtype) else eq.set_index("Date").Equity.resample("ME").last().pct_change().dropna()
    wins=tr[tr.get("return",pd.Series(dtype=float))>0] if not tr.empty else tr
    losses=tr[tr.get("return",pd.Series(dtype=float))<0] if not tr.empty else tr
    pf=(wins["return"].sum()/abs(losses["return"].sum())) if not losses.empty else float("inf")
    return {"total_return_pct":(e.iloc[-1]/initial-1)*100,
            "cagr_pct":((e.iloc[-1]/initial)**(1/years)-1)*100,
            "max_drawdown_pct":dd.min()*100,
            "sharpe":(r.mean()/r.std()*np.sqrt(252)) if r.std()>0 else 0,
            "sortino":(r.mean()/r[r<0].std()*np.sqrt(252)) if (r<0).any() and r[r<0].std()>0 else 0,
            "win_rate_pct":(len(wins)/len(tr)*100) if len(tr) else 0,
            "profit_factor":pf,"trades":int(len(tr)),
            "best_month_pct":(monthly.max()*100 if len(monthly) else 0),
            "worst_month_pct":(monthly.min()*100 if len(monthly) else 0),
            "months_over_30pct":int((monthly>=.30).sum()) if len(monthly) else 0}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tickers",default="tickers.txt"); ap.add_argument("--period",default="5y")
    ap.add_argument("--output",default="docs/backtest_30pct.json"); ap.add_argument("--workers",type=int,default=10)
    args=ap.parse_args()
    panel=load_universe(args.tickers,args.period,args.workers)
    panel["Date"]=pd.to_datetime(panel.Date)
    dates=np.sort(panel.Date.unique()); n=len(dates)
    train_end=dates[int(n*.60)]; val_end=dates[int(n*.80)]
    train=panel[panel.Date<train_end].copy(); val=panel[(panel.Date>=train_end)&(panel.Date<val_end)].copy(); test=panel[panel.Date>=val_end].copy()
    # Train once on the historical training window. Validation is used ONLY to choose execution parameters.
    val_probs,_=fit_predict(train,val); val_probs=pd.Series(val_probs,index=val.index)
    grid=[]
    for prob in (.55,.60,.65,.70,.75):
      for sl in (0.8,1.0,1.25,1.5,2.0):
       for tp in (3.0,5.0,7.0,10.0):
        for hold in (1,2,3,5):
         p=Params(prob,sl,tp,hold,top_n=5,risk_per_trade=.02)
         eq,tr=simulate(val,val_probs,p); m=metrics(eq,tr,1_000_000)
         # Reward return but penalize drawdown and low trade counts.
         score=m["total_return_pct"]-0.60*abs(m["max_drawdown_pct"])-max(0,20-m["trades"])*0.25
         grid.append((score,p,m))
    grid.sort(key=lambda x:x[0],reverse=True); best_score,best,best_val=grid[0]
    # Refit only after parameters are frozen. Test remains untouched by tuning.
    trainval=pd.concat([train,val],ignore_index=True)
    test_probs,_=fit_predict(trainval,test); test_probs=pd.Series(test_probs,index=test.index)
    eq,tr=simulate(test,test_probs,best); test_metrics=metrics(eq,tr,1_000_000)
    result={"research_target_monthly_pct":30,"split":{"train_end":str(train_end.date()),"validation_end":str(val_end.date()),"test_start":str(val_end.date()),"test_end":str(dates[-1].date())},
            "best_params":asdict(best),"validation_score":best_score,"validation_metrics":best_val,"test_metrics":test_metrics,
            "model":"LightGBM pooled cross-sectional classifier","target":"next session high >= 5% above next session open",
            "execution":"next-open entry; daily OHLC SL/TP; both hit assumes stop first; commission 3bps + slippage 5bps",
            "notes":["30% monthly is a target for research, not a guaranteed return.","Current ticker file is a present-day universe and may contain survivorship bias.","Daily OHLC cannot know intraday order of SL/TP except by conservative both-hit rule.","Do not deploy from validation results; evaluate untouched test first."]}
    os.makedirs(os.path.dirname(args.output) or ".",exist_ok=True)
    with open(args.output,"w") as f: json.dump(result,f,indent=2,default=str)
    print(json.dumps(result,indent=2,default=str))

if __name__=="__main__": main()
