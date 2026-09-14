#!/usr/bin/env python3
"""Walk-forward ML portfolio research engine.

Research target: test whether a realistic long-only Indian-equity strategy can
approach 30% monthly returns without look-ahead. The target is not a promise.
Features use information known at the signal day's close; entries occur at the
next open. Daily OHLC is used conservatively: if both SL and TP are touched,
STOP is assumed first.
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

FEATURES=["ret_1","ret_2","ret_3","ret_5","ret_10","ret_21","co","hl","gap","natr","rsi","bb_width","bb_pos","vol_ratio","vol_z","ema_diff","breakout_20","breakout_60","atr_pct","trend_strength"]

@dataclass(frozen=True)
class Params:
    probability: float
    stop_atr: float
    target_pct: float
    max_hold: int
    top_n: int
    risk_per_trade: float

def features(df):
    x=df.copy().sort_index(); c,h,l,o,v=x.Close,x.High,x.Low,x.Open,x.Volume
    for n in (1,2,3,5,10,21): x[f"ret_{n}"]=np.log(c/c.shift(n))
    x["co"]=(c-o)/o; x["hl"]=(h-l)/c; x["gap"]=(o-c.shift(1))/c.shift(1)
    tr=pd.concat([(h-l),(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
    atr=tr.rolling(14).mean(); x["natr"]=atr/c; x["atr_pct"]=atr/c
    d=c.diff(); gain=d.clip(lower=0).rolling(14).mean(); loss=(-d.clip(upper=0)).rolling(14).mean(); x["rsi"]=100-100/(1+gain/(loss+1e-9))
    ma=c.rolling(20).mean(); sd=c.rolling(20).std(); x["bb_width"]=(4*sd)/(ma+1e-9); x["bb_pos"]=(c-(ma-2*sd))/(4*sd+1e-9)
    adv=v.rolling(20).mean(); x["vol_ratio"]=v/(adv+1e-9); x["vol_z"]=(v-adv)/(v.rolling(20).std()+1e-9)
    e9=c.ewm(span=9,adjust=False).mean(); e21=c.ewm(span=21,adjust=False).mean(); x["ema_diff"]=(e9-e21)/c
    x["breakout_20"]=c/c.rolling(20).max().shift(1)-1; x["breakout_60"]=c/c.rolling(60).max().shift(1)-1; x["trend_strength"]=(e9-e21)/(atr+1e-9)
    x["target"]=((h.shift(-1)-o.shift(-1))/o.shift(-1)>=.05).astype(int)
    return x.replace([np.inf,-np.inf],np.nan).dropna(subset=FEATURES+["target"])

def load_one(ticker,period):
    try:
        d=yf.download(ticker,period=period,interval="1d",auto_adjust=False,progress=False,threads=False)
        if d is None or len(d)<250:return ticker,None
        if isinstance(d.columns,pd.MultiIndex):d.columns=d.columns.get_level_values(0)
        d=d[["Open","High","Low","Close","Volume"]].dropna(); d["Ticker"]=ticker
        return ticker,features(d)
    except Exception:return ticker,None

def load_universe(path,period,workers):
    with open(path) as f: raw=[z.strip().upper() for z in f if z.strip()]
    tickers=[z if z.endswith(".NS") else z+".NS" for z in raw]; out=[]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs=[ex.submit(load_one,t,period) for t in tickers]
        for i,f in enumerate(as_completed(futs),1):
            _,d=f.result()
            if d is not None and len(d)>200:out.append(d)
            if i%50==0:print(f"Downloaded {i}/{len(tickers)}")
    if not out:raise RuntimeError("No usable market data returned")
    return pd.concat(out).reset_index(names="Date").sort_values(["Date","Ticker"])

def fit_predict(train,pred):
    model=LGBMClassifier(n_estimators=350,learning_rate=.025,num_leaves=31,max_depth=6,min_child_samples=40,subsample=.85,colsample_bytree=.85,reg_alpha=.2,reg_lambda=1.0,random_state=42,verbosity=-1)
    model.fit(train[FEATURES],train.target); p=model.predict_proba(pred[FEATURES])[:,1]
    auc=roc_auc_score(pred.target,p) if pred.target.nunique()>1 else float("nan"); return p,float(auc)

def simulate(df,probs,params,capital=1_000_000,commission_bps=3.,slippage_bps=5.):
    """Simulate next-open entries and check every held position on every OHLC bar."""
    z=df.copy(); z["prob"]=probs; dates=np.sort(z.Date.unique()); equity=capital; curve=[]; trades=[]; positions=[]
    for i,date in enumerate(dates):
        day=z[z.Date==date].set_index("Ticker")
        # First process existing positions on today's bar. This fixes the prior
        # bug where multi-day positions were only checked at time exit.
        remaining=[]
        for p in positions:
            if p["ticker"] not in day.index:
                remaining.append(p); continue
            row=day.loc[p["ticker"]]; hi,lo=float(row.High),float(row.Low)
            exit_px=None; reason=None
            if lo<=p["stop"] and hi>=p["target"]: exit_px=p["stop"]; reason="STOP_AND_TARGET_CONSERVATIVE"
            elif lo<=p["stop"]: exit_px=p["stop"]; reason="STOP"
            elif hi>=p["target"]: exit_px=p["target"]; reason="TARGET"
            elif i-p["entry_i"]>=p["max_hold"]: exit_px=float(row.Open); reason="TIME_OPEN"
            if exit_px is None:
                p["last_close"]=float(row.Close); remaining.append(p); continue
            ret=exit_px/p["entry_px"]-1-commission_bps/10000; equity*=1+p["weight"]*ret
            trades.append({**p,"exit_date":date,"exit_px":exit_px,"return":ret,"reason":reason});
        positions=remaining
        # Signals from today's close can only enter tomorrow, so no look-ahead.
        if i>=len(dates)-1: curve.append((date,equity)); continue
        next_date=dates[i+1]; nxt=z[z.Date==next_date].set_index("Ticker"); held={p["ticker"] for p in positions}
        cand=day[(day.prob>=params.probability)&(~day.index.isin(held))].nlargest(params.top_n,"prob")
        for t,sig in cand.iterrows():
            if t not in nxt.index:continue
            row=nxt.loc[t]; entry=float(row.Open)*(1+slippage_bps/10000); atr=float(sig.natr)*float(sig.Close)
            stop=entry-params.stop_atr*atr; target=entry*(1+params.target_pct/100)
            risk_frac=min(params.risk_per_trade,params.risk_per_trade*.05/(float(sig.natr)*params.stop_atr+1e-9)); weight=max(0.,min(1./params.top_n,risk_frac))
            lo,hi=float(row.Low),float(row.High)
            if lo<=stop and hi>=target:exit_px=stop;reason="STOP_AND_TARGET_CONSERVATIVE"
            elif lo<=stop:exit_px=stop;reason="STOP"
            elif hi>=target:exit_px=target;reason="TARGET"
            else:
                positions.append({"ticker":t,"signal_date":date,"entry_date":next_date,"entry_px":entry,"weight":weight,"max_hold":params.max_hold,"entry_i":i+1,"stop":stop,"target":target,"last_close":float(row.Close),"prob":float(sig.prob)})
                continue
            ret=exit_px/entry-1-commission_bps/10000; equity*=1+weight*ret
            trades.append({"ticker":t,"signal_date":date,"entry_date":next_date,"entry_px":entry,"exit_date":next_date,"exit_px":exit_px,"weight":weight,"return":ret,"reason":reason,"prob":float(sig.prob)})
        curve.append((date,equity))
    eq=pd.DataFrame(curve,columns=["Date","Equity"]).drop_duplicates("Date"); tr=pd.DataFrame(trades); return eq,tr

def metrics(eq,tr,initial):
    if eq.empty:return {"total_return_pct":0,"cagr_pct":0,"max_drawdown_pct":0,"sharpe":0,"sortino":0,"win_rate_pct":0,"profit_factor":0,"trades":0}
    e=eq.set_index("Date").Equity; r=e.pct_change().dropna(); years=max((e.index[-1]-e.index[0]).days/365.25,1/365.25); dd=e/e.cummax()-1; monthly=e.resample("ME").last().pct_change().dropna()
    wins=tr[tr["return"]>0] if not tr.empty else tr; losses=tr[tr["return"]<0] if not tr.empty else tr; pf=wins["return"].sum()/abs(losses["return"].sum()) if not losses.empty else float("inf")
    return {"total_return_pct":(e.iloc[-1]/initial-1)*100,"cagr_pct":((e.iloc[-1]/initial)**(1/years)-1)*100,"max_drawdown_pct":dd.min()*100,"sharpe":r.mean()/r.std()*np.sqrt(252) if r.std()>0 else 0,"sortino":r.mean()/r[r<0].std()*np.sqrt(252) if (r<0).any() and r[r<0].std()>0 else 0,"win_rate_pct":len(wins)/len(tr)*100 if len(tr) else 0,"profit_factor":pf,"trades":int(len(tr)),"best_month_pct":monthly.max()*100 if len(monthly) else 0,"worst_month_pct":monthly.min()*100 if len(monthly) else 0,"months_over_30pct":int((monthly>=.30).sum()) if len(monthly) else 0}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--tickers",default="tickers.txt"); ap.add_argument("--period",default="5y"); ap.add_argument("--output",default="docs/backtest_30pct.json"); ap.add_argument("--workers",type=int,default=10); args=ap.parse_args()
    panel=load_universe(args.tickers,args.period,args.workers); panel["Date"]=pd.to_datetime(panel.Date); dates=np.sort(panel.Date.unique()); n=len(dates); train_end=dates[int(n*.60)]; val_end=dates[int(n*.80)]
    train=panel[panel.Date<train_end].copy(); val=panel[(panel.Date>=train_end)&(panel.Date<val_end)].copy(); test=panel[panel.Date>=val_end].copy()
    val_probs,_=fit_predict(train,val); val_probs=pd.Series(val_probs,index=val.index); grid=[]
    for prob in (.55,.60,.65,.70,.75):
      for sl in (.8,1.,1.25,1.5,2.):
       for tp in (3.,5.,7.,10.):
        for hold in (1,2,3,5):
         p=Params(prob,sl,tp,hold,5,.02); eq,tr=simulate(val,val_probs,p); m=metrics(eq,tr,1_000_000); score=m["total_return_pct"]-.60*abs(m["max_drawdown_pct"])-max(0,20-m["trades"])*.25; grid.append((score,p,m))
    grid.sort(key=lambda x:x[0],reverse=True); best_score,best,best_val=grid[0]; trainval=pd.concat([train,val],ignore_index=True); test_probs,_=fit_predict(trainval,test); test_probs=pd.Series(test_probs,index=test.index); eq,tr=simulate(test,test_probs,best); test_metrics=metrics(eq,tr,1_000_000)
    result={"research_target_monthly_pct":30,"split":{"train_end":str(train_end.date()),"validation_end":str(val_end.date()),"test_start":str(val_end.date()),"test_end":str(dates[-1].date())},"best_params":asdict(best),"validation_score":best_score,"validation_metrics":best_val,"test_metrics":test_metrics,"model":"LightGBM pooled cross-sectional classifier","target":"next session high >= 5% above next session open","execution":"next-open entry; every held position checked daily against OHLC; both SL/TP hit assumes stop first; commission 3bps + slippage 5bps","notes":["30% monthly is a research target, not a guaranteed return.","Current ticker file is a present-day universe and may contain survivorship bias.","Daily OHLC cannot know intraday order of SL/TP except by conservative both-hit rule.","Validation is used only for parameter selection; test remains untouched."]}
    os.makedirs(os.path.dirname(args.output) or ".",exist_ok=True)
    with open(args.output,"w") as f:json.dump(result,f,indent=2,default=str)
    print(json.dumps(result,indent=2,default=str))
if __name__=="__main__":main()
