#!/usr/bin/env python3
"""ML intraday + BTST research/signal pipeline for Indian equities.

Designed for GitHub Actions. Research mode uses yfinance; a broker/data adapter can
replace DataClient without changing the ML/risk/backtest layers.
"""
from __future__ import annotations
import argparse, json, logging, math, os, pathlib, time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

LOG = logging.getLogger("mltrading")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

@dataclass
class Regime:
    name: str
    vol: float
    percentile: float
    size_multiplier: float

class DataClient:
    def __init__(self, cfg):
        self.cfg = cfg
        import yfinance as yf
        self.yf = yf
    @staticmethod
    def _flat(df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df
    def daily(self, ticker: str, period: str | None = None) -> pd.DataFrame:
        d = self.yf.download(ticker, period=period or self.cfg["data"]["daily_period"], interval="1d", auto_adjust=False, progress=False, threads=False)
        if d is None or d.empty: return pd.DataFrame()
        d = self._flat(d).dropna(subset=["Open","High","Low","Close","Volume"]).copy()
        d.index = pd.to_datetime(d.index).tz_localize(None)
        return d
    def intraday(self, ticker: str) -> pd.DataFrame:
        d = self.yf.download(ticker, period=self.cfg["data"]["intraday_period"], interval=self.cfg["data"]["intraday_interval"], auto_adjust=False, progress=False, threads=False, prepost=False)
        if d is None or d.empty: return pd.DataFrame()
        d = self._flat(d).dropna(subset=["Open","High","Low","Close","Volume"]).copy()
        d.index = pd.to_datetime(d.index).tz_localize(None)
        return d

def rsi(s, n=14):
    delta=s.diff(); up=delta.clip(lower=0).rolling(n).mean(); dn=(-delta.clip(upper=0)).rolling(n).mean()
    return 100 - 100/(1 + up/(dn+1e-12))

def atr(df, n=14):
    pc=df.Close.shift(1); tr=pd.concat([df.High-df.Low,(df.High-pc).abs(),(df.Low-pc).abs()],axis=1).max(axis=1)
    return tr.rolling(n).mean()

def macd(s):
    e12=s.ewm(span=12,adjust=False).mean(); e26=s.ewm(span=26,adjust=False).mean(); line=e12-e26; sig=line.ewm(span=9,adjust=False).mean()
    return line, sig, line-sig

def add_features(df: pd.DataFrame, market: pd.DataFrame | None = None, intraday=False) -> pd.DataFrame:
    x=df.copy(); c=x.Close; o=x.Open; h=x.High; l=x.Low; v=x.Volume
    for n in ([1,2,3,5,10,21,63] if not intraday else [1,2,4,8,16,32]): x[f"ret_{n}"]=np.log(c/c.shift(n))
    x["co"]=(c-o)/o; x["hl"]=(h-l)/c; x["gap"]=(o-c.shift(1))/c.shift(1)
    x["atr14"]=atr(x,14); x["natr14"]=x.atr14/c; x["rsi14"]=rsi(c,14)
    for n in ([9,21,50,100,200] if not intraday else [9,21,50]): x[f"ema_gap_{n}"]=(c/c.ewm(span=n,adjust=False).mean())-1
    for n in [14,21,50]: x[f"sma_gap_{n}"]=c/c.rolling(n).mean()-1
    m,s,hist=macd(c); x["macd"]=m/c; x["macd_sig"]=s/c; x["macd_hist"]=hist/c
    ll=l.rolling(14).min(); hh=h.rolling(14).max(); x["stoch_k"]=(c-ll)/(hh-ll+1e-12); x["wpr"]=(hh-c)/(hh-ll+1e-12)
    adv=v.rolling(20).mean(); x["adv20"]=adv; x["vol_ratio"]=v/(adv+1e-12); x["vol_z"]=(v-adv)/(v.rolling(20).std()+1e-12)
    x["range_z"]=((h-l)-(h-l).rolling(20).mean())/((h-l).rolling(20).std()+1e-12)
    x["drawdown_63"]=c/c.rolling(63).max()-1
    if intraday:
        sess=x.index.date; tp=(h+l+c)/3; x["cum_vol"]=v.groupby(sess).cumsum(); x["cum_pv"]=(tp*v).groupby(sess).cumsum(); x["vwap"]=x.cum_pv/(x.cum_vol+1e-12); x["vwap_gap"]=c/x.vwap-1
        mins=x.index.hour*60+x.index.minute; x["tod_sin"]=np.sin(2*np.pi*(mins-555)/375); x["tod_cos"]=np.cos(2*np.pi*(mins-555)/375)
    if market is not None and not market.empty:
        mc=market.Close.reindex(x.index,method="ffill"); x["mkt_ret_1"]=mc.pct_change(); x["rel_ret_5"]=x.ret_5 - mc.pct_change(5); x["rel_ret_21"]=x.ret_21 - mc.pct_change(21)
    return x.replace([np.inf,-np.inf],np.nan)

def triple_barrier_labels(x: pd.DataFrame, horizon: int, stop_mult: float, target_mult: float) -> pd.Series:
    close=x.Close.to_numpy(); atrv=x.atr14.to_numpy(); y=np.full(len(x), np.nan)
    for i in range(len(x)-horizon-1):
        entry=close[i]; stop=entry-stop_mult*atrv[i]; target=entry+target_mult*atrv[i]
        if not np.isfinite(stop) or not np.isfinite(target): continue
        path_hi=x.High.to_numpy()[i+1:i+1+horizon]; path_lo=x.Low.to_numpy()[i+1:i+1+horizon]
        hit_t=np.where(path_hi>=target)[0]; hit_s=np.where(path_lo<=stop)[0]
        if len(hit_t) and (not len(hit_s) or hit_t[0] <= hit_s[0]): y[i]=1
        elif len(hit_s): y[i]=0
        else: y[i]=1 if close[i+horizon] > entry else 0
    return pd.Series(y,index=x.index,name="target")

def btst_labels(x: pd.DataFrame, stop_mult: float, target_mult: float) -> pd.Series:
    y=np.full(len(x), np.nan); close=x.Close.to_numpy(); a=x.atr14.to_numpy(); hi=x.High.to_numpy(); lo=x.Low.to_numpy()
    for i in range(len(x)-1):
        if not np.isfinite(a[i]) or a[i] <= 0: continue
        entry=close[i]; stop=entry-stop_mult*a[i]; target=entry+target_mult*a[i]
        if hi[i+1]>=target and lo[i+1]<=stop: y[i]=0
        elif hi[i+1]>=target: y[i]=1
        elif lo[i+1]<=stop: y[i]=0
        else: y[i]=1 if close[i+1]>entry else 0
    return pd.Series(y,index=x.index,name="target")

class ModelEnsemble:
    def __init__(self, cfg):
        rs=cfg["model"]["random_state"]
        self.models={
            "lgbm":LGBMClassifier(n_estimators=180,max_depth=5,num_leaves=24,learning_rate=0.03,subsample=.85,colsample_bytree=.85,random_state=rs,verbosity=-1,n_jobs=2),
            "rf":RandomForestClassifier(n_estimators=160,max_depth=9,min_samples_leaf=12,class_weight="balanced_subsample",random_state=rs,n_jobs=2),
            "lr":Pipeline([("s",RobustScaler()),("m",LogisticRegression(C=.15,max_iter=1200,class_weight="balanced",random_state=rs))])}
        self.weights={}
    def fit_walk_forward(self, data: pd.DataFrame, features: List[str], folds=4) -> Tuple[Dict,str,float]:
        data=data.sort_values("time").reset_index(drop=True); times=np.array(sorted(data.time.unique())); splits=np.array_split(times,folds+1); oof={k:[] for k in self.models}; yo=[]; min_auc=.0
        for j in range(1,len(splits)):
            tr_times=np.concatenate(splits[:j]); va_times=splits[j]
            if j>0 and len(tr_times)>8: tr_times=tr_times[:-8]
            tr=data.time.isin(tr_times); va=data.time.isin(va_times)
            if tr.sum()<1000 or va.sum()<50: continue
            Xtr=data.loc[tr,features]; ytr=data.loc[tr,"target"]; Xv=data.loc[va,features]; yv=data.loc[va,"target"]
            for name,m in self.models.items():
                m.fit(Xtr,ytr); p=m.predict_proba(Xv)[:,1]; oof[name].extend(p.tolist())
            yo.extend(yv.tolist())
        if len(set(yo))<2: raise RuntimeError("Insufficient class diversity in walk-forward validation")
        for name in self.models:
            auc=roc_auc_score(yo,oof[name]); self.weights[name]=max(0.0,auc-.5); LOG.info("%s OOF AUC %.4f",name,auc); min_auc=max(min_auc,auc)
        if sum(self.weights.values())==0: self.weights={k:1 for k in self.models}
        total=sum(self.weights.values()); self.weights={k:v/total for k,v in self.weights.items()}
        return self.weights,"walk_forward",min_auc
    def fit(self, data, features):
        if len(data)>250000: data=data.sample(250000,random_state=42).sort_values("time")
        for m in self.models.values(): m.fit(data[features],data.target)
    def predict(self, x):
        return float(sum(self.weights.get(k,1/len(self.models))*m.predict_proba(x)[:,1][0] for k,m in self.models.items()))

def load_symbols(path, max_symbols):
    syms=[s.strip().upper() for s in pathlib.Path(path).read_text().splitlines() if s.strip() and not s.strip().startswith("#")]
    syms=[s if s.endswith(".NS") else f"{s}.NS" for s in syms]
    return syms[:max_symbols]

def market_regime(client,cfg):
    d=client.daily(cfg["data"]["market_ticker"],"2y"); r=np.log(d.Close/d.Close.shift(1)); rv=r.rolling(21).std()*np.sqrt(252); cur=float(rv.iloc[-1]); hist=rv.dropna(); pct=float((hist<cur).mean()) if len(hist) else .5
    if pct>cfg["risk"]["regime_high_vol_pct"]: return Regime("HIGH_VOL",cur,pct,cfg["risk"]["high_vol_size_multiplier"])
    if pct<cfg["risk"]["regime_low_vol_pct"]: return Regime("LOW_VOL",cur,pct,cfg["risk"]["low_vol_size_multiplier"])
    return Regime("NORMAL",cur,pct,cfg["risk"]["normal_vol_size_multiplier"])

def stops(entry, atrv, ex):
    stop=entry-ex["stop_atr_mult"]*atrv if ex["stop_mode"]=="ATR" else entry*(1-ex["stop_pct"]/100)
    target=entry+ex["target_atr_mult"]*atrv if ex["target_mode"]=="ATR" else entry*(1+ex["target_pct"]/100)
    return stop,target

def position_qty(capital, entry, stop, adv, risk_cfg, regime_mult):
    risk_cash=capital*risk_cfg["risk_per_trade"]*regime_mult; per_share=max(entry-stop,entry*0.0025); qty=max(0,int(risk_cash/per_share))
    qty=min(qty,int(adv*risk_cfg["max_participation_pct_adv"]/100)); qty=min(qty,int(capital*risk_cfg["max_gross_exposure"]/entry)); return qty

def backtest_intraday(df, prob, cfg):
    ex=cfg["execution"]; trades=[]; i=0
    while i<len(df)-ex["intraday_max_hold_bars"]-1:
        if prob[i] < cfg["model"]["signal_probability"]: i+=1; continue
        entry=float(df.Close.iloc[i]); atrv=float(df.atr14.iloc[i]); stop,target=stops(entry,atrv,ex); exit_px=float(df.Close.iloc[min(i+ex["intraday_max_hold_bars"],len(df)-1)]); reason="TIME"
        for j in range(i+1,min(i+1+ex["intraday_max_hold_bars"],len(df))):
            if df.Low.iloc[j]<=stop: exit_px=stop; reason="STOP"; break
            if df.High.iloc[j]>=target: exit_px=target; reason="TARGET"; break
        cost=(ex["slippage_bps_per_side"]+ex["transaction_cost_bps_per_side"])*2/10000; ret=(exit_px*(1-cost)/entry)-1; trades.append(ret); i=j
    return trades

def simple_backtest_btst(df, model, features, cfg):
    ex=cfg["execution"]; out=[]
    for i in range(len(df)-1):
        p=model.predict(df.iloc[[i]][features])
        if p<cfg["model"]["signal_probability"]: continue
        entry=float(df.Open.iloc[i+1]); a=float(df.atr14.iloc[i]); stop,target=stops(entry,a,ex); hi=float(df.High.iloc[i+1]); lo=float(df.Low.iloc[i+1]); close=float(df.Close.iloc[i+1])
        if lo<=stop: exit_px=stop
        elif hi>=target: exit_px=target
        else: exit_px=close
        cost=(ex["slippage_bps_per_side"]+ex["transaction_cost_bps_per_side"])*2/10000; out.append((exit_px*(1-cost)/entry)-1)
    return out

def train_daily_btst(client,cfg,regime):
    market=client.daily(cfg["data"]["market_ticker"],cfg["data"]["daily_period"]); rows=[]; cache=[]
    for s in load_symbols(cfg["universe_file"],cfg["max_symbols"]):
        try:
            d=client.daily(s,cfg["data"]["daily_period"])
            if len(d)<cfg["data"]["min_daily_history"]: continue
            f=add_features(d,market,False); f["target"]=btst_labels(f,cfg["execution"]["stop_atr_mult"],cfg["execution"]["target_atr_mult"]); f["symbol"]=s; f["time"]=f.index; rows.append(f); cache.append((s,f))
        except Exception as e: LOG.warning("BTST %s failed: %s",s,e)
    if not rows: return [], {"btst_auc":None,"btst_trades":0,"btst_return":0.0}
    panel=pd.concat(rows,ignore_index=True).dropna(subset=["target"]); drop={"target","symbol","time"}; features=[c for c in panel.columns if c not in drop and c not in ["Open","High","Low","Close","Adj Close"] and pd.api.types.is_numeric_dtype(panel[c])]; panel=panel.dropna(subset=features)
    model=ModelEnsemble(cfg); _,_,auc=model.fit_walk_forward(panel,features,cfg["model"]["folds"]); model.fit(panel,features); signals=[]; rets=[]
    for s,f in cache:
        ff=f.dropna(subset=features)
        if ff.empty: continue
        i=len(ff)-1; row=ff.iloc[i]; p=model.predict(ff.iloc[[i]][features]); e=float(row.Close); a=float(row.atr14); st,tg=stops(e,a,cfg["execution"]); adv=float(row.adv20); qty=position_qty(cfg["risk"]["initial_capital"],e,st,adv,cfg["risk"],regime.size_multiplier) if np.isfinite(adv) else 0
        signals.append({"symbol":s,"time":str(ff.index[i]),"probability":round(p,5),"entry_reference":round(e,2),"stop":round(st,2),"target":round(tg,2),"qty":qty,"side":"BTST_LONG" if p>=cfg["model"]["signal_probability"] and qty>0 else "NO_TRADE","mode":"BTST"}); rets.extend(simple_backtest_btst(ff.tail(cfg["backtest"]["btst_days"]),model,features,cfg))
    return signals, {"btst_auc":float(auc),"btst_trades":len(rets),"btst_return":float(np.prod([1+r for r in rets])-1) if rets else 0.0}

def run(cfg):
    client=DataClient(cfg); symbols=load_symbols(cfg["universe_file"],cfg["max_symbols"]); regime=market_regime(client,cfg); LOG.info("Regime %s vol=%.3f pct=%.1f",regime.name,regime.vol,100*regime.percentile)
    intraday_market=client.intraday(cfg["data"]["market_ticker"]); rows_i=[]; cache_i=[]
    for k,s in enumerate(symbols,1):
        try:
            d=client.intraday(s)
            if len(d)<cfg["data"]["min_intraday_bars"]: continue
            f=add_features(d,intraday_market,True); f["target"]=triple_barrier_labels(f,cfg["model"]["horizon_bars_intraday"],cfg["execution"]["stop_atr_mult"],cfg["execution"]["target_atr_mult"]); f["symbol"]=s; f["time"]=f.index; rows_i.append(f); cache_i.append((s,f))
        except Exception as e: LOG.warning("intraday %s failed: %s",s,e)
        if k%20==0: LOG.info("loaded %d/%d intraday symbols",k,len(symbols))
    if not rows_i: raise RuntimeError("No intraday data available")
    panel=pd.concat(rows_i,ignore_index=True).dropna(subset=["target"]); drop={"target","symbol","time"}; feat=[c for c in panel.columns if c not in drop and c not in ["Open","High","Low","Close","Adj Close"] and pd.api.types.is_numeric_dtype(panel[c])]; panel=panel.dropna(subset=feat)
    model=ModelEnsemble(cfg); _,_,auc=model.fit_walk_forward(panel,feat,cfg["model"]["folds"]); model.fit(panel,feat); intraday_signals=[]; hist=[]
    for s,f in cache_i:
        ff=f.dropna(subset=feat)
        if ff.empty: continue
        row=ff.iloc[-1]; p=model.predict(ff.iloc[[-1]][feat]); e=float(row.Close); a=float(row.atr14); st,tg=stops(e,a,cfg["execution"]); adv=float(row.adv20); qty=position_qty(cfg["risk"]["initial_capital"],e,st,adv,cfg["risk"],regime.size_multiplier) if np.isfinite(adv) else 0
        intraday_signals.append({"symbol":s,"time":str(ff.index[-1]),"probability":round(p,5),"entry":round(e,2),"stop":round(st,2),"target":round(tg,2),"qty":qty,"side":"LONG" if p>=cfg["model"]["signal_probability"] and qty>0 else "NO_TRADE","mode":"INTRADAY"})
        ff2=ff.tail(cfg["backtest"]["intraday_days"]*26); probs=[model.predict(ff2.iloc[[i]][feat]) for i in range(len(ff2))]; hist += [{"symbol":s,"return":float(r)} for r in backtest_intraday(ff2,probs,cfg)]
    btst_signals,btst_metrics=train_daily_btst(client,cfg,regime); metrics={"generated_at":pd.Timestamp.utcnow().isoformat(),"intraday_auc":float(auc),"regime":asdict(regime),"intraday_signals":len(intraday_signals),"qualified_intraday":sum(x["side"]=="LONG" for x in intraday_signals),"intraday_trades":len(hist),"intraday_return":float(np.prod([1+r["return"] for r in hist])-1) if hist else 0.0,**btst_metrics}; signals=intraday_signals+btst_signals
    pathlib.Path("docs").mkdir(exist_ok=True); pathlib.Path(cfg["output"]["signal_json"]).write_text(json.dumps({"generated_at":metrics["generated_at"],"metrics":metrics,"signals":signals},indent=2,default=str)); pathlib.Path(cfg["output"]["backtest_json"]).write_text(json.dumps(metrics,indent=2,default=str)); pathlib.Path(cfg["output"]["history_json"]).write_text(json.dumps(hist,indent=2,default=str)); pd.DataFrame(signals).to_csv(cfg["output"]["signals_csv"],index=False); LOG.info("done: %d signals, %d intraday trades, %d BTST trades",len(signals),len(hist),btst_metrics["btst_trades"]); return metrics

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="config/trading.yaml"); args=ap.parse_args(); cfg=yaml.safe_load(open(args.config)); run(cfg)
if __name__=="__main__": main()
