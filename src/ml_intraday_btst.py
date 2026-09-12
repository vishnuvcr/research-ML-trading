#!/usr/bin/env python3
"""Leakage-aware ML intraday + BTST research/signal engine for Indian equities.

Signal generation only. No broker orders are placed. yfinance is supported for
research/live snapshots; for serious multi-year intraday research configure a CSV
provider with externally collected historical 15-minute bars.
"""
from __future__ import annotations
import argparse, json, logging, pathlib
from dataclasses import dataclass, asdict
from typing import List
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from lightgbm import LGBMClassifier

LOG = logging.getLogger("mltrading")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

@dataclass
class Regime:
    name: str
    vol: float
    percentile: float
    size_multiplier: float

def clean_ohlcv(d: pd.DataFrame) -> pd.DataFrame:
    if d is None or d.empty: return pd.DataFrame()
    d=d.copy()
    if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
    cols={str(c).strip().lower():c for c in d.columns}; rename={}
    for want in ["open","high","low","close","volume"]:
        if want in cols: rename[cols[want]]=want.title() if want!="volume" else "Volume"
    d=d.rename(columns=rename); needed=["Open","High","Low","Close","Volume"]
    if any(c not in d.columns for c in needed): return pd.DataFrame()
    d=d.dropna(subset=needed).copy(); d.index=pd.to_datetime(d.index)
    if getattr(d.index,"tz",None) is not None: d.index=d.index.tz_convert("Asia/Kolkata").tz_localize(None)
    return d.sort_index()[needed]

class DataClient:
    def __init__(self,cfg):
        self.cfg=cfg; self.source=str(cfg["data"].get("intraday_source","yfinance")).lower(); self.yf=None
        if self.source=="yfinance":
            import yfinance as yf; self.yf=yf
    def daily(self,ticker,period=None):
        import yfinance as yf
        d=yf.download(ticker,period=period or self.cfg["data"]["daily_period"],interval="1d",auto_adjust=False,progress=False,threads=False)
        return clean_ohlcv(d)
    def intraday(self,ticker):
        if self.source=="csv":
            root=pathlib.Path(self.cfg["data"]["intraday_csv_dir"]); p=root/f"{ticker.replace('.', '_')}.csv"
            if not p.exists(): p=root/f"{ticker}.csv"
            if not p.exists(): return pd.DataFrame()
            d=pd.read_csv(p,parse_dates=[0]).set_index(lambda x:x) if False else pd.read_csv(p,parse_dates=[0])
            d=d.set_index(d.columns[0]); return clean_ohlcv(d)
        d=self.yf.download(ticker,period=self.cfg["data"]["intraday_period"],interval=self.cfg["data"]["intraday_interval"],auto_adjust=False,progress=False,threads=False,prepost=False)
        return clean_ohlcv(d)

def rsi(s,n=14):
    delta=s.diff(); up=delta.clip(lower=0).rolling(n).mean(); dn=(-delta.clip(upper=0)).rolling(n).mean(); return 100-100/(1+up/(dn+1e-12))

def atr(df,n=14):
    pc=df.Close.shift(1); tr=pd.concat([(df.High-df.Low),(df.High-pc).abs(),(df.Low-pc).abs()],axis=1).max(axis=1); return tr.rolling(n).mean()

def macd(s):
    e12=s.ewm(span=12,adjust=False).mean(); e26=s.ewm(span=26,adjust=False).mean(); line=e12-e26; sig=line.ewm(span=9,adjust=False).mean(); return line,sig,line-sig

def add_features(df,market=None,intraday=False):
    x=df.copy(); c=x.Close; o=x.Open; h=x.High; l=x.Low; v=x.Volume; ns=[1,2,3,5,10,21,63] if not intraday else [1,2,4,8,16,32]
    for n in ns: x[f"ret_{n}"]=np.log(c/c.shift(n))
    x["co"]=(c-o)/o; x["hl"]=(h-l)/c; x["gap"]=(o-c.shift(1))/c.shift(1); x["atr14"]=atr(x); x["natr14"]=x.atr14/c; x["rsi14"]=rsi(c)
    for n in ([9,21,50,100,200] if not intraday else [9,21,50]): x[f"ema_gap_{n}"]=c/c.ewm(span=n,adjust=False).mean()-1
    for n in [14,21,50]: x[f"sma_gap_{n}"]=c/c.rolling(n).mean()-1
    m,s,hist=macd(c); x["macd"]=m/c; x["macd_sig"]=s/c; x["macd_hist"]=hist/c
    ll=l.rolling(14).min(); hh=h.rolling(14).max(); x["stoch_k"]=(c-ll)/(hh-ll+1e-12); x["wpr"]=(hh-c)/(hh-ll+1e-12)
    adv=v.rolling(20).mean(); x["adv20"]=adv; x["vol_ratio"]=v/(adv+1e-12); x["vol_z"]=(v-adv)/(v.rolling(20).std()+1e-12); x["range_z"]=((h-l)-(h-l).rolling(20).mean())/((h-l).rolling(20).std()+1e-12); x["drawdown_63"]=c/c.rolling(63).max()-1
    if intraday:
        sess=x.index.date; tp=(h+l+c)/3; x["cum_vol"]=v.groupby(sess).cumsum(); x["cum_pv"]=(tp*v).groupby(sess).cumsum(); x["vwap"]=x.cum_pv/(x.cum_vol+1e-12); x["vwap_gap"]=c/x.vwap-1
        mins=x.index.hour*60+x.index.minute; x["tod_sin"]=np.sin(2*np.pi*(mins-555)/375); x["tod_cos"]=np.cos(2*np.pi*(mins-555)/375)
    if market is not None and not market.empty:
        mc=market.Close.reindex(x.index,method="ffill"); x["mkt_ret_1"]=mc.pct_change()
        if intraday: x["mkt_ret_5"]=mc.pct_change(5); x["rel_ret_5"]=x.ret_4-x.mkt_ret_5
        else: x["mkt_ret_5"]=mc.pct_change(5); x["mkt_ret_21"]=mc.pct_change(21); x["rel_ret_5"]=x.ret_5-x.mkt_ret_5; x["rel_ret_21"]=x.ret_21-x.mkt_ret_21
    return x.replace([np.inf,-np.inf],np.nan)

def triple_barrier_labels(x,horizon,stop_mult,target_mult):
    c=x.Close.to_numpy(); a=x.atr14.to_numpy(); hi=x.High.to_numpy(); lo=x.Low.to_numpy(); y=np.full(len(x),np.nan)
    for i in range(max(0,len(x)-horizon-1)):
        if not np.isfinite(a[i]) or a[i]<=0: continue
        entry=c[i]; stop=entry-stop_mult*a[i]; target=entry+target_mult*a[i]; th=np.where(hi[i+1:i+1+horizon]>=target)[0]; sh=np.where(lo[i+1:i+1+horizon]<=stop)[0]
        if len(th) and len(sh): y[i]=1 if th[0]<=sh[0] else 0
        elif len(th): y[i]=1
        elif len(sh): y[i]=0
        else: y[i]=1 if c[min(i+horizon,len(x)-1)]>entry else 0
    return pd.Series(y,index=x.index,name="target")

def btst_labels(x,stop_mult,target_mult):
    y=np.full(len(x),np.nan); c=x.Close.to_numpy(); a=x.atr14.to_numpy(); hi=x.High.to_numpy(); lo=x.Low.to_numpy()
    for i in range(len(x)-1):
        if not np.isfinite(a[i]) or a[i]<=0: continue
        entry=c[i]; stop=entry-stop_mult*a[i]; target=entry+target_mult*a[i]
        if hi[i+1]>=target and lo[i+1]<=stop: y[i]=0
        elif hi[i+1]>=target: y[i]=1
        elif lo[i+1]<=stop: y[i]=0
        else: y[i]=1 if c[i+1]>entry else 0
    return pd.Series(y,index=x.index,name="target")

class ModelEnsemble:
    def __init__(self,cfg):
        rs=cfg["model"]["random_state"]
        self.models={"lgbm":LGBMClassifier(n_estimators=220,max_depth=5,num_leaves=24,learning_rate=.03,subsample=.85,colsample_bytree=.85,random_state=rs,verbosity=-1,n_jobs=2),"rf":RandomForestClassifier(n_estimators=180,max_depth=9,min_samples_leaf=12,class_weight="balanced_subsample",random_state=rs,n_jobs=2),"lr":Pipeline([("s",RobustScaler()),("m",LogisticRegression(C=.15,max_iter=1200,class_weight="balanced",random_state=rs))])}
        self.weights={k:1/len(self.models) for k in self.models}; self.oof_auc={}
    def validation(self,data,features,folds,purge_bars):
        data=data.sort_values("time").reset_index(drop=True); times=np.array(sorted(data.time.unique())); splits=np.array_split(times,folds+1); oof={k:[] for k in self.models}; yy=[]
        for j in range(1,len(splits)):
            trt=np.concatenate(splits[:j]); vat=splits[j]
            if purge_bars and len(trt)>purge_bars: trt=trt[:-purge_bars]
            tr=data.time.isin(trt); va=data.time.isin(vat)
            if tr.sum()<500 or va.sum()<50: continue
            for name,m in self.models.items(): m.fit(data.loc[tr,features],data.loc[tr,"target"]); oof[name].extend(m.predict_proba(data.loc[va,features])[:,1].tolist())
            yy.extend(data.loc[va,"target"].astype(int).tolist())
        if len(yy)<50 or len(set(yy))<2: raise RuntimeError("Insufficient walk-forward validation samples/classes")
        for k in self.models: self.oof_auc[k]=float(roc_auc_score(yy,oof[k]))
        w={k:max(0.0,a-.5) for k,a in self.oof_auc.items()}; total=sum(w.values()); self.weights={k:(w[k]/total if total else 1/len(w)) for k in w}; return np.array(yy),{k:np.array(v) for k,v in oof.items()}
    def fit(self,data,features):
        if len(data)>250000: data=data.sort_values("time").tail(250000)
        for m in self.models.values(): m.fit(data[features],data.target)
    def predict(self,x,features): return float(sum(self.weights[k]*m.predict_proba(x[features])[:,1][0] for k,m in self.models.items()))

def choose_threshold(y,preds,cfg):
    fixed=cfg["model"].get("signal_probability",.58)
    if cfg["model"].get("threshold_mode","validation")!="validation": return float(fixed)
    best=(fixed,-np.inf)
    for t in np.arange(.50,.81,.01):
        mask=preds>=t; n=int(mask.sum())
        if n<int(cfg["model"].get("min_validation_signals",30)): continue
        score=float(y[mask].mean())*np.sqrt(n)
        if score>best[1]: best=(float(t),score)
    return best[0]

def load_symbols(path,max_symbols):
    syms=[s.strip().upper() for s in pathlib.Path(path).read_text().splitlines() if s.strip() and not s.startswith("#")]; return [s if s.endswith(".NS") else s+".NS" for s in syms][:max_symbols]

def market_regime(client,cfg):
    d=client.daily(cfg["data"]["market_ticker"],"2y")
    if d.empty: return Regime("UNKNOWN",0,.5,cfg["risk"]["normal_vol_size_multiplier"])
    rv=np.log(d.Close/d.Close.shift(1)).rolling(21).std()*np.sqrt(252); hist=rv.dropna(); cur=float(hist.iloc[-1]); pct=float((hist<cur).mean()); r=cfg["risk"]
    if pct>r["regime_high_vol_pct"]: return Regime("HIGH_VOL",cur,pct,r["high_vol_size_multiplier"])
    if pct<r["regime_low_vol_pct"]: return Regime("LOW_VOL",cur,pct,r["low_vol_size_multiplier"])
    return Regime("NORMAL",cur,pct,r["normal_vol_size_multiplier"])

def stops(entry,atrv,ex):
    stop=entry-ex["stop_atr_mult"]*atrv if ex["stop_mode"]=="ATR" else entry*(1-ex["stop_pct"]/100); target=entry+ex["target_atr_mult"]*atrv if ex["target_mode"]=="ATR" else entry*(1+ex["target_pct"]/100); return stop,target

def net_return(entry,exit_px,ex):
    cost=(ex["slippage_bps_per_side"]+ex["transaction_cost_bps_per_side"])*2/10000; return exit_px*(1-cost)/entry-1

def position_qty(capital,entry,stop,adv,risk_cfg,regime_mult):
    if not np.isfinite(adv) or adv<risk_cfg["min_adv_shares"]: return 0
    risk_cash=capital*risk_cfg["risk_per_trade"]*regime_mult; per_share=max(entry-stop,entry*.0025); qty=max(0,int(risk_cash/per_share)); qty=min(qty,int(adv*risk_cfg["max_participation_pct_adv"]/100)); qty=min(qty,int(capital*risk_cfg["max_gross_exposure"]/entry)); return max(0,qty)

def trade_metrics(rets):
    r=np.asarray(rets,dtype=float)
    if len(r)==0: return {"trades":0,"win_rate":None,"profit_factor":None,"expectancy":0.0,"total_return":0.0,"max_drawdown":0.0,"sharpe":None,"avg_win":None,"avg_loss":None}
    eq=np.cumprod(1+r); dd=eq/np.maximum.accumulate(eq)-1; wins=r[r>0]; losses=r[r<0]
    return {"trades":int(len(r)),"win_rate":float((r>0).mean()),"profit_factor":float(wins.sum()/abs(losses.sum())) if len(losses) else None,"expectancy":float(r.mean()),"total_return":float(eq[-1]-1),"max_drawdown":float(dd.min()),"sharpe":float(r.mean()/r.std(ddof=1)*np.sqrt(len(r))) if len(r)>1 and r.std(ddof=1)>0 else None,"avg_win":float(wins.mean()) if len(wins) else None,"avg_loss":float(losses.mean()) if len(losses) else None}

def backtest_intraday_oos(df,model,features,threshold,cfg):
    ex=cfg["execution"]; maxhold=ex["intraday_max_hold_bars"]; rets=[]; trades=[]; probs=[model.predict(df.iloc[[i]],features) for i in range(len(df))]; i=0
    while i<len(df)-maxhold-1:
        if probs[i]<threshold: i+=1; continue
        entry=float(df.Open.iloc[i+1]); a=float(df.atr14.iloc[i])
        if not np.isfinite(a) or a<=0: i+=1; continue
        stop,target=stops(entry,a,ex); exit_px=float(df.Close.iloc[i+maxhold]); reason="TIME"; exit_i=i+maxhold
        for j in range(i+1,min(i+1+maxhold,len(df))):
            if df.Low.iloc[j]<=stop and df.High.iloc[j]>=target: exit_px=stop; reason="STOP_SAME_BAR_CONSERVATIVE"; exit_i=j; break
            if df.Low.iloc[j]<=stop: exit_px=stop; reason="STOP"; exit_i=j; break
            if df.High.iloc[j]>=target: exit_px=target; reason="TARGET"; exit_i=j; break
        r=net_return(entry,exit_px,ex); rets.append(r); trades.append({"time":str(df.index[i]),"entry_time":str(df.index[i+1]),"entry":entry,"exit":exit_px,"return":r,"reason":reason}); i=exit_i+1
    return rets,trades

def backtest_btst_oos(df,model,features,threshold,cfg):
    ex=cfg["execution"]; rets=[]; trades=[]
    for i in range(len(df)-1):
        if model.predict(df.iloc[[i]],features)<threshold: continue
        entry=float(df.Open.iloc[i+1]); a=float(df.atr14.iloc[i])
        if not np.isfinite(a) or a<=0: continue
        stop,target=stops(entry,a,ex); hi=float(df.High.iloc[i+1]); lo=float(df.Low.iloc[i+1]); close=float(df.Close.iloc[i+1])
        if lo<=stop and hi>=target: exit_px=stop; reason="STOP_SAME_BAR_CONSERVATIVE"
        elif lo<=stop: exit_px=stop; reason="STOP"
        elif hi>=target: exit_px=target; reason="TARGET"
        else: exit_px=close; reason="CLOSE"
        r=net_return(entry,exit_px,ex); rets.append(r); trades.append({"time":str(df.index[i]),"entry_time":str(df.index[i+1]),"entry":entry,"exit":exit_px,"return":r,"reason":reason})
    return rets,trades

def build_panel(client,cfg,intraday,market):
    rows=[]; cache={}
    for s in load_symbols(cfg["universe_file"],cfg["max_symbols"]):
        try:
            d=client.intraday(s) if intraday else client.daily(s,cfg["data"]["daily_period"]); minbars=cfg["data"]["min_intraday_bars"] if intraday else cfg["data"]["min_daily_history"]
            if len(d)<minbars: continue
            f=add_features(d,market,intraday); f["target"]=triple_barrier_labels(f,cfg["model"]["horizon_bars_intraday"],cfg["execution"]["stop_atr_mult"],cfg["execution"]["target_atr_mult"]) if intraday else btst_labels(f,cfg["execution"]["stop_atr_mult"],cfg["execution"]["target_atr_mult"]); f["symbol"]=s; f["time"]=f.index; rows.append(f); cache[s]=f
        except Exception as e: LOG.warning("%s %s failed: %s",'intraday' if intraday else 'BTST',s,e)
    if not rows: return pd.DataFrame(),{}
    panel=pd.concat(rows,ignore_index=True).dropna(subset=["target"]); drop={"target","symbol","time","Open","High","Low","Close","Adj Close"}; features=[c for c in panel.columns if c not in drop and pd.api.types.is_numeric_dtype(panel[c])]; return panel.dropna(subset=features),{s:f.dropna(subset=features) for s,f in cache.items()}

def run_mode(client,cfg,intraday,regime):
    market=client.intraday(cfg["data"]["market_ticker"]) if intraday else client.daily(cfg["data"]["market_ticker"],cfg["data"]["daily_period"]); panel,cache=build_panel(client,cfg,intraday,market)
    if panel.empty: return {"metrics":trade_metrics([]),"auc":{},"threshold":None,"signals":[],"status":"NO_DATA"}
    features=[c for c in panel.columns if c not in {"target","symbol","time","Open","High","Low","Close","Adj Close"} and pd.api.types.is_numeric_dtype(panel[c])]; model=ModelEnsemble(cfg); y,oof=model.validation(panel,features,cfg["model"]["folds"],cfg["model"]["horizon_bars_intraday"] if intraday else 1); ensemble=np.zeros(len(y))
    for k,v in oof.items(): ensemble+=model.weights[k]*v
    threshold=choose_threshold(y,ensemble,cfg); min_auc=min(model.oof_auc.values())
    if min_auc<cfg["model"]["min_auc"]: return {"metrics":trade_metrics([]),"auc":model.oof_auc,"threshold":threshold,"signals":[],"status":"AUC_GATE_FAILED"}
    times=np.array(sorted(panel.time.unique())); oos_days=cfg["backtest"]["intraday_days"] if intraday else cfg["backtest"]["btst_days"]; cutoff=times[max(0,len(times)-oos_days)] if len(times)>oos_days else times[max(1,len(times)//5)]; train=panel[panel.time<cutoff]; oos=panel[panel.time>=cutoff]
    if len(train)<500 or oos.empty: return {"metrics":trade_metrics([]),"auc":model.oof_auc,"threshold":threshold,"signals":[],"status":"INSUFFICIENT_OOS"}
    model.fit(train,features); all_rets=[]
    for s,f in cache.items():
        fo=f[f.index>=pd.Timestamp(cutoff)]
        if fo.empty: continue
        rets,_=(backtest_intraday_oos(fo,model,features,threshold,cfg) if intraday else backtest_btst_oos(fo,model,features,threshold,cfg)); all_rets.extend(rets)
    metrics=trade_metrics(all_rets); model.fit(panel,features); candidates=[]
    for s,f in cache.items():
        if f.empty: continue
        row=f.iloc[[-1]]; p=model.predict(row,features); adv=float(row.adv20.iloc[0]) if "adv20" in row else np.nan
        if p<threshold or not np.isfinite(adv) or adv<cfg["risk"]["min_adv_shares"]: continue
        entry=float(row.Close.iloc[0]); a=float(row.atr14.iloc[0]);
        if not np.isfinite(a) or a<=0: continue
        stop,target=stops(entry,a,cfg["execution"]); qty=position_qty(cfg["risk"]["initial_capital"],entry,stop,adv,cfg["risk"],regime.size_multiplier)
        if qty<=0: continue
        candidates.append({"symbol":s,"mode":"INTRADAY" if intraday else "BTST","signal_time":str(row.index[0]),"probability":round(p,5),"entry_reference":entry,"stop":round(stop,4),"target":round(target,4),"qty":qty,"notional":round(qty*entry,2),"regime":regime.name})
    return {"metrics":metrics,"auc":model.oof_auc,"threshold":threshold,"signals":sorted(candidates,key=lambda z:z["probability"],reverse=True)[:cfg["risk"]["max_positions"]],"status":"OK","oos_start":str(cutoff),"oos_rows":int(len(oos))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="config/trading.yaml"); args=ap.parse_args(); cfg=yaml.safe_load(pathlib.Path(args.config).read_text()); client=DataClient(cfg); regime=market_regime(client,cfg); modes=[]
    if cfg["mode"] in ("both","intraday"): modes.append(("intraday",True))
    if cfg["mode"] in ("both","btst"): modes.append(("btst",False))
    result={"generated_at":pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),"regime":asdict(regime),"modes":{}}; signals=[]
    for name,is_intraday in modes:
        try: r=run_mode(client,cfg,is_intraday,regime); result["modes"][name]=r; signals.extend(r.get("signals",[]))
        except Exception as e: LOG.exception("%s pipeline failed",name); result["modes"][name]={"status":"ERROR","error":str(e),"metrics":trade_metrics([]),"signals":[]}
    signals=sorted(signals,key=lambda z:z["probability"],reverse=True); out=cfg["output"]; pathlib.Path(out["signal_json"]).write_text(json.dumps({"generated_at":result["generated_at"],"regime":result["regime"],"signals":signals},indent=2,default=str)); pathlib.Path(out["backtest_json"]).write_text(json.dumps(result,indent=2,default=str))
    history=pathlib.Path(out["history_json"]); hist=[]
    if history.exists():
        try: hist=json.loads(history.read_text())
        except Exception: hist=[]
    hist.append({"generated_at":result["generated_at"],"regime":result["regime"],"intraday":result["modes"].get("intraday",{}).get("metrics",{}),"btst":result["modes"].get("btst",{}).get("metrics",{})}); history.write_text(json.dumps(hist[-1000:],indent=2,default=str)); pd.DataFrame(signals).to_csv(out["signals_csv"],index=False); LOG.info("Generated %d signals; regime=%s",len(signals),regime.name)

if __name__=="__main__": main()
