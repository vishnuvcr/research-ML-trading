#!/usr/bin/env python3
"""Audit the two Kaggle datasets used by the historical intraday research pipeline.

This script deliberately does not modify raw data. It discovers CSVs, samples them,
normalizes common column names, checks timestamp/price/volume quality and writes a
small JSON manifest that can be consumed by later backtest jobs.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
from typing import Iterable

import pandas as pd


def csv_files(root: pathlib.Path) -> Iterable[pathlib.Path]:
    yield from root.rglob("*.csv")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for c in df.columns:
        k = re.sub(r"[^a-z0-9]", "", str(c).lower())
        if k in {"datetime", "timestamp", "date", "time", "datatime"}:
            rename[c] = "timestamp"
        elif k == "open": rename[c] = "open"
        elif k == "high": rename[c] = "high"
        elif k == "low": rename[c] = "low"
        elif k in {"close", "closingprice"}: rename[c] = "close"
        elif k in {"volume", "vol"}: rename[c] = "volume"
    return df.rename(columns=rename)


def inspect_file(path: pathlib.Path, sample_rows: int) -> dict:
    try:
        df = pd.read_csv(path, nrows=sample_rows)
        df = normalize_columns(df)
        out = {"file": str(path), "sample_rows": len(df), "columns": list(df.columns), "error": None}
        if "timestamp" in df:
            ts = pd.to_datetime(df["timestamp"], errors="coerce")
            out["timestamp_valid_fraction"] = float(ts.notna().mean())
            out["sample_start"] = str(ts.min()) if ts.notna().any() else None
            out["sample_end"] = str(ts.max()) if ts.notna().any() else None
            out["duplicate_timestamp_fraction"] = float(ts.duplicated().mean())
        for c in ("open", "high", "low", "close"):
            if c in df:
                out[f"{c}_nonpositive_fraction"] = float((pd.to_numeric(df[c], errors="coerce") <= 0).mean())
        if "volume" in df:
            vol = pd.to_numeric(df["volume"], errors="coerce")
            out["zero_volume_fraction"] = float((vol == 0).mean())
        if all(c in df for c in ("high", "low", "open", "close")):
            h = pd.to_numeric(df.high, errors="coerce")
            l = pd.to_numeric(df.low, errors="coerce")
            o = pd.to_numeric(df.open, errors="coerce")
            c = pd.to_numeric(df.close, errors="coerce")
            out["ohlc_invalid_fraction"] = float(((h < l) | (h < o) | (h < c) | (l > o) | (l > c)).mean())
        return out
    except Exception as exc:
        return {"file": str(path), "sample_rows": 0, "columns": [], "error": repr(exc)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True, help="Kaggle dataset directories")
    ap.add_argument("--output", default="docs/kaggle_data_manifest.json")
    ap.add_argument("--sample-rows", type=int, default=5000)
    args = ap.parse_args()

    roots = [pathlib.Path(x).expanduser().resolve() for x in args.roots]
    records = []
    for root in roots:
        files = list(csv_files(root)) if root.exists() else []
        for path in sorted(files):
            records.append(inspect_file(path, args.sample_rows))

    manifest = {
        "dataset_roots": [str(x) for x in roots],
        "csv_count": len(records),
        "records": records,
        "quality_rules": {
            "reject_invalid_ohlc": True,
            "reject_invalid_timestamps": True,
            "deduplicate_timestamps": True,
            "do_not_impute_price_bars": True,
            "zero_volume_is_flagged_not_silently_replaced": True,
        },
    }
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"csv_count": len(records), "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
