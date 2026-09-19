from __future__ import annotations
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def phase_00() -> None:
    print("Phase 00 is validation-only; preflight is the execution gate.")

def phase_01() -> None:
    required = ["^NSEI", "NIFTYBEES.NS"]
    out = ROOT / "data" / "README.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "# Data Phase
\n"
        "Configured primary signal series: NIFTY 50 (^NSEI)\n"
        "Configured tradable proxy: NIFTYBEES.NS\n"
        "Data acquisition must reuse valid caches and write manifests/checksums.\n"
        "The execution implementation is intentionally fail-closed until the data-source/cache adapter is validated.\n",
        encoding="utf-8",
    )
    print("Phase 01 data scaffold validated:", ", ".join(required))

def phase_02() -> None:
    marker = ROOT / "data" / "processed"
    marker.mkdir(parents=True, exist_ok=True)
    (marker / "FEATURES_PENDING.md").write_text(
        "# Features pending execution\n\n"
        "CPR definitions are frozen in research/PROTOCOL.md. "
        "Feature generation must pass leakage checks before backtesting.\n",
        encoding="utf-8",
    )
    print("Phase 02 feature scaffold validated.")

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", required=True)
    a = p.parse_args()
    funcs = {
        "phase-00-protocol": phase_00,
        "phase-01-data": phase_01,
        "phase-02-features": phase_02,
    }
    if a.phase not in funcs:
        raise SystemExit(
            f"{a.phase} is intentionally fail-closed: implementation must be added before execution."
        )
    funcs[a.phase]()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
