from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

REQUIRED = [
    "RESEARCH_INSTRUCTIONS.md",
    "research/PROTOCOL.md",
    "research/STATUS.json",
    "research/RUN_LOG.jsonl",
    "research/ERROR_LOG.md",
    "research/CONVERSATION_LOG.md",
    "research/ANALYSIS_PLAN.md",
    "research/DATA_CONTRACT.md",
    "research/COST_MODEL.yaml",
]

PHASES = [
    "phase-00-protocol",
    "phase-01-data",
    "phase-02-features",
    "phase-03-backtest",
    "phase-04-robustness",
    "phase-05-statistics",
    "phase-06-manuscript",
    "phase-07-audit",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument("--branch", required=True)
    args = parser.parse_args()

    errors: list[str] = []

    if args.branch != f"research/{args.phase}":
        errors.append(
            f"Wrong branch for {args.phase}: expected research/{args.phase}, got {args.branch}"
        )

    for rel in REQUIRED:
        if not (ROOT / rel).exists():
            errors.append(f"Missing required repository control file: {rel}")

    status_path = ROOT / "research/STATUS.json"
    status = {}
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text())
        except Exception as exc:
            errors.append(f"STATUS.json is invalid JSON: {exc}")

    protocol = (ROOT / "research/PROTOCOL.md").read_text() if (ROOT / "research/PROTOCOL.md").exists() else ""
    protocol_version = status.get("protocol_version")
    if not protocol_version or f"Protocol version: {protocol_version}" not in protocol:
        errors.append("Protocol version mismatch between STATUS.json and PROTOCOL.md")

    if args.phase not in status.get("phases", {}):
        errors.append(f"Phase is absent from STATUS.json: {args.phase}")

    # Cost inputs are only a hard prerequisite from the backtest phase onward.
    if args.phase in PHASES[3:]:
        cost_text = (ROOT / "research/COST_MODEL.yaml").read_text()
        if 'validation:\n  status: "READY"' not in cost_text:
            errors.append("Cost model is not validated for a cost-sensitive phase")

    if errors:
        print("PREFLIGHT_FAILED")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(2)

    print("PREFLIGHT_OK")
    print(f"phase={args.phase}")
    print(f"branch={args.branch}")
    print(f"protocol_version={protocol_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
