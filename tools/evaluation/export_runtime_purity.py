"""Write a versioned runtime-purity snapshot for Phase 3 closeout evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from .bootstrap import V7_ROOT
from .runtime_purity import audit_runtime_purity


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "-C", str(V7_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-pure", action="store_true")
    args = parser.parse_args(argv)

    purity = audit_runtime_purity().as_dict()
    payload = {
        "schema": "stage_vla_v7.phase3_runtime_purity.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "runtime_purity": purity,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if args.require_pure and not purity["verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
