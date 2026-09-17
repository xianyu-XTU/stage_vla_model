"""Write a versioned runtime-purity snapshot for Phase 3 closeout evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from .bootstrap import V7_ROOT
from .provenance import capture_source_snapshot
from .runtime_purity import audit_runtime_purity, install_v5_import_blocker


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-pure", action="store_true")
    args = parser.parse_args(argv)

    source_snapshot = capture_source_snapshot(V7_ROOT)
    install_v5_import_blocker()
    purity = audit_runtime_purity().as_dict()
    payload = {
        "schema": "stage_vla_v7.phase3_runtime_purity.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        **source_snapshot.as_dict(),
        "git_commit": source_snapshot.commit,
        "git_worktree_dirty": not source_snapshot.worktree_clean,
        "git_status": list(source_snapshot.git_status),
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
