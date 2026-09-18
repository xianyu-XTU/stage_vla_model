"""Generate one frozen Phase 4 generalization layout manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from .manifest import generate_layout_manifest, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42024)
    parser.add_argument("--layout-count", type=int, default=20)
    args = parser.parse_args()
    output = write_manifest(
        args.output,
        generate_layout_manifest(seed=args.seed, layout_count=args.layout_count),
    )
    print(output)


if __name__ == "__main__":
    main()
