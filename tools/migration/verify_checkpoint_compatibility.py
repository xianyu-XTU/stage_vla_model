"""Compare frozen V5 TorchScript outputs with the refactored V7 action path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from stage_vla_v7.action import (  # noqa: E402
    ActionRouter,
    ActionService,
    SafetyProjector,
    build_v5_cube_bundle,
)
from stage_vla_v7.interfaces import (  # noqa: E402
    ActionRequest,
    ObjectProfile,
    RobotAction,
    SKILL_SEQUENCE,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path("config/artifacts.lock.json"),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    import torch

    lock = json.loads(args.lock.resolve().read_text(encoding="utf-8"))
    records = lock["artifacts"]
    expected_hashes = {
        skill: records[skill.value.lower()]["sha256"] for skill in SKILL_SEQUENCE
    }
    bundle = build_v5_cube_bundle(
        args.artifact_root,
        device=args.device,
        expected_hashes=expected_hashes,
    )
    service = ActionService(ActionRouter({"v5": bundle}))
    profile = ObjectProfile(
        "compatibility_cube",
        "box",
        (0.04, 0.04, 0.04),
        0.05,
        stackable=True,
    )
    projector = SafetyProjector()
    comparisons: dict[str, object] = {}
    all_compatible = True

    for skill in SKILL_SEQUENCE:
        record = records[skill.value.lower()]
        dimension = int(record["observation_dim"])
        values = torch.linspace(-0.25, 0.25, dimension, dtype=torch.float32)
        checkpoint = (args.artifact_root / record["logical_path"]).resolve()
        legacy_model = torch.jit.load(str(checkpoint), map_location=args.device).eval()
        with torch.inference_mode():
            raw = legacy_model(values.to(args.device).unsqueeze(0))
        legacy_action = RobotAction.from_values(
            torch.as_tensor(raw).detach().cpu().reshape(-1).tolist()
        )
        expected = projector.project(skill, legacy_action)
        actual = service.act(
            ActionRequest(
                skill,
                tuple(values.tolist()),
                profile,
                profile,
            )
        ).action
        max_abs_error = max(abs(left - right) for left, right in zip(expected.values, actual.values))
        compatible = max_abs_error <= 1e-7
        all_compatible &= compatible
        comparisons[skill.value] = {
            "checkpoint": str(checkpoint),
            "sha256": record["sha256"],
            "observation_dim": dimension,
            "action_dim": len(actual.values),
            "legacy_projected_action": expected.values,
            "refactored_action": actual.values,
            "max_abs_error": max_abs_error,
            "compatible": compatible,
        }

    report = {
        "schema": "stage_vla_v7.checkpoint_compatibility.v1",
        "artifact_bundle": lock["bundle"],
        "comparison": "direct_v5_torchscript_plus_frozen_safety_vs_refactored_action_service",
        "tolerance": 1e-7,
        "all_compatible": all_compatible,
        "skills": comparisons,
    }
    text = json.dumps(report, indent=2)
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not all_compatible:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
