"""Example composition of immutable V5 policies with the V7 action boundary."""

from pathlib import Path

from stage_vla_v7.action import ActionRouter, ActionService, build_v5_cube_bundle


V5_BUNDLE = Path("E:/stage_vla_v5/outputs/v5_generalized_cube_bc_v1")


def build_action_service(device: str = "cuda:0") -> ActionService:
    bundle = build_v5_cube_bundle(V5_BUNDLE, device=device)
    return ActionService(ActionRouter({"v5_cube": bundle}))


if __name__ == "__main__":
    service = build_action_service(device="cpu")
    print(service.router.bundles["v5_cube"].descriptor)
