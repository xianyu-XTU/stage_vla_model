"""Train the oracle known-size GRASP/LIFT pressure residual with StagePPO.

This is a focused skill experiment.  It keeps the v5 five-dimensional policy
interface, but the default ``--arm_locked`` mode trains only the gripper
pressure residual.  A meaningful quality run should provide real pregrasp
entries via ``--entry_dir``; running without entries is limited to an interface
smoke from the task's reset distribution.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_size(path: Path):
    from stage_vla.rl.known_size_grasp import KnownSizeGraspConfig

    payload = json.loads(path.read_text(encoding="utf-8"))
    obj = payload["object"]
    grip = payload["gripper"]
    return KnownSizeGraspConfig(
        width_m=float(obj["width_m"]),
        depth_m=float(obj["depth_m"]),
        height_m=float(obj["height_m"]),
        mass_kg=float(obj["mass_kg"]),
        **{key: float(grip[key]) for key in (
            "jaw_clearance_m", "max_compression_m", "friction_coefficient",
            "safety_factor", "min_force_n", "max_force_n",
            "pressure_tolerance_n", "force_balance_tolerance_n",
            "residual_force_range_n",
            "residual_action_penalty",
            "lift_acceleration_mps2", "joint_min_m", "joint_max_m",
        )},
    )


def _load_object_metadata(path: Path) -> tuple[str, tuple[int, int, int]]:
    obj = json.loads(path.read_text(encoding="utf-8"))["object"]
    return str(obj.get("geometry", "box")), tuple(int(value) for value in obj.get("color_rgb", (220, 40, 40)))


def _load_grasp_profile(path: Path, geometry: str):
    from stage_vla.stages.grasp_geometry import profile_from_config

    return profile_from_config(path, geometry=geometry)


def _load_physical_domain(path: Path):
    from stage_vla.rl.object_physics import PhysicalDomainConfig

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "stage_vla_v5.physical_domain.v1":
        raise ValueError("unsupported physical-domain schema")
    return PhysicalDomainConfig.from_mapping(payload["domain"])


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/v5_modular_box_grasp.json")
    parser.add_argument(
        "--physical_domain_config",
        type=Path,
        default=ROOT / "config/v5_modular_box_physics.json",
        help="size/mass ranges for the shared geometry-bundle policy",
    )
    parser.add_argument("--entry_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--skill",
        choices=(
            "GRASP", "LIFT", "TRANSPORT", "ALIGN", "DESCEND",
            "RELEASE_STABILIZE", "RETREAT",
        ),
        default="GRASP",
    )
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--episode_steps", type=int, default=80)
    parser.add_argument("--stable_steps", type=int, default=3)
    parser.add_argument("--max_compression_m", type=float, default=None,
                        help="optional skill-specific GeometryAdapter compression travel")
    parser.add_argument(
        "--snapshot_assignment", choices=("auto", "random", "sequential"), default="auto",
        help="auto binds physical snapshots sequentially; legacy snapshots remain random",
    )
    parser.add_argument("--seed", type=int, default=7105)
    parser.add_argument("--arm_locked", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--lift_vertical_only", action=argparse.BooleanOptionalAction, default=False,
        help="for LIFT, lock x/y/yaw and learn only vertical motion plus grip",
    )
    parser.add_argument("--lift_prepress_steps", type=int, default=0,
                        help="for LIFT, hold position and close before motion")
    parser.add_argument("--lift_prepress_grip", type=float, default=-1.0,
                        help="normalized grip command during LIFT prepress")
    parser.add_argument("--entrance_prepress_steps", type=int, default=0,
                        help="for carrying skills after LIFT, hold and close before motion")
    parser.add_argument("--entrance_prepress_grip", type=float, default=-1.0,
                        help="normalized grip command during carrying-skill prepress")
    parser.add_argument("--entrance_gravity_compensation",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="cancel payload gravity only during carrying-skill prepress")
    parser.add_argument(
        "--entrance_gravity_compensation_until_contact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keep carrying-skill gravity compensation until physical grasp and pressure recover",
    )
    parser.add_argument("--entrance_contact_refresh_steps", type=int, default=0,
                        help="initial steps that hold restored physical finger positions")
    parser.add_argument("--entrance_servo_target_blend", type=float, default=1.0,
                        help="blend restored joints (0) with saved servo target (1)")
    parser.add_argument(
        "--entrance_servo_target_max_delta_m", type=float, default=None,
        help="maximum trusted saved-servo delta; stale targets use bounded geometric recovery",
    )
    parser.add_argument("--entrance_arm_warmup_steps", type=int, default=0,
                        help="gradually hand arm control to ALIGN/DESCEND after entrance prepress")
    parser.add_argument("--entrance_arm_warmup_scale", type=float, default=1.0,
                        help="initial arm-action scale during entrance warmup")
    parser.add_argument(
        "--carry_leveling_max_angle_rad", type=float, default=0.0,
        help="bounded roll/pitch correction that keeps the parallel-jaw axis level",
    )
    parser.add_argument("--transport_planar_only", action=argparse.BooleanOptionalAction, default=True,
                        help="for TRANSPORT, lock vertical and yaw channels")
    parser.add_argument("--transport_xy_m", type=float, default=0.045,
                        help="TRANSPORT XY success radius")
    parser.add_argument("--transport_speed_mps", type=float, default=0.05,
                        help="maximum payload linear speed at TRANSPORT handoff")
    parser.add_argument("--transport_angular_speed_radps", type=float, default=1.0,
                        help="maximum payload angular speed at TRANSPORT handoff")
    parser.add_argument("--transport_reward_variant",
                        choices=("absolute_v1", "progress_direction_v2", "progress_safety_v3",
                                 "progress_height_hold_v4", "progress_settle_v5"),
                        default="progress_direction_v2",
                        help="TRANSPORT reward shaping variant")
    parser.add_argument("--transport_translation_limit_m", type=float, default=0.005,
                        help="TRANSPORT XY translation limit per control step")
    parser.add_argument("--align_xy_m", type=float, default=0.010)
    parser.add_argument("--align_inner_xy_m", type=float, default=0.0075)
    parser.add_argument("--align_height_target_m", type=float, default=0.0618)
    parser.add_argument("--align_height_tolerance_m", type=float, default=0.015)
    parser.add_argument("--align_speed_mps", type=float, default=0.05)
    parser.add_argument("--align_translation_limit_m", type=float, default=0.005)
    parser.add_argument(
        "--init_policy", type=Path, default=None,
        help="optional deterministic BC state-dict or StagePPO checkpoint used to initialize the actor",
    )
    parser.add_argument(
        "--init_std", type=float, default=0.10,
        help="initial Gaussian actor std when --init_policy is used",
    )
    parser.add_argument(
        "--actor_std_override", type=float, default=None,
        help="optional actor exploration std for either fresh or initialized training",
    )
    parser.add_argument(
        "--learning_rate_override", type=float, default=None,
        help="optional PPO learning-rate override for conservative fine-tuning",
    )
    parser.add_argument(
        "--entropy_coef_override", type=float, default=None,
        help="optional PPO entropy coefficient override",
    )
    parser.add_argument("--smoke", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if min(args.num_envs, args.iterations, args.episode_steps, args.stable_steps) < 1:
        parser.error("counts must be positive")
    if min(args.lift_prepress_steps, args.entrance_prepress_steps,
           args.entrance_contact_refresh_steps) < 0:
        parser.error("prepress step counts must be non-negative")
    if args.entrance_arm_warmup_steps < 0:
        parser.error("entrance_arm_warmup_steps must be non-negative")
    if not 0.0 < args.entrance_arm_warmup_scale <= 1.0:
        parser.error("entrance_arm_warmup_scale must lie in (0,1]")
    if not -1.0 <= args.entrance_prepress_grip <= 1.0:
        parser.error("entrance_prepress_grip must be in [-1,1]")
    if not 0.0 <= args.entrance_servo_target_blend <= 1.0:
        parser.error("entrance_servo_target_blend must lie in [0,1]")
    if (
        args.entrance_servo_target_max_delta_m is not None
        and args.entrance_servo_target_max_delta_m <= 0
    ):
        parser.error("entrance_servo_target_max_delta_m must be positive")
    if not 0.0 <= args.carry_leveling_max_angle_rad <= 1.5707963267948966:
        parser.error("carry_leveling_max_angle_rad must lie in [0,pi/2]")
    if args.max_compression_m is not None and args.max_compression_m <= 0:
        parser.error("max_compression_m must be positive")
    if min(args.transport_speed_mps, args.transport_angular_speed_radps) <= 0:
        parser.error("TRANSPORT speed thresholds must be positive")
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {out}")
    size = _load_size(args.config.resolve())
    if args.max_compression_m is not None:
        size = replace(size, max_compression_m=float(args.max_compression_m))
    object_geometry, object_color_rgb = _load_object_metadata(args.config.resolve())
    grasp_profile = _load_grasp_profile(args.config.resolve(), object_geometry)
    from stage_vla.stages.grasp_geometry import local_width_ratio
    size = replace(
        size,
        grasp_width_m=size.width_m * float(local_width_ratio(
            grasp_profile.geometry,
            grasp_profile.height_ratio,
            explicit_width_ratio=grasp_profile.width_ratio,
        ).item()),
    )
    entry_paths = sorted(args.entry_dir.glob("*.pt")) if args.entry_dir else []
    count = min(args.num_envs, 8) if args.smoke else args.num_envs
    iterations = min(args.iterations, 3) if args.smoke else args.iterations
    physical_domain = _load_physical_domain(args.physical_domain_config.resolve())
    from stage_vla.rl.object_physics import (
        PhysicalObjectBatch, snapshot_has_physical_context,
    )
    import torch

    entry_payloads = [
        torch.load(path, map_location="cpu", weights_only=False) for path in entry_paths
    ]
    context_flags = [snapshot_has_physical_context(payload) for payload in entry_payloads]
    if any(context_flags) and not all(context_flags):
        raise ValueError("entry_dir cannot mix legacy and physical-context snapshots")
    if context_flags and all(context_flags):
        if args.snapshot_assignment == "random":
            raise ValueError(
                "random assignment would mismatch snapshot poses and physical profiles"
            )
        snapshot_assignment = "sequential"
        physical_batch = PhysicalObjectBatch.from_snapshot_payloads(
            entry_payloads, count
        )
        if physical_batch.geometry_bundle != physical_domain.geometry_bundle:
            raise ValueError("entry and configured physical-domain geometry bundles differ")
        physical_profile_source = "entry_snapshots"
        physical_profile_seed = None
    else:
        snapshot_assignment = (
            "random" if args.snapshot_assignment == "auto" else args.snapshot_assignment
        )
        physical_batch = physical_domain.sample(count, seed=args.seed + 10_000)
        physical_profile_source = "domain_sample"
        physical_profile_seed = args.seed + 10_000
    out.mkdir(parents=True, exist_ok=True)

    app = AppLauncher(args).app
    raw = wrapped = None
    try:
        from rsl_rl.runners import OnPolicyRunner
        from stage_vla.rl.known_size_grasp_vecenv import KnownSizeGraspVecEnv
        from stage_vla.rl.rsl_rl_runtime import build_m9a_runner_cfg
        from tools.stageppo_known_size_grasp_env import make_known_size_grasp_env

        torch.manual_seed(args.seed)
        raw = make_known_size_grasp_env(
            device=args.device,
            num_envs=count,
            seed=args.seed,
            log_dir=out,
            known_size=size,
            effort_limit=size.max_force_n,
            object_geometry=object_geometry,
            object_color_rgb=object_color_rgb,
            physical_batch=physical_batch,
        )
        wrapped = KnownSizeGraspVecEnv(
            raw,
            known_size=size,
            skill=args.skill,
            snapshots=entry_paths,
            episode_steps=args.episode_steps,
            stable_steps=args.stable_steps,
            arm_locked=args.arm_locked,
            lift_vertical_only=args.lift_vertical_only,
            lift_prepress_steps=args.lift_prepress_steps,
            lift_prepress_grip=args.lift_prepress_grip,
            entrance_prepress_steps=args.entrance_prepress_steps,
            entrance_prepress_grip=args.entrance_prepress_grip,
            entrance_gravity_compensation=args.entrance_gravity_compensation,
            entrance_gravity_compensation_until_contact=(
                args.entrance_gravity_compensation_until_contact
            ),
            entrance_contact_refresh_steps=args.entrance_contact_refresh_steps,
            entrance_servo_target_blend=args.entrance_servo_target_blend,
            entrance_servo_target_max_delta_m=(
                args.entrance_servo_target_max_delta_m
            ),
            entrance_arm_warmup_steps=args.entrance_arm_warmup_steps,
            entrance_arm_warmup_scale=args.entrance_arm_warmup_scale,
            carry_leveling_max_angle_rad=args.carry_leveling_max_angle_rad,
            transport_planar_only=args.transport_planar_only,
            transport_xy_m=args.transport_xy_m,
            transport_speed_mps=args.transport_speed_mps,
            transport_angular_speed_radps=args.transport_angular_speed_radps,
            transport_reward_variant=args.transport_reward_variant,
            transport_translation_limit_m=args.transport_translation_limit_m,
            grasp_profile=grasp_profile,
            align_xy_m=args.align_xy_m,
            align_inner_xy_m=args.align_inner_xy_m,
            align_height_target_m=args.align_height_target_m,
            align_height_tolerance_m=args.align_height_tolerance_m,
            align_speed_mps=args.align_speed_mps,
            align_translation_limit_m=args.align_translation_limit_m,
            physical_batch=physical_batch,
            include_physical_context=True,
            seed=args.seed + 1,
            snapshot_assignment=snapshot_assignment,
        )
        wrapped.reset()
        runner_cfg, _ = build_m9a_runner_cfg(
            config_file=ROOT / "config/default.yaml",
            seed=args.seed,
            max_iterations=iterations,
        )
        runner_cfg.device = args.device
        runner_cfg.experiment_name = "stage_vla_v5_known_size_grasp"
        runner_cfg.run_name = out.name
        runner_cfg.algorithm.gamma = 0.98
        if args.learning_rate_override is not None:
            if args.learning_rate_override <= 0:
                raise ValueError("learning_rate_override must be positive")
            runner_cfg.algorithm.learning_rate = float(args.learning_rate_override)
        else:
            runner_cfg.algorithm.learning_rate = 1e-4
        if args.entropy_coef_override is not None:
            if args.entropy_coef_override < 0:
                raise ValueError("entropy_coef_override must be non-negative")
            runner_cfg.algorithm.entropy_coef = float(args.entropy_coef_override)
        config_dict = runner_cfg.to_dict()
        (out / "runner_config.json").write_text(json.dumps(config_dict, indent=2) + "\n", encoding="utf-8")
        runner = OnPolicyRunner(wrapped, config_dict, log_dir=str(out), device=args.device)
        fixed_bc_output_head_init = False

        # The supervised transport diagnostic has the same 55->256->128->64->5
        # MLP as the PPO actor, but its last layer is followed by Tanh.  Copying
        # the linear weights gives PPO the same local action geometry; the env's
        # documented action clip keeps the unbounded PPO mean inside the valid
        # command range during this conservative warm start.  A full StagePPO
        # checkpoint can also be loaded directly when its actor shape matches.
        if args.init_policy is not None:
            init_path = args.init_policy.resolve()
            if not init_path.is_file():
                raise FileNotFoundError(init_path)
            payload = torch.load(str(init_path), map_location=args.device, weights_only=False)
            if isinstance(payload, dict) and "actor_state_dict" in payload:
                actor_state = payload["actor_state_dict"]
                runner.alg.actor.load_state_dict(actor_state, strict=True)
            elif isinstance(payload, dict) and "0.weight" in payload:
                actor_state = runner.alg.actor.state_dict()
                for layer_idx in (0, 2, 4, 6):
                    actor_state[f"mlp.{layer_idx}.weight"].copy_(payload[f"{layer_idx}.weight"])
                    actor_state[f"mlp.{layer_idx}.bias"].copy_(payload[f"{layer_idx}.bias"])
                runner.alg.actor.load_state_dict(actor_state, strict=True)
            elif isinstance(payload, dict) and "net.0.weight" in payload:
                # Context-aware BC actors use a named ``net`` Sequential while
                # the StagePPO actor stores the same MLP as ``mlp``.  Preserve
                # the learned geometry when warming up a modular skill.
                actor_state = runner.alg.actor.state_dict()
                for layer_idx in (0, 2, 4, 6):
                    actor_state[f"mlp.{layer_idx}.weight"].copy_(
                        payload[f"net.{layer_idx}.weight"]
                    )
                    actor_state[f"mlp.{layer_idx}.bias"].copy_(
                        payload[f"net.{layer_idx}.bias"]
                    )
                if args.skill == "DESCEND":
                    # DescendContextPolicy fixes yaw/grip in forward(), so its
                    # final two stored rows never receive gradients.  Do not
                    # import those random rows into PPO's fully learned head.
                    actor_state["mlp.6.weight"][3:].zero_()
                    actor_state["mlp.6.bias"][3] = 0.0
                    actor_state["mlp.6.bias"][4] = -0.05
                    fixed_bc_output_head_init = True
                runner.alg.actor.load_state_dict(actor_state, strict=True)
            else:
                raise ValueError(
                    f"unsupported init policy format: {init_path}; expected BC model.pt or StagePPO model_final.pt"
                )
            if args.init_std <= 0:
                raise ValueError("init_std must be positive")
            with torch.no_grad():
                runner.alg.actor.distribution.std_param.fill_(float(args.init_std))
        if args.actor_std_override is not None:
            if args.actor_std_override <= 0:
                raise ValueError("actor_std_override must be positive")
            with torch.no_grad():
                runner.alg.actor.distribution.std_param.fill_(
                    float(args.actor_std_override)
                )
        torch.save({**runner.alg.save(), "iter": 0, "infos": None}, out / "initial_policy.pt")
        runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=False)
        runner.save(str(out / "model_final.pt"))
        runner.export_policy_to_jit(str(out), filename="policy.ts")
        result = {
            "status": "complete",
            "skill": args.skill,
            "config": str(args.config.resolve()),
            "known_size": asdict(size),
            "physical_domain": asdict(physical_domain),
            "physical_domain_config": str(args.physical_domain_config.resolve()),
            "physical_profile_source": physical_profile_source,
            "physical_profile_seed": physical_profile_seed,
            "physical_profile": {
                "object_size_min_m": physical_batch.object_size_m.amin(dim=0).tolist(),
                "object_size_max_m": physical_batch.object_size_m.amax(dim=0).tolist(),
                "support_size_min_m": physical_batch.support_size_m.amin(dim=0).tolist(),
                "support_size_max_m": physical_batch.support_size_m.amax(dim=0).tolist(),
                "object_mass_min_kg": float(physical_batch.object_mass_kg.min()),
                "object_mass_max_kg": float(physical_batch.object_mass_kg.max()),
                "support_mass_min_kg": float(physical_batch.support_mass_kg.min()),
                "support_mass_max_kg": float(physical_batch.support_mass_kg.max()),
            },
            "object_geometry": object_geometry,
            "grasp_profile": {
                "geometry": grasp_profile.geometry,
                "height_ratio": grasp_profile.height_ratio,
                "width_ratio": grasp_profile.width_ratio,
            },
            "entry_paths": [str(path.resolve()) for path in entry_paths],
            "default_reset_distribution": not bool(entry_paths),
            "requested_snapshot_assignment": args.snapshot_assignment,
            "snapshot_assignment": snapshot_assignment,
            "arm_locked": args.arm_locked,
            "num_envs": count,
            "iterations": iterations,
            "episode_steps": args.episode_steps,
            "stable_steps": args.stable_steps,
            "observation_dim": wrapped.observation_dim,
            "physical_context_version": physical_batch.context_version,
            "action_dim": 5,
            "statistics": wrapped.statistics(),
            "quality_claim": False if not entry_paths else "focused oracle skill only",
            "init_policy": str(args.init_policy.resolve()) if args.init_policy is not None else None,
            "init_std": float(args.init_std) if args.init_policy is not None else None,
            "fixed_bc_output_head_init": fixed_bc_output_head_init,
            "actor_std_override": args.actor_std_override,
            "learning_rate": float(runner_cfg.algorithm.learning_rate),
            "entropy_coef": float(runner_cfg.algorithm.entropy_coef),
            "entrance_arm_warmup_steps": args.entrance_arm_warmup_steps,
            "entrance_arm_warmup_scale": args.entrance_arm_warmup_scale,
            "entrance_servo_target_max_delta_m": (
                args.entrance_servo_target_max_delta_m
            ),
            "carry_leveling_max_angle_rad": args.carry_leveling_max_angle_rad,
            "entrance_gravity_compensation_until_contact": (
                args.entrance_gravity_compensation_until_contact
            ),
        }
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
    finally:
        if wrapped is not None:
            wrapped.close()
        elif raw is not None:
            raw.close()
        app.close()


if __name__ == "__main__":
    main()
