# PHASE-2 REFACTOR AUDIT

Audit baseline: commit `b2e14c95557fff88839856865918dcbc7efd4be2`.
The findings below come from source/import/call searches and a physical run,
not from README claims.

## Regression baseline

- Isaac Lab root: `E:\work\IsaacLab`
- Unit/boundary/integration suite: `78 passed`
- Eight locked TorchScript policies: hashes valid, action dimension 5, maximum
  legacy-vs-V7 projected-action error `0.0`
- Physical seed: `2012` using the frozen layout whose filename records seed
  `61081`
- Physical task: `1/1` passed, no mid-episode reset
- Prepared Skill tokens: `8/8`; exact state-preserving handoffs: `7/7`
- V7 Vision calls: `729`; invalid frames: `0`
- Reference Skill calls: `0`; reference recovery: `0`
- `v7_chain.verified`: `true`

The full pre-change run is ignored under
`outputs/phase2_baseline_seed61081.json`; the locked policy comparison is
`evidence/phase2_checkpoint_compatibility_baseline.json`.

## Current Simulation ownership

`src/stage_vla_v7/simulation` currently owns dependency-free descriptions and
registries for environments, scenes, Franka, cubes, RGB/depth sensors,
materials, contacts, limits, seeded positions, and the Isaac-facing adapters.
The `isaac_lab` package owns camera/observation/action conversion,
`PipelineActionSource`, and a callback runtime.

The concrete physical environment is not yet V7-owned.  The function
`simulation.isaac_lab.runtime.make_legacy_v5_known_size_grasp_env` imports
`tools.stageppo_known_size_grasp_env.make_known_size_grasp_env` from the
vendored V5 tree and returns that environment unchanged.

## Active V5 dependency map

Direct physical evaluator roots are:

| V5 path | Live responsibility | Phase-2 disposition |
|---|---|---|
| `tools/stageppo_known_size_grasp_env.py` | Isaac task configuration and environment construction | MIGRATE to `simulation/isaac_lab/env_factory.py`; retain a compatibility re-export |
| `tools/train_known_size_grasp.py` | size/object metadata loading | ADAPT until the configuration contract is migrated |
| `stage_vla/action_output.py` | retained vector action-output binding | ADAPT |
| `stage_vla/data/v4_2_runtime.py` | batched RGB/depth readers | ADAPT through the camera boundary |
| `stage_vla/envs/state_readers.py` | tensor/state conversion | ADAPT |
| `stage_vla/rl/reach_policy.py` | frozen REACH observation/reference helpers | KEEP/ADAPT; checkpoint semantics are frozen |
| `stage_vla/rl/v5_skill_contracts.py` | optional diagnostic reference action | KEEP; fail-closed V7 runs do not select it |
| `stage_vla/rl/known_size_grasp_vecenv.py` | physical vector wrapper and retained state | ADAPT |
| `stage_vla/rl/known_size_grasp.py` | frozen physical grasp configuration/check | ADAPT |
| `stage_vla/rl/skill_action_safety.py` | frozen physical geometry projection helpers | ADAPT |
| `stage_vla/rl/skill_demonstrations.py` | optional data collection | KEEP |
| `stage_vla/stages/grasp_geometry.py` | frozen grasp geometry | ADAPT |
| `stage_vla/stages/object_motion.py` | frozen motion limits from config | ADAPT |
| `stage_vla/vision.py` | compact calibrated RGB-D detector | ADAPT behind V7 VisionService |
| `stage_vla/evaluation/four_cube_stack.py` | frozen layout manifest reader | MIGRATE pure manifest loading |

Environment creation also transitively uses V5 contact-sensor installation,
known-size gripper action, fixed-pose events, physical profiles and object
physics.  These remain ADAPT until equivalent V7-owned runtime code has both
unit and physical regression evidence.  No V5 file qualifies for DELETE at
the start of this phase.

## Evaluator responsibility map

`tools/eval_v7_multicube_chain.py` is 1,929 lines and currently performs all of
the following in one `main` function:

1. CLI declaration, parsing and validation.
2. V5/V7 import-path bootstrapping.
3. language command and task-pair construction.
4. artifact-lock and checkpoint validation.
5. layout loading and random XY generation.
6. Isaac application and physical environment creation.
7. Franka/cube/contact/gripper wrapper configuration.
8. Vision camera calibration and RGB-D reads.
9. V7 pipeline/action-source construction and all Skill loops.
10. handoff, tracing, demonstration and DAgger collection.
11. per-Skill and final-stack success evaluation.
12. observer frame capture, overlay drawing and MP4 encoding.
13. result assembly, JSON writing and application shutdown.

## Camera and recording dependency map

Both cameras are currently one conditional camera.  The evaluator chooses
either `v7_multicube_camera` for Vision or `v5_chain_video_camera` for video,
then passes one name, one resolution and one data-type tuple to the V5 factory.
The factory creates that single `CameraCfg` with `setattr(cfg.scene, name, ...)`.

Consequently `--use_vision` and `--video` cannot describe two sensors.  The
explicit parser error is only the visible guard; the root cause is the
single-camera factory contract and shared resolution/data-type branch.  It is
not evidence of a fundamental Isaac Lab rendering limitation.

- Vision camera creation: evaluator camera branch plus V5 factory `CameraCfg`.
- Vision reads: V5 `read_rgb_u8_batch` and `read_depth_m_batch`, then V7
  `VisionService`.
- Video camera creation: the same V5 factory branch under a different name.
- Video reads: evaluator closure `capture_video_frame`.
- Overlay/MP4: evaluator lines 1724-1752 using Pillow and imageio/ffmpeg.

## Physical construction ownership

- Environment entry: V7 `make_legacy_v5_known_size_grasp_env` -> V5
  `make_known_size_grasp_env` -> Isaac Lab `gym.make`.
- Franka: Isaac Lab's registered Franka stack task; V5 factory adjusts action
  scale, hand effort and contact sensors.  V7 currently owns only a descriptor.
- Cubes: Isaac Lab task cube configs; V5 factory adjusts geometry, color,
  mass/friction, extra instances and reset events.  V7 owns descriptors only.
- Randomization: evaluator random XY/layout logic plus V5/Isaac reset events.
- Physics/contact: V5 factory, `stage_vla.envs`, and retained vector wrapper;
  V7 physics modules are descriptive.
- Success evaluation: V7 `action/evaluation` adapter delegates frozen tensor
  predicates to V5; the evaluator controls stage/final aggregation.

This audit fixes the Phase-2 migration boundary: move pure utilities,
recording, camera declarations and environment construction first; keep the
frozen control/vector/state stack behind explicit adapters until physical
equivalence is re-proven.

## Post-Phase-2 disposition

The audit actions were applied without changing the eight Skills, five-value
`RobotAction`, observation ordering, success predicates, or checkpoints:

- Concrete environment construction moved to
  `simulation/isaac_lab/env_factory.py`; the V5 file is now a compatibility
  re-export.
- Layout loading and seeded pair sampling moved to
  `simulation/randomization/object_pose.py`.
- Recording moved to `simulation/recording`, with streaming encoding and
  explicit strict/best-effort behavior.
- Vision and Observer cameras now have independent IDs, outputs, bindings, and
  consumers. The Observer path cannot enter `VisionService`.
- The original evaluator path is a 17-line entry; episode execution and JSON
  persistence live in `tools/evaluation`.
- Direct V5 imports remain for 19 unique module paths, so V5 runtime migration
  is PARTIAL. See `docs/VENDOR_CLASSIFICATION.md` for the exact list.

Post-change evidence is 96 passing tests, 8/8 checkpoint comparisons with
maximum error `0.0`, a passing post-factory RGB-D physical run, and a passing
headless RGB-D + Observer MP4 run with 730 Vision calls, 768 video frames,
8/8 Skills, 7/7 state-exact handoffs, no reference/recovery calls, and
`v7_chain.verified=true`.
