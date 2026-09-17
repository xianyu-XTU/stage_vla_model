# Phase 3 V5 Runtime Migration

Closeout baseline: `main` at
`94006de24ecdcb4725394f24edf36d8d9b4c23c1`.
Clean-HEAD Recovery source:
`0b4060423fa226a4bb428a40b6b4961706359c12`.
Final audit date: 2026-09-17. The inventory below was regenerated from the
current Python imports, import state, and physical evaluator call graph.

## Result

Stage VLA V7 Phase 3 runtime-isolation implementation is complete for the
locked cube-policy physical path. Recovery verification passed from a clean
source commit:

- formal `src/stage_vla_v7` V5 Python imports: **0**;
- formal `tools/evaluation` V5 Python imports: **0**;
- formal bootstrap V5 source paths: **0**;
- loaded `stage_vla` / `stage_vla.*` modules in the physical smoke: **0**;
- exposed V5 package paths in the physical smoke: **0**;
- active V5 import blocker in the physical smoke: **enabled**;
- canonical `stage_vla_v7.action.evaluation` V5 adapters: **0**;
- evaluator calls to private environment methods: **0**;
- evaluator mutations of protected environment state: **0**;
- `vendor/stage_vla_v5` remains unchanged for provenance, regression oracles,
  and historical training;
- the eight V5-trained checkpoints remain external artifacts and were not
  retrained or rewritten;
- the clean-HEAD V5-isolated strict Vision + Video physical smoke passed with
  runtime purity and the V7 chain both verified.

`tools/evaluation/bootstrap.py` now exposes only `repo/src`. It does not define
`V5_ROOT`, inspect `STAGE_VLA_V5_ROOT`, or add `vendor/stage_vla_v5` to
`sys.path`. Parity and migration tools obtain V5 only through explicit,
regression-only setup outside the formal evaluator.

## Dependency count

These milestone counts track the 19 direct V5 Python module paths in the Phase
2 inventory. Temporary helper imports discovered while opening the old VecEnv
were migrated and are listed separately below.

| Milestone | Remaining original runtime modules | Result |
|---|---:|---|
| Phase 2 baseline | 19 | audited at `947d27a` |
| P0 strict Vision policy | 19 | behavior fixed before dependency replacement |
| P1 native success plus Vision | 16 | success, RGB/depth, detector and calibration switched |
| P2 action bridge plus native environment | 14 | action output and V5 VecEnv switched |
| P3 helpers, configuration, collection and public environment API | 0 | final formal runtime |
| P3 closeout runtime isolation | 0 | zero path exposure, zero loaded modules, canonical adapter removed |

The final direct V5-import count in both formal trees is zero. The only retained
`legacy_vectorized_skill_success` implementation is the explicit migration
oracle in `tools/migration/v5_success_adapter.py`; parity tests and migration
tools are intentionally outside the formal runtime.

## Phase 2 dependency disposition

| V5 module | Consumer before migration | Purpose | V7 target | Status | Parity test | Physical smoke |
|---|---|---|---|---|---|---|
| `tools.train_known_size_grasp` | `episode_runner.py` | size/object metadata loading | `simulation/config.py` | MIGRATED | configuration tests | PASS |
| `stage_vla.action_output` | `episode_runner.py` | policy/reference output and active masks | `action/output.py`, `action/reference.py` | MIGRATED | 8-Skill action parity, max error `0.0` | PASS |
| `stage_vla.data.v4_2_runtime` | `episode_runner.py` | RGB/depth tensor conversion | `simulation/isaac_lab/camera_adapter.py` | MIGRATED | reader shape/dtype/value parity | PASS |
| `stage_vla.envs` | `env_factory.py` | fingertip contact sensor installation | `simulation/isaac_lab/contact_sensors.py` | MIGRATED | exact config parity | PASS |
| `stage_vla.envs.fixed_object_pose` | `env_factory.py` | deterministic pose events | `simulation/isaac_lab/fixed_object_pose.py` | MIGRATED | exact pose/event parity | PASS |
| `stage_vla.envs.fixed_tilt_ik` | `env_factory.py` | optional fixed-tilt action term | `simulation/isaac_lab/fixed_tilt_ik.py` | MIGRATED | action/config parity | branch disabled in locked smoke |
| `stage_vla.envs.known_size_grasp_action` | factory and VecEnv | force-conditioned gripper control | `simulation/isaac_lab/gripper_action.py` | MIGRATED | exact action/controller parity | PASS |
| `stage_vla.envs.physical_profiles` | `env_factory.py` | USD scale, mass and inertia writes | `simulation/isaac_lab/physical_profiles.py` | MIGRATED | exact USD/profile parity | PASS |
| `stage_vla.envs.state_readers` | evaluator, factory and VecEnv | Isaac tensor/state sampling | `simulation/isaac_lab/state_reader.py` | MIGRATED | exact state parity | PASS |
| `stage_vla.rl.known_size_grasp` | evaluator, factory and VecEnv | known-size config, pressure, control and motion helpers | `simulation/config.py`, `simulation/physics/known_size.py`, `action/safety.py` | MIGRATED | exact helper parity | PASS |
| `stage_vla.rl.known_size_grasp_vecenv` | `episode_runner.py` | 55-D observation, lifecycle, control, reward and terminals | `simulation/isaac_lab/known_size_environment.py` plus V7 components | MIGRATED | exact 55-D observation and runtime behavior tests | PASS |
| `stage_vla.rl.object_physics` | factory and VecEnv | per-environment physical profiles | `simulation/physics/object_profiles.py` | MIGRATED | shape/dtype/value parity | PASS |
| `stage_vla.rl.reach_policy` | `episode_runner.py` | 52-D REACH observation/state/action | `simulation/environments/reach_observation.py`, `simulation/isaac_lab/reach_state.py`, `action_adapter.py` | MIGRATED | max observation error `0.0` | PASS |
| `stage_vla.rl.skill_action_safety` | evaluator and VecEnv | action projections and failure guards | `action/safety.py`, `action/evaluation/physical_runtime.py` | MIGRATED | boolean/action parity | PASS |
| `stage_vla.rl.skill_demonstrations` | `episode_runner.py` | demonstration and DAgger buffer | `tools/evaluation/data_collection.py` | MIGRATED | buffer/manifest parity | not enabled in locked smoke |
| `stage_vla.rl.v5_skill_contracts` | evaluator and Action evaluation | success and reference action | native success/reference modules | LEGACY_ONLY | native parity retained against this oracle | not in runtime |
| `stage_vla.stages.grasp_geometry` | evaluator and VecEnv | target and parallel-jaw geometry | `simulation/physics/grasp_geometry.py` | MIGRATED | exact geometry parity | PASS |
| `stage_vla.stages.object_motion` | `cli.py` | frozen motion limits | `simulation/config.py` | MIGRATED | exact config parity | PASS |
| `stage_vla.vision` | `episode_runner.py` | calibration and compact RGB-D detector | `vision/geometry`, `vision/providers/color_depth_detector.py` | MIGRATED | detector/calibration parity | PASS |

## Hidden VecEnv dependencies

Opening the V5 VecEnv exposed additional transitive responsibilities. They
were not left behind as adapters.

| V5 module | Purpose | V7 target | Status | Parity test | Physical smoke |
|---|---|---|---|---|---|
| `stage_vla.stages.physical_grasp` | contact geometry and physical grasp | `simulation/physics/physical_grasp.py` | MIGRATED | exact diagnostics parity | PASS |
| `stage_vla.rl.align_reward` | ALIGN reward terms | `simulation/environments/align_reward.py` | MIGRATED | exact reward parity | PASS |
| `stage_vla.rl.transport_handoff` | transport settle/reward terms | `simulation/environments/transport_handoff.py` | MIGRATED | exact reward/handoff parity | PASS |
| `stage_vla.rl.fixed_tilt_reference` | quaternion math and reference state | `simulation/physics/fixed_tilt.py` | MIGRATED | exact math/reset/clipping parity | optional branch disabled |
| `stage_vla.rl.place_snapshot` | single-state batch expansion | `simulation/isaac_lab/snapshot_state.py` | MIGRATED | exact tensor expansion parity | PASS |

## Public environment boundary

`KnownSizeGraspEnvironment` now owns the former evaluator-side bookkeeping.
The supported evaluation surface is:

```text
reset / observe / step
configure_evaluation / configure_skill
synchronize_after_external_step
get_physical_state / physical_state
status / gripper_diagnostics
object_size_m / current_skill / stability_speed_source
set_object_roles / set_visual_object_positions
```

`configure_skill` preserves the historical operation order: measure, change
Skill, mark continuous handoff, unlock the arm, update horizons, clear
counters and terminal masks, mark inactive environments finished, measure
again, require exact physical equality, and refresh the LIFT target when
needed. AST tests reject future evaluator calls to `env._...` and direct
mutation of these protected fields.

## Final validation

| Acceptance item | Result |
|---|---|
| Oracle fallback strict mode | PASS |
| `--require_v7_chain` strict Vision | PASS |
| Native `SuccessChecker` runtime | PASS |
| Success parity | PASS |
| Native RGB reader | PASS |
| Native depth reader | PASS |
| Native detector | PASS |
| Native calibration | PASS |
| Action execution bridge | PASS |
| Native vector environment | PASS |
| Physical helper migration | PASS |
| 8 checkpoint compatibility | PASS, max error `0.0` |
| Action output parity | PASS |
| Vision + Video | PASS |
| Physical stack smoke | PASS, 1/1 |
| V7 chain | PASS, 8/8 Skills and 7/7 exact handoffs |
| Remaining V5 runtime imports | **0** |
| Formal bootstrap exposes V5 path | **NO** |
| Canonical action-evaluation V5 adapter | **NO** |
| Loaded V5 module count | **0** |
| Vendor path exposed at runtime | **NO** |
| Runtime purity | **PASS** |
| V5 import blocker | **ENABLED** |
| Historical V5-isolated physical smoke | **PASS**, dirty-worktree provenance |
| Clean-HEAD V5-isolated physical smoke | **PASS** |

Clean-HEAD regression evidence:

- `291` tests passed, `0` failed, `0` skipped.
- `evidence/phase3_closeout_clean_head_checkpoint_compatibility.json`: all eight checkpoints,
  action dimension 5, maximum absolute error `0.0`.
- `evidence/phase3_closeout_clean_head_runtime_purity.json`: a clean-source
  snapshot with no loaded V5 module, no exposed V5 path, and the blocker enabled.
- `evidence/phase3_closeout_clean_head_seed61081.summary.json`: strict Vision, 729 valid
  calls, zero invalid frames, zero oracle fallback, no reference/recovery,
  `runtime_purity.import_blocker_enabled=true`,
  `runtime_purity.verified=true`, and `v7_chain.verified=true`.
- `evidence/phase3_closeout_clean_head_seed61081.mp4`: independently decoded `767/767`
  nonblank frames at 640x480; decoded pixel SHA-256
  `8345c24593cfa76d984420f828ab91946043881fae0a6f167b7398972a6898e6`.

The clean-HEAD result JSON records the source commit, pre-run and end-of-run
worktree state, UTC timestamps, artifact-lock path and SHA-256, and all eight
actual checkpoint hashes.

V5-trained checkpoints are allowed. V5 historical/regression source is
allowed. A V5 formal Python runtime dependency is forbidden.

The smoke proves preserved wiring and one locked seed. It is not a 20/50/100
seed success-rate claim and does not extend the trained cube policy domain.

**PHASE 3 COMPLETE**

`READY_FOR_PHASE4 = true`
