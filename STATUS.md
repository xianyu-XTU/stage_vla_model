# Stage VLA V7 status

Updated: 2026-09-16

## Implemented

- Dependency-free `interfaces` contracts and public provider/simulator ports.
- Independent Vision, Language, Action, Simulation, and Orchestration modules.
- Replaceable Vision and Language registries; Chinese, English, and stack DSL.
- Canonical registry and independent definitions for all eight Skills.
- Isolated Action network, SHA-validating checkpoint loader, training, and
  success-evaluation packages.
- Simulator model registries for Franka, cube, RGB, and depth; `StackScene`,
  seeded non-overlap randomization, and `RedOnBlueEnvironment`.
- Canonical Isaac Lab adapters for camera, observation, action, runtime, and
  audited batched pipeline dispatch.
- V7-owned concrete Isaac environment factory with the old V5 path retained as
  a compatibility re-export.
- Separate 128 x 128 RGB-D Vision and 640 x 480 RGB Observer cameras, with
  role/ID validation and an observer-only data path.
- Streaming `VideoRecorder`, independent `FrameCapture` and overlays, plus
  explicit strict and best-effort failure modes.
- Evaluator compatibility entry reduced from 1,929 lines to 17 lines. CLI,
  runtime assembly, Skill execution, task aggregation, data collection, trace,
  and result writing now have separate owners under `tools/evaluation`; no
  evaluator module exceeds 700 lines.
- Split owner-specific configuration and train/evaluate/simulation/smoke entry
  directories, with compatibility wrappers for existing imports and scripts.
- Conservative V5 vendor classification; no vendor files deleted and 19
  remaining direct V5 module dependencies enumerated.

## Verified

- 99 unit, boundary, compatibility, Simulation, and integration tests pass
  under Isaac Sim Python 3.12.13.
- `compileall` passes for `src`, `tests`, `scripts`, and `tools`.
- All 145 package modules import under the Isaac Python validation environment
  without starting Isaac Lab or a simulator process; top-level
  `import stage_vla_v7` also passes under bare Python 3.11.
- Static tests enforce dependency-free Interfaces, Vision/Language/Action
  isolation, no Isaac dependency in Action/Orchestration, and no Simulation
  dependency in the pipeline.
- Chinese CLI resolves red-on-blue to the canonical eight-Skill sequence.
- All eight real V5 checkpoints match locked hashes; direct legacy inference
  plus frozen safety and refactored ActionService outputs have maximum absolute
  error `0.0`.
- Post-environment-migration physical RGB-D smoke passes 1/1 through
  `stage_vla_v7.simulation.isaac_lab` with the artifact lock enforced.
- Post-evaluator-split dual-camera headless physical smoke passes 1/1 with a
  decodable 640 x 480, 20 FPS MP4 containing 767 frames; Observer data was not
  used for Vision.
- Dual-camera audit: 729 V7 Vision calls, zero invalid frames, all eight prepared
  tokens exercised, zero mid-episode resets, no reference Skills, no recovery,
  and `v7_chain.verified=true` with `--require_v7_chain`.

## Evidence

- `evidence/phase2_checkpoint_compatibility_baseline.json`
- `evidence/phase2_checkpoint_compatibility_post.json`
- `evidence/phase2_vision_video_physical.json`
- `evidence/v7_vla_smoke_refactor_seed61081.summary.json`
- `docs/VALIDATION.md`

## Phase-2 acceptance

| Item | Status | Evidence |
|---|---|---|
| Core boundary | PASS | dependency-boundary tests |
| 8 Skill compatibility | PASS | registry tests and 8/8 physical execution |
| Checkpoint compatibility | PASS | 8/8, maximum action error `0.0` |
| Simulation registry | PASS | registry/fail-closed tests |
| V5 dependency migration | PARTIAL | factory migrated; 19 direct V5 module paths remain |
| Evaluator split | PASS | 17-line entry; 7 focused evaluation modules, all <= 700 lines |
| VideoRecorder | PASS | unit tests and decoded physical MP4 |
| Dual Camera | PASS | isolated bindings and physical two-camera run |
| Vision + Video | PASS | 729 Vision calls and 767 video frames together |
| V7 Chain | PASS | `v7_chain.verified=true` |
| Physical task | PASS | 1/1 stable stack episode |

## Current limitations

- Policies remain V5-trained artifacts; no retraining was performed.
- The concrete factory is V7-owned, but the vector environment, gripper,
  state readers, contacts, physical profiles, detector, and frozen helpers
  remain adapted from `vendor/stage_vla_v5`.
- Language is deterministic; no remote LLM/VLM is connected.
- Vision supplies object positions while proprioception, orientation, contacts,
  and physical terminal feedback still come from Isaac state.
- The validated policy domain remains rigid 4 cm, 0.05 kg cubes.
- One physical seed verifies preserved wiring and behavior, not a multi-seed
  generalization rate.
- The executor still adapts retained V5 physical helpers through an explicit
  context; replacing that compatibility boundary belongs to a later migration.
- Ruff was not installed in the available Python or system environment;
  syntax/import validation used `compileall` plus the full test suite.
