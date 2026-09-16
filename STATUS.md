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
- Split owner-specific configuration and train/evaluate/simulation/smoke entry
  directories, with compatibility wrappers for existing imports and scripts.
- Conservative V5 vendor classification; no vendor files deleted.

## Verified

- 78 unit, boundary, compatibility, Simulation, and integration tests pass
  under Isaac Sim Python 3.12.13.
- `compileall` passes for `src`, `tests`, `scripts`, and `tools`.
- All 136 package modules import under standalone Python 3.11 without starting
  Isaac Lab or requiring a simulator process.
- Static tests enforce dependency-free Interfaces, Vision/Language/Action
  isolation, no Isaac dependency in Action/Orchestration, and no Simulation
  dependency in the pipeline.
- Chinese CLI resolves red-on-blue to the canonical eight-Skill sequence.
- All eight real V5 checkpoints match locked hashes; direct legacy inference
  plus frozen safety and refactored ActionService outputs have maximum absolute
  error `0.0`.
- Post-refactor physical RGB-D smoke passes 1/1 through
  `stage_vla_v7.simulation.isaac_lab` with the artifact lock enforced.
- Physical audit: 729 V7 Vision calls, zero invalid frames, all eight prepared
  tokens exercised, zero mid-episode resets, no reference Skills, no recovery,
  and `v7_chain.verified=true` with `--require_v7_chain`.

## Evidence

- `evidence/checkpoint_compatibility.json`
- `evidence/v7_vla_smoke_refactor_seed61081.summary.json`
- `docs/VALIDATION.md`

## Current limitations

- Policies remain V5-trained artifacts; no retraining was performed.
- The physical Isaac factory, vector environment, state readers, and contact
  stack remain adapted from `vendor/stage_vla_v5`.
- Language is deterministic; no remote LLM/VLM is connected.
- Vision supplies object positions while proprioception, orientation, contacts,
  and physical terminal feedback still come from Isaac state.
- The validated policy domain remains rigid 4 cm, 0.05 kg cubes.
- One physical seed verifies preserved wiring and behavior, not a multi-seed
  generalization rate.
- Ruff was not installed in the available Python or system environment;
  syntax/import validation used `compileall` plus the full test suite.
