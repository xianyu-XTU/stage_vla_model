# Phase 3 Closeout Audit

## Recovery notice

The original final-result section below records a successful run from a dirty
worktree at commit `94006de`. It is retained as historical recovery evidence,
not clean-HEAD closeout proof. Current implementation recovery is complete;
clean-HEAD physical verification is pending. See
`CURRENT_MAIN_VS_CLOSEOUT_EVIDENCE_AUDIT.md`.

Audit baseline: `main` at `94006de24ecdcb4725394f24edf36d8d9b4c23c1`
(`Complete native V7 physical runtime migration`).

Audit date: 2026-09-17.

This audit was produced before changing the physical runtime. It classifies the
results of a repository-wide search for `import stage_vla`, `from stage_vla`,
`STAGE_VLA_V5_ROOT`, `V5_ROOT`, `VENDORED_V5_ROOT`,
`vendor/stage_vla_v5`, `legacy_vectorized_skill_success`, `sys.path`, and
`importlib`. Broad `V5` and `legacy` names were also reviewed so checkpoint
provenance is not confused with a Python runtime dependency.

## Executive finding

Phase 3 was **not complete at the audit baseline**. Three formal-runtime
dependencies remained:

1. `tools/evaluation/bootstrap.py` selected a V5 source root and inserted it in
   `sys.path` on every evaluator import.
2. `stage_vla_v7.action.evaluation` publicly exported
   `legacy_vectorized_skill_success`, whose call path dynamically imported
   `stage_vla.rl.v5_skill_contracts`.
3. `scripts/smoke/run_v7_smoke.ps1` loaded the evaluation configuration from
   `vendor/stage_vla_v5`, so the locked physical smoke could not run with the
   vendor tree unavailable.

The formal evaluator otherwise uses native V7 Vision, action output, safety,
success evaluation, physical helpers, environment construction, camera
adapters, and recording. No other direct `stage_vla` import was found in
`src/stage_vla_v7` or `tools/evaluation`.

## A. Formal Runtime

| Reference | Baseline classification | Closeout disposition |
|---|---|---|
| `tools/evaluation/bootstrap.py` | `STILL_RUNTIME_DEPENDENCY`: defines `VENDORED_V5_ROOT`/`V5_ROOT`, reads `STAGE_VLA_V5_ROOT`, and inserts V5 in `sys.path` | Replace with a V7-only source bootstrap |
| `src/stage_vla_v7/action/evaluation/skill_evaluator.py` | `STILL_RUNTIME_DEPENDENCY`: lazy `from stage_vla.rl.v5_skill_contracts import skill_success` | Move the regression adapter outside the canonical package |
| `src/stage_vla_v7/action/evaluation/__init__.py` | `PUBLIC_V5_ADAPTER`: exports the lazy adapter | Remove it from the canonical API |
| `scripts/smoke/run_v7_smoke.ps1` | `STILL_RUNTIME_DEPENDENCY`: reads the runtime config below `vendor/stage_vla_v5` | Point to a V7-owned config under `config/` |
| `src/stage_vla_v7/action/network/v5_factory.py`, `action/adapters/v5.py`, and their public re-exports | `CHECKPOINT_PROVENANCE`: pure V7 loader for eight V5-trained TorchScript files; no V5 Python import | Retain; checkpoints may remain V5-trained |
| `src/stage_vla_v7/action/actions/*.py`, `simulation/environments/known_size_observation.py`, and `interfaces/contracts/physical_state.py` | `CHECKPOINT_SCHEMA_COMPATIBILITY`: frozen 52/55-D schemas and legacy field aliases, implemented in V7 | Retain |
| `src/stage_vla_v7/action/evaluation/{success_checker,vectorized_success}.py` | `NATIVE_V7_PARITY_SEMANTICS`: descriptions mention V5 compatibility but code is V7-owned | Retain |
| `src/stage_vla_v7/vision/providers/legacy_provider.py` and compatibility re-exports | `OPTIONAL_OBJECT_ADAPTER`: wraps an injected detector object and never imports `stage_vla`; not used by the formal physical evaluator | Retain; not a Python runtime dependency |
| `src/stage_vla_v7/simulation/isaac_lab/runtime.py` compatibility factory name | `V7_COMPATIBILITY_ALIAS`: delegates only to the V7 factory | Retain; not used by the formal evaluator |
| `tools/evaluation/data_collection.py` | `DATA_SCHEMA_COMPATIBILITY`: validates legacy dataset metadata only | Retain |
| `scripts/run_v7_smoke.ps1` and `scripts/smoke/run_v7_vision_video_smoke.ps1` | V5-trained artifact/calibration defaults only | Retain as artifact provenance; neither path is added to Python import state |

The initial formal-runtime direct V5-import count is **1**, the initial formal
V5-path exposure count is **1**, and the initial vendor runtime-resource
dependency count is **1**.

## B. Regression / Parity

The following tests explicitly add `vendor/stage_vla_v5` to `sys.path` and
import frozen V5 implementations as comparison oracles. They are permitted
regression-only consumers and are not formal-runtime entry points:

- `tests/action/test_action_output_parity.py`
- `tests/action/test_physical_safety_parity.py`
- `tests/evaluation/test_demonstration_buffer_parity.py`
- `tests/evaluation/test_physical_runtime_success_parity.py`
- `tests/evaluation/test_success_checker_parity.py`
- `tests/simulation/test_camera_reader_parity.py`
- `tests/simulation/test_fixed_tilt_ik_parity.py`
- `tests/simulation/test_gripper_action_contract_parity.py`
- `tests/simulation/test_isaac_factory_helpers_parity.py`
- `tests/simulation/test_known_size_control_parity.py`
- `tests/simulation/test_known_size_observation_parity.py`
- `tests/simulation/test_motion_config_parity.py`
- `tests/simulation/test_object_profiles_parity.py`
- `tests/simulation/test_physical_grasp_parity.py`
- `tests/simulation/test_physical_profiles_parity.py`
- `tests/simulation/test_reach_runtime_parity.py`
- `tests/simulation/test_reward_terms_parity.py`
- `tests/simulation/test_snapshot_state_parity.py`
- `tests/simulation/test_state_reader_parity.py`
- `tests/test_vision_detector_parity.py`

These imports must remain explicit and isolated. They do not justify restoring
V5 discovery in the formal bootstrap.

## C. Historical Training

- `vendor/stage_vla_v5/` contains 179 Python files (180 files total) and is
  retained unchanged as historical source, training provenance, and a parity
  oracle.
- `src/stage_vla_v7/action/training/legacy_v5.py` builds an explicit command for
  the retained historical trainer. It is not imported by `stage_vla_v7`, the
  physical evaluator, or the smoke entry point.
- `TRAINING.md` and training-related references describe provenance; they are
  not runtime imports.

Historical training is outside the formal evaluation call graph. No PPO, BC,
DAgger, or Stage-PPO run is part of this closeout.

## D. Migration Tool

- `tools/migration/verify_checkpoint_compatibility.py` loads the same frozen
  TorchScript files twice through V7 contracts to verify hashes, observation
  dimensions, finite five-dimensional actions, and safety-projected parity. It
  does not import the V5 Python package.
- `examples/integrate_v5_yolo.py` explicitly inserts an external V5 source root
  and imports `stage_vla.vision`. This is an opt-in migration example, not a
  formal runtime entry point.
- `examples/integrate_v5.py` demonstrates the V7-owned V5-trained checkpoint
  bundle loader and has no V5 Python import.

The removed success adapter may live in `tools/migration/` if an explicit
regression caller still needs it; canonical `stage_vla_v7.action.evaluation`
must not import or export it.

## E. Tests

Existing structural tests enforce the public environment boundary and reject
private evaluator access or protected state mutation. Closeout tests must add:

- AST guards over `src/stage_vla_v7/` and `tools/evaluation/` for exact
  `stage_vla` imports;
- subprocess import guards for `tools.evaluation.cli`,
  `tools.evaluation.episode_runner`, and `stage_vla_v7`;
- public-API checks for `stage_vla_v7.action.evaluation`;
- loaded-module, vendor-path, and `stage_vla_v7` false-positive purity tests;
- fail-closed `--require_v7_chain` purity cases;
- a sanitized subprocess with a meta-path blocker for `stage_vla`.

Parity tests listed in section B are an explicit whitelist by directory and
purpose. A whole-repository zero-match assertion would be incorrect.

## F. Documentation

V5 references in `README.md`, `ACTIONS.md`, `SIMULATION.md`, `STATUS.md`,
`ARCHITECTURE.md`, `docs/MIGRATION_FROM_V5.md`,
`docs/PHASE2_REFACTOR_AUDIT.md`, `docs/REFACTORING.md`,
`docs/VALIDATION.md`, `docs/VENDOR_CLASSIFICATION.md`, and
`docs/VENDOR_MIGRATION_PHASE3.md` describe history, compatibility, or the old
completion definition. They must distinguish:

- **V5-trained checkpoint**: allowed;
- **V5 historical/regression source**: allowed;
- **V5 formal Python runtime dependency**: forbidden.

Existing JSON under `evidence/` records earlier Phase 2/3 runs and checkpoint
provenance. It is historical evidence only and cannot satisfy this closeout.
New evidence must record runtime purity, current code provenance, artifact lock
and checkpoint hashes, a new locked physical smoke, and actual test totals.

## Closeout acceptance state

At baseline:

| Gate | State |
|---|---|
| Formal `src/stage_vla_v7` V5 imports | **NOT ZERO (1)** |
| Formal `tools/evaluation` V5 imports/path bootstrap | **NOT ZERO / EXPOSED** |
| Canonical action-evaluation V5 adapter | **EXPOSED** |
| Runtime loaded-module purity gate | **MISSING** |
| Runtime vendor-path purity gate | **MISSING** |
| V5-isolated physical smoke | **NOT RUN FOR CLOSEOUT** |

Therefore the baseline decision is **PHASE 3 NOT COMPLETE**. The document will
be updated with final evidence only after all code, import guards, checkpoint
compatibility, physical smoke, video, full tests, and `compileall` have been
verified.

## Historical dirty-worktree closeout result

The three baseline defects were resolved without deleting the vendor tree,
retraining a policy, widening the 4 cm / 0.05 kg rigid-cube domain, enabling a
reference/recovery controller, or entering Phase 4:

- `tools/evaluation/bootstrap.py` now adds only `repo/src`.
- The V5 success oracle moved to `tools/migration/v5_success_adapter.py` and is
  reachable only through the explicit regression-only
  `with_v5_reference_path` context.
- `stage_vla_v7.action.evaluation` neither imports nor exports that adapter.
- The formal smoke uses `config/evaluation/known_size_cube.json`; it no longer
  reads a runtime config from the vendor tree.
- `tools/evaluation/runtime_purity.py` checks exact V5 module names and exposed
  package paths and installs an active meta-path blocker before Isaac Lab
  starts. The result is persisted in every completed physical result.
- `verify_v7_chain` requires runtime purity and fails closed on a loaded V5
  module, an exposed V5 source path, a missing blocker, or missing/malformed
  purity evidence.
- Static AST guards, isolated import probes, and a meta-path V5 blocker protect
  the formal import graph.

### Phase 3 final acceptance table

| Acceptance item | Final value |
|---|---|
| Formal `src/stage_vla_v7` V5 imports | **0** |
| Formal `tools/evaluation` V5 imports | **0** |
| Formal bootstrap exposes V5 path | **NO** |
| Canonical V7 action-evaluation API exposes V5 adapter | **NO** |
| Loaded V5 module count | **0** |
| Vendor path exposed at runtime | **NO** |
| Runtime purity | **PASS** |
| V5 import blocker during physical run | **ENABLED** |
| Native `SuccessChecker` | **PASS** |
| Native Vision | **PASS** |
| Native `KnownSizeGraspEnvironment` | **PASS** |
| Eight checkpoint load | **8 / 8** |
| Checkpoint action dimension | **5** |
| Checkpoint compatibility | **PASS**, max error `0.0`, all outputs finite |
| Strict Vision | **PASS** |
| Invalid Vision frames | **0** |
| Oracle fallback | **0** |
| Reference Skill calls | **0** |
| Recovery calls | **0** |
| Skill execution | **8 / 8** |
| State-exact Skill handoffs | **7 / 7** |
| Physical stable stack | **PASS**, 1 / 1 |
| Vision + Video | **PASS**, 768 / 768 nonblank decoded frames |
| V7 chain | **PASS** |
| V5-isolated physical smoke | **PASS** |
| Tests | **288 passed / 0 failed / 0 skipped** |
| `compileall` | **PASS** |

Historical closeout evidence:

- `evidence/phase3_closeout_runtime_purity.json`
- `evidence/phase3_closeout_checkpoint_compatibility.json`
- `evidence/phase3_closeout_seed61081.summary.json`
- `evidence/phase3_closeout_seed61081.mp4`

The physical result records the baseline Git commit, generation timestamp,
working-tree state, artifact-lock identity, and all eight actual checkpoint
hashes. It reports 730 strict V7 Vision calls, zero invalid frames, zero oracle
fallback, 8/8 learned Skills, 7/7 exact handoffs, zero reference/recovery use,
zero loaded V5 modules, no exposed V5 path, the V5 import blocker enabled, and
both purity and V7-chain gates verified. The MP4 independently decoded all
768 frames as nonblank at 640x480; its decoded-pixel SHA-256 is
`97944f34b5adaed6642e07d13f082730eaa16f8e0c8d77980d83c3b419ec6fec`.

V5-trained checkpoints remain allowed. V5 historical/regression source remains
allowed. V5 formal Python runtime dependency is forbidden and absent.

**PHASE 3 CLOSEOUT RECOVERY VERIFICATION PENDING**

`READY_FOR_PHASE4 = false`

## Files changed by the closeout

- `ACTIONS.md`
- `ARCHITECTURE.md`
- `PHASE3_CLOSEOUT_AUDIT.md`
- `STATUS.md`
- `config/evaluation/known_size_cube.json`
- `config/simulation/smoke_layout_seed61081.json`
- `docs/REFACTORING.md`
- `docs/VALIDATION.md`
- `docs/VENDOR_CLASSIFICATION.md`
- `docs/VENDOR_MIGRATION_PHASE3.md`
- `evidence/phase3_closeout_checkpoint_compatibility.json`
- `evidence/phase3_closeout_runtime_purity.json`
- `evidence/phase3_closeout_seed61081.mp4`
- `evidence/phase3_closeout_seed61081.summary.json`
- `scripts/smoke/run_v7_smoke.ps1`
- `src/stage_vla_v7/action/evaluation/__init__.py`
- `src/stage_vla_v7/action/evaluation/skill_evaluator.py`
- `src/stage_vla_v7/action/output.py`
- `tests/evaluation/test_runtime_purity.py`
- `tests/evaluation/test_vision_fail_policy.py`
- `tools/evaluation/audit.py`
- `tools/evaluation/bootstrap.py`
- `tools/evaluation/episode_runner.py`
- `tools/evaluation/export_runtime_purity.py`
- `tools/evaluation/provenance.py`
- `tools/evaluation/result_writer.py`
- `tools/evaluation/runtime_purity.py`
- `tools/migration/v5_bootstrap.py`
- `tools/migration/v5_success_adapter.py`
- `tools/migration/verify_checkpoint_compatibility.py`
