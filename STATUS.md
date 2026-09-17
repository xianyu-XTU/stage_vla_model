# Stage VLA V7 status

Updated: 2026-09-17

## Implemented

- Dependency-free public contracts and isolated Vision, Language, Action,
  Orchestration, and Simulation ownership.
- Eight independently registered Skills with the frozen 5D
  `[dx, dy, dz, dyaw, grip]` action ABI.
- SHA-validating TorchScript checkpoint loading and native V7 action routing,
  reference actions, safety projections, and success/failure evaluation.
- Native RGB/depth reading, calibration, compact color-depth detection, and
  strict/debug-oracle Vision failure policies.
- Native Isaac factory, contact/state readers, gripper controller, object and
  physical profiles, fixed pose/tilt support, snapshot expansion, REACH state
  and action adapters, and `KnownSizeGraspEnvironment`.
- Typed `PhysicalState`, exact legacy-compatible 52-D/55-D observations, and
  public environment APIs for observation, external-step synchronization,
  Skill handoff, terminal status, and gripper diagnostics.
- Separate Vision and Observer cameras plus strict streaming MP4 recording.
- Split evaluator modules for CLI/preflight, assembly, execution, aggregation,
  collection, trace, and result persistence. AST tests prevent evaluator use
  of private environment methods or protected state mutation.
- Native demonstration/DAgger buffer behavior used by evaluation collection.
- Conservative V5 retention for history, regression and checkpoint provenance;
  no vendor source or model artifact was deleted or rewritten.
- V7-only evaluation bootstrap, exact V5 module/path runtime auditing, an
  active `stage_vla` import blocker during physical evaluation, and a
  fail-closed runtime-purity condition in `--require_v7_chain`.

## Phase 3 verification

- Full suite: **288 passed, 0 failed, 0 skipped** under `E:\work\IsaacLab` Isaac Sim Python.
- `compileall`: `src`, `tests`, `scripts`, and `tools` pass.
- Dependency-boundary and evaluator-public-API guards pass.
- Formal physical runtime V5 imports: **0**.
- Formal bootstrap V5 path exposure: **0**; it adds only `repo/src`.
- Canonical `stage_vla_v7.action.evaluation` V5 adapters: **0**. The optional
  regression oracle moved to `tools/migration/v5_success_adapter.py`.
- V5-isolated import and physical processes report **0 loaded V5 modules** and
  `vendor_path_exposed=false`.
- All eight locked checkpoints load with their expected hashes, retain action
  dimension 5, and produce maximum action error **0.0**.
- Strict seed-61081 Vision + Video smoke: 1/1 physical task, 8/8 Skills, 7/7
  exact handoffs, zero mid-episode resets, 730 valid Vision calls, zero invalid
  frames, zero oracle fallback, no reference/recovery calls,
  `runtime_purity.import_blocker_enabled=true`,
  `runtime_purity.verified=true`, and `v7_chain.verified=true`.
- Independent MP4 decode: 768/768 frames, 640x480, all frames nonblank; decoded
  pixel SHA-256
  `97944f34b5adaed6642e07d13f082730eaa16f8e0c8d77980d83c3b419ec6fec`.

## Evidence

- `docs/VENDOR_MIGRATION_PHASE3.md`
- `PHASE3_CLOSEOUT_AUDIT.md`
- `evidence/phase3_closeout_checkpoint_compatibility.json`
- `evidence/phase3_closeout_runtime_purity.json`
- `evidence/phase3_closeout_seed61081.summary.json`
- `evidence/phase3_closeout_seed61081.mp4`
- `docs/VALIDATION.md`

## Phase 3 acceptance

| Item | Status | Evidence |
|---|---|---|
| Strict Vision, no oracle fallback | PASS | 730/730 valid calls; fallback count 0 |
| Native SuccessChecker | PASS | parity tests and physical runtime |
| Native Vision runtime | PASS | reader/detector/calibration parity and smoke |
| Native action bridge | PASS | 8-Skill parity; max error `0.0` |
| Native vector environment | PASS | exact observation tests and public API guard |
| Physical helpers | PASS | cluster parity tests and final smoke |
| Checkpoint compatibility | PASS | 8/8; maximum error `0.0` |
| Vision + Video | PASS | strict smoke and complete independent decode |
| V7 chain | PASS | 8/8 Skills; 7/7 exact handoffs |
| Remaining formal V5 runtime imports | PASS | 0 |
| Formal V5 path exposure | PASS | bootstrap exposes only `repo/src` |
| Loaded V5 modules | PASS | 0 in the isolated physical result |
| Physical V5 import blocker | PASS | enabled for the complete physical run |
| Canonical action-evaluation V5 adapter | PASS | absent from module and `__all__` |
| V5-isolated physical smoke | PASS | runtime purity and V7 chain both verified |

## Current limitations

- Policies remain V5-trained artifacts; no retraining was performed.
- The validated policy domain remains rigid 4 cm, 0.05 kg cubes.
- Vision supplies object positions; robot proprioception, orientation, contact,
  and physical terminal feedback still come from Isaac state.
- Language is deterministic; no remote LLM or VLM is connected.
- The final physical regression uses one locked seed. It validates migration
  fidelity, not a multi-seed generalization or success-rate claim.
- The optional fixed-tilt branch has behavior parity but is disabled by the
  locked physical smoke configuration.
- `vendor/stage_vla_v5` remains intentionally available for historical
  training, regression tests, and provenance.

## Completion decision

V5-trained checkpoints are allowed. V5 historical/regression source is
allowed. A V5 formal Python runtime dependency is forbidden and is now absent
from the locked physical path.

**PHASE 3 COMPLETE**

`READY_FOR_PHASE4 = true`
