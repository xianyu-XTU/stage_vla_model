# Stage VLA V7 status

Updated: 2026-09-16

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

## Phase 3 verification

- Full suite: **270 passed** under `E:\work\IsaacLab` Isaac Sim Python.
- `compileall`: `src`, `tests`, `scripts`, and `tools` pass.
- Dependency-boundary and evaluator-public-API guards pass.
- Formal physical runtime V5 imports: **0**.
- Source-only lazy V5 imports: **1**, explicitly `LEGACY_ONLY` in the
  regression adapter.
- All eight locked checkpoints load with their expected hashes, retain action
  dimension 5, and produce maximum action error **0.0**.
- Strict seed-61081 Vision + Video smoke: 1/1 physical task, 8/8 Skills, 7/7
  exact handoffs, zero mid-episode resets, 730 valid Vision calls, zero invalid
  frames, zero oracle fallback, no reference/recovery calls, and
  `v7_chain.verified=true`.
- Independent MP4 decode: 768/768 frames, 640x480, all frames nonblank; decoded
  pixel SHA-256
  `a4a94a8f7511c4c36220f9a43a830dfa6e8d49b6253f54915418d4e2a4c93d71`.

## Evidence

- `docs/VENDOR_MIGRATION_PHASE3.md`
- `evidence/phase3_checkpoint_compatibility.json`
- `evidence/phase3_runtime_migration_seed61081.summary.json`
- `outputs/phase3_p3_public_env_api_seed61081.json` (local, ignored)
- `outputs/phase3_p3_public_env_api_seed61081.mp4` (local, ignored)
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
