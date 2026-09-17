# V5 vendor classification

Phase-3 implementation status: **runtime migration complete for the locked
cube-policy path**. Clean-HEAD closeout verification is pending. No vendor file
is classified for deletion.

## Current classification

| Vendor area | Current caller | Handling | Status |
|---|---|---|---|
| `stage_vla/rl/v5_skill_contracts.py` | explicit migration adapter and parity tests | retained success oracle | LEGACY_ONLY |
| Python helpers covered by Phase-3 parity tests | tests only | retained comparison oracle | LEGACY_ONLY |
| `config/v5_generalized_cube_eval.json` | regression tests only | historical schema/provenance input | LEGACY_ONLY |
| historical training tools and datasets | explicit training adapters or no current caller | retained; no training migration claimed | KEEP |
| checkpoint provenance and original source | documentation and audits | retained | KEEP |
| all other vendor packages/tools | historical or unproven capabilities | retained conservatively | KEEP |

## Runtime audit

- Direct V5 Python modules in the Phase-2 physical runtime: 19.
- Direct V5 Python modules in the final formal physical runtime: 0.
- V5 imports present in formal V7-owned source: 0.
- V5 imports in parity tests: intentional and outside runtime.
- V5 import in `tools/migration/v5_success_adapter.py`: explicit
  regression-only oracle, outside runtime.
- Vendor root insertion by evaluation bootstrap: 0; the bootstrap exposes only
  `repo/src`.
- Historical isolated physical smoke: active V5 import blocker, 0 loaded V5
  modules, and no exposed V5 package path, with dirty-worktree provenance.

The module-level migration targets and parity/smoke evidence are recorded in
`docs/VENDOR_MIGRATION_PHASE3.md`.

## Migrated capabilities

| Capability | V7 owner |
|---|---|
| config and object metadata | `simulation/config.py` |
| action output and reference actions | `action/output.py`, `action/reference.py` |
| native success/failure predicates | `action/evaluation/*` |
| RGB/depth reads and calibration | `simulation/isaac_lab/camera_adapter.py`, `vision/geometry/*` |
| compact RGB-D detection | `vision/providers/color_depth_detector.py` |
| physical state and 52-D/55-D observations | `interfaces/contracts/physical_state.py`, `simulation/environments/*observation.py` |
| Isaac state and REACH sampling | `simulation/isaac_lab/state_reader.py`, `reach_state.py` |
| known-size environment | `simulation/isaac_lab/known_size_environment.py` |
| gripper, contacts, poses, profiles, fixed tilt, snapshots | `simulation/isaac_lab/*` |
| geometry, grasp, object profiles and control math | `simulation/physics/*` |
| action safety and physical runtime helpers | `action/safety.py`, `action/evaluation/physical_runtime.py` |
| evaluation collection | `tools/evaluation/data_collection.py` |

## Delete policy

**DELETE: none.** Runtime independence is not deletion evidence. Removal would
require proving that historical training, external consumers, regression
comparisons, checkpoint provenance, and retained schemas no longer need the
file family. Phase 3 deliberately does not make that claim.
