# Phase 3 Closeout Final Integration Report

Verification date: 2026-09-18.

Physical-runtime source commit:
`978763dc4af3530b398d821e29a3ec6451610bc0`.

## Integration result

The current `main` already contained the V7-only bootstrap, strict Vision and
runtime-purity chain gates, the expanded `ResultContext`, provenance capture,
and the normal physical-result wiring described by the closeout request. This
integration added the remaining failure-path coverage: Isaac launch failures
and later physical exceptions now write a failed result containing the failure
reason, source provenance, runtime purity, and observed reference/recovery call
counts before the exception is re-raised.

- `tools/evaluation/bootstrap.py` exposes only `repo/src`; it does not read or
  inject a V5 source root.
- `tools/evaluation/audit.py` requires strict Vision, positive V7 Vision calls,
  zero invalid/oracle/reference/recovery use, full prepared-Skill exercise, no
  loaded V5 modules, no exposed vendor path, an enabled import blocker, and a
  verified purity snapshot.
- `tools/evaluation/episode_runner.py` captures source provenance, rejects an
  already polluted runtime, installs the V5 blocker before Isaac starts,
  audits purity after the physical episode, and supplies the complete
  `ResultContext` from measured runtime values.
- `tools/evaluation/provenance.py` remains the single Git/source-state
  implementation. No duplicate Git-state collector was added.
- `reference_skill_calls`, `recovery_calls`, and `physical_success` come from
  the existing action instrumentation and task evaluator. No values or
  thresholds were hard-coded for closeout.

## Static V5 audit

AST-backed runtime-purity tests report zero `stage_vla` imports in both formal
trees. A text scan finds V5-root identifiers only in
`tools/evaluation/runtime_purity.py`, where they are required to detect an
exposed path or environment variable. The formal bootstrap contains no V5
root handling or path injection. Migration-only access remains under
`tools/migration/`.

## Software gates

| Gate | Result |
|---|---:|
| Runtime-purity tests | 21 passed, 0 failed |
| Evaluation tests | 84 passed, 0 failed |
| Full pytest | 292 passed, 0 failed, 0 skipped |
| `compileall` | PASS |
| Formal V7 V5 imports | 0 |
| Formal evaluator V5 imports | 0 |
| Checkpoint load | 8 / 8 |
| Checkpoint parity | PASS |
| Action dimension | 5 |
| Finite checkpoint output | PASS |

The full test run emitted 37 third-party deprecation warnings from TorchScript
and Isaac Lab; they did not fail a test.

## Clean-source physical smoke

The final run used the locked seed-61081 layout, strict RGB-D Vision, all eight
existing checkpoints, the original physical parameters and success criteria,
`--require_v7_chain`, and strict observer-video recording. No network was
trained and no Phase 4 work was performed.

| Check | Result |
|---|---:|
| Source commit | `978763dc4af3530b398d821e29a3ec6451610bc0` |
| Source clean before smoke | true; status `[]` |
| Physical success | PASS, 1 / 1 |
| Stable stack | PASS |
| Skills | 8 / 8 |
| Handoffs | 7 / 7 |
| Vision service calls | 729 |
| Invalid Vision frames | 0 |
| Oracle fallback | 0 |
| Reference Skill calls | 0 |
| Recovery calls | 0 |
| Vendor path exposed | false |
| Loaded V5 modules | 0 |
| Import blocker enabled | true |
| Runtime purity verified | true |
| V7 chain verified | true |

## Video verification

The recorder reported a completed 767-frame MP4 at 640x480 and 20 FPS. An
independent OpenCV pass decoded all 767 frames; every frame was nonblank. The
minimum frame pixel range was 255, the minimum frame standard deviation was
62.19341099683714, and the decoded-pixel SHA-256 was
`a4bb10c9c70be426c2f7ed17938395bc08ba6e626eaa2c8eb9c05405520ba5af`.

New evidence was written outside the repository so the recorded source tree
remained clean before and throughout the smoke:

- `C:\Users\Mayn\Documents\Codex\2026-09-18\is\outputs\phase3_closeout_clean_checkpoint_compatibility.json`
- `C:\Users\Mayn\Documents\Codex\2026-09-18\is\outputs\phase3_closeout_clean_runtime_purity.json`
- `C:\Users\Mayn\Documents\Codex\2026-09-18\is\outputs\phase3_closeout_clean_seed61081.summary.json`
- `C:\Users\Mayn\Documents\Codex\2026-09-18\is\outputs\phase3_closeout_clean_seed61081.mp4`

## Final acceptance

```text
bootstrap V7-only                 PASS

Formal V7 V5 imports             0
Formal evaluator V5 imports      0

Runtime Import Blocker           PASS
Runtime Purity                   PASS

vendor_path_exposed              false
loaded_v5_module_count           0
import_blocker_enabled           true

strict Vision gate               PASS
require_v7_chain purity gate     PASS

ResultContext integration        PASS
Provenance integration           PASS

runtime purity tests             21 passed / 0 failed
evaluation tests                 84 passed / 0 failed
full pytest                      292 passed / 0 failed / 0 skipped
compileall                       PASS

checkpoint load                  8 / 8
checkpoint parity                PASS
action dimension                 5

source commit                    978763dc4af3530b398d821e29a3ec6451610bc0
source clean before smoke        true

physical success                 PASS
stable stack                     PASS

Skills                           8 / 8
handoffs                         7 / 7

invalid Vision frames            0
oracle fallback                  0
reference skill calls            0
recovery calls                   0

Vision + Video                   PASS

V5-isolated physical smoke       PASS
V7 chain                         PASS

PHASE 3 COMPLETE                 YES
READY_FOR_PHASE4                 true
```

This result closes Phase 3 only for the locked one-seed cube-policy path. It is
not a multi-seed generalization result and does not expand the trained domain.
