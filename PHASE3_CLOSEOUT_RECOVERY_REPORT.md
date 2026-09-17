# Phase 3 Closeout Recovery Report

Recovery date: 2026-09-17

## Outcome

Phase 3 clean-HEAD closeout verification passed. The verified source commit is
`0b4060423fa226a4bb428a40b6b4961706359c12`; both the pre-run source snapshot
and the post-run Git snapshot were clean. No training, threshold, physics,
oracle, reference, recovery, multi-seed, or Phase 4 work was performed.

## 1. Root cause on current main

The original closeout behavior was run successfully while its implementation
was still uncommitted. The old physical summary identifies commit `94006de` and
records `git_worktree_dirty=true` together with modified runtime, tests, and
documentation. Commit `80af30c` later consolidated the runtime-isolation code,
but the committed evidence still described the earlier dirty execution.

There was one remaining provenance defect: `collect_evidence_provenance()` read
Git state only while serializing the result. It could not prove which commit and
worktree state existed before Isaac Lab launched, and evidence written inside
the repository could itself make the end-of-run status dirty.

## 2. Previously uncommitted closeout files

The old evidence reports that `bootstrap.py`, `audit.py`, `episode_runner.py`,
`result_writer.py`, canonical action evaluation, migration verification,
tests, scripts, and docs were uncommitted during the successful run. At the
start of this Recovery, all of those runtime-isolation changes were already
present in clean commit `80af30c`; none required reimplementation.

## 3. Files changed by Recovery

Runtime/evidence changes:

- `tools/evaluation/provenance.py`
- `tools/evaluation/episode_runner.py`
- `tools/evaluation/export_runtime_purity.py`
- `tools/migration/verify_checkpoint_compatibility.py`
- `tests/evaluation/test_runtime_purity.py`

Audit/status changes:

- `CURRENT_MAIN_VS_CLOSEOUT_EVIDENCE_AUDIT.md`
- `PHASE3_CLOSEOUT_AUDIT.md`
- `PHASE3_CLOSEOUT_RECOVERY_REPORT.md`
- `STATUS.md`
- `docs/VALIDATION.md`
- `docs/VENDOR_CLASSIFICATION.md`
- `docs/VENDOR_MIGRATION_PHASE3.md`

The four new clean-HEAD evidence artifacts are listed below.

## 4. Formal bootstrap before and after

The historical pre-closeout bootstrap exposed both V7 and V5 source roots.
Current `tools/evaluation/bootstrap.py`, already present at Recovery baseline,
exposes only `repo/src`; it does not read `STAGE_VLA_V5_ROOT` or insert
`vendor/stage_vla_v5`. Recovery preserved this implementation unchanged.

## 5. `verify_v7_chain` before and after

The historical implementation lacked runtime-purity and explicit strict-Vision
gates. Current `verify_v7_chain`, already present at Recovery baseline, requires
strict Vision, positive V7 Vision calls, zero invalid/oracle/reference/recovery
use, all prepared Skills, no V5 path/module exposure, an enabled import blocker,
and verified runtime purity. Recovery preserved these gates unchanged.

## 6. Episode-runner purity integration

`episode_runner.py` already installed the fail-closed V5 blocker before
`AppLauncher` and audited purity after execution. Recovery added an immutable
source snapshot at the function entry, before blocker installation and Isaac
Lab launch, and reuses it for success and failure result serialization.

## 7. Canonical V5 adapter cleanup

`stage_vla_v7.action.evaluation` contains and exports no V5 adapter. Formal V7
and evaluator imports contain zero `stage_vla` / `stage_vla.*` imports.

## 8. Migration-only V5 entry

Historical comparison remains isolated under `tools/migration/`, specifically
`v5_bootstrap.py` and `v5_success_adapter.py`. V5-trained checkpoint files are
allowed data artifacts; no V5 Python module is a formal runtime dependency.

## 9. Runtime-purity tests

`tests/evaluation/test_runtime_purity.py`: **21 passed, 0 failed, 0 skipped**.
New coverage proves that a clean pre-run snapshot remains clean even if the
repository becomes dirty later, that capture precedes `AppLauncher`, and that
the standalone exporter enables the import blocker and records source fields.

## 10. Full pytest

Full suite under Isaac Sim Python: **291 passed, 0 failed, 0 skipped**. The 37
warnings are PyTorch/Isaac deprecation warnings; no test warning represents a
failed closeout gate.

## 11. Compile validation

`python -m compileall -q src tests scripts tools`: **PASS**.

## 12. Eight-checkpoint compatibility

All **8/8** locked checkpoints loaded with their expected SHA-256 values. The
action dimension remained **5**, all outputs were finite, and the maximum
absolute V5-frozen-safety versus V7-action-path difference was **0.0**.

## 13. Clean source commit

- Source commit: `0b4060423fa226a4bb428a40b6b4961706359c12`
- Source worktree clean before smoke: `true`
- Source Git status before smoke: `[]`
- End-of-run worktree dirty: `false`
- End-of-run Git status: `[]`

The smoke summary, standalone purity snapshot, and checkpoint report all name
the same source commit and clean pre-run state.

## 14. Clean-HEAD physical smoke

The single locked seed-61081 regression passed with stable physical success
`1/1`, all `8/8` learned Skills, `7/7` state-exact handoffs, and zero
mid-episode resets. Per-Skill inference rows were REACH 374, GRASP 31, LIFT 60,
TRANSPORT 101, ALIGN 31, DESCEND 20, RELEASE_STABILIZE 7, and RETREAT 103.

## 15. Vision and video

Strict RGB-D Vision made **729** V7 service calls with **0** invalid frames and
**0** oracle fallbacks. The Observer camera was separate from Vision. OpenCV
independently decoded **767/767** MP4 frames as nonblank at **640x480, 20 FPS**;
the minimum per-frame pixel range was `254` and the minimum standard deviation
was `62.19127421237001`.

Decoded-pixel SHA-256:
`8345c24593cfa76d984420f828ab91946043881fae0a6f167b7398972a6898e6`

MP4 file SHA-256:
`ec97a6cdb00c80eb80f4279de099f0aa6d21d7f01cf24d38286dee27f39cf6a2`

## 16. Runtime purity

The physical process and standalone exporter both report no exposed V5 path,
zero loaded V5 modules, no `STAGE_VLA_V5_ROOT`, an enabled import blocker, and
`verified=true`. The physical `v7_chain.verified` value is also `true`.

## 17. New evidence

- `evidence/phase3_closeout_clean_head_checkpoint_compatibility.json`
- `evidence/phase3_closeout_clean_head_runtime_purity.json`
- `evidence/phase3_closeout_clean_head_seed61081.summary.json`
- `evidence/phase3_closeout_clean_head_seed61081.mp4`

The old `phase3_closeout_*` files remain historical dirty-worktree evidence and
were not overwritten.

## 18. Updated status

`STATUS.md`, `PHASE3_CLOSEOUT_AUDIT.md`, and the Phase 3 validation/migration
documents now distinguish the historical dirty run from this clean-HEAD proof.

## 19. Phase decision

All Phase 3 closeout gates pass. `PHASE 3 COMPLETE = YES` and
`READY_FOR_PHASE4 = true`. This report establishes readiness only; Phase 4 was
not started.

## Final acceptance table

| Gate | Result |
|---|---|
| Formal bootstrap exposes V5 | **NO** |
| Formal V7 V5 imports | **0** |
| Formal evaluator V5 imports | **0** |
| Canonical V5 adapter | **REMOVED** |
| Runtime import blocker | **PASS** |
| Runtime purity | **PASS** |
| `vendor_path_exposed` | **false** |
| `loaded_v5_module_count` | **0** |
| `require_v7_chain` purity gate | **PASS** |
| Strict Vision gate | **PASS** |
| Runtime-purity tests | **21 passed / 0 failed** |
| Full pytest | **291 passed / 0 failed / 0 skipped** |
| `compileall` | **PASS** |
| Checkpoint load | **8 / 8** |
| Checkpoint parity | **PASS**, max error `0.0` |
| Action dim | **5** |
| Source commit | `0b4060423fa226a4bb428a40b6b4961706359c12` |
| Source clean before smoke | **true** |
| Vision invalid frames | **0** |
| Oracle fallback | **0** |
| Reference Skill calls | **0** |
| Recovery calls | **0** |
| Skills | **8 / 8** |
| Handoffs | **7 / 7** |
| Stable physical stack | **PASS**, 1 / 1 |
| Video | **PASS**, 767 / 767 nonblank decoded frames |
| V5-isolated physical smoke | **PASS** |
| V7 chain | **PASS** |
| PHASE 3 COMPLETE | **YES** |
| `READY_FOR_PHASE4` | **true** |

