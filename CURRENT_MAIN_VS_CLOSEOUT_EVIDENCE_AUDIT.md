# Current Main vs Closeout Evidence Audit

Audit date: 2026-09-17

Audited baseline: `main` at
`80af30ccae93321c7e6ada1b389ce451915777f3` with a clean worktree.

## Decision

The Phase 3 runtime-isolation implementation is present on current `main` and
matches the closeout behavior described by the tests. The committed closeout
evidence does not prove that implementation from a clean source checkout: its
physical summary names commit `94006de24ecdcb4725394f24edf36d8d9b4c23c1`
and records a dirty worktree containing the then-uncommitted closeout source.

The Recovery scope is therefore evidence provenance and clean-HEAD
re-verification, not a second runtime implementation. Until that run passes,
`PHASE 3 COMPLETE` and `READY_FOR_PHASE4` must remain false.

## Requirement comparison

| Closeout requirement | Current `80af30c` implementation | Existing test requirement | Existing evidence | Consistent? | Recovery action |
|---|---|---|---|---|---|
| Formal bootstrap exposes only V7 | `tools/evaluation/bootstrap.py` adds only `repo/src` | `test_formal_bootstrap_does_not_expose_v5` | Physical run reports no vendor path | Implementation yes; provenance no | Re-run from clean committed source |
| Formal V7/evaluator imports contain no V5 | Exact-name AST guard covers `src/stage_vla_v7` and `tools/evaluation` | `test_formal_runtime_has_no_v5_imports` | Loaded V5 module count is 0 | Implementation yes; provenance no | Re-run static/full suites and formal audit |
| Canonical action evaluation has no V5 adapter | Adapter is isolated in `tools/migration/v5_success_adapter.py` | Public API and isolated-import guards | Physical run loads no V5 | Implementation yes; provenance no | Preserve migration-only entry and re-verify |
| Import blocker is fail-closed before Isaac Lab | `episode_runner.py` installs `V5ImportBlocker` before `AppLauncher` | Clean/dirty-start and import-block tests | Physical summary records blocker enabled | Behavior yes; source was dirty | Capture source snapshot before launch and re-run |
| Runtime purity gates `--require_v7_chain` | `verify_v7_chain` requires strict Vision, no fallback/reference/recovery, no V5 path/module, and blocker enabled | Purity and strict-Vision rejection tests | Purity and V7 chain are true | Behavior yes; source was dirty | Re-run unchanged gate from clean commit |
| Runtime purity is persisted | `result_writer.py` writes full purity plus closeout metrics | Result construction tests | Summary contains required fields | Behavior yes; source was dirty | Generate a new summary, do not overwrite history |
| Eight checkpoints remain compatible | Migration verifier checks hashes, dimensions, finiteness, and exact projected actions | Action/checkpoint regression coverage | 8/8, action dim 5, max error 0.0 | Behavior yes; evidence names old commit | Regenerate compatibility evidence from clean commit |
| Strict Vision + Video physical smoke passes | Locked seed-61081 script still uses all eight policies and no reference/recovery path | Vision fail policy and recording tests | 1/1 stack, 8/8 Skills, 7/7 handoffs, valid video | Behavior yes; source was dirty | Run once from clean commit with outputs outside repo |
| Evidence proves the source was clean before launch | Provenance was collected only when serializing the result | No pre-run provenance test on baseline | `git_worktree_dirty=true`; long dirty status | **No** | Add immutable pre-run source snapshot and tests |
| Documentation cites valid final evidence | Docs cite the dirty-worktree artifacts as final | Not test-enforced | Claims lead the actual proof | **No** | Mark pending, then update only after clean-head verification |

## Historical evidence boundary

The following files are retained as historical recovery inputs only:

- `evidence/phase3_closeout_seed61081.summary.json`
- `evidence/phase3_closeout_runtime_purity.json`
- `evidence/phase3_closeout_checkpoint_compatibility.json`
- `evidence/phase3_closeout_seed61081.mp4`

They establish that the local closeout implementation once passed the locked
physical path, but they do not establish clean-HEAD reproducibility. New
evidence must use distinct `phase3_closeout_clean_head_*` names and record:

- `source_commit`;
- `source_worktree_clean_before_run=true`;
- `source_git_status_before_run=[]`;
- `git_worktree_dirty=false` and `git_status=[]` when outputs are outside the
  repository;
- an enabled V5 import blocker and verified runtime purity.

## Recovery change boundary

Required changes are limited to pre-run provenance capture, evidence exporters,
focused tests, and closeout documentation. This Recovery must not retrain,
change success thresholds or physics, enable oracle/reference/recovery paths,
run multiple seeds, or begin Phase 4.

