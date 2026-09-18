# Phase 4-A Infrastructure Recovery Audit

## Scope

- Audited baseline: `f569d15e672fcdcc7bd43feaecd1e69c4146dfb3` (`main` before recovery work)
- Phase 3/Phase 4 harness reference: `0b03cafbacdbef2546aa5ff821ac12d2c4058007`
- Comparison: `git diff --no-ext-diff 0b03caf..f569d15 -- <file>`
- Remote state at audit start: local `main` and `origin/main` both resolved to `f569d15`.

The six requested evaluator files had no byte-level differences between the two commits. The only changes after `0b03caf` and before this recovery were the V1 Pilot report plus Phase 4 analysis/harness updates:

- `PHASE4_PILOT_REPORT.md`
- `tests/generalization/test_analysis.py`
- `tools/generalization/analysis.py`
- `tools/generalization/run_pilot.py`

No Phase 3 evaluator file was checked out or mechanically overwritten.

## File Classification

| File | `0b03caf` vs baseline `main` | Classification | Audit result |
|---|---|---|---|
| `tools/evaluation/bootstrap.py` | Identical | Correct Phase 3 closeout behavior | Exposes only the repository `src` root; no V5/vendor path injection. |
| `tools/evaluation/audit.py` | Identical | Correct Phase 3 closeout behavior | `verify_v7_chain` remains fail-closed on strict Vision, Vision invalid frames, oracle/reference/recovery use, incomplete Skill exercise, and runtime-purity violations. |
| `tools/evaluation/episode_runner.py` | Identical | Correct Phase 3 closeout behavior | Captures source provenance before launch, installs the V5 import blocker before Isaac, audits runtime purity after execution, and passes measured purity/provenance/call counts into result construction. |
| `tools/evaluation/result_writer.py` | Identical | Correct Phase 3 closeout behavior | Serializes measured physical success, runtime purity, provenance, strict Vision telemetry, reference calls, and recovery calls. |
| `tools/evaluation/runtime_purity.py` | Identical | Correct Phase 3 closeout behavior | Detects V5 modules and vendor paths and keeps the active V5 import blocker auditable. |
| `tools/evaluation/provenance.py` | Identical | Correct Phase 3 closeout behavior | Preserves the clean-before-run source snapshot separately from post-run Git state and records checkpoint/lock hashes. |

## Invariant Verification

The pre-change Phase 3 purity suite passed: `21 passed` in `tests/evaluation/test_runtime_purity.py`.

The audited baseline satisfies:

- Formal bootstrap is V7-only and exposes `repo/src` only.
- Formal runtime does not import `stage_vla` or expose `vendor/stage_vla_v5`.
- The V5 import blocker is installed before Isaac launch.
- Runtime purity and source provenance are measured rather than hard-coded.
- Strict Vision, zero oracle fallback, zero reference calls, and zero recovery calls are required by `verify_v7_chain`.
- `v7_chain_verified` remains fail-closed when any Phase 3 closeout gate is false.

## Recovery Changes

No baseline regression required restoration in `bootstrap.py`, `audit.py`, `runtime_purity.py`, or `provenance.py`.

Phase 4-A makes additive changes to `episode_runner.py` and `result_writer.py` only where required for per-environment strict Vision isolation and independent outcomes. Those changes preserve the Phase 3 call order and audit inputs. Supporting lifecycle, action masking, telemetry, analysis, and tests are implemented in dedicated modules rather than replacing or duplicating the evaluator control loop.

## Conclusion

- Correct Phase 3 closeout behavior: present and retained.
- Legitimate later Phase 4 additions: V1 report/analysis/harness changes and the additive Phase 4-A isolation path.
- Regression requiring repair: none found in the six audited files.
- Mechanical checkout from `0b03caf`: not performed.
