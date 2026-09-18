# Phase 4-A Infrastructure Recovery and Pilot V2

Evaluation date: 2026-09-18.

## Evidence and Configuration

- Source commit: `b33e1ed43dd105647b090248c40a676431a84fd3`.
- Source worktree clean before Pilot: `true`.
- Final raw evidence: `outputs/phase4_pilot_v2_final` outside the repository.
- The earlier `outputs/phase4_pilot_v2` attempt is preserved but excluded. Its
  replay rows exposed the fixed contradiction between `VISION_FAILURE` and
  `stable_success=true`.
- Frozen manifest: `evaluation/generalization/phase4_layouts_v1_20.json`.
- Manifest SHA-256:
  `E024B2E79205576FDB60879B5BCEA468F20875B453B1420532ED4DFBA755CD2C`.
- Manifest seed/count: `42024`, 20 layouts.
- Batch size: 4, in five sequential Isaac processes.
- Strict RGB-D Vision; no Vision retry, debug oracle, reference Skill, or recovery.
- The same eight frozen checkpoint hashes, cameras, physics, task, and Skill
  thresholds as V1 were retained and verified by the evaluator.
- Every failed batch case was replayed with the same layout and seed using
  `num_envs=1`, strict Vision, trace, observer video, and `--require_v7_chain`.

## Recovery Validation

The Phase 3 comparison is documented in `PHASE4_INFRA_RECOVERY_AUDIT.md`. No
regression was found in the six requested evaluator files relative to `0b03caf`;
the V7-only bootstrap, V5 import blocker, measured runtime-purity audit, and
fail-closed V7-chain gates remain active.

| Check | Result |
|---|---:|
| Runtime-purity tests | 21 passed |
| Per-environment Vision isolation tests | 8 passed |
| Full `tests/evaluation` | 92 passed |
| Full `tests/generalization` | 8 passed |
| Full pytest | 308 passed |
| `compileall` | PASS |
| `git diff --check` | PASS |
| `ruff` | unavailable in the Isaac Python environment |

The final run produced 19 result files and 34 environment-outcome rows across
batches, replays, and determinism checks. Independent post-run validation found:

- zero PASS rows with a first failure or non-PASS outcome;
- zero failed rows with `physical_success` or `stable_success` set to true;
- zero source-provenance mismatches or dirty-before-run records;
- zero runtime-purity failures, loaded V5 modules, or exposed vendor paths; and
- zero oracle, reference, or recovery calls.

## Pilot Results

| Metric | V2 result |
|---|---:|
| Independently accounted layouts | 20 / 20 |
| Physical success | 9 / 20 (45.0%) |
| Stable success | 9 / 20 (45.0%) |
| Stable-success Wilson 95% CI | [25.8%, 65.8%] |
| V7-chain valid | 13 / 20 |
| Strict Vision enabled | 20 / 20 |
| Vision-valid | 13 / 20 |
| Runtime-pure | 20 / 20 |
| Peer-aborted | 0 / 20 |
| VISION_FAILURE | 7 |
| TIMEOUT | 4 |
| RUNTIME_ERROR | 0 |
| GLOBAL_RUNTIME_ERROR | 0 |
| Oracle / reference / recovery calls | 0 / 0 / 0 |

## V1 vs V2

| Metric | V1 | V2 |
|---|---:|---:|
| Stable success | 4 / 20 | 9 / 20 |
| Vision failures | 3 | 7 |
| RUNTIME_ERROR | 9 | 0 |
| Peer aborts | 9 | 0 |
| Batch/replay exact agreement | 2 / 16 | 6 / 11 |
| ALIGN conditional success | 62.5% | 76.9% |
| DESCEND conditional success | 80.0% | 90.0% |

V1's nine peer `RUNTIME_ERROR` rows were infrastructure contamination. V2
independently accounts for all 20 cases and reduces both peer aborts and runtime
errors to zero. The increase from three to seven Vision failures reflects direct
per-environment attribution rather than failure propagation to healthy peers.

## Skill Funnel

| Skill | Entered | Success | Conditional success |
|---|---:|---:|---:|
| REACH | 20 | 16 | 80.0% |
| GRASP | 16 | 16 | 100.0% |
| LIFT | 16 | 16 | 100.0% |
| TRANSPORT | 14 | 14 | 100.0% |
| ALIGN | 13 | 10 | 76.9% |
| DESCEND | 10 | 9 | 90.0% |
| RELEASE_STABILIZE | 9 | 9 | 100.0% |
| RETREAT | 9 | 9 | 100.0% |

First failures were Vision 7, ALIGN 3, and DESCEND 1. All other Skill,
final-stability, runtime-error, and unknown buckets were zero.

## Layout Outcomes

| Layout | Result | First failure | Taxonomy |
|---|---|---|---|
| layout_001 | PASS | - | PASS |
| layout_002 | PASS | - | PASS |
| layout_003 | FAIL | VISION | VISION_FAILURE |
| layout_004 | FAIL | VISION | VISION_FAILURE |
| layout_005 | FAIL | VISION | VISION_FAILURE |
| layout_006 | FAIL | VISION | VISION_FAILURE |
| layout_007 | PASS | - | PASS |
| layout_008 | FAIL | VISION | VISION_FAILURE |
| layout_009 | FAIL | ALIGN | TIMEOUT |
| layout_010 | FAIL | ALIGN | TIMEOUT |
| layout_011 | PASS | - | PASS |
| layout_012 | PASS | - | PASS |
| layout_013 | FAIL | DESCEND | TIMEOUT |
| layout_014 | FAIL | VISION | VISION_FAILURE |
| layout_015 | PASS | - | PASS |
| layout_016 | FAIL | VISION | VISION_FAILURE |
| layout_017 | PASS | - | PASS |
| layout_018 | FAIL | ALIGN | TIMEOUT |
| layout_019 | PASS | - | PASS |
| layout_020 | PASS | - | PASS |

## Failure Replays and Determinism

All 11 failed cases were replayed, and all 11 videos decoded with every frame
nonblank at 640 x 480 and 20 FPS.

| Layout | Batch first | Replay result | Replay first | Exact match |
|---|---|---|---|---:|
| layout_003 | VISION | FAIL | VISION | yes |
| layout_004 | VISION | FAIL | VISION | yes |
| layout_005 | VISION | FAIL | VISION | yes |
| layout_006 | VISION | FAIL | VISION | yes |
| layout_008 | VISION | FAIL | VISION | yes |
| layout_009 | ALIGN | PASS | - | no |
| layout_010 | ALIGN | PASS | - | no |
| layout_013 | DESCEND | FAIL | VISION | no |
| layout_014 | VISION | FAIL | VISION | yes |
| layout_016 | VISION | PASS | - | no |
| layout_018 | ALIGN | PASS | - | no |

Exact agreement was 6/11. Focused determinism checks were internally stable:

- `layout_001`: batch PASS and two repeated single-env runs PASS.
- `layout_003`: batch VISION failure and two repeated single-env runs VISION failure.

The five remaining batch/single mismatches are too many to treat vectorized and
single-env outcomes as generally interchangeable. They should be investigated
through camera, environment, Vision tensor, Skill state, randomization, success
checker, and action-dispatch indexing before a 50-case run.

## Acceptance

```text
Phase 3 Closeout restored           PASS
Formal bootstrap V7-only            PASS
Runtime purity                      PASS (20 / 20)
Loaded V5 modules                   0
Vendor path exposed                 false

Per-env Vision isolation            PASS
Single-env Vision failure
aborts healthy peers                NO
Peer-abort count                    0

Oracle fallback                     0
Reference calls                     0
Recovery calls                      0

Runtime purity tests                21 passed / 0 failed
Per-env failure tests               8 passed / 0 failed
Evaluation tests                    92 passed / 0 failed
Generalization tests                8 passed / 0 failed
Full pytest                         308 passed / 0 failed
Compileall                          PASS

Source commit                       b33e1ed43dd105647b090248c40a676431a84fd3
Source clean before Pilot           true
Pilot manifest                      phase4_layouts_v1_20.json
Evaluated cases                     20 / 20

Physical success                    9 / 20
Stable success                      9 / 20
Stable success rate                 45.0%
95% Wilson CI                       [25.8%, 65.8%]

VISION_FAILURE                      7
RUNTIME_ERROR                       0
GLOBAL_RUNTIME_ERROR                0

REACH success                       16 / 20
GRASP success                       16 / 16
LIFT success                        16 / 16
TRANSPORT success                   14 / 14
ALIGN success                       10 / 13
DESCEND success                     9 / 10
RELEASE_STABILIZE success           9 / 9
RETREAT success                     9 / 9

Failed cases replayed               11 / 11
Replay videos valid                 11 / 11
Batch/replay matched                6 / 11

Evaluation infrastructure          PASS
READY_FOR_PHASE4_50                 false
```

The infrastructure recovery objective is complete: one environment's strict
Vision failure no longer terminates or misclassifies healthy peers. No training,
checkpoint, threshold, fallback, camera, or task-definition change was made.
Phase 4-A stops here because batch/single consistency is not yet acceptable.
