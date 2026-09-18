# Phase 4-A Random Layout Generalization Pilot

Evaluation date: 2026-09-18.

## A. Experiment Configuration

- Physical source commit: `0b03cafbacdbef2546aa5ff821ac12d2c4058007`.
- Layout manifest: `evaluation/generalization/phase4_layouts_v1_20.json`.
- Layout seed/count: `42024`, 20 frozen layouts.
- Batch size: 4 (five sequential Isaac processes).
- Artifact-lock SHA-256:
  `AED7B1DE2CFB610C086046B85797B3F933C2166EDD5D5520B1283812D62F4C2D`.
- All eight locked checkpoint hashes were verified by the evaluator.
- Vision policy: strict RGB-D; no debug oracle.
- Action policy: all eight learned checkpoints; no reference Skill or recovery.
- `--require_v7_chain` enabled; batch video disabled.
- Failure replays: the identical frozen layout, `num_envs=1`, trace and strict
  observer video enabled.

The manifest uses the current evaluator's frozen randomization contract: X in
`[0.4, 0.6]`, Y in `[-0.1, 0.1]`, Z equal to `0.0203 m`, and minimum pairwise
XY separation `0.1 m`. All object, support, and distractor coordinates passed
finite, workspace, fixed-Z, row-count, and non-overlap validation.

## B. Overall Results

| Metric | Result |
|---|---:|
| Evaluated layouts | 20 / 20 |
| Batch fail-closed physical success | 4 / 20 (20.0%) |
| Physical Wilson 95% CI | [8.1%, 41.6%] |
| Batch fail-closed stable success | 4 / 20 (20.0%) |
| Stable Wilson 95% CI | [8.1%, 41.6%] |
| V7-chain verified | 8 / 20 |
| Strict Vision policy enabled | 20 / 20 |
| Vision-valid batch cases | 17 / 20 |
| Runtime purity verified | 20 / 20 |
| Oracle fallback / reference / recovery | 0 / 0 / 0 |
| Replay-resolved stable success | 12 / 20 |
| Replay-resolved strict Vision failure | 8 / 20 |

Three batches were aborted by one strict Vision invalid detection each. The
nine peer environments in those batches have `RUNTIME_ERROR` as their batch
first failure because the evaluator correctly failed the whole process closed;
the peers did not themselves record a Vision miss. The batch 4/20 rate therefore
includes peer-abort propagation and is not an unbiased policy-only success-rate
estimate. The Wilson interval describes that measured batch outcome only.

Replacing only failed batch rows with their required single-environment replay
gives 12 successes and 8 strict Vision failures. This is a diagnostic resolution,
not a rewrite of the original result.

## C. Skill Funnel

| Skill | Entered | Success | Conditional success |
|---|---:|---:|---:|
| REACH | 8 | 8 | 100.0% |
| GRASP | 8 | 8 | 100.0% |
| LIFT | 8 | 8 | 100.0% |
| TRANSPORT | 8 | 8 | 100.0% |
| ALIGN | 8 | 5 | 62.5% |
| DESCEND | 5 | 4 | 80.0% |
| RELEASE_STABILIZE | 4 | 4 | 100.0% |
| RETREAT | 4 | 4 | 100.0% |

The funnel contains only telemetry actually produced before each fail-closed
termination. It does not treat unentered downstream Skills as failures.

## D. First Failure

| First failure | Count |
|---|---:|
| REACH | 0 |
| GRASP | 0 |
| LIFT | 0 |
| TRANSPORT | 0 |
| ALIGN | 3 |
| DESCEND | 1 |
| RELEASE_STABILIZE | 0 |
| RETREAT | 0 |
| VISION | 3 |
| FINAL_STABILITY | 0 |
| RUNTIME_ERROR (peer batch abort) | 9 |
| UNKNOWN | 0 |

## E. Failure Taxonomy

| Failure type | Count |
|---|---:|
| VISION_FAILURE | 3 |
| TIMEOUT | 4 |
| RUNTIME_ERROR (peer batch abort) | 9 |
| All other defined taxonomy values | 0 |

The three ALIGN rows and one DESCEND row ended with explicit timeout telemetry.
No geometry subtype was inferred after timeout because that would overstate the
available evidence.

## F. Layout Table

| Layout | Object XY | Support XY | Distance | Batch result | First failure |
|---|---|---|---:|---|---|
| layout_001 | (0.4003, -0.0387) | (0.5376, -0.0065) | 0.1410 | FAIL | RUNTIME_ERROR |
| layout_002 | (0.4548, 0.0110) | (0.5660, -0.0103) | 0.1132 | FAIL | RUNTIME_ERROR |
| layout_003 | (0.5855, -0.0993) | (0.4377, -0.0434) | 0.1581 | FAIL | VISION |
| layout_004 | (0.5460, -0.0694) | (0.4238, -0.0263) | 0.1296 | FAIL | RUNTIME_ERROR |
| layout_005 | (0.5768, 0.0817) | (0.5029, -0.0983) | 0.1945 | FAIL | VISION |
| layout_006 | (0.4445, -0.0310) | (0.5540, 0.0830) | 0.1580 | FAIL | RUNTIME_ERROR |
| layout_007 | (0.5429, -0.0858) | (0.4551, 0.0765) | 0.1845 | FAIL | RUNTIME_ERROR |
| layout_008 | (0.5102, -0.0557) | (0.4873, 0.0474) | 0.1056 | FAIL | RUNTIME_ERROR |
| layout_009 | (0.4181, 0.0854) | (0.4258, -0.0842) | 0.1698 | FAIL | ALIGN |
| layout_010 | (0.4635, 0.0662) | (0.4516, -0.0410) | 0.1078 | FAIL | ALIGN |
| layout_011 | (0.4401, -0.0798) | (0.4366, 0.0471) | 0.1270 | PASS | - |
| layout_012 | (0.5932, -0.0952) | (0.4639, 0.0180) | 0.1718 | PASS | - |
| layout_013 | (0.5623, -0.0467) | (0.4125, -0.0458) | 0.1498 | FAIL | RUNTIME_ERROR |
| layout_014 | (0.5727, 0.0757) | (0.4212, 0.0859) | 0.1519 | FAIL | RUNTIME_ERROR |
| layout_015 | (0.4229, 0.0637) | (0.5668, -0.0659) | 0.1936 | FAIL | RUNTIME_ERROR |
| layout_016 | (0.5148, 0.0009) | (0.5676, 0.0935) | 0.1065 | FAIL | VISION |
| layout_017 | (0.5723, -0.0163) | (0.4073, 0.0406) | 0.1746 | FAIL | DESCEND |
| layout_018 | (0.4776, 0.0657) | (0.5373, -0.0526) | 0.1325 | FAIL | ALIGN |
| layout_019 | (0.4274, 0.0857) | (0.5551, -0.0940) | 0.2204 | PASS | - |
| layout_020 | (0.4368, -0.0260) | (0.5685, 0.0455) | 0.1499 | PASS | - |

The machine-readable aggregate contains full XYZ coordinates, relative dx/dy,
distance, per-Skill rows, episode steps, and position-conditioned buckets.

## G. Replay and Determinism

All 16 failed batch cases were replayed with the same manifest row. Every MP4
independently decoded as nonblank.

| Layout | Batch failure | Single-env result | Reproduced exactly |
|---|---|---|---:|
| layout_001 | RUNTIME_ERROR | PASS | no |
| layout_002 | RUNTIME_ERROR | PASS | no |
| layout_003 | VISION | VISION | yes |
| layout_004 | RUNTIME_ERROR | VISION | no |
| layout_005 | VISION | VISION | yes |
| layout_006 | RUNTIME_ERROR | VISION | no |
| layout_007 | RUNTIME_ERROR | PASS | no |
| layout_008 | RUNTIME_ERROR | VISION | no |
| layout_009 | ALIGN | PASS | no |
| layout_010 | ALIGN | PASS | no |
| layout_013 | RUNTIME_ERROR | VISION | no |
| layout_014 | RUNTIME_ERROR | VISION | no |
| layout_015 | RUNTIME_ERROR | PASS | no |
| layout_016 | VISION | PASS | no |
| layout_017 | DESCEND | VISION | no |
| layout_018 | ALIGN | PASS | no |

Exact first-failure reproduction was 2/16. Single-env replay outcomes were
8/16 PASS and 8/16 strict Vision failure. This is severe batch/single outcome
variation, even though two focused determinism probes were internally stable:

- Successful `layout_011`: batch PASS and two repeated single runs PASS.
- Failing `layout_003`: batch VISION and two repeated single runs VISION.

## H. Conclusion

The dominant Pilot limitation is strict Vision invalid detection plus
process-wide fail-closed propagation across parallel environments. Among the
eight cases with complete Skill telemetry, ALIGN is the policy-side bottleneck
at 5/8 conditional success, followed by DESCEND at 4/5. There is no evidence
of a runtime-purity, oracle, reference, recovery, checkpoint, or GPU-memory
violation.

The infrastructure is not ready for a 50-case run because 3/5 batches aborted,
only 17/20 cases were Vision-valid in batch, and batch/single attribution was
not stable. Phase 4-A stops here. No Skill, threshold, reward, policy, checkpoint,
Vision fallback, or physical task definition was changed.

## Acceptance Table

```text
Source commit                       0b03cafbacdbef2546aa5ff821ac12d2c4058007
Layout manifest                    evaluation/generalization/phase4_layouts_v1_20.json
Layout count                       20
Batch size                         4

Strict Vision policy               PASS
Vision-valid cases                 17 / 20
Runtime purity                     PASS (20 / 20)
Oracle fallback                    0
Reference calls                    0
Recovery calls                     0

Evaluated cases                    20 / 20
Physical success                   4 / 20
Physical success rate              20.0%
Stable success                     4 / 20
Stable success rate                20.0%
95% Wilson CI                      [8.1%, 41.6%]

REACH success                      8 / 8
GRASP success                      8 / 8
LIFT success                       8 / 8
TRANSPORT success                  8 / 8
ALIGN success                      5 / 8
DESCEND success                    4 / 5
RELEASE_STABILIZE success          4 / 4
RETREAT success                    4 / 4

First failure:
REACH                              0
GRASP                              0
LIFT                               0
TRANSPORT                          0
ALIGN                              3
DESCEND                            1
RELEASE_STABILIZE                  0
RETREAT                            0
VISION                             3
FINAL_STABILITY                    0
UNKNOWN                            0
RUNTIME_ERROR                      9

Failed cases replayed              16 / 16
Single-env replay failures         8 / 16
Failures reproduced exactly        2 / 16
Replay videos valid                16 / 16

Evaluation infrastructure          FAIL
READY_FOR_PHASE4_50                false
```

Raw batches, aggregate JSON, failure JSON, replay results, traces, logs, videos,
and determinism records are stored outside the repository under the run's
`outputs/phase4_pilot` directory.
