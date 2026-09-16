# V7 validation record

Validation date: 2026-09-16

## Environment

- Isaac Lab root: `E:\work\IsaacLab`
- Isaac Sim Python: 3.12.13
- PyTorch: 2.10.0+cu128
- GPU: NVIDIA GeForce RTX 4060
- Simulator device: `cuda:0`
- Physics/control step: 0.01 s / 0.05 s
- Checkpoints: `E:\stage_vla_v5\outputs\v5_generalized_cube_bc_v1`

## Automated regression

```text
270 passed
```

The suite covers public contracts, dependency boundaries, parsing and provider
registries, eight Skill definitions, checkpoint routing, action safety and
output parity, native success predicates, strict Vision failure handling,
RGB/depth/calibration/detector parity, typed physical state, exact 52-D and 55-D
observation parity, gripper/control/geometry/reward/profile/snapshot helpers,
evaluation collection, the public environment boundary, camera isolation, and
recording behavior.

`python -m compileall -q src tests scripts tools` passes. The dependency guard
also confirms that the evaluator neither calls `env._...` nor directly mutates
protected environment bookkeeping.

## Runtime dependency audit

The Phase 2 baseline directly imported 19 V5 Python module paths. The final
physical evaluator call graph imports none. A single lazy source import remains
inside `legacy_vectorized_skill_success`; it is a regression-only oracle and is
never selected by the physical runtime. Parity tests intentionally import V5
implementations from `vendor/stage_vla_v5`.

See `docs/VENDOR_MIGRATION_PHASE3.md` for the complete module-by-module audit.

## Checkpoint compatibility

The verifier loaded all eight locked artifacts, checked their SHA-256 values,
and compared direct TorchScript plus frozen safety with the V7
`ActionRouter -> ActionService -> SafetyProjector` path.

| Skill | Observation dim | Action dim | Max absolute error |
|---|---:|---:|---:|
| REACH | 52 | 5 | 0.0 |
| GRASP | 55 | 5 | 0.0 |
| LIFT | 55 | 5 | 0.0 |
| TRANSPORT | 55 | 5 | 0.0 |
| ALIGN | 55 | 5 | 0.0 |
| DESCEND | 55 | 5 | 0.0 |
| RELEASE_STABILIZE | 55 | 5 | 0.0 |
| RETREAT | 55 | 5 | 0.0 |

The machine-readable result is
`evidence/phase3_checkpoint_compatibility.json`.

## Final physical smoke

The final run used the frozen seed-61081 layout, strict RGB-D Vision, all eight
real policies, the required V7 audit gate, the Observer camera, strict MP4
recording, and no reference or recovery controller.

```powershell
scripts\smoke\run_v7_smoke.ps1 -IsaacLabRoot E:\work\IsaacLab -ArtifactRoot E:\stage_vla_v5\outputs\v5_generalized_cube_bc_v1 -Calibration E:\stage_vla_v5\outputs\vision_rgbd_mapping_calibration_train_20260910.json -Output outputs\phase3_p3_public_env_api_seed61081.json -Video -VideoPath outputs\phase3_p3_public_env_api_seed61081.mp4 -Headless
```

| Check | Result |
|---|---:|
| Physical task | 1/1 passed |
| V7 chain gate | passed |
| Prepared Skill tokens | 8/8 |
| State-exact handoffs | 7/7 |
| Mid-episode resets | 0 |
| V7 Vision calls / invalid frames | 730 / 0 |
| Oracle fallback count | 0 |
| Reference Skills / recovery calls | 0 / 0 |
| Observer frames | 768 |
| Observer output | 640x480, 20 FPS, 38.4 s |
| Observer used for Vision | false |

Per-Skill inference rows were REACH 374, GRASP 31, LIFT 60, TRANSPORT 101,
ALIGN 32, DESCEND 20, RELEASE_STABILIZE 7, and RETREAT 103. Every Skill passed
for the single validation environment.

## Independent video decode

OpenCV reopened the completed MP4 and decoded the entire stream independently
of the recorder:

```text
decoded frames: 768 / 768
resolution: 640 x 480
minimum frame standard deviation: 62.16835394099387
maximum frame standard deviation: 67.63165749786386
decoded pixel SHA-256:
a4a94a8f7511c4c36220f9a43a830dfa6e8d49b6253f54915418d4e2a4c93d71
```

The positive per-frame standard-deviation floor rejects an all-black or blank
frame. This decode check is independent of the frame count written into the
evaluation JSON.

## Reproducible commands

Run from the repository root in PowerShell:

```powershell
# Dependency boundaries and public environment boundary
E:\work\IsaacLab\_isaac_sim\python.bat -m pytest tests\test_dependency_boundaries.py tests\evaluation\test_evaluator_structure.py

# Complete regression suite
E:\work\IsaacLab\_isaac_sim\python.bat -m pytest tests

# Syntax/bytecode validation
E:\work\IsaacLab\_isaac_sim\python.bat -m compileall -q src tests scripts tools

# Eight-checkpoint compatibility
E:\work\IsaacLab\_isaac_sim\python.bat tools\migration\verify_checkpoint_compatibility.py --artifact-root E:\stage_vla_v5\outputs\v5_generalized_cube_bc_v1 --lock config\artifacts.lock.json --output evidence\phase3_checkpoint_compatibility.json
```

## Claim boundary

This proves Phase 3 runtime purity for the locked cube-policy path, exact
checkpoint/action compatibility, preserved physical handoffs, strict Vision,
and a complete observer recording for one seed. It does not claim new weights,
non-cube policy support, fully visual proprioception/contact feedback, or a
multi-seed success rate.
