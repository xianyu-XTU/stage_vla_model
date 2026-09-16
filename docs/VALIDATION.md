# V7 validation record

Validation date: 2026-09-16

## Environment

- Isaac Lab root: `E:\work\IsaacLab`
- Isaac Sim Python: 3.12.13
- PyTorch: 2.10.0+cu128
- GPU: NVIDIA GeForce RTX 4060
- Simulator device: `cuda:0`
- Physics/control step: 0.01 s / 0.05 s

## Automated tests

```text
99 passed
```

Coverage includes dependency-free contracts, strict import boundaries,
Chinese/English/DSL parsing, provider registries, routing and safety, all eight
Skill definitions, success/failure predicates, temporary TorchScript loading,
hash mismatch rejection, simulation model/scene/environment registries, seeded
randomization, Isaac camera/observation/action/runtime adapters, dual-camera
isolation, strict/best-effort recording, overlays, compatibility imports, and
the complete Chinese-to-eight-Skill adapter chain.

`python -m compileall -q src tests scripts tools` also passed. Ruff could not be
run because it was not installed in either available Python environment.
Isaac Python imported the package root and all 144 discovered
`stage_vla_v7` submodules (145 modules total) without an Isaac launcher
or simulator process. Bare Python 3.11, which lacks the optional NumPy
recording dependency, still imports top-level `stage_vla_v7` successfully.
This confirms that simulator startup and recording packages are not
import-time core dependencies.

## Checkpoint compatibility

The verifier loaded the eight artifacts at
`E:\stage_vla_v5\outputs\v5_generalized_cube_bc_v1`, checked every SHA256 from
`config/artifacts.lock.json`, and used one deterministic input with each frozen
observation dimension.

```text
direct V5 TorchScript -> frozen SafetyProjector
refactored checkpoint loader -> ActionRouter -> ActionService -> SafetyProjector
```

All eight comparisons passed at tolerance `1e-7`; every maximum absolute error
was exactly `0.0` before and after Phase 2. Full actions and hashes are in
`evidence/phase2_checkpoint_compatibility_baseline.json` and
`evidence/phase2_checkpoint_compatibility_post.json`.

## Chinese CLI

`python -m stage_vla_v7 --command "把红色方块放到蓝色方块上"` produced one
`red_cube -> blue_cube` relation and the ordered Skills `REACH`, `GRASP`,
`LIFT`, `TRANSPORT`, `ALIGN`, `DESCEND`, `RELEASE_STABILIZE`, `RETREAT`.

The Windows smoke runner passes the same command as ASCII Base64 and strictly
decodes UTF-8 in Python. This avoids Windows PowerShell 5.1 changing non-ASCII
arguments while preserving the ordinary `--command` option.

## Post-refactor physical smoke

Canonical runner:

```powershell
scripts\smoke\run_v7_smoke.ps1 `
  -Output outputs\v7_vla_smoke_refactor_seed61081.json
```

The frozen `config/simulation/smoke_layout_seed61081.json` scene used:

- the Chinese command above through V7 Language;
- 128 x 128 RGB-D detection through V7 Vision;
- `PipelineActionSource` from `stage_vla_v7.simulation.isaac_lab`;
- all eight real TorchScript policies through
  `StageVLAPipeline.act -> ActionService`;
- `config/artifacts.lock.json` enforced before policy execution;
- centralized vectorized success evaluation adapter;
- no reference Skills and no recovery controller;
- `--require_v7_chain` enabled.

| Check | Result |
|---|---:|
| Physical chain | 1/1 passed |
| V7 chain gate | passed |
| RGB-D service calls | 729 |
| Invalid vision frames | 0 |
| Mid-episode resets | 0 |
| Prepared Skill tokens exercised | 8/8 |
| Inter-Skill handoffs preserving state | 7/7 |

Per-Skill inference rows were REACH 374, GRASP 31, LIFT 60, TRANSPORT 101,
ALIGN 32, DESCEND 20, RELEASE_STABILIZE 7, and RETREAT 102. The compact record
is `evidence/v7_vla_smoke_refactor_seed61081.summary.json`; the full diagnostic
JSON remains under ignored `outputs/`.

After moving concrete environment construction into
`simulation/isaac_lab/env_factory.py`, the same physical smoke passed 1/1 with
731 V7 Vision calls, zero invalid frames, 8/8 Skills, 7/7 state-exact handoffs,
no reference or recovery calls, and `v7_chain.verified=true`. Its full local
output is `outputs/phase2_post_env_factory_seed61081.json`.

After decomposing the evaluator into focused modules, a fresh physical RGB-D
run again passed 1/1 with 729 Vision calls, zero invalid frames, 8/8 Skills,
7/7 state-exact handoffs, no reference or recovery calls, and
`v7_chain.verified=true`. Its ignored full output is
`outputs/phase2_post_evaluator_split_seed61081.json`.

## Vision and observer-video smoke

The following real Isaac run enabled RGB-D Vision, strict observer recording,
the full learned eight-Skill chain, the V7 audit gate, and headless rendering
at the same time:

```powershell
scripts\smoke\run_v7_vision_video_smoke.ps1 `
  -IsaacLabRoot E:\work\IsaacLab `
  -ArtifactRoot E:\stage_vla_v5\outputs\v5_generalized_cube_bc_v1 `
  -Output outputs\phase2_post_evaluator_split_vision_video_seed61081.json `
  -VideoPath outputs\phase2_post_evaluator_split_vision_video_seed61081.mp4
```

| Check | Result |
|---|---:|
| Physical task | 1/1 passed |
| V7 chain gate | passed |
| Prepared Skill tokens | 8/8 |
| State-exact handoffs | 7/7 |
| V7 Vision calls / invalid frames | 729 / 0 |
| Reference Skills / recovery calls | 0 / 0 |
| Observer frames | 767 |
| Observer output | 640 x 480, 20 FPS, 38.35 s |
| Observer used for Vision | false |

Imageio reopened the post-evaluator-split MP4, counted all 767 frames, and
reported 20 FPS, a 640 x 480 stream, and a `480 x 640 x 3` decoded frame. The
MP4 was 9,936,276 bytes with SHA256
`2864b2f7a94891b4953a2818471f8609aed925029b79af5bfa73478fc472ba6b`.
The tracked compact record is
`evidence/phase2_vision_video_physical.json`; full output and media stay under
ignored `outputs/`; the source result was
`outputs/phase2_post_evaluator_split_vision_video_seed61081.json` with SHA256
`4e1030815c462e80ae2814d8ba0ba546da05817ea17e3bd3825402e5be92dbe0`.

## Reproducible validation commands

Run from the repository root in PowerShell:

```powershell
# 1. Dependency boundaries
E:\work\IsaacLab\_isaac_sim\python.bat -m pytest tests\test_dependency_boundaries.py -q

# 2. Complete unit/compatibility suite
E:\work\IsaacLab\_isaac_sim\python.bat -m pytest tests -q

# 3. Checkpoint compatibility
E:\work\IsaacLab\_isaac_sim\python.bat `
  tools\migration\verify_checkpoint_compatibility.py `
  --artifact-root E:\stage_vla_v5\outputs\v5_generalized_cube_bc_v1 `
  --output evidence\phase2_checkpoint_compatibility_post.json

# 4. Dependency-light Simulation tests
E:\work\IsaacLab\_isaac_sim\python.bat -m pytest tests\simulation -q

# 5. Physical RGB-D V7 chain smoke
scripts\smoke\run_v7_smoke.ps1 `
  -IsaacLabRoot E:\work\IsaacLab `
  -Output outputs\v7_chain.json

# 6. Physical RGB-D + Observer MP4 V7 chain smoke
scripts\smoke\run_v7_vision_video_smoke.ps1 `
  -IsaacLabRoot E:\work\IsaacLab `
  -Output outputs\v7_vision_video.json `
  -VideoPath outputs\v7_vision_video.mp4
```

`--headless` remains compatible with the installed Isaac Lab and produced the
verified MP4, although the launcher warns that the flag is deprecated. For
future direct evaluator commands, prefer `--viz none` when supported.

## Claim boundary

This proves module wiring, legacy checkpoint equivalence, one successful
physical episode with two isolated cameras, a decodable observer video, and the
unbroken V7 audit gate. It does not claim new trained weights, complete visual
state estimation, complete removal of V5 runtime code, or a multi-seed success
rate.
