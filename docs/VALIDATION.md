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
78 passed
```

Coverage includes dependency-free contracts, strict import boundaries,
Chinese/English/DSL parsing, provider registries, routing and safety, all eight
Skill definitions, success/failure predicates, temporary TorchScript loading,
hash mismatch rejection, simulation model/scene/environment registries, seeded
randomization, Isaac camera/observation/action/runtime adapters, compatibility
imports, and the complete Chinese-to-eight-Skill adapter chain.

`python -m compileall -q src tests scripts tools` also passed. Ruff could not be
run because it was not installed in either available Python environment.
Standalone Python 3.11 imported all 136 `stage_vla_v7` modules without an
Isaac launcher, confirming that simulator startup is not an import-time core
dependency.

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
was exactly `0.0`. Full actions and hashes are in
`evidence/checkpoint_compatibility.json`.

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

## Claim boundary

This proves module wiring, legacy checkpoint equivalence, one successful
physical episode, and the unbroken V7 audit gate. It does not claim new trained
weights, complete visual state estimation, or a multi-seed success rate.
