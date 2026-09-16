# Stage VLA V7

Stage VLA V7 is an auditable vision-language-action stack for Isaac Lab. It
keeps vision, language, action, and orchestration independent while providing a
real simulator bridge that routes every learned action through the same V7
pipeline.

```text
Chinese/English/DSL command -> LanguageService -> TaskPlan
Isaac Lab RGB-D frame       -> VisionService   -> SceneState
TaskPlan + SceneState + robot observation
                             -> StageVLAPipeline.act
                             -> ActionService / domain router
                             -> per-skill TorchScript policy
                             -> safe 5-D RobotAction
                             -> Isaac Lab adapter
```

## What is included

- Dependency-free immutable V7 contracts.
- Deterministic Chinese, English, and stack-DSL language provider.
- Static, callable, and legacy RGB-D vision adapters.
- Callable and TorchScript action policies with fail-closed physical routing.
- Eight-skill stack scheduler: `REACH`, `GRASP`, `LIFT`, `TRANSPORT`,
  `ALIGN`, `DESCEND`, `RELEASE_STABILIZE`, and `RETREAT`.
- `PipelineActionSource`, which converts Isaac Lab batches into audited calls
  to `StageVLAPipeline.act`.
- The V5 simulator application runtime vendored under `vendor/stage_vla_v5` so
  the repository contains the complete source needed by the evaluation tool.
- A frozen one-environment layout and a reproducible physical smoke runner.

Model checkpoints, Isaac Sim, and Isaac Lab are runtime artifacts and are not
committed. Their expected hashes are recorded in `config/artifacts.lock.json`.

## Unit check

```powershell
$env:PYTHONPATH = "E:\stage_vla_v7\src"
E:\work\IsaacLab\_isaac_sim\python.bat -m pytest E:\stage_vla_v7\tests -q
E:\work\IsaacLab\_isaac_sim\python.bat -m stage_vla_v7 `
  --command "把红色方块放到蓝色方块上"
```

## Physical VLA check

```powershell
E:\stage_vla_v7\scripts\run_v7_smoke.ps1
```

The runner enables `--require_v7_chain`. A result can only claim
`v7_chain.verified=true` when RGB-D uses the V7 vision service, language
planning succeeds, all eight prepared skill tokens execute through the V7
pipeline, no reference skill is selected, and no reference recovery is used.

The verified local run completed 1/1 without a simulator reset. See
`docs/VALIDATION.md` and `evidence/v7_vla_smoke_seed61081.summary.json`.

## Layout

```text
src/stage_vla_v7/        V7 contracts, services, adapters, orchestration
tools/                   Isaac Lab evaluation and report tools
scripts/                 Reproducible Windows runners
tests/                   Unit and boundary tests
config/                  Component, artifact, and frozen-layout manifests
evidence/                Small committed verification records
vendor/stage_vla_v5/     Source-only migrated simulator runtime
```

The migrated checkpoints remain V5-trained artifacts. V7 claims orchestration
and end-to-end integration, not newly trained policy weights or a learned LLM.
