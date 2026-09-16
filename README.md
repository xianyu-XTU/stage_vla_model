# Stage VLA V7

Stage VLA V7 is an auditable, modular vision-language-action stack. The VLA
core is independent of Isaac Lab; Isaac Lab is one optional Simulation backend.
The refactor preserves the existing V7 behavior, the frozen five-parameter
action convention, V5-trained TorchScript checkpoints, and the
`--require_v7_chain` acceptance gate.

```text
Chinese / English / DSL            Isaac RGB-D
            |                           |
     LanguageService                VisionService
            |                           |
         TaskPlan                    SceneState
             \                         /
              +--- StageVLAPipeline --+
                         |
                 registered SkillToken
                         |
             ActionService / domain router
                         |
               per-skill parameter policy
                         |
        RobotAction[dx, dy, dz, dyaw, grip]
                         |
                Simulation adapter
                         |
                  Isaac Lab / Franka
```

## Canonical modules

- `interfaces`: dependency-free immutable contracts and Protocols.
- `vision`: RGB-D providers, preprocessing boundary, and provider registry.
- `language`: Chinese, English, and stack-DSL parsing plus provider registry.
- `action`: eight independent Skill definitions, routing, policies, training,
  evaluation, and safety projection.
- `orchestration`: task preparation, scheduling, execution context, and audit.
- `simulation`: robot/object/sensor models, scenes, environments, physics,
  randomization, recording, and the Isaac Lab adapter/factory.
- `tools/evaluation`: whole-episode execution and JSON result writing. The
  historical `tools/eval_v7_multicube_chain.py` path is now a small CLI
  compatibility entry.

`contracts`, module-local `interfaces.py`, legacy `adapters`, and
`integrations.isaaclab` remain as compatibility imports. New code should use
the canonical packages above.

Simulation models under `simulation/models` describe physical entities such as
Franka, cubes, and cameras. Learned models live under `vision`, `language`, and
`action/network`; these are deliberately different concepts.

## Configuration and artifacts

Runtime configuration is split by owner:

```text
config/vision/       detector and RGB-D configuration
config/language/     parser/provider configuration
config/action/       policy bundle, dimensions, and action convention
config/simulation/   robot, object, sensor, scene, physics, and layout
config/tasks/        semantic task definitions
```

Large checkpoints and simulator assets remain outside Git.
`config/artifacts.lock.json` records logical paths, SHA256, model type, Skill,
version, observation dimension, action dimension, and training source.

## Unit and compatibility checks

```powershell
$env:PYTHONPATH = (Resolve-Path "src").Path
E:\work\IsaacLab\_isaac_sim\python.bat -m pytest tests\test_dependency_boundaries.py -q
E:\work\IsaacLab\_isaac_sim\python.bat -m pytest tests -q

E:\work\IsaacLab\_isaac_sim\python.bat `
  tools\migration\verify_checkpoint_compatibility.py `
  --artifact-root E:\stage_vla_v5\outputs\v5_generalized_cube_bc_v1 `
  --output evidence\phase2_checkpoint_compatibility_post.json

E:\work\IsaacLab\_isaac_sim\python.bat -m stage_vla_v7 `
  --command "把红色方块放到蓝色方块上"
```

The checkpoint verifier compares the original TorchScript output plus the
frozen safety projection against the refactored `ActionService` path for all
eight skills.

## Physical VLA smoke

```powershell
scripts\smoke\run_v7_smoke.ps1 `
  -IsaacLabRoot E:\work\IsaacLab `
  -Output outputs\v7_chain.json

scripts\smoke\run_v7_vision_video_smoke.ps1 `
  -IsaacLabRoot E:\work\IsaacLab `
  -Output outputs\v7_vision_video.json `
  -VideoPath outputs\v7_vision_video.mp4
```

The legacy `scripts\run_v7_smoke.ps1` path forwards to this runner. A physical
result is accepted only when RGB-D uses V7 Vision, language produces a valid
plan, all eight prepared tokens execute through `ActionService`, no reference
Skill is selected, no recovery controller runs, and
`v7_chain.verified=true`.

The Vision + Video runner creates two independent sensors in the same Isaac
scene. `v7_multicube_camera` supplies 128 x 128 RGB-D only to
`VisionService`; `v7_observer_camera` supplies 640 x 480 RGB only to
`FrameCapture -> VideoRecorder`. Observer frames never enter model input.
Recording defaults to strict failure handling; direct evaluator users may pass
`--recording_best_effort` to warn, close the encoder, and continue control.
The runner currently accepts `--headless`; Isaac Lab reports that flag as
deprecated, so future direct invocations should prefer the installed version's
`--viz none` equivalent when available.

## Documentation

- `ARCHITECTURE.md`: dependency direction and runtime flow.
- `INTERFACES.md`: public contracts and provider boundaries.
- `ACTIONS.md`: all eight Skills, policies, dimensions, and terminal rules.
- `TRAINING.md`: independent BC and retained V5 training adapters.
- `SIMULATION.md`: supported backend, models, scene, and extension process.
- `docs/REFACTORING.md`: before/after trees and migration table.
- `docs/VENDOR_CLASSIFICATION.md`: conservative V5 KEEP/MIGRATE/ADAPT/DELETE audit.
- `docs/VALIDATION.md`: test, checkpoint, CLI, and physical evidence.
- `docs/PHASE2_REFACTOR_AUDIT.md`: source-derived Phase-2 dependency audit.

No retraining or control-algorithm change is part of this refactor.
