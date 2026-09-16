# Refactoring record

## Current architecture audit before changes

The audit was performed against commit `6048395` (`v7.0.0`), using source and
import searches rather than README claims.

- Vision: `vision/service.py`, module-local interfaces, static and legacy
  adapters. The physical evaluator constructed the RGB-D detector.
- Language: `language/service.py`, callable/deterministic adapters, with
  Chinese/English/DSL parsing combined in one implementation.
- Action: service, domain/router, safety, callable/TorchScript/V5 adapters. The
  eight Skills were enum values and policy keys, not independent modules.
- Contracts: `contracts/models.py` held scene, instruction, Skill, object,
  action, and provider data together.
- Orchestration: `pipeline.py` and `catalog.py`; `PreparedTask` was colocated.
- Isaac bridge: `integrations/isaaclab.py` owned
  `PipelineActionSource` and the migrated V5 cube service factory.
- Environment and entity configuration: physical environment, Franka, cube,
  camera, contact, and state-reading code lived in the evaluator and retained
  `vendor/stage_vla_v5` runtime.
- Training/evaluation: V5 training tools and tensor terminal checks remained in
  vendor; the evaluator imported `skill_success` directly.
- Checkpoint loader: TorchScript adapter loaded paths directly without a
  reusable hash-validating boundary.
- Baseline: 15 tests passed. The evaluator was 1862 lines.
- Vendor reachability audit: 75 of 179 Python modules were statically reachable
  from physical evaluation roots; the other 104 were not proven deletable.

## Pre-refactor tracked tree

```text
config/
  artifacts.lock.json
  components.example.json
  smoke_layout_seed61081.json
scripts/
  run_v7_smoke.ps1
src/stage_vla_v7/
  __init__.py
  __main__.py
  cli.py
  contracts/{__init__.py,errors.py,models.py}
  vision/
    __init__.py interfaces.py service.py
    adapters/{__init__.py,legacy.py,static.py}
  language/
    __init__.py interfaces.py service.py
    adapters/{__init__.py,callable.py,deterministic.py}
  action/
    __init__.py domain.py interfaces.py safety.py service.py
    adapters/{__init__.py,callable.py,torchscript.py,v5.py}
  orchestration/{__init__.py,catalog.py,pipeline.py}
  integrations/{__init__.py,isaaclab.py}
  plugins/{__init__.py,registry.py}
tests/
  conftest.py
  test_action.py test_contracts.py test_dependency_boundaries.py
  test_isaaclab_bridge.py test_language.py test_pipeline.py
  test_registry.py test_vision.py
tools/
  eval_v7_multicube_chain.py
  summarize_stack_benchmark.py
vendor/stage_vla_v5/  (179 audited Python modules; unchanged)
```

## Phase-1 post-refactor tree

```text
config/
  action/v5_generalized_cube.json
  language/deterministic.json
  simulation/{red_on_blue.json,smoke_layout_seed61081.json}
  tasks/red_on_blue_stack.json
  vision/default.json
  artifacts.lock.json components.example.json
  smoke_layout_seed61081.json                 # compatibility copy
scripts/
  train/train_skill_bc.py
  evaluate/run_v7_multicube_chain.ps1
  simulation/run_red_on_blue.ps1
  smoke/run_v7_smoke.ps1
  run_v7_smoke.ps1                            # compatibility entry
src/stage_vla_v7/
  interfaces/
    action_interface.py language_interface.py pipeline_interface.py
    simulation_interface.py vision_interface.py errors.py
    contracts/
      _validation.py action.py instruction.py model_descriptor.py
      object_profile.py observation.py scene.py simulation.py skill.py task.py
  vision/
    service.py registry.py
    providers/{legacy_provider.py,static_provider.py}
    adapters/                                  # compatibility imports
  language/
    service.py registry.py
    parser/{chinese.py,dsl.py,english.py,normalization.py}
    providers/{callable.py,deterministic.py}
    adapters/                                  # compatibility imports
  action/
    action_list.py router.py scheduler.py safety.py service.py
    actions/{base.py,reach.py,grasp.py,lift.py,transport.py,align.py,
             descend.py,release_stabilize.py,retreat.py}
    domains/{action_bundle.py,registry.py}
    network/{callable_policy.py,checkpoint_loader.py,parameter_network.py,
             torchscript_policy.py,v5_factory.py}
    training/{bc.py,checkpoint.py,dataset.py,legacy_v5.py,trainer.py}
    evaluation/{metrics.py,report.py,skill_evaluator.py,
                success_checker.py,task_evaluator.py}
    adapters/ domain.py interfaces.py           # compatibility imports
  orchestration/
    pipeline.py prepared_task.py task_scheduler.py
    execution_context.py audit.py catalog.py
  simulation/
    isaac_lab/{action_adapter.py,adapter.py,camera_adapter.py,
               observation_adapter.py,pipeline_action_source.py,runtime.py}
    models/
      registry.py assets/manifests.py robots/franka.py objects/cube.py
      sensors/{rgb_camera.py,depth_camera.py}
    scenes/{base.py,registry.py,stack_scene.py}
    environments/{base.py,registry.py,red_on_blue.py}
    physics/{contacts.py,limits.py,materials.py}
    randomization/{object_pose.py,seeds.py}
  contracts/ integrations/                     # compatibility imports
  plugins/ cli.py __main__.py
tests/
  action/{test_skill_registry.py,test_torchscript_network.py}
  simulation/{test_models_and_scene.py,test_isaac_lab_adapters.py}
  integration/test_chinese_isaac_chain.py
  test_compatibility_surface.py test_config_manifests.py
  test_action.py test_contracts.py test_dependency_boundaries.py
  test_isaaclab_bridge.py test_language.py test_pipeline.py
  test_registry.py test_vision.py conftest.py
tools/
  eval_v7_multicube_chain.py
  migration/verify_checkpoint_compatibility.py
  summarize_stack_benchmark.py
vendor/stage_vla_v5/                           # retained, no deletion
```

Every package directory also contains its normal `__init__.py`; they are
omitted from the tree only to keep the ownership structure readable.

## Migration map

| Old location | Canonical location | Compatibility |
|---|---|---|
| `contracts/models.py` | `interfaces/contracts/*.py` | old contracts re-export canonical classes |
| `contracts/errors.py` | `interfaces/errors.py` | old path re-exports errors |
| `vision/interfaces.py` | `interfaces/vision_interface.py` | wrapper retained |
| `vision/adapters/static.py` | `vision/providers/static_provider.py` | wrapper retained |
| `vision/adapters/legacy.py` | `vision/providers/legacy_provider.py` | wrapper retained |
| `language/interfaces.py` | `interfaces/language_interface.py` | wrapper retained |
| `language/adapters/deterministic.py` | `language/providers/deterministic.py` and `language/parser/*` | wrapper retained |
| `language/adapters/callable.py` | `language/providers/callable.py` | wrapper retained |
| `action/interfaces.py` | `interfaces/action_interface.py` | wrapper retained |
| `action/domain.py` | `action/domains/*` and `action/router.py` | wrapper retained |
| `action/adapters/callable.py` | `action/network/callable_policy.py` | wrapper retained |
| `action/adapters/torchscript.py` | `action/network/{checkpoint_loader.py,torchscript_policy.py}` | wrapper retained |
| `action/adapters/v5.py` | `action/network/v5_factory.py` | wrapper retained |
| enum-only Skill handling | `action/actions/*.py` and `action/action_list.py` | frozen enum/order retained |
| colocated scheduler | `action/scheduler.py` plus orchestration facade | old imports retained |
| V5 success call sites | `action/evaluation/*` | native vectorized implementation; V5 adapter retained for regression only |
| `orchestration/pipeline.py` internals | `prepared_task.py`, `execution_context.py`, `audit.py`, `task_scheduler.py` | public pipeline API retained |
| `integrations/isaaclab.py` | `simulation/isaac_lab/*` | old integration path re-exports bridge |
| evaluator V5 environment import | `simulation/isaac_lab/env_factory.py` | old V5 factory path re-exports the V7 owner |
| evaluator layout helpers | `simulation/randomization/object_pose.py` | schemas and seeded behavior retained |
| evaluator MP4/overlay logic | `simulation/recording/*` | streaming recorder with strict/best-effort modes |
| single shared camera branch | distinct Vision and Observer `CameraSpec` values | Observer is excluded from Vision input |
| monolithic evaluator | `tools/evaluation/{cli,episode_runner,task_executor,task_evaluator,data_collection,trace,result_writer}.py` | 17-line compatibility entry; all owner modules <= 700 lines |
| mixed component JSON | split `config/{vision,language,action,simulation,tasks}` | example aggregate retained |
| root smoke implementation | `scripts/smoke/run_v7_smoke.ps1` | root script forwards |

## Deletions

No source file or vendored file was deleted. Existing import paths were retained
as wrappers because downstream users may still rely on them. The former V5
environment-factory module is now a compatibility re-export of the V7-owned
implementation. No artifact was copied into the repository and no checkpoint
was rewritten.

## Compatibility result

`tools/migration/verify_checkpoint_compatibility.py` checked all eight real
V5-trained policies against their locked SHA256 values. For a deterministic
input per declared observation dimension, direct TorchScript inference plus the
frozen safety projection matched refactored `ActionService` output with maximum
absolute error `0.0` for every Skill. See
`evidence/phase2_checkpoint_compatibility_post.json`.

## Phase-2 result

Phase 2 started from commit `b2e14c95557fff88839856865918dcbc7efd4be2`.
The concrete Isaac environment factory, pure layout handling, camera
declarations, and all video responsibilities moved to `simulation`. The old
1,929-line `tools/eval_v7_multicube_chain.py` is now a 17-line compatibility
entry. Argument/preflight handling, runtime assembly, Skill execution, task
aggregation, optional collection/trace, and result writing are separate modules
under `tools/evaluation`; Action success semantics remain under
`action/evaluation`.

The successful dual-camera run proves the two live paths are independent:

```text
Vision Camera RGB-D -> VisionService -> SceneState -> StageVLAPipeline
Observer Camera RGB -> FrameCapture -> VideoRecorder -> MP4
```

At the Phase-2 cutoff migration was intentionally incomplete. The new factory
and episode runner still used 19 unique direct V5 module paths for
gripper/contact/state/vector runtime behavior and frozen helpers. That
historical inventory is superseded by the Phase-3 result below.

## Phase-3 result

Phase 3 starts from the Phase-2 state described above and closes its physical
runtime boundary without changing the eight Skills, 5D action ABI, observation
ordering, thresholds, timing, success predicates, checkpoints, or cube policy
domain.

Each dependency cluster followed the same sequence: source audit, failing
parity/boundary test, native V7 implementation, runtime switch, full tests, and
strict Vision + Video physical smoke. The resulting ownership is:

```text
V7 VisionService + native RGB-D detector
  -> SceneState
V7 LanguageService
  -> TaskPlan
SceneState + TaskPlan
  -> StageVLAPipeline -> ActionService -> 8 TorchScript policies
  -> RobotAction[5] -> V7 Isaac adapters
  -> KnownSizeGraspEnvironment -> Isaac Lab -> Franka
```

Native V7 modules now own success evaluation, action output/reference logic,
camera reads/calibration/detection, `PhysicalState`, 52-D and 55-D observations,
the known-size environment, state sampling, gripper control, contacts, fixed
poses, physical/object profiles, fixed-tilt math and action setup, snapshot
expansion, REACH sampling/actions, grasp/control/safety/reward helpers, motion
configuration, and demonstration/DAgger buffers.

The last evaluator coupling was removed through the public environment API:
`observe`, `configure_evaluation`, `configure_skill`,
`synchronize_after_external_step`, typed physical state, terminal status,
gripper diagnostics, object size, current Skill, and stability-source access.
Static tests now reject private environment calls and direct protected-state
mutation from `tools/evaluation`.

Final Phase-3 evidence is 270 passing tests, 8/8 checkpoint compatibility with
maximum error `0.0`, zero formal V5 runtime imports, and a strict seed-61081
physical run with 8/8 Skills, 7/7 exact handoffs, 730/730 valid Vision calls,
zero oracle fallback, and a fully decoded 768-frame Observer MP4. The one V5
source import that remains is a lazy `LEGACY_ONLY` regression adapter.

## Phase-2 complete scoped tree

```text
src/stage_vla_v7/simulation/
  __init__.py
  config.py
  environments/
    __init__.py
    base.py
    red_on_blue.py
    registry.py
  isaac_lab/
    __init__.py
    action_adapter.py
    adapter.py
    camera_adapter.py
    env_factory.py
    observation_adapter.py
    pipeline_action_source.py
    runtime.py
  models/
    __init__.py
    registry.py
    assets/
      __init__.py
      manifests.py
    objects/
      __init__.py
      cube.py
    robots/
      __init__.py
      franka.py
    sensors/
      __init__.py
      depth_camera.py
      observer_camera.py
      rgb_camera.py
  physics/
    __init__.py
    contacts.py
    limits.py
    materials.py
  randomization/
    __init__.py
    object_pose.py
    seeds.py
  recording/
    __init__.py
    config.py
    frame_capture.py
    overlays.py
    video_recorder.py
  scenes/
    __init__.py
    base.py
    registry.py
    stack_scene.py

tools/
  eval_v7_multicube_chain.py
  summarize_stack_benchmark.py
  evaluation/
    __init__.py
    artifacts.py
    bootstrap.py
    camera_setup.py
    cli.py
    constants.py
    data_collection.py
    episode_runner.py
    result_writer.py
    task_evaluator.py
    task_executor.py
    trace.py
  migration/
    verify_checkpoint_compatibility.py

scripts/
  run_v7_smoke.ps1
  evaluate/
    run_v7_multicube_chain.ps1
  simulation/
    run_red_on_blue.ps1
  smoke/
    run_v7_smoke.ps1
    run_v7_vision_video_smoke.ps1
  train/
    train_skill_bc.py

tests/
  conftest.py
  test_action.py
  test_compatibility_surface.py
  test_config_manifests.py
  test_contracts.py
  test_dependency_boundaries.py
  test_isaaclab_bridge.py
  test_language.py
  test_pipeline.py
  test_registry.py
  test_vision.py
  action/
    test_skill_registry.py
    test_torchscript_network.py
  evaluation/
    test_evaluator_structure.py
  integration/
    test_chinese_isaac_chain.py
  simulation/
    test_dual_camera.py
    test_isaac_lab_adapters.py
    test_models_and_scene.py
    test_recording.py
```
