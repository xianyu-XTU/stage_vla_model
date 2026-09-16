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

## Post-refactor tree

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
| V5 success call sites | `action/evaluation/*` | vectorized adapter preserves V5 tensors |
| `orchestration/pipeline.py` internals | `prepared_task.py`, `execution_context.py`, `audit.py`, `task_scheduler.py` | public pipeline API retained |
| `integrations/isaaclab.py` | `simulation/isaac_lab/*` | old integration path re-exports bridge |
| evaluator V5 environment import | `simulation.isaac_lab.runtime` adapter | retained V5 factory called explicitly |
| mixed component JSON | split `config/{vision,language,action,simulation,tasks}` | example aggregate retained |
| root smoke implementation | `scripts/smoke/run_v7_smoke.ps1` | root script forwards |

## Deletions

No source file or vendored file was deleted. Existing import paths were retained
as wrappers because downstream users and the physical evaluator may still rely
on them. No artifact was copied into the repository and no checkpoint was
rewritten.

## Compatibility result

`tools/migration/verify_checkpoint_compatibility.py` checked all eight real
V5-trained policies against their locked SHA256 values. For a deterministic
input per declared observation dimension, direct TorchScript inference plus the
frozen safety projection matched refactored `ActionService` output with maximum
absolute error `0.0` for every Skill. See
`evidence/checkpoint_compatibility.json`.
