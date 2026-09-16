# Architecture

## Dependency rule

`interfaces` is the only public contract layer and imports only the Python
standard library. The dependency direction is:

```text
                         interfaces
                    /        |        \
               vision     language    action
                    \        |        /
                     orchestration
                           |
                     RobotAction[5]
                           |
                       simulation
                           |
                       Isaac Lab
                           |
                         Franka
```

The practical rules enforced by AST boundary tests are:

- Vision, Language, and Action do not import one another.
- Action and Orchestration do not import Isaac Lab or Simulation.
- The pipeline does not import a concrete simulator.
- Simulation may depend on Interfaces and Orchestration because it is the
  outer application/infrastructure layer.
- Torch, Isaac tensors, OpenCV, and model implementations never enter public
  contracts.

## Runtime flow

1. `VisionService.observe(VisionRequest)` returns `VisionResult(SceneState)`.
2. `LanguageService.interpret(LanguageRequest)` returns
   `LanguageResult(TaskPlan)`.
3. `StageVLAPipeline.prepare` validates object labels/profiles and asks the
   registry-driven `TaskScheduler` for `SkillToken` values.
4. `StageVLAPipeline.act` creates one `ActionRequest` for one scheduled token.
5. `ActionService` requires the registered Skill, routes one physical-domain
   bundle, checks the exact observation dimension, runs one policy, and applies
   the shared safety projection.
6. The result is a finite normalized
   `RobotAction[dx, dy, dz, dyaw, grip]`.
7. A Simulation adapter converts that contract to a backend command. Only the
   environment/application loop may call `step`.

## Action structure

`action/action_list.py` is the canonical registry. The eight files in
`action/actions` own Skill metadata and transitions. `action/network` owns
parameter prediction and checkpoint loading; `action/training` owns training;
`action/evaluation` owns all success/failure predicates. None of these owns a
simulator lifecycle.

Skill selection and parameter prediction remain separate:

```text
TaskPlan -> TaskScheduler -> SkillToken -> selected policy
         -> [dx, dy, dz, dyaw, grip] -> SafetyProjector
```

Unknown Skills, missing policies, ambiguous domains, wrong observation sizes,
hash mismatches, and non-finite actions fail closed.

## Simulation structure

`simulation/models` describes physical entities, not neural networks:

- `robots/franka.py`: joints, links, limits, default pose, controller support.
- `objects/cube.py`: geometry, mass, material, collision, and ObjectProfile.
- `sensors`: RGB and metric depth camera descriptions.

`StackScene` composes robot, objects, sensors, table, and light.
`RedOnBlueEnvironment` owns the task environment boundary and requires an
explicit backend. Without a backend it raises instead of pretending to run
physics. Registries for models, scenes, and environments fail closed.

`simulation/isaac_lab` owns conversions between raw camera/tensor payloads and
the public contracts. `PipelineActionSource` converts every batch row to
`RobotObservation`, calls `StageVLAPipeline.act`, and converts only the final
validated actions back to a Torch tensor.

The physical environment factory is still implemented by the retained V5
runtime. `make_legacy_v5_known_size_grasp_env` is the explicit adapter while
that source remains under `vendor/stage_vla_v5`; no silent fallback exists.

## Audit chain

`PipelineActionSource.audit()` records language and vision providers, action
providers and bundles, inference rows per Skill and token, batch calls, and
safety projections. The evaluator's `--require_v7_chain` gate additionally
requires all prepared tokens, no reference Skill, no reference recovery, a
valid visual path, and a physically successful task.

The evaluator now obtains vectorized terminal predicates through
`action.evaluation.legacy_vectorized_skill_success`. That adapter deliberately
calls the retained V5 tensor implementation so physical semantics are unchanged
while ownership is centralized.

## Replaceability

- Register a new `VisionProvider` without changing Language or Action.
- Register a new `LanguageProvider` without changing Vision or Action.
- Implement `ActionPolicy` or `ParameterNetwork` without changing semantics.
- Register a robot/object/sensor/scene/environment without rewriting the VLA
  pipeline.
- Add a ninth Skill by adding its definition, registering it, configuring a
  policy and evaluator, and extending the scheduler sequence.
