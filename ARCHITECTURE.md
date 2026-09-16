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
- `sensors`: RGB, metric depth, and observer-camera descriptions.

`StackScene` composes robot, objects, sensors, table, and light.
`RedOnBlueEnvironment` owns the task environment boundary and requires an
explicit backend. Without a backend it raises instead of pretending to run
physics. Registries for models, scenes, and environments fail closed.

`simulation/isaac_lab` owns environment construction and conversions between
raw camera/tensor payloads and the public contracts. `SimulationEnvironmentFactory`
fails closed on unknown environments. `PipelineActionSource` converts every batch row to
`RobotObservation`, calls `StageVLAPipeline.act`, and converts only the final
validated actions back to a Torch tensor.

The concrete environment factory now lives in
`simulation/isaac_lab/env_factory.py`. The former V5 factory is a compatibility
re-export. Gripper control, contact sensors, state readers, object/physical
profiles, fixed poses, fixed-tilt IK, snapshot expansion, REACH state sampling,
and the known-size environment are all V7-owned. The formal physical runtime
does not import V5 Python modules.

`KnownSizeGraspEnvironment` orchestrates the frozen cube-policy behavior while
delegating typed state, observations, success predicates, geometry, rewards,
control conversion, and physics helpers to their owning V7 modules. Evaluation
uses only its public `observe`, `configure_skill`, external-step synchronization,
status, and diagnostics surface; AST tests reject private-method access and
direct mutation of protected environment state.

## Camera and recording flow

```text
Isaac StackScene
  +-- Vision Camera (128 x 128 RGB-D)
  |     -> IsaacCameraAdapter -> VisionService -> SceneState
  +-- Observer Camera (640 x 480 RGB)
        -> IsaacCameraAdapter -> FrameCapture -> overlay -> VideoRecorder -> MP4
```

Camera IDs and roles are validated as unique. `CameraBindings.vision` is the
only handle passed into the visual observation path; the observer handle is
read only by recording code. `VideoRecorder` streams frames to the encoder and
does not retain the whole episode in memory. Strict mode raises on capture or
encoding failure. Best-effort mode records the error, closes the encoder,
warns, and leaves the control loop unchanged.

## Evaluation structure

`tools/eval_v7_multicube_chain.py` is a 17-line compatibility entry that only
establishes the repository import path and invokes the CLI. Evaluation is split
by responsibility:

- `cli.py`: arguments, validation, task/layout/artifact resolution;
- `episode_runner.py`: runtime assembly, application lifetime, and subsystem
  coordination;
- `task_executor.py`: REACH plus seven downstream Skills and state-exact
  handoffs;
- `task_evaluator.py`: relation and final-stack aggregation;
- `data_collection.py` and `trace.py`: optional output collection;
- `result_writer.py`: stable result schema and JSON persistence.

No evaluation module exceeds 700 lines. Environment construction, layout
parsing/randomization, camera adaptation, overlay rendering, and MP4 encoding
are owned by their Simulation modules. Action terminal semantics remain in
`action/evaluation`.

## Audit chain

`PipelineActionSource.audit()` records language and vision providers, action
providers and bundles, inference rows per Skill and token, batch calls, and
safety projections. The evaluator's `--require_v7_chain` gate additionally
requires all prepared tokens, no reference Skill, no reference recovery, zero
Vision-invalid frames, zero oracle fallback, a valid visual path, and a
physically successful task.

The physical environment calls native V7 `SuccessChecker.evaluate_batch` for
vectorized terminal predicates. `legacy_vectorized_skill_success` remains only
as an explicit regression oracle and is not reachable from the formal runtime.

## Replaceability

- Register a new `VisionProvider` without changing Language or Action.
- Register a new `LanguageProvider` without changing Vision or Action.
- Implement `ActionPolicy` or `ParameterNetwork` without changing semantics.
- Register a robot/object/sensor/scene/environment without rewriting the VLA
  pipeline.
- Add a ninth Skill by adding its definition, registering it, configuring a
  policy and evaluator, and extending the scheduler sequence.
