# Public interfaces

All canonical cross-module types live under `stage_vla_v7.interfaces`. They use
dataclasses, tuples, mappings, primitives, and Protocols only. The old
`stage_vla_v7.contracts` and module-local `interfaces.py` paths re-export these
types for compatibility.

## Contracts

| Contract | Meaning |
|---|---|
| `ObjectDetection` | label, XYZ metres, yaw, confidence, optional feature |
| `SceneState` | immutable detections and optional held-object label |
| `StackRelation` | semantic object-on-support relation |
| `TaskPlan` | validated ordered relations |
| `Skill` / `SkillToken` | registered Skill and its relation binding |
| `ObjectProfile` | geometry, size, mass, friction, and capabilities |
| `RobotObservation` | finite tuple plus explicit schema identity |
| `RobotAction` | finite `[dx, dy, dz, dyaw, grip]` values |
| `ModelDescriptor` | learned/provider identity and capabilities |
| `SimulationModelDescriptor` | backend/entity identity and capabilities |
| `SimulationObservation` | robot contract plus opaque RGB/depth payloads |
| `SimulationAction` | RobotAction plus backend control mode |
| `SimulationState` | episode, step, observation, termination state |

An Isaac Lab tensor is never a contract. Simulation adapters unwrap tensors
before core calls and create tensors only after `RobotAction` validation.

## Vision

```python
class VisionProvider(Protocol):
    descriptor: ModelDescriptor
    def detect(self, request: VisionRequest) -> VisionResult: ...
```

`VisionRequest` owns opaque RGB/depth, frame metadata, and calibration.
`VisionResult` owns `SceneState`. Vision localizes objects; it does not schedule
Skills, judge task success, or step a simulator.

## Language

```python
class LanguageProvider(Protocol):
    descriptor: ModelDescriptor
    def interpret(self, request: LanguageRequest) -> LanguageResult: ...
```

`LanguageResult` contains only `TaskPlan`. It cannot contain continuous action
parameters, torques, or joint commands. The deterministic provider supports
Chinese, English, and stack DSL.

## Action

```python
class ActionPolicy(Protocol):
    descriptor: ModelDescriptor
    observation_dim: int
    def predict(self, request: ActionRequest) -> ActionResult: ...
```

`ActionRequest` binds one Skill, one observation, and explicit object/support
profiles. `ActionResult` contains one `RobotAction`. `ActionRouter` selects
exactly one registered physical domain; zero or multiple matches fail.

## Pipeline

```python
prepared = pipeline.prepare(command, frame)
for token in prepared.tokens:
    result = pipeline.act(prepared, token, robot_observation)
```

The application owns timing and lifecycle. Orchestration owns data flow and
semantic binding, but not model training or physics.

## Simulation

```python
class SimulationAdapter(Protocol):
    descriptor: SimulationModelDescriptor
    def to_vision_request(self, observation): ...
    def to_robot_observation(self, observation): ...
    def to_simulation_action(self, action): ...

class SimulationEnvironment(Protocol):
    descriptor: SimulationModelDescriptor
    def reset(self, seed=None): ...
    def observe(self): ...
    def step(self, action): ...
    def close(self): ...
```

The concrete Isaac path is:

```text
raw RGB-D -> IsaacCameraAdapter -> VisionRequest
policy tensor row -> IsaacObservationAdapter -> RobotObservation
StageVLAPipeline -> RobotAction -> IsaacActionAdapter -> tensor/environment
```

`PipelineActionSource` is located at
`stage_vla_v7.simulation.isaac_lab.pipeline_action_source`. The old
`stage_vla_v7.integrations` import is a compatibility re-export.
