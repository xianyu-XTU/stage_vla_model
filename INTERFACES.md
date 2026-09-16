# Public interfaces

## Shared contracts

| Type | Purpose |
|---|---|
| `ObjectDetection` | label, XYZ [m], yaw [rad], confidence |
| `SceneState` | immutable collection of detections plus optional held label |
| `StackRelation` | object/support semantic relation |
| `TaskPlan` | one or more relations in validated execution order |
| `SkillToken` | one skill bound to one relation |
| `ObjectProfile` | size, mass, geometry, grasp and support properties |
| `RobotAction` | normalized `[dx, dy, dz, dyaw, grip]` |
| `ModelDescriptor` | provider identity, version and capabilities |

## Vision port

```python
class VisionProvider(Protocol):
    descriptor: ModelDescriptor
    def detect(self, request: VisionRequest) -> VisionResult: ...
```

`VisionRequest` carries an opaque RGB/depth frame and calibration metadata.
`VisionResult` carries a validated `SceneState`. Implementations may be a
compact CNN, YOLO+RGB-D geometry, a remote service, or an oracle test provider.

## Language port

```python
class LanguageProvider(Protocol):
    descriptor: ModelDescriptor
    def interpret(self, request: LanguageRequest) -> LanguageResult: ...
```

`LanguageResult` contains only `TaskPlan`. It cannot contain continuous robot
actions or physical parameters. The current deterministic provider supports
strict DSL plus constrained Chinese and English commands.

## Action port

```python
class ActionPolicy(Protocol):
    descriptor: ModelDescriptor
    observation_dim: int
    def predict(self, request: ActionRequest) -> ActionResult: ...
```

`ActionRequest` contains one skill, a numeric observation vector, and explicit
object/support profiles. `ActionResult` contains one normalized `RobotAction`.
The `ActionRouter` selects exactly one `ActionBundle` by physical domain.

## Orchestration API

```python
prepared = pipeline.prepare(command="stack red cube on blue cube", frame=frame)
token = prepared.tokens[0]
result = pipeline.act(prepared, token, observation, finished=False)
```

The application owns control-loop timing and terminal-condition evaluation.
V7 owns semantic binding, skill selection, domain routing and action safety.

## Isaac Lab batch adapter

```python
source = PipelineActionSource(pipeline, prepared)
source.bind_relation(0)
actions = source.action("REACH", observation_batch)
audit = source.audit()
```

The adapter intentionally lives outside the core ports. Torch tensors do not
cross into contracts, and every row still follows the ordinary V7 action API.
