# Architecture

## Dependency rule

The dependency direction is fixed:

```text
contracts <- vision
contracts <- language
contracts <- action
contracts + ports <- orchestration
ports <- adapters
```

`vision`, `language`, and `action` must not import one another. This allows a
YOLO detector, a compact CNN, an LLM, a deterministic parser, or a new policy
backend to be replaced independently.

## Runtime flow

```text
1. LanguageService.interpret(text)
   -> TaskPlan(StackRelation...)

2. VisionService.observe(frame)
   -> SceneState(ObjectDetection...)

3. StageVLAPipeline.prepare(...)
   -> PreparedTask(TaskPlan + SceneState + scheduled SkillToken values)

4. StageVLAPipeline.act(...)
   -> ActionService.act(ActionRequest)
   -> domain router -> one skill policy -> safety projector
   -> normalized RobotAction[dx, dy, dz, dyaw, grip]
```

The pipeline does not own Isaac Lab. Simulator integration belongs in an
adapter/application loop so unit tests remain fast and dependency-free.

## Isaac Lab bridge

`PipelineActionSource` is the only learned-action source configured by the V7
physical evaluator. It binds an Isaac Lab relation index to prepared
`SkillToken` values, converts each tensor row to the dependency-free V7
contract, calls `StageVLAPipeline.act`, and converts the validated action back
to a simulator tensor. The environment-specific adapter may apply a second,
more restrictive projection before stepping physics, but it cannot invoke a
checkpoint directly.

The `--require_v7_chain` acceptance gate additionally requires RGB-D through
`VisionService`, a valid language plan, every prepared token to be exercised,
and zero reference-controller or recovery usage.

## Extension points

- Add a detector by implementing `VisionProvider`.
- Add an LLM/VLM semantics adapter by implementing `LanguageProvider`.
- Add a policy backend by implementing `ActionPolicy`.
- Add a physical domain by registering another `ActionBundle`.
- Add an object type by registering an `ObjectProfile` in `ObjectCatalog`.
- Add a skill by extending `Skill`, the scheduler, and an action bundle. Missing
  policies fail closed during bundle validation.

## Safety and auditability

- Every provider exposes a `ModelDescriptor` with name, version and capabilities.
- Every result records its provider and diagnostics.
- Actions are finite, exactly five-dimensional and projected to configured limits.
- Domain routing rejects unsupported and ambiguously supported object pairs.
- Vision/language cannot bypass the action service.
- Reference controllers must be registered as explicit action policies; fallback
  is never silent.
