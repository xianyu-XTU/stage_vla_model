# Extending V7

## Add a vision model

Implement `VisionProvider.detect(VisionRequest) -> VisionResult`. Convert SDK
objects to `ObjectDetection` before returning. Register the provider in an
application-owned `ProviderRegistry[VisionProvider]`.

```python
class MyDetector:
    descriptor = ModelDescriptor("my-detector", "1", "vision", ("rgbd",))

    def detect(self, request):
        detections = (...,)
        return VisionResult(SceneState(detections), self.descriptor)
```

## Add an LLM

Implement `LanguageProvider` or wrap an SDK callable with
`CallableLanguageProvider`. The callable must return a validated `TaskPlan`.
Keep prompting, retries and JSON decoding inside the adapter.

```python
provider = CallableLanguageProvider(
    llm_client_to_task_plan,
    name="my-llm",
    version="model-revision",
)
```

The LLM output must not contain actions, masses, friction or grasp forces.

## Add an action policy

Implement `ActionPolicy` for a new backend, then create a complete eight-skill
`ActionBundle`. Declare its physical `PolicyDomain` from evaluation evidence.
The router will reject missing, unsupported or overlapping coverage.

```python
bundle = ActionBundle(descriptor, validated_domain, policies_by_skill)
router = ActionRouter({"new-domain": bundle})
```

Do not expand a domain merely because the model returns finite actions. Domain
limits must come from held-out physical validation.

## Add a new skill

Adding an enum value is intentionally not plug-and-play because skill ordering
is a behavioral contract. Update `Skill`, `SKILL_SEQUENCE`, scheduler tests,
every complete action bundle, terminal-condition handling, and acceptance
evidence together.
