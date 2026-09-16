# Migrating V5 components

V7 does not modify checkpoints or datasets from V5. Wrap them at the boundary.

| V5 component | V7 destination |
|---|---|
| `stage_vla.vision.LearnedRGBDDetector` | `LegacyDetectorAdapter` |
| `stage_vla.vision.YoloCubeRGBDDetector` | `LegacyDetectorAdapter` |
| `InstructionParser` | `DeterministicLanguageProvider` |
| `outputs/v5_generalized_cube_bc_v1/*/policy.ts` | `TorchScriptActionPolicy` instances |
| `V5ActionModelRuntime` | `ActionBundle` + `ActionRouter` + `ActionService` |
| `V5ManipulationPipeline` | `StageVLAPipeline` |

## Vision adapter

Construct the V5 detector in the application, then pass it to
`LegacyDetectorAdapter`. Calibration is supplied through adapter keyword
arguments. The V7 vision package does not import V5 or Ultralytics.

## Action adapter

Create one `TorchScriptActionPolicy` per skill. Put the eight policies into an
`ActionBundle` with the exact validated cube `PolicyDomain`. Do not broaden the
domain merely because an object label can be parsed.

## Acceptance rule

Imported V5 evidence remains V5 evidence. A V7 result should record provider
descriptors, bundle name, checkpoint paths, seeds, reset count, fallback usage
and whether every action was produced by a registered action policy.
