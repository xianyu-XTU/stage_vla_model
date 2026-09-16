# Stage VLA V7 status

Updated: 2026-09-16

## Implemented

- Independent vision, language, action, orchestration, and plugin packages.
- Immutable cross-module contracts and explicit provider descriptors.
- Chinese, English, single-relation DSL, and multi-relation DSL planning.
- RGB-D detector adaptation through `VisionService` on every simulator frame.
- Complete eight-policy TorchScript cube bundle with physical-domain routing.
- `PipelineActionSource` batch bridge from Isaac Lab to `StageVLAPipeline.act`.
- Per-token action audit counts, provider identities, bundle identity, and
  safety-projection counts in physical result JSON.
- Frozen smoke layout and reproducible PowerShell runner.
- Source-only V5 simulator runtime vendored for repository portability.

## Verified

- 15 unit and boundary tests pass under Isaac Sim Python 3.12.13.
- Chinese command expands to the canonical eight-skill sequence.
- All eight migrated TorchScript policies load and execute through V7.
- Physical RGB-D VLA smoke: 1/1 successful, 730 V7 vision calls, zero invalid
  frames, zero mid-episode resets, no reference skills, no recovery controller.
- Every prepared token was exercised and `v7_chain.verified=true`.

## Current limitations

- The action checkpoints are migrated V5 cube policies, not newly trained V7
  policies.
- The language provider is deterministic; no remote LLM/VLM is connected.
- Vision supplies object positions; robot proprioception, object orientation,
  and physical terminal checks still use Isaac Lab state.
- The validated action domain remains rigid 4 cm, 0.05 kg cubes.
- The one-seed smoke proves wiring and physical execution, not statistical
  generalization. The previous 20-seed V6 benchmark should be rerun through V7
  before making multi-seed performance claims.
