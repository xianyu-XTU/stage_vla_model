# V7 validation record

Validation date: 2026-09-16

## Environment

- Isaac Lab root: `E:\work\IsaacLab`
- Isaac Sim Python: 3.12.13
- PyTorch: 2.10.0+cu128
- GPU: NVIDIA GeForce RTX 4060
- Simulator device: `cuda:0`
- Physics/control step: 0.01 s / 0.05 s

## Automated tests

```text
15 passed
```

The tests cover contracts, dependency boundaries, language parsing, legacy
vision normalization, action routing, safety projection, orchestration, plugin
registration, and batched Isaac Lab-to-pipeline dispatch.

## Physical vision-language-action smoke

The frozen `config/smoke_layout_seed61081.json` scene ran with:

- command: `把红色方块放到蓝色方块上`;
- V7 deterministic language provider;
- 128×128 RGB-D compact color/depth detection through `VisionService`;
- all eight migrated TorchScript policies through
  `StageVLAPipeline.act -> ActionService`;
- no reference skills and no reference recovery;
- `--require_v7_chain` enabled.

Result:

| Check | Result |
|---|---:|
| Physical chain | 1/1 passed |
| V7 chain gate | passed |
| RGB-D service calls | 730 |
| Invalid vision frames | 0 |
| Mid-episode resets | 0 |
| Prepared skill tokens exercised | 8/8 |
| Inter-skill handoffs preserving state | 7/7 |

Per-skill learned inference calls were REACH 374, GRASP 31, LIFT 60,
TRANSPORT 101, ALIGN 32, DESCEND 20, RELEASE_STABILIZE 7, and RETREAT 103.

The compact committed record is
`evidence/v7_vla_smoke_seed61081.summary.json`. The full local diagnostic JSON
is generated under `outputs/` and intentionally ignored by Git.

## Claim boundary

This run verifies end-to-end wiring and one successful physical episode. It
does not claim a new trained model, full visual state estimation, or a
multi-seed success rate.
