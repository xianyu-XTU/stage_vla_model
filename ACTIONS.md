# Actions

## Common contract

All Skills output exactly five normalized parameters in this fixed order:

```text
[dx, dy, dz, dyaw, grip]
```

`dx/dy/dz` are relative Cartesian commands, `dyaw` is relative yaw, and
positive `grip` opens while negative closes. `SafetyProjector` clamps every
parameter to its configured bound and emits a stationary Skill-specific grip
command after completion. Non-finite or non-five-dimensional output fails.

`action/action_list.py` is the only canonical Skill registry. Each definition
records the numeric ID, implementation module, observation schema, legacy
dimension, required fields, parameter schema, supported domain, policy slot,
entrance/success/failure conditions, and transition.

## Skills

| ID | Skill / file | Input and checkpoint | Success condition | Failure behavior | Next |
|---:|---|---|---|---|---|
| 0 | `REACH` / `actions/reach.py` | 52 values: end effector, object/support poses, fingertips, gripper; `reach/policy.ts` | Open fingertip midpoint is within 0.012 m XY and 0.003 m Z error of the 0.010 m pregrasp target | Invalid pose or end-effector/object distance above 0.40 m fails; no fallback | `GRASP` |
| 1 | `GRASP` / `actions/grasp.py` | 55 values: end effector, object, fingertips, force, gripper; `grasp/policy.ts` | Physical hold and end-effector/object distance below 0.022 m remain stable | Lost hold, object too low, excessive support distance, or timeout fails | `LIFT` |
| 2 | `LIFT` / `actions/lift.py` | 55 values: end effector, object/support, held flag, velocity; `lift/policy.ts` | Held object clears support by more than 0.060 m | Lost hold, object too low, excessive support distance, or timeout fails | `TRANSPORT` |
| 3 | `TRANSPORT` / `actions/transport.py` | same 55-value schema; `transport/policy.ts` | Held object is within 0.045 m XY of support, stable, below speed limits | Lost hold, object too low, excessive support distance, or timeout fails | `ALIGN` |
| 4 | `ALIGN` / `actions/align.py` | same 55-value schema; `align/policy.ts` | Held object is within 0.010 m XY and 0.015 m height error at low speed | Lost hold, object too low, excessive support distance, or timeout fails | `DESCEND` |
| 5 | `DESCEND` / `actions/descend.py` | same 55-value schema; `descend/policy.ts` | Held object is within 0.012 m XY and 0.004 m stack-height error at low speed | Lost hold, object too low, excessive XY error, or timeout fails | `RELEASE_STABILIZE` |
| 6 | `RELEASE_STABILIZE` / `actions/release_stabilize.py` | 55 values: end effector, object/support, gripper, velocity; `release_stabilize/policy.ts` | Open gripper, XY below 0.040 m, Z error below 0.010 m, stable low speed | Broken/out-of-domain stack or timeout fails | `RETREAT` |
| 7 | `RETREAT` / `actions/retreat.py` | same 55-value schema; `retreat/policy.ts` | End effector clears object by 0.100 m total and 0.080 m vertically | Invalid/disturbed stack or timeout fails | terminal |

The checkpoint names above are logical paths under the external artifact root;
hashes and full metadata are in `config/artifacts.lock.json`.

## Policy and parameter network

Scheduling selects a Skill before continuous inference. The parameter network
does not relearn Skill selection:

```text
TaskPlan -> TaskScheduler -> SkillToken -> policy slot
         -> ParameterNetwork -> five parameters -> SafetyProjector
```

`ParameterNetwork` is a minimal protocol with an explicit observation dimension
and `predict_parameters`. `TorchScriptActionPolicy` is the current backend;
callable and future lightweight networks can replace it without changes to
Vision or Language.

Checkpoint loading verifies file existence and optionally SHA256 before
`torch.jit.load`. `build_v5_cube_bundle` loads all eight policies and preserves
the exact 4 cm, 0.05 kg rigid-box routing domain. A missing Skill policy makes
bundle construction fail.

## Success evaluation

`action/evaluation` owns the dependency-free eight-Skill predicates,
full-task/stable-stack evaluation, metrics, and reporting. Training and
Simulation call this layer rather than defining new criteria.

The current physical V5 vector environment uses tensor-specific terminal
logic. `legacy_vectorized_skill_success` is an explicit adapter to that frozen
implementation, preserving current physical results while keeping the call
site centralized. It is not a fallback controller.
