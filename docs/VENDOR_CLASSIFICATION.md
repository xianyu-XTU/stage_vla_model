# V5 vendor classification

`vendor/stage_vla_v5` is retained intact. Classification is conservative: a
file is not deletable merely because the current evaluator does not import it.
The static audit found 75 of 179 Python modules reachable from physical V7
evaluation roots; 104 modules lacked enough replacement and regression evidence
for deletion.

## ADAPT

These module families remain active behind explicit V7 adapters:

| V5 family | Current use |
|---|---|
| `tools/stageppo_known_size_grasp_env.py` | actual Isaac Lab environment factory, called through `simulation.isaac_lab.runtime` |
| `tools/train_known_size_grasp.py` | physical size/config helpers and explicit legacy training entry |
| `stage_vla/action_output.py` | binds the V7 `PipelineActionSource` to the retained vector loop |
| `stage_vla/data` | RGB/depth camera readers and runtime data contracts |
| `stage_vla/envs` | Isaac scene state, gripper, contacts, physical profiles, and controls |
| `stage_vla/rl` | observations, vector wrapper, safety, demonstrations, physical terminal state, and retained training |
| `stage_vla/stages` | frozen grasp geometry and stage-specific physical helpers |
| `stage_vla/vision` | calibrated physical RGB-D detector used by the legacy provider |
| `stage_vla/evaluation` | frozen layout loading and physical benchmark helpers |

These files cannot be deleted until an owned V7 backend reproduces the complete
Isaac scene, state readers, action manager, contact sensing, and physical tests.

## MIGRATE

Capabilities below now have canonical V7 ownership, while V5 source remains for
physical compatibility:

| V5 capability | V7 owner | Remaining dependency |
|---|---|---|
| vectorized `skill_success` | `action/evaluation` | explicit adapter still calls V5 tensor predicate |
| checkpoint inference boundary | `action/network` | original checkpoint files remain V5-trained |
| VLA-to-Isaac batch bridge | `simulation/isaac_lab` | vector environment consumes the bridge |
| Franka/cube/camera metadata | `simulation/models` | concrete Isaac configs remain in V5 factory/task registry |
| scene/environment ownership | `simulation/scenes` and `simulation/environments` | backend lifecycle delegates to retained runtime |
| deterministic object randomization | `simulation/randomization` | frozen physical layouts remain compatible |

Migration here means public ownership has moved; it does not authorize deleting
the source that still supplies runtime behavior.

## KEEP

The following module families are not required by the one physical evaluation
entry or are historical/training capabilities, but they remain because no full
replacement plus regression suite proves deletion safe:

```text
stage_vla/core
stage_vla/il
stage_vla/imitation
stage_vla/integration
stage_vla/lightweight_vla
stage_vla/openvla_action_chunk
stage_vla/policies
stage_vla/semantic_action
stage_vla/task_dsl
stage_vla/transition
stage_vla/vla_teacher
all other vendor tools, configs, package metadata, and documentation
```

Transitive files inside the ADAPT families are also KEEP when not reached by the
current evaluator; keeping the whole family avoids an unsafe partial vendor
rewrite.

## DELETE

None. The required conditions of no callers, complete replacement coverage,
and passing physical regression tests have not all been established for any
vendor category. This refactor therefore removes zero V5 files.
