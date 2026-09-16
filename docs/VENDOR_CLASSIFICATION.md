# V5 vendor classification

Phase-2 status: **PARTIAL**. The concrete environment-factory implementation
has moved into V7 Simulation, but the physical run still imports 19 unique V5
module paths directly, excluding their transitive imports. No V5 file is
classified DELETE.

## VENDOR STATUS

| File/module | Original purpose | Current caller | Handling | Migration target | Final status |
|---|---|---|---|---|---|
| `tools/stageppo_known_size_grasp_env.py` | build the concrete Isaac environment | legacy/external imports only | compatibility re-export | `simulation/isaac_lab/env_factory.py` | ADAPT; implementation migrated |
| `tools/train_known_size_grasp.py` | load object and size configuration | `tools/evaluation/episode_runner.py` | explicit import | Simulation configuration | ADAPT |
| `stage_vla/action_output.py` | bind policy output to vector execution | episode runner | explicit bridge | V7 evaluator/runtime | ADAPT |
| `stage_vla/data/v4_2_runtime.py` | RGB/depth tensor reads | episode runner | camera-boundary helper | Isaac camera adapter | ADAPT |
| `stage_vla/envs/*` active subset | gripper action, contacts, state reads, fixed poses, physical profiles | V7 environment factory and episode runner | lazy imports after launcher start | V7 Isaac runtime | ADAPT |
| `stage_vla/rl/*` active subset | vector wrapper, frozen reach/Skill helpers, physical object/config behavior | V7 factory, evaluator, Action success adapter | explicit compatibility imports | V7 Simulation/Action adapters | ADAPT |
| `stage_vla/stages/grasp_geometry.py` | frozen grasp geometry | episode runner | explicit helper | V7 Simulation geometry | ADAPT |
| `stage_vla/stages/object_motion.py` | frozen motion limits | episode runner | explicit helper | V7 Simulation configuration | ADAPT |
| `stage_vla/vision.py` | calibrated compact RGB-D detector | V7 legacy Vision provider in episode runner | behind `VisionService` | native V7 provider | ADAPT |
| `stage_vla/evaluation/four_cube_stack.py` | layout manifest loading | no Phase-2 runtime caller | replaced by V7 parser | `simulation/randomization/object_pose.py` | MIGRATE complete; retained |
| all other vendor packages/tools | training, historical, or unproven capabilities | not required by the verified entry, or transitive only | retained conservatively | none in this phase | KEEP |

## Exact direct dependency inventory

The following 19 V5 Python module paths are directly imported by V7-owned
`src` or `tools/evaluation` runtime code:

```text
1.  tools.train_known_size_grasp
2.  stage_vla.action_output
3.  stage_vla.data.v4_2_runtime
4.  stage_vla.envs
5.  stage_vla.envs.fixed_object_pose
6.  stage_vla.envs.fixed_tilt_ik
7.  stage_vla.envs.known_size_grasp_action
8.  stage_vla.envs.physical_profiles
9.  stage_vla.envs.state_readers
10. stage_vla.rl.known_size_grasp
11. stage_vla.rl.known_size_grasp_vecenv
12. stage_vla.rl.object_physics
13. stage_vla.rl.reach_policy
14. stage_vla.rl.skill_action_safety
15. stage_vla.rl.skill_demonstrations
16. stage_vla.rl.v5_skill_contracts
17. stage_vla.stages.grasp_geometry
18. stage_vla.stages.object_motion
19. stage_vla.vision
```

The environment factory uses V5 gripper/contact/state/physical components;
the episode runner uses the retained vector wrapper and frozen physical
helpers; `action/evaluation/skill_evaluator.py` still calls the V5 tensor
predicate to preserve success semantics. Checkpoints also remain V5-trained
external artifacts, though loading and routing are V7-owned.

## Migrated capabilities

| Capability | V7 owner | Evidence |
|---|---|---|
| concrete environment construction | `simulation/isaac_lab/env_factory.py` | post-migration physical smoke passed |
| deterministic layout parsing/sampling | `simulation/randomization/object_pose.py` | unit tests and physical frozen layout |
| multi-camera declarations and binding | `simulation/config.py`, `camera_adapter.py` | dual-camera unit and physical smoke |
| Observer model/config | `simulation/models/sensors/observer_camera.py` | registry/config tests and physical MP4 |
| frame conversion, overlay, MP4 streaming | `simulation/recording/*` | strict/best-effort tests and decoded MP4 |
| whole-task Skill execution and handoffs | `tools/evaluation/task_executor.py` | 8/8 physical Skill execution and 7/7 exact handoffs |
| final relation/stack aggregation | `tools/evaluation/task_evaluator.py` | structure tests and physical task result |
| whole-task JSON writing | `tools/evaluation/result_writer.py` | physical result output |

Migration of a capability does not authorize deleting retained V5 modules that
still serve other callers or supply transitive physical behavior.

## KEEP

Historical, training, and currently unproven module families remain in the
vendor tree. The earlier reachability audit found 75 of 179 Python modules
statically reachable from physical evaluation roots; lack of reachability is
not sufficient deletion evidence. No training implementation, artifact, or
checkpoint was rewritten in Phase 2.

## DELETE

None. Deletion requires no runtime or test caller, full V7 replacement, and
passing regression plus physical evidence. Those conditions have not been
established for an entire V5 file family.
