# Simulation

Simulation is an outer infrastructure module. The VLA core can be imported and
tested without Isaac Lab, Torch, OpenCV, or a GPU.

## Supported backend

The physical backend is Isaac Lab using the registered task
`Isaac-Stack-Cube-Franka-IK-Rel-v0`. The validated local environment uses Isaac
Sim Python 3.12.13 and PyTorch 2.10.0+cu128. Simulator installation remains an
external prerequisite.

The actual environment factory and low-level state readers are retained under
`vendor/stage_vla_v5`. V7 reaches them only through
`make_legacy_v5_known_size_grasp_env`; this is an explicit compatibility
adapter, not a hidden fallback. `CallbackIsaacLabRuntime` exposes reset,
observe, step, and close callbacks through the public `SimulationEnvironment`
port.

## Supported entities

| Kind | Registered name | Implementation |
|---|---|---|
| Robot | `franka` | joint/link names, limits, default pose, relative differential IK |
| Object | `cube` | 0.04 m rigid cube, 0.05 kg, material/collision properties |
| Sensor | `rgb` | 128 x 128 pinhole RGB description |
| Sensor | `depth` | 128 x 128 metric `distance_to_image_plane` description |
| Scene | `stack` | Franka, table, light, three cubes, RGB and depth sensors |
| Environment | `red_on_blue` | seeded red-cube-on-blue-cube lifecycle boundary |

Model, scene, and environment registries reject unknown names. `StackScene`
loads a manifest and samples deterministic, non-overlapping tabletop positions.
`RedOnBlueEnvironment` requires an attached backend for physics operations and
raises when none is attached.

## VLA boundary

```text
Isaac RGB/depth -> IsaacCameraAdapter -> VisionRequest
Isaac policy row -> IsaacObservationAdapter -> RobotObservation
RobotObservation -> StageVLAPipeline -> RobotAction
RobotAction -> IsaacActionAdapter -> SimulationAction / Torch tensor
```

`PipelineActionSource` owns batched dispatch and audit counters. It does not
load a reference controller and it cannot call `env.step`.

## Adding entities

To add a robot, create a descriptor under `simulation/models/robots`, including
asset reference, joints, links, default pose, limits, and supported controller,
then register it with `SimulationModelRegistry.register_robot`.

To add an object, define geometry, mass, material, collision properties, and an
`ObjectProfile` adapter under `simulation/models/objects`, then register it.
The Action domain must explicitly support the new profile or routing will fail.

To add a sensor, define its raw output and calibration metadata under
`simulation/models/sensors`, register it, and adapt its payload to
`VisionRequest`. Vision must not create the simulator sensor.

To add a scene, compose registered entities under `simulation/scenes` and
register it with `SceneRegistry`. To add an environment, implement the public
lifecycle port under `simulation/environments`, attach an explicit backend, and
register it with `EnvironmentRegistry`.

## Configuration

`config/simulation/red_on_blue.json` owns backend, robot, object, sensor, scene,
physics, and randomization selections. The frozen physical smoke layout is
`config/simulation/smoke_layout_seed61081.json`. Large USD assets remain
external and are referenced by logical manifests.
