# Simulation

Simulation is an outer infrastructure module. The VLA core can be imported and
tested without Isaac Lab, Torch, OpenCV, or a GPU.

## Supported backend

The physical backend is Isaac Lab using the registered task
`Isaac-Stack-Cube-Franka-IK-Rel-v0`. The validated local environment uses Isaac
Sim Python 3.12.13 and PyTorch 2.10.0+cu128. Simulator installation remains an
external prerequisite.

The actual environment factory is now
`stage_vla_v7.simulation.isaac_lab.SimulationEnvironmentFactory`. It accepts
only registered environment names and lazily starts Isaac dependencies after
the application launcher is active. The historical V5 factory module forwards
to this implementation. Low-level gripper actions, contact/state readers,
physical profiles, and the vector wrapper remain V5 runtime dependencies;
therefore the migration status is PARTIAL. `CallbackIsaacLabRuntime` exposes
reset, observe, step, and close callbacks through the public
`SimulationEnvironment` port.

## Supported entities

| Kind | Registered name | Implementation |
|---|---|---|
| Robot | `franka` | joint/link names, limits, default pose, relative differential IK |
| Object | `cube` | 0.04 m rigid cube, 0.05 kg, material/collision properties |
| Sensor | `rgb` | 128 x 128 pinhole RGB description |
| Sensor | `depth` | 128 x 128 metric `distance_to_image_plane` description |
| Sensor | `observer` | configurable third-person RGB recording camera |
| Scene | `stack` | Franka, table, light, cubes, Vision camera, optional Observer camera |
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

## Dual cameras

The evaluator declares cameras with dependency-light `CameraSpec` values and
the environment factory creates both sensors in one scene:

```text
v7_multicube_camera  RGB + distance_to_image_plane  128 x 128
    -> VisionService -> SceneState -> StageVLAPipeline

v7_observer_camera   RGB only                       640 x 480 @ 20 FPS
    -> FrameCapture -> VideoRecorder -> MP4
```

The IDs and roles must be unique. The observer camera is absent unless video
is requested, is never passed to `VisionService`, and cannot affect
`SceneState`, Skill selection, or `RobotAction`. Its defaults live in
`config/simulation/observer_camera.json`; width, height, pose, optics, and FPS
are validated before Isaac scene construction.

## Video recording

`simulation/recording` separates four responsibilities:

- `FrameCapture` converts a camera payload to a validated `uint8` RGB array.
- `draw_overlay` renders seed, environment, step, Skill, and optional status.
- `VideoRecorder` streams frames to imageio/ffmpeg and returns path, frame
  count, FPS, resolution, duration, completion, and error fields.
- `RecordingConfig.strict` selects explicit failure behavior.

Recording uses NumPy, Pillow, imageio, and an imageio-compatible ffmpeg binary;
these are available in the validated Isaac Python environment and remain
optional to the dependency-free VLA core.

With strict recording, camera, conversion, encoder, or output failures abort
the evaluation. With best-effort recording, the recorder warns, closes the
encoder, marks the video incomplete, and control continues without changing
the VLA action path.

```powershell
scripts\smoke\run_v7_vision_video_smoke.ps1 `
  -IsaacLabRoot E:\work\IsaacLab `
  -Output outputs\v7_vision_video.json `
  -VideoPath outputs\v7_vision_video.mp4
```

For a direct evaluator invocation, use `--use_vision`, `--enable_cameras`,
`--video`, and `--require_v7_chain` together. Add
`--recording_best_effort` only when an
incomplete observer video is acceptable. The smoke runner uses strict mode.
Headless rendering is verified; Isaac Lab currently warns that `--headless` is
deprecated, so use `--viz none` for direct commands on versions that support
that spelling.

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
`config/simulation/smoke_layout_seed61081.json`; observer defaults are
`config/simulation/observer_camera.json`. Large USD assets remain external and
are referenced by logical manifests.

## Remaining V5 runtime boundary

There are 19 unique direct V5 module paths imported by the current V7 physical
factory/evaluator/success adapter, excluding their transitive imports. They own
the known-size gripper, contacts, state reads, physical object profiles, vector
environment, frozen reach/Skill helpers, compact detector, and metadata
loading. The exact inventory and disposition are maintained in
`docs/VENDOR_CLASSIFICATION.md`. No vendor source is classified DELETE.
