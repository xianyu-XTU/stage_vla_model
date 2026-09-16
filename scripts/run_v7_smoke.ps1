param(
    [string]$IsaacLabRoot = "E:\work\IsaacLab",
    [string]$ArtifactRoot = "E:\stage_vla_v5\outputs\v5_generalized_cube_bc_v1",
    [string]$Calibration = "E:\stage_vla_v5\outputs\vision_rgbd_mapping_calibration_train_20260910.json",
    [string]$Output = ""
)

# Compatibility entry. The canonical smoke runner lives under scripts/smoke/.
$runner = Join-Path $PSScriptRoot "smoke\run_v7_smoke.ps1"
& $runner `
    -IsaacLabRoot $IsaacLabRoot `
    -ArtifactRoot $ArtifactRoot `
    -Calibration $Calibration `
    -Output $Output
