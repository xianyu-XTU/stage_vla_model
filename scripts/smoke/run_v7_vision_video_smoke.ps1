param(
    [string]$IsaacLabRoot = "E:\work\IsaacLab",
    [string]$ArtifactRoot = "E:\stage_vla_v5\outputs\v5_generalized_cube_bc_v1",
    [string]$Calibration = "E:\stage_vla_v5\outputs\vision_rgbd_mapping_calibration_train_20260910.json",
    [string]$Output = "",
    [string]$VideoPath = ""
)

$ErrorActionPreference = "Stop"
$runner = Join-Path $PSScriptRoot "run_v7_smoke.ps1"
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $Output) {
    $Output = Join-Path $repoRoot "outputs\v7_vla_vision_video_seed61081.json"
}
if (-not $VideoPath) {
    $VideoPath = Join-Path $repoRoot "outputs\v7_vla_vision_video_seed61081.mp4"
}

& $runner `
    -IsaacLabRoot $IsaacLabRoot `
    -ArtifactRoot $ArtifactRoot `
    -Calibration $Calibration `
    -Output $Output `
    -Video `
    -VideoPath $VideoPath `
    -Headless

if ($LASTEXITCODE -ne 0) {
    throw "V7 Vision + Video smoke failed with exit code $LASTEXITCODE"
}
