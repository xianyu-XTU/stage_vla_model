param(
    [string]$IsaacLabRoot = "E:\work\IsaacLab",
    [string]$ArtifactRoot = "E:\stage_vla_v5\outputs\v5_generalized_cube_bc_v1",
    [string]$Calibration = "E:\stage_vla_v5\outputs\vision_rgbd_mapping_calibration_train_20260910.json",
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$python = Join-Path $IsaacLabRoot "_isaac_sim\python.bat"
$evaluator = Join-Path $repoRoot "tools\eval_v7_multicube_chain.py"
$config = Join-Path $repoRoot "vendor\stage_vla_v5\config\v5_generalized_cube_eval.json"
$layout = Join-Path $repoRoot "config\simulation\smoke_layout_seed61081.json"
$artifactLock = Join-Path $repoRoot "config\artifacts.lock.json"
if (-not $Output) {
    $Output = Join-Path $repoRoot "outputs\v7_vla_smoke_seed61081.json"
}

$required = @($python, $evaluator, $config, $layout, $artifactLock, $Calibration)
$required += @(
    "reach", "grasp", "lift", "transport", "align", "descend",
    "release_stabilize", "retreat"
) | ForEach-Object { Join-Path $ArtifactRoot "$_\policy.ts" }
foreach ($path in $required) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required V7 smoke artifact is missing: $path"
    }
}

$arguments = @(
    $evaluator,
    "--config", $config,
    "--reach_checkpoint", (Join-Path $ArtifactRoot "reach\policy.ts"),
    "--grasp_checkpoint", (Join-Path $ArtifactRoot "grasp\policy.ts"),
    "--lift_checkpoint", (Join-Path $ArtifactRoot "lift\policy.ts"),
    "--transport_checkpoint", (Join-Path $ArtifactRoot "transport\policy.ts"),
    "--align_checkpoint", (Join-Path $ArtifactRoot "align\policy.ts"),
    "--descend_checkpoint", (Join-Path $ArtifactRoot "descend\policy.ts"),
    "--release_checkpoint", (Join-Path $ArtifactRoot "release_stabilize\policy.ts"),
    "--retreat_checkpoint", (Join-Path $ArtifactRoot "retreat\policy.ts"),
    "--output", $Output,
    "--artifact_lock", $artifactLock,
    "--use_vision", "--require_v7_chain", "--enable_cameras",
    "--camera_calibration", $Calibration,
    "--geometry_bias_m", "-0.0099", "0.00214", "0.0",
    "--num_envs", "1", "--scene_cube_count", "3",
    "--asset_layout_file", $layout,
    "--grasp_steps", "160", "--lift_steps", "500",
    "--transport_steps", "500", "--align_steps", "650",
    "--descend_steps", "400", "--release_steps", "300",
    "--retreat_steps", "500",
    "--lift_translation_limit_m", "0.005",
    "--transport_translation_limit_m", "0.005",
    "--align_translation_limit_m", "0.005",
    "--descend_translation_limit_m", "0.003",
    "--release_translation_limit_m", "0.003",
    "--retreat_translation_limit_m", "0.005"
)

$previousEncodedCommand = $env:STAGE_VLA_COMMAND_UTF8_BASE64
try {
    # ASCII Base64 survives PowerShell -> cmd.exe -> kit.exe without code-page loss.
    $env:STAGE_VLA_COMMAND_UTF8_BASE64 = (
        "5oqK57qi6Imy5pa55Z2X5pS+5Yiw6JOd6Imy5pa55Z2X5LiK"
    )
    & $python @arguments
    $smokeExitCode = $LASTEXITCODE
}
finally {
    if ($null -eq $previousEncodedCommand) {
        Remove-Item Env:STAGE_VLA_COMMAND_UTF8_BASE64 -ErrorAction SilentlyContinue
    }
    else {
        $env:STAGE_VLA_COMMAND_UTF8_BASE64 = $previousEncodedCommand
    }
}
if ($smokeExitCode -ne 0) {
    throw "V7 physical smoke failed with exit code $smokeExitCode"
}

$result = Get-Content -LiteralPath $Output -Raw -Encoding UTF8 | ConvertFrom-Json
if ($result.status -ne "passed" -or -not $result.v7_chain.verified) {
    throw "V7 output did not pass the physical and chain gates: $Output"
}
Write-Output "V7 VLA smoke passed: $Output"
