param(
    [string]$IsaacLabRoot = "E:\work\IsaacLab",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$EvaluatorArguments
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$python = Join-Path $IsaacLabRoot "_isaac_sim\python.bat"
$evaluator = Join-Path $repoRoot "tools\eval_v7_multicube_chain.py"
foreach ($path in @($python, $evaluator)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required evaluation entry is missing: $path"
    }
}
& $python $evaluator @EvaluatorArguments
if ($LASTEXITCODE -ne 0) {
    throw "V7 evaluation failed with exit code $LASTEXITCODE"
}
