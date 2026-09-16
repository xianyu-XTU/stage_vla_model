param(
    [string]$IsaacLabRoot = "E:\work\IsaacLab",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$EvaluatorArguments
)

$runner = Join-Path (Split-Path -Parent $PSScriptRoot) "evaluate\run_v7_multicube_chain.ps1"
& $runner -IsaacLabRoot $IsaacLabRoot @EvaluatorArguments
