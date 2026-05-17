param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $Root "venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    $Python = Join-Path $Root ".venv\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Expected venv Python not found under venv or .venv."
}

Push-Location -LiteralPath $Root
try {
    & $Python (Join-Path $Root "tools\run_phase53_active_shadow_cycle.py") @ExtraArgs
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}
