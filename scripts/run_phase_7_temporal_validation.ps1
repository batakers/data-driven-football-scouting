$ErrorActionPreference = "Stop"

function Invoke-PythonStep {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,

        [Parameter(Mandatory = $true)]
        [string]$ScriptPath
    )

    Write-Host "Running $Name..."
    python $ScriptPath
    if ($LASTEXITCODE -ne 0) {
        Write-Error "$Name failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
}

Invoke-PythonStep -Name "Phase 7 temporal backtest" -ScriptPath "src/temporal_backtest.py"
Invoke-PythonStep -Name "Phase 7 temporal validation" -ScriptPath "src/validate_temporal_backtest.py"
