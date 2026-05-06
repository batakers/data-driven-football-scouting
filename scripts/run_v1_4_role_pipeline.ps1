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

Invoke-PythonStep -Name "Kaggle enrichment" -ScriptPath "src/enrich_similarity.py"
Invoke-PythonStep -Name "Enrichment validation" -ScriptPath "src/validate_enrichment.py"
Invoke-PythonStep -Name "Transfermarkt role enrichment" -ScriptPath "src/enrich_roles.py"
Invoke-PythonStep -Name "Role-aware similarity engine build" -ScriptPath "src/similarity.py"
Invoke-PythonStep -Name "Role validation" -ScriptPath "src/validate_roles.py"
