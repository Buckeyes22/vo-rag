# vo-rag OCR farm (VOR-1427): run one bootstrap step on a Windows worker with
# a time limit enforced on this host. On timeout it stops the process tree
# rooted at the exact PID it started (taskkill /T /F /PID $($p.Id)) and nothing else,
# then exits 124; otherwise it exits with the step's own exit code. A
# -TimeoutSeconds outside 1..32767 (Wait-Process's range) exits 2 before
# anything starts. Invoked as one line from cmd over ssh:
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File farm_step.ps1 `
#     -TimeoutSeconds 3600 -FilePath uv -Arguments "pip install ..."
param(
    [Parameter(Mandatory = $true)][int]$TimeoutSeconds,
    [Parameter(Mandatory = $true)][string]$FilePath,
    [Parameter(Mandatory = $true)][string]$Arguments
)
$ErrorActionPreference = "Stop"

if ($TimeoutSeconds -lt 1 -or $TimeoutSeconds -gt 32767) {
    [Console]::Error.WriteLine("farm_step: -TimeoutSeconds must be 1..32767 (Wait-Process limit), got $TimeoutSeconds")
    exit 2
}

$stdout = Join-Path $PSScriptRoot ("farm_step." + $PID + ".out")
$stderr = Join-Path $PSScriptRoot ("farm_step." + $PID + ".err")

function Write-StepOutput {
    foreach ($pair in @(@($stdout, [Console]::Out), @($stderr, [Console]::Error))) {
        if (Test-Path $pair[0]) {
            $pair[1].Write([string](Get-Content -Raw -ErrorAction SilentlyContinue $pair[0]))
            Remove-Item -Force -ErrorAction SilentlyContinue $pair[0]
        }
    }
}

$p = Start-Process -FilePath $FilePath -ArgumentList $Arguments -PassThru -NoNewWindow `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr
$null = $p.Handle  # keep the handle so ExitCode is readable after the process exits

Wait-Process -Id $p.Id -Timeout $TimeoutSeconds -ErrorAction SilentlyContinue
if (-not $p.HasExited) {
    # The tree rooted at the exact PID we started (uv's children too); never by name.
    taskkill /T /F /PID $($p.Id) | Out-Null
    $p.WaitForExit()
    Write-StepOutput
    [Console]::Error.WriteLine("farm_step: timed out after $TimeoutSeconds s; stopped the tree of PID $($p.Id)")
    exit 124
}
$p.WaitForExit()
Write-StepOutput
exit $p.ExitCode
