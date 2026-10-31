# Pin the CPU clock for benchmarking, without touching the user's Balanced plan.
# Clones the active scheme, forces PROCTHROTTLEMIN=100% (never downclock) and
# boost mode = Aggressive, then activates it. Revert with:
#     powercfg /setactive SCHEME_BALANCED
$ErrorActionPreference = 'Continue'

$cur = (powercfg /getactivescheme)
if ($cur -notmatch 'GUID:\s*([0-9a-fA-F-]{36})') { throw "cannot read active scheme: $cur" }
$base = $Matches[1]
Write-Host "base scheme : $cur"

$dup = (powercfg /duplicatescheme $base) 2>&1 | Out-String
if ($dup -notmatch '([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})') {
    Write-Host "duplicatescheme failed (needs elevation?): $dup"
    exit 1
}
$g = $Matches[1]
Write-Host "cloned to   : $g"

powercfg /changename $g "LiGGUF Bench (clock pinned)" | Out-Null

# 100% minimum processor state on both AC and DC -> no SpeedStep ramp-down.
foreach ($v in @('PROCTHROTTLEMIN', 'PROCTHROTTLEMAX')) {
    powercfg /setacvalueindex $g SUB_PROCESSOR $v 100 | Out-Null
    powercfg /setdcvalueindex $g SUB_PROCESSOR $v 100 | Out-Null
}
# MaximumPerformance=1 on the performance hint knobs, and Aggressive turbo.
powercfg /setacvalueindex $g SUB_PROCESSOR PERFBOOSTMODE 2 2>$null | Out-Null
powercfg /setdcvalueindex $g SUB_PROCESSOR PERFBOOSTMODE 2 2>$null | Out-Null

powercfg /setactive $g
if ($LASTEXITCODE -ne 0) { Write-Host "setactive failed (needs elevation)"; exit 1 }

Write-Host ""
Write-Host "=== pins now in effect ==="
powercfg /q SCHEME_CURRENT SUB_PROCESSOR |
    Select-String -Pattern 'PROCTHROTTLEMIN|PROCTHROTTLEMAX|PERFBOOSTMODE|Current A[CD]' |
    ForEach-Object { $_.Line.Trim() }

Write-Host ""
Write-Host "=== battery ==="
Get-CimInstance Win32_Battery | Select-Object BatteryStatus, EstimatedChargeRemaining | Format-List

Write-Host "revert with:  powercfg /setactive SCHEME_BALANCED"
