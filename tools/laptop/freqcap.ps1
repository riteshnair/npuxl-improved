# Pin the CPU to a fixed absolute frequency (default 2.4 GHz) on the active scheme.
#   PROCFREQMAX is an absolute MHz cap (hidden by default); PROCTHROTTLEMAX is only
#   a % of nominal base, so it cannot express >1.7 GHz here.
#   PROCTHROTTLEMIN=100 stops ramp-down; PERFBOOSTMODE=0 stops turbo overshoot.
# Revert:  powercfg /setactive SCHEME_BALANCED
param([double]$GHz = 2.4)
$ErrorActionPreference = 'Continue'

$mhz = [int]($GHz * 1000)

powercfg -attributes SUB_PROCESSOR PROCFREQMAX -ATTRIB_HIDE | Out-Null

$cur = (powercfg /getactivescheme)
if ($cur -notmatch 'GUID:\s*([0-9a-fA-F-]{36})') { throw "cannot read active scheme" }
$g = $Matches[1]
Write-Host "scheme : $g"
Write-Host "cap    : $mhz MHz"

powercfg /setacvalueindex $g SUB_PROCESSOR PROCTHROTTLEMIN 100 | Out-Null
powercfg /setdcvalueindex $g SUB_PROCESSOR PROCTHROTTLEMIN 100 | Out-Null
powercfg /setacvalueindex $g SUB_PROCESSOR PROCFREQMAX    $mhz | Out-Null
powercfg /setdcvalueindex $g SUB_PROCESSOR PROCFREQMAX    $mhz | Out-Null
powercfg /setacvalueindex $g SUB_PROCESSOR PERFBOOSTMODE  0 2>$null | Out-Null
powercfg /setdcvalueindex $g SUB_PROCESSOR PERFBOOSTMODE  0 2>$null | Out-Null
powercfg /setactive $g | Out-Null

Write-Host ""
Write-Host "=== settings now in effect ==="
powercfg /q SCHEME_CURRENT SUB_PROCESSOR |
    Select-String 'PROCFREQMAX|PROCTHROTTLEMIN|PERFBOOSTMODE|Current A[CD]' |
    ForEach-Object { "  " + $_.Line.Trim() }

# ---- verify: 4 spinners pinned to the P-core logical mask ----
$spin = '$e=(Get-Date).AddSeconds(12); while((Get-Date) -lt $e){ $x=0; for($i=0;$i -lt 200000;$i++){ $x+=$i }; if($x -lt 0){ Write-Host $x } }'
$PMASK = [IntPtr]0xFC03   # P-core logicals: 0,1 and 10..15
Write-Host ""
Write-Host "=== verify: 4 threads pinned to P-core mask 0xFC03 ==="
$procs = 1..4 | ForEach-Object {
    $p = Start-Process -FilePath 'powershell' -ArgumentList @('-NoProfile', '-Command', $spin) -PassThru -WindowStyle Hidden
    $p.ProcessorAffinity = $PMASK
    $p
}
Start-Sleep -Milliseconds 2500
$perf = (Get-Counter '\Processor Information(*)\% Processor Performance' -SampleInterval 1 -MaxSamples 3).CounterSamples
$util = (Get-Counter '\Processor Information(*)\% Processor Time'          -SampleInterval 1 -MaxSamples 3).CounterSamples
$names = $perf | Where-Object { $_.InstanceName -notmatch '_Total' } | Select-Object -ExpandProperty InstanceName -Unique
$rows = foreach ($nm in $names) {
    $pv = @($perf | Where-Object { $_.InstanceName -eq $nm } | ForEach-Object { $_.CookedValue })
    $uv = @($util | Where-Object { $_.InstanceName -eq $nm } | ForEach-Object { $_.CookedValue })
    if ($pv.Count -eq 0) { continue }
    [pscustomobject]@{
        cpu  = $nm
        util = [math]::Round(($uv | Measure-Object -Average).Average, 0)
        perf = [math]::Round(($pv | Measure-Object -Average).Average, 0)
        GHz  = [math]::Round((($pv | Measure-Object -Average).Average * 1.7 / 100), 2)
    }
}
$rows | Sort-Object util -Descending | Select-Object -First 6 | Format-Table -AutoSize | Out-String | Write-Host
$hot = $rows | Where-Object { $_.util -gt 50 }
if ($hot) {
    Write-Host ("  busy-core clock: {0:N2} - {1:N2} GHz  ({2} cores busy)" -f `
        (($hot | Measure-Object -Property GHz -Minimum).Minimum),
        (($hot | Measure-Object -Property GHz -Maximum).Maximum),
        $hot.Count)
} else {
    Write-Host "  no core above 50% - load did not land"
}
$procs | ForEach-Object { $_.Kill() }
Write-Host "revert:  powercfg /setactive SCHEME_BALANCED"
