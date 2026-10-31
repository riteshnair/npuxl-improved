# Compare the ALL-CORE P-core clock with turbo on vs turbo off.
# Turbo-on measured 2.0 GHz all-core earlier; turbo-off measured 2.31 GHz
# single-thread. If turbo-off also holds ~2.3 GHz with 4 threads, it is both
# the more reproducible AND the faster setting for a 4-core GEMM.
$ErrorActionPreference = 'SilentlyContinue'
$PMASK = [IntPtr]0xFC03     # P-core logicals 0,1,10..15

function PinnedClock([int]$threads) {
    $spin = '$e=(Get-Date).AddSeconds(5); while((Get-Date) -lt $e){ $x=0; for($i=0;$i -lt 200000;$i++){ $x+=$i }; if($x -lt 0){Write-Host $x} }'
    $procs = 1..$threads | ForEach-Object {
        $p = Start-Process -FilePath 'powershell' -ArgumentList @('-NoProfile','-Command',$spin) -PassThru -WindowStyle Hidden
        $p.ProcessorAffinity = $PMASK
        $p
    }
    Start-Sleep -Milliseconds 2200
    $c = (Get-Counter '\Processor Information(*)\% Processor Performance' -SampleInterval 1 -MaxSamples 3).CounterSamples
    # Only the P-core logicals we actually pinned to. Ranking by instantaneous
    # perf% across ALL cores picks idle LP-E cores, which report nonsense.
    $maskNames = @('0,0','0,1','0,10','0,11','0,12','0,13','0,14','0,15')
    $hot = $c | Where-Object { $_.InstanceName -in $maskNames } |
           Group-Object InstanceName |
           ForEach-Object {
               [pscustomobject]@{
                   cpu  = $_.Name
                   perf = ($_.Group | Measure-Object -Property CookedValue -Average).Average
               }
           } | Sort-Object perf -Descending | Select-Object -First $threads
    $procs | ForEach-Object { $_.Kill() }
    $avg = ($hot | Measure-Object -Property perf -Average).Average
    return [pscustomobject]@{
        GHz = [math]::Round(($avg * 1.7 / 100), 2)
        Cores = ($hot | ForEach-Object { '{0}={1:N0}%' -f $_.cpu, $_.perf }) -join '  '
    }
}

foreach ($mode in @(@{n='turbo ON  (boost=2)'; v=2}, @{n='turbo OFF (boost=0)'; v=0})) {
    powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PERFBOOSTMODE $mode.v | Out-Null
    powercfg /setdcvalueindex SCHEME_CURRENT SUB_PROCESSOR PERFBOOSTMODE $mode.v | Out-Null
    powercfg /setactive SCHEME_CURRENT | Out-Null
    Start-Sleep -Milliseconds 500
    $r = PinnedClock 4
    Write-Host ("{0}  1-thread: {1,5:N2} GHz   4-thread: {2,5:N2} GHz   [{3}]" -f `
        $mode.n, (PinnedClock 1).GHz, $r.GHz, $r.Cores)
}
# leave turbo ON
powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PERFBOOSTMODE 2 | Out-Null
powercfg /setdcvalueindex SCHEME_CURRENT SUB_PROCESSOR PERFBOOSTMODE 2 | Out-Null
powercfg /setactive SCHEME_CURRENT | Out-Null
Write-Host "left on: turbo ON"
