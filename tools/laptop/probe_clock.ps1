# READ-ONLY clock probe. Does not touch any power setting - safe to run while
# the user is deciding what to set. Measures what the CPU actually delivers to
# threads pinned to the P-cores, for 1 and 4 concurrent threads.
$ErrorActionPreference = 'SilentlyContinue'
$PMASK = [IntPtr]0xFC03                                   # P-core logicals 0,1,10..15
$maskNames = @('0,0','0,1','0,10','0,11','0,12','0,13','0,14','0,15')

function PinnedClock([int]$threads) {
    $spin = '$e=(Get-Date).AddSeconds(7); while((Get-Date) -lt $e){ $x=0; for($i=0;$i -lt 200000;$i++){ $x+=$i }; if($x -lt 0){Write-Host $x} }'
    $procs = 1..$threads | ForEach-Object {
        $p = Start-Process -FilePath 'powershell' -ArgumentList @('-NoProfile','-Command',$spin) -PassThru -WindowStyle Hidden
        $p.ProcessorAffinity = $PMASK
        $p
    }
    Start-Sleep -Milliseconds 4000                         # let the P-state settle
    $c = (Get-Counter '\Processor Information(*)\% Processor Performance' -SampleInterval 1 -MaxSamples 3).CounterSamples
    $hot = $c | Where-Object { $_.InstanceName -in $maskNames } |
           Group-Object InstanceName |
           ForEach-Object {
               [pscustomobject]@{ cpu = $_.Name; perf = ($_.Group | Measure-Object -Property CookedValue -Average).Average }
           } | Sort-Object perf -Descending | Select-Object -First $threads
    $procs | ForEach-Object { $_.Kill() }
    $avg = ($hot | Measure-Object -Property perf -Average).Average
    return [pscustomobject]@{
        GHz    = [math]::Round(($avg * 1.7 / 100), 2)
        Detail = (($hot | ForEach-Object { '{0}={1:N0}%' -f $_.cpu, $_.perf }) -join '  ')
    }
}

Write-Host "=== active scheme ==="
powercfg /getactivescheme
$k = 'HKLM:\SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes'
$ov = (Get-ItemProperty -Path $k -Name 'ActiveOverlayAcPowerScheme' -ErrorAction SilentlyContinue).ActiveOverlayAcPowerScheme
Write-Host "power-mode overlay: $ov"
Write-Host ""
$one  = PinnedClock 1
$four = PinnedClock 4
Write-Host ("1-thread : {0,5:N2} GHz   [{1}]" -f $one.GHz,  $one.Detail)
Write-Host ("4-thread : {0,5:N2} GHz   [{1}]" -f $four.GHz, $four.Detail)
