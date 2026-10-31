# Per-core clock under load. _Total averages in idle E-cores, so read instances.
# % Processor Performance: 100 == 1.7 GHz nominal, 250 ~= 4.25 GHz.
$ErrorActionPreference = 'SilentlyContinue'

function Spin([int]$seconds) {
    $end = (Get-Date).AddSeconds($seconds)
    while ((Get-Date) -lt $end) {
        $x = 0
        for ($i = 0; $i -lt 200000; $i++) { $x += $i }
        if ($x -lt 0) { Write-Host $x }
    }
}

function Avg([object]$samples, [string]$name) {
    $vals = @($samples | Where-Object { $_.InstanceName -eq $name } | ForEach-Object { $_.CookedValue })
    if ($vals.Count -eq 0) { return 0 }
    return ($vals | Measure-Object -Average).Average
}

foreach ($n in 1, 4) {
    Write-Host ("=== {0} busy thread(s) ===" -f $n)
    $jobs = 1..$n | ForEach-Object { Start-Job -ScriptBlock ${function:Spin} -ArgumentList 10 }
    Start-Sleep -Milliseconds 2500
    $perf = (Get-Counter '\Processor Information(*)\% Processor Performance' -SampleInterval 1 -MaxSamples 3).CounterSamples
    $util = (Get-Counter '\Processor Information(*)\% Processor Time'          -SampleInterval 1 -MaxSamples 3).CounterSamples
    $names = $perf | Where-Object { $_.InstanceName -notmatch '_Total' } |
             Select-Object -ExpandProperty InstanceName -Unique
    $rows = foreach ($nm in $names) {
        [pscustomobject]@{
            cpu   = $nm
            util  = [math]::Round((Avg $util $nm), 0)
            perf  = [math]::Round((Avg $perf $nm), 0)
            GHz   = [math]::Round(((Avg $perf $nm) * 1.7 / 100), 2)
        }
    }
    $busy = $rows | Sort-Object util -Descending | Select-Object -First 6
    $busy | Format-Table -AutoSize | Out-String | Write-Host
    $m = ($rows | Measure-Object -Property perf -Maximum).Maximum
    Write-Host ("  peak core: {0:N0}%  ~{1:N2} GHz    busy cores: {2}" -f `
        $m, ($m * 1.7 / 100), (($rows | Where-Object { $_.util -gt 50 }).Count))
    $jobs | Stop-Job
    $jobs | Remove-Job
    Write-Host ""
}
