# Measure the ACTUAL clock under load, and read the Win11 power-mode overlay.
# "% Processor Performance" is actual/nominal: 100 == base (1.7 GHz), 250 ~= 4.25 GHz.
$ErrorActionPreference = 'SilentlyContinue'

Write-Host "=== active power scheme ==="
powercfg /getactivescheme

Write-Host ""
Write-Host "=== Win11 power-mode overlay (separate from the scheme) ==="
$k = 'HKLM:\SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes'
$map = @{
  '961cc777-2547-4f9d-8174-7d86181b8a7a' = 'Best power efficiency'
  '00000000-0000-0000-0000-000000000000' = 'Balanced'
  'ded574b5-45a0-4f42-8737-46345c09c238' = 'Best performance'
}
foreach ($n in @('ActiveOverlayAcPowerScheme','ActiveOverlayDcPowerScheme')) {
    $v = (Get-ItemProperty -Path $k -Name $n -ErrorAction SilentlyContinue).$n
    if ($null -eq $v) { $v = '(not set)' }
    $label = if ($map.ContainsKey($v.ToLower())) { $map[$v.ToLower()] } else { '?' }
    "{0,-26} {1}  {2}" -f $n, $v, $label
}

Write-Host ""
Write-Host "=== nominal ==="
Get-CimInstance Win32_Processor | Select-Object Name, CurrentClockSpeed, MaxClockSpeed | Format-List

function Spin([int]$seconds) {
    $end = (Get-Date).AddSeconds($seconds)
    while ((Get-Date) -lt $end) {
        $x = 0
        for ($i = 0; $i -lt 200000; $i++) { $x += $i }
        if ($x -lt 0) { Write-Host $x }
    }
}

foreach ($njob in 1, 4) {
    Write-Host ("=== under {0}-thread load ===" -f $njob)
    $jobs = 1..$njob | ForEach-Object { Start-Job -ScriptBlock ${function:Spin} -ArgumentList 8 }
    Start-Sleep -Milliseconds 2500
    $perf = (Get-Counter '\Processor Information(_Total)\% Processor Performance' -SampleInterval 1 -MaxSamples 4).CounterSamples
    $freq = (Get-Counter '\Processor Information(*)\Processor Frequency' -SampleInterval 1 -MaxSamples 4).CounterSamples |
            Where-Object { $_.InstanceName -notmatch '_Total' }
    $pmax = ($perf | Measure-Object -Property CookedValue -Maximum).Maximum
    $pmin = ($perf | Measure-Object -Property CookedValue -Minimum).Minimum
    "  % Processor Performance : min {0:N1}%   max {1:N1}%   -> approx {2:N2}-{3:N2} GHz" -f `
        $pmin, $pmax, ($pmin * 1.7 / 100), ($pmax * 1.7 / 100)
    "  top core frequency      : {0:N0} MHz" -f (($freq | Measure-Object -Property CookedValue -Maximum).Maximum)
    $jobs | Stop-Job
    $jobs | Remove-Job
}
