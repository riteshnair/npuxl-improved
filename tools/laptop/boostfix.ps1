# Turbo was accidentally disabled by freqcap.ps1 (PERFBOOSTMODE 0). Measure the
# single-thread turbo clock, re-enable boost, measure again.
$ErrorActionPreference = 'SilentlyContinue'

function ClockGHz {
    $spin = '$e=(Get-Date).AddSeconds(4); while((Get-Date) -lt $e){ $x=0; for($i=0;$i -lt 200000;$i++){ $x+=$i }; if($x -lt 0){Write-Host $x} }'
    $p = Start-Process -FilePath 'powershell' -ArgumentList @('-NoProfile', '-Command', $spin) -PassThru -WindowStyle Hidden
    $p.ProcessorAffinity = [IntPtr]0xFC03     # a P-core logical
    Start-Sleep -Milliseconds 1800
    $c = (Get-Counter '\Processor Information(*)\% Processor Performance' -SampleInterval 1 -MaxSamples 3).CounterSamples
    $m = ($c | Where-Object { $_.InstanceName -notmatch '_Total' } |
          Measure-Object -Property CookedValue -Maximum).Maximum
    $p.Kill()
    if ($null -eq $m) { return 0 }
    return [math]::Round(($m * 1.7 / 100), 2)
}

Write-Host ("single-thread turbo clock BEFORE: {0:N2} GHz" -f (ClockGHz))

# 0 = Disabled, 1 = Enabled, 2 = Aggressive
powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PERFBOOSTMODE 2 | Out-Null
powercfg /setdcvalueindex SCHEME_CURRENT SUB_PROCESSOR PERFBOOSTMODE 2 | Out-Null
powercfg /setactive SCHEME_CURRENT | Out-Null

Write-Host ("single-thread turbo clock AFTER : {0:N2} GHz" -f (ClockGHz))
