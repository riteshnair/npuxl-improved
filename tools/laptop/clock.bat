@echo off
REM Report the current power scheme and the processor-performance settings that
REM control whether this CPU is allowed to vary its clock (SpeedStep / turbo).
echo === active power scheme ===
powercfg /getactivescheme
echo.
echo === processor settings: alias + AC/DC values (current scheme) ===
powercfg /q SCHEME_CURRENT SUB_PROCESSOR | findstr /C:"Alias" /C:"Current AC" /C:"Current DC"
echo.
echo === nominal clocks ===
powershell -NoProfile -Command "Get-CimInstance Win32_Processor | Select-Object Name,CurrentClockSpeed,MaxClockSpeed | Format-List"
echo === per-core current clock (MHz) ===
powershell -NoProfile -Command "(Get-Counter '\Processor Information(*)\Processor Frequency' -ErrorAction SilentlyContinue).CounterSamples | Where-Object { $_.InstanceName -notmatch '_Total|_Total' } | Select-Object InstanceName,CookedValue | Format-Table -AutoSize"
