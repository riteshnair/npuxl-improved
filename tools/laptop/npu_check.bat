@echo off
REM Query OpenVINO for available devices (expect CPU, GPU, NPU)
call "C:\Users\rina0423\Desktop\openvino_toolkit\setupvars.bat" >nul 2>&1
set "PYTHONPATH=C:\Users\rina0423\Desktop\openvino_toolkit\python;%PYTHONPATH%"
python -c "import openvino as ov; c=ov.Core(); print('openvino', ov.__version__); print('devices', c.available_devices); [print(' ',d, c.get_property(d,'SUPPORTED_PROPERTIES') if False else '') for d in c.available_devices]"
