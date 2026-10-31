@echo off
call "C:\Users\rina0423\Desktop\openvino_toolkit\setupvars.bat" >nul 2>&1
set "PYTHONPATH=C:\Users\rina0423\Desktop\openvino_toolkit\python;%PYTHONPATH%"
python "%~dp0npu_matmul.py"
