@echo off
REM duplex-voice 启动（Windows）——跨平台逻辑见 start.py
REM 用法：start.bat [--vad omni|rule]
cd /d "%~dp0"
python start.py %*
