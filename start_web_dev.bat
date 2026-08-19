@echo off
setlocal
cd /d "%~dp0"
set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo 未找到 .venv\Scripts\python.exe。请先创建并安装 FunASR 虚拟环境。
  exit /b 1
)
start "FunASR_e2e API" cmd /k ""%PYTHON%" scripts\launch_web.py --app-data app_data_test --port 8000"
start "FunASR_e2e Frontend" cmd /k "npm --prefix frontend run dev -- --host 127.0.0.1"
start "FunASR_e2e Web" http://127.0.0.1:5173
