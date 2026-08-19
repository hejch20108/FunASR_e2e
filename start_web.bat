@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo 未找到 .venv\Scripts\python.exe。请先完成 Python 与 FunASR 环境安装。
  exit /b 1
)

if not exist frontend\dist\index.html (
  echo 未找到生产前端 frontend\dist\index.html。请先运行 setup_web.bat。
  exit /b 1
)

start "FunASR_e2e Web" http://127.0.0.1:8000
"%PYTHON%" scripts\launch_web.py
