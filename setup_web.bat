@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo 未找到 .venv\Scripts\python.exe。
  echo 请先创建项目虚拟环境，并安装与本机硬件匹配的 PyTorch 和 FunASR。
  exit /b 1
)

where node >nul 2>nul
if errorlevel 1 (
  echo 未找到 Node.js。请安装 Node.js 20.19 或更高版本后重试。
  exit /b 1
)

where npm >nul 2>nul
if errorlevel 1 (
  echo 未找到 npm。请重新安装包含 npm 的 Node.js 后重试。
  exit /b 1
)

if not exist frontend\package-lock.json (
  echo 未找到 frontend\package-lock.json，无法执行可重复的前端依赖安装。
  exit /b 1
)

"%PYTHON%" -m ensurepip --upgrade
if errorlevel 1 goto :python_error

"%PYTHON%" -m pip install -e .
if errorlevel 1 goto :python_error

npm --prefix frontend ci
if errorlevel 1 goto :frontend_install_error

npm --prefix frontend run build
if errorlevel 1 goto :frontend_build_error

echo Web Python 依赖与生产前端构建已就绪。
exit /b 0

:python_error
echo Python Web 依赖安装失败。请检查 .venv、pip 和网络后重试。
exit /b 1

:frontend_install_error
echo 前端依赖安装失败。请检查 Node.js、npm、package-lock.json 和网络后重试。
exit /b 1

:frontend_build_error
echo 前端构建失败。请检查上方 npm 输出后重试。
exit /b 1
