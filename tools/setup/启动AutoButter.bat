@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

rem 先做一次只读自检；缺运行环境时询问是否安装，装完再启动。
rem 想直接启动（跳过检查）：双击 MFAAvalonia.exe。

if not exist "MFAAvalonia.exe" (
    echo [错误] 找不到 MFAAvalonia.exe，请在发布包根目录运行本脚本。
    pause
    exit /b 1
)
if not exist "setup\install_env.ps1" (
    echo [错误] 找不到 setup\install_env.ps1，发布包可能不完整。
    pause
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "setup\install_env.ps1" -CheckOnly -NoPause >nul 2>&1
if not errorlevel 1 goto run

echo.
echo 检测到运行环境不完整，明细如下：
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "setup\install_env.ps1" -CheckOnly -NoPause
echo.
choice /c YN /m "是否现在自动安装缺失的运行环境（需要管理员权限）"
if errorlevel 2 goto run

echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "setup\install_env.ps1"
if errorlevel 1 (
    echo.
    echo 运行环境仍未就绪，已取消启动。
    pause
    exit /b 1
)

:run
echo.
echo 正在启动 MFAAvalonia...
start "" "MFAAvalonia.exe"
exit /b 0
