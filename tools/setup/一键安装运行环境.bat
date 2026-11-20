@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

rem 首次使用：双击本文件即可补齐 .NET / VC++ / Python / MaaFw / adb。
rem 参数会原样传给 setup\install_env.ps1，例如：一键安装运行环境.bat -CheckOnly

if not exist "%~dp0setup\install_env.ps1" (
    echo [错误] 找不到 setup\install_env.ps1，发布包可能不完整。
    pause
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup\install_env.ps1" %*
exit /b %ERRORLEVEL%
