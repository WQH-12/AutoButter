#requires -Version 5.1
<#
.SYNOPSIS
    AutoButter 运行环境自检 / 一键安装。
.DESCRIPTION
    检测并补齐运行 MFAAvalonia + agent 所需的 Windows 运行环境，全部幂等（已装即跳过）：
      [1] .NET 10 运行时      MFAAvalonia.runtimeconfig.json 依赖 Microsoft.NETCore.App 10.0.0
      [2] VC++ 2015-2022 x64  MaaFramework 原生 DLL 依赖
      [3] Python 3.10+        interface.json 的 agent.child_exec = python
      [4] pip 包              MaaFw==5.8.0 / opencv-python-headless（agent 直接 import maa/cv2/numpy）
      [5] adb                 PATH 或包内 platform-tools\，缺则下载并加入用户 PATH
.PARAMETER CheckOnly
    只检测不安装。退出码 0 = 全部就绪，1 = 有缺失。
.PARAMETER SkipAdb
    不检测 / 不安装 adb。
.PARAMETER NoElevate
    缺管理员权限时不自动提权，只报告。
.PARAMETER NoPause
    结束时不等待回车。
.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File setup\install_env.ps1 -CheckOnly
#>
[CmdletBinding()]
param(
    [switch]$CheckOnly,
    [switch]$SkipAdb,
    [switch]$NoElevate,
    [switch]$NoPause,
    [switch]$PreferMirror,
    [string]$Root = ''
)

$ErrorActionPreference = 'Continue'
try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch { }

if ($Root) {
    $PkgRoot = (Resolve-Path $Root).Path
} else {
    $PkgRoot = Split-Path -Parent $PSScriptRoot
    if (-not (Test-Path (Join-Path $PkgRoot 'MFAAvalonia.exe'))) {
        $up = Split-Path -Parent $PkgRoot
        if ($up -and (Test-Path (Join-Path $up 'MFAAvalonia.exe'))) { $PkgRoot = $up }
    }
}

$Order = @('dotnet', 'vc', 'python', 'maafw', 'maamatch', 'adb')
$Names = @{
    dotnet   = '.NET 10 运行时'
    vc       = 'VC++ 2015-2022 x64'
    python   = 'Python 3.10+'
    maafw    = 'pip MaaFw/cv2'
    maamatch = 'MaaFw 与内置 MaaFramework'
    adb      = 'adb (platform-tools)'
}

# ---------------------------------------------------------------- 通用工具

function Test-Admin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $pr = New-Object Security.Principal.WindowsPrincipal($id)
    return $pr.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Get-DisplayWidth([string]$Text) {
    $w = 0
    foreach ($ch in $Text.ToCharArray()) {
        if ([int]$ch -gt 0x2E80) { $w += 2 } else { $w += 1 }
    }
    return $w
}

function Pad-Display([string]$Text, [int]$Width) {
    $pad = $Width - (Get-DisplayWidth $Text)
    if ($pad -lt 0) { $pad = 0 }
    return $Text + (' ' * $pad)
}

function Write-Step([string]$Text) {
    Write-Host ''
    Write-Host ('== ' + $Text + ' ==') -ForegroundColor Cyan
}

function Update-SessionPath {
    $m = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $u = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:PATH = (@($m, $u) | Where-Object { $_ }) -join ';'
}

function Add-UserPath([string]$Dir) {
    $cur = [Environment]::GetEnvironmentVariable('Path', 'User')
    if (-not $cur) { $cur = '' }
    $parts = @($cur.Split(';') | Where-Object { $_ -and $_.Trim() })
    if ($parts -contains $Dir) { return }
    $new = (@($parts) + $Dir) -join ';'
    [Environment]::SetEnvironmentVariable('Path', $new, 'User')
}

function Get-RemoteFile([string]$Url, [string]$OutFile) {
    $dir = Split-Path -Parent $OutFile
    if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    if (Test-Path $OutFile) { Remove-Item -LiteralPath $OutFile -Force -ErrorAction SilentlyContinue }
    try {
        $wc = New-Object Net.WebClient
        $wc.Headers.Add('User-Agent', 'AutoButter-setup/1.0')
        $wc.DownloadFile($Url, $OutFile)
        return (Test-Path $OutFile)
    } catch {
        Write-Host ('    下载失败：' + $_.Exception.Message) -ForegroundColor Yellow
        return $false
    }
}

function Get-Sha256([string]$Path) {
    $sha = [Security.Cryptography.SHA256]::Create()
    $fs = [IO.File]::OpenRead($Path)
    try {
        $hash = $sha.ComputeHash($fs)
    } finally {
        $fs.Dispose()
        $sha.Dispose()
    }
    return ([BitConverter]::ToString($hash) -replace '-', '').ToLowerInvariant()
}

function Expand-Zip([string]$Zip, [string]$Dest) {
    try {
        Expand-Archive -LiteralPath $Zip -DestinationPath $Dest -Force -ErrorAction Stop
        return
    } catch { }
    Add-Type -AssemblyName System.IO.Compression.FileSystem -ErrorAction SilentlyContinue
    [IO.Compression.ZipFile]::ExtractToDirectory($Zip, $Dest)
}

function Get-UrlOrder([string]$Official, [string]$Mirror) {
    if (-not $Mirror) { return @($Official) }
    if ($PreferMirror) { return @($Mirror, $Official) }
    return @($Official, $Mirror)
}

function Get-RemoteFileFrom([string[]]$Urls, [string]$OutFile) {
    foreach ($u in $Urls) {
        Write-Host ('    下载 ' + $u) -ForegroundColor DarkGray
        if (Get-RemoteFile $u $OutFile) { return $true }
    }
    return $false
}

function Test-Winget {
    return [bool](Get-Command winget -ErrorAction SilentlyContinue)
}

function Install-WingetPackage([string]$Id) {
    Write-Host ('    winget install ' + $Id) -ForegroundColor DarkGray
    & winget install --id $Id -e --silent --accept-package-agreements --accept-source-agreements --disable-interactivity
    return ($LASTEXITCODE -eq 0)
}

# ---------------------------------------------------------------- 检测

function Get-DotNetState {
    $exe = $null
    $c = Get-Command dotnet -ErrorAction SilentlyContinue
    if ($c) { $exe = $c.Source }
    if (-not $exe) {
        $p = Join-Path $env:ProgramFiles 'dotnet\dotnet.exe'
        if (Test-Path $p) { $exe = $p }
    }
    if (-not $exe) { return @{ Ok = $false; Detail = '未找到 dotnet 主机（运行时未安装）' } }
    $lines = & $exe --list-runtimes 2>$null
    $hit = @($lines | Where-Object { $_ -match '^Microsoft\.(WindowsDesktop|NETCore)\.App 10\.' })
    if ($hit.Count -gt 0) {
        return @{ Ok = $true; Detail = (($hit | ForEach-Object { ($_ -split ' ')[1] }) -join ', ') }
    }
    $any = @($lines | Where-Object { $_ -match '^Microsoft\.' })
    if ($any.Count -gt 0) {
        return @{ Ok = $false; Detail = ('只有 ' + (($any | ForEach-Object { ($_ -split ' ')[1] }) -join ', ') + '，缺 10.x') }
    }
    return @{ Ok = $false; Detail = '未安装任何 .NET 运行时' }
}

function Get-VCState {
    $installed = $null
    $ver = ''
    try {
        $key = [Microsoft.Win32.Registry]::LocalMachine.OpenSubKey('SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64')
        if ($key) {
            $installed = $key.GetValue('Installed')
            $ver = [string]$key.GetValue('Version')
            $key.Close()
        }
    } catch { }
    if ($installed -eq 1) { return @{ Ok = $true; Detail = ('x64 ' + $ver) } }
    return @{ Ok = $false; Detail = '未安装 x64 运行库' }
}

function Get-PythonState {
    $cands = New-Object System.Collections.ArrayList
    $c = Get-Command python -ErrorAction SilentlyContinue
    if ($c -and $c.Source -notlike '*\WindowsApps\*') { [void]$cands.Add($c.Source) }
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        $p = & $py.Source -3 -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $p) { [void]$cands.Add($p.Trim()) }
    }
    $guess = @(
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'),
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python311\python.exe'),
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python310\python.exe'),
        (Join-Path $env:ProgramFiles 'Python312\python.exe'),
        (Join-Path $env:ProgramFiles 'Python311\python.exe'),
        (Join-Path $env:ProgramFiles 'Python310\python.exe')
    )
    foreach ($p in $guess) { if (Test-Path $p) { [void]$cands.Add($p) } }
    foreach ($p in $cands) {
        if (-not (Test-Path $p)) { continue }
        $v = & $p -c "import sys; print('%d.%d.%d' % sys.version_info[:3])" 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $v) { continue }
        $v = $v.Trim()
        $sp = $v.Split('.')
        if ([int]$sp[0] -eq 3 -and [int]$sp[1] -ge 10) {
            return @{ Ok = $true; Path = $p; Version = $v; Detail = ('Python ' + $v + '  ' + $p) }
        }
    }
    if ($cands.Count -gt 0) {
        return @{ Ok = $false; Detail = ('版本低于 3.10：' + $cands[0]) }
    }
    return @{ Ok = $false; Detail = 'PATH 与常见安装目录都没有 python' }
}

function Get-MaaFwState([string]$Py) {
    $ver = & $Py -c "import importlib.metadata as m; print(m.version('MaaFw'))" 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $ver) { return @{ Ok = $false; Detail = 'MaaFw 未安装' } }
    $ver = $ver.Trim()
    $cv = & $Py -c "import cv2; print(cv2.__version__)" 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $cv) {
        return @{ Ok = $false; Detail = ('MaaFw ' + $ver + ' 已装，缺 opencv-python-headless') }
    }
    $np = & $Py -c "import numpy; print(numpy.__version__)" 2>$null
    if ($ver -ne '5.8.0') {
        return @{ Ok = $false; Detail = ('MaaFw ' + $ver + '（需要 5.8.0，与内置 MaaFramework 对齐）') }
    }
    return @{ Ok = $true; Detail = ('MaaFw ' + $ver + ' / cv2 ' + $cv.Trim() + ' / numpy ' + $np.Trim()) }
}

function Get-MaaNativeMatch([string]$Py) {
    $native = Join-Path $PkgRoot 'runtimes\win-x64\native'
    if (-not (Test-Path $native)) { return @{ Ok = $true; Detail = '包内无 runtimes\native，跳过' } }
    $bin = & $Py -c "import maa, os; print(os.path.join(os.path.dirname(maa.__file__), 'bin'))" 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $bin) { return @{ Ok = $false; Detail = 'MaaFw 未安装' } }
    $bin = $bin.Trim()
    if (-not (Test-Path $bin)) { return @{ Ok = $false; Detail = ('MaaFw bin 目录不存在：' + $bin) } }
    $diff = @()
    $n = 0
    foreach ($f in Get-ChildItem -LiteralPath $native -Filter *.dll) {
        $n++
        $b = Join-Path $bin $f.Name
        if (-not (Test-Path $b)) { $diff += $f.Name; continue }
        try {
            $h1 = Get-Sha256 $f.FullName
            $h2 = Get-Sha256 $b
            if ($h1 -ne $h2) { $diff += $f.Name }
        } catch {
            $diff += ($f.Name + '(读取失败)')
        }
    }
    if ($diff.Count -eq 0) { return @{ Ok = $true; Detail = ([string]$n + ' 个 dll 全部一致') } }
    return @{ Ok = $false; Detail = ('协议版本不一致：' + ($diff -join ', ')) }
}

function Get-AdbState {
    $c = Get-Command adb -ErrorAction SilentlyContinue
    if ($c) { return @{ Ok = $true; Path = $c.Source; Detail = $c.Source } }
    $local = Join-Path $PkgRoot 'platform-tools\adb.exe'
    if (Test-Path $local) { return @{ Ok = $true; Path = $local; Detail = ('包内 ' + $local) } }
    return @{ Ok = $false; Detail = 'PATH 与包内都没有 adb' }
}

function Get-DetectState {
    $st = @{}
    $st.dotnet = Get-DotNetState
    $st.vc = Get-VCState
    $st.python = Get-PythonState
    if ($st.python.Ok) {
        $st.maafw = Get-MaaFwState $st.python.Path
        $st.maamatch = Get-MaaNativeMatch $st.python.Path
    } else {
        $st.maafw = @{ Ok = $false; Detail = 'Python 未就绪' }
        $st.maamatch = @{ Ok = $false; Detail = 'Python 未就绪' }
    }
    if ($SkipAdb) { $st.adb = @{ Ok = $true; Detail = '已跳过 (-SkipAdb)' } }
    else { $st.adb = Get-AdbState }
    return $st
}

function Show-Summary($st) {
    Write-Host ''
    Write-Host '================ 检查结果 ================' -ForegroundColor White
    foreach ($k in $Order) {
        $it = $st[$k]
        $name = Pad-Display $Names[$k] 30
        if ($it.Ok) {
            Write-Host ('  [ OK ] ' + $name + $it.Detail) -ForegroundColor Green
        } else {
            Write-Host ('  [缺失] ' + $name + $it.Detail) -ForegroundColor Red
        }
    }
    $bad = @($Order | Where-Object { -not $st[$_].Ok })
    Write-Host ''
    if ($bad.Count -eq 0) {
        Write-Host '全部就绪，可以运行 MFAAvalonia.exe（建议先双击「启动AutoButter.bat」）。' -ForegroundColor Green
        return $true
    }
    Write-Host ('还有 ' + $bad.Count + ' 项缺失。') -ForegroundColor Yellow
    return $false
}

# ---------------------------------------------------------------- 安装

function Install-DotNet {
    Write-Step '安装 .NET 10 桌面运行时'
    if (Test-Winget) {
        if (Install-WingetPackage 'Microsoft.DotNet.DesktopRuntime.10') { return }
        Write-Host '    winget 失败，改用官方直链' -ForegroundColor Yellow
    }
    $exe = Join-Path $env:TEMP 'autobutter-dotnet-desktop-10-x64.exe'
    if (-not (Get-RemoteFile 'https://aka.ms/dotnet/10.0/windowsdesktop-runtime-win-x64.exe' $exe)) { return }
    & $exe /install /quiet /norestart
    Write-Host ('    安装程序退出码 ' + $LASTEXITCODE + '（3010 = 需要重启）') -ForegroundColor DarkGray
    Remove-Item -LiteralPath $exe -Force -ErrorAction SilentlyContinue
}

function Install-VC {
    Write-Step '安装 VC++ 2015-2022 x64 运行库'
    if (Test-Winget) {
        if (Install-WingetPackage 'Microsoft.VCRedist.2015+.x64') { return }
        Write-Host '    winget 失败，改用官方直链' -ForegroundColor Yellow
    }
    $exe = Join-Path $env:TEMP 'autobutter-vc-redist-x64.exe'
    if (-not (Get-RemoteFile 'https://aka.ms/vs/17/release/vc_redist.x64.exe' $exe)) { return }
    & $exe /install /quiet /norestart
    Write-Host ('    安装程序退出码 ' + $LASTEXITCODE + '（3010 = 需要重启）') -ForegroundColor DarkGray
    Remove-Item -LiteralPath $exe -Force -ErrorAction SilentlyContinue
}

function Install-Python {
    Write-Step '安装 Python 3.12'
    if (Test-Winget) {
        if (Install-WingetPackage 'Python.Python.3.12') { return }
        Write-Host '    winget 失败，改用官方直链' -ForegroundColor Yellow
    }
    $exe = Join-Path $env:TEMP 'autobutter-python-3.12-amd64.exe'
    $file = 'python-3.12.10-amd64.exe'
    $official = 'https://www.python.org/ftp/python/3.12.10/' + $file
    $mirror = 'https://mirrors.huaweicloud.com/python/3.12.10/' + $file
    if (-not (Get-RemoteFileFrom (Get-UrlOrder $official $mirror) $exe)) {
        Write-Host '    自动下载失败，请手动安装 Python 3.10+：https://www.python.org/downloads/windows/' -ForegroundColor Red
        return
    }
    & $exe /quiet InstallAllUsers=1 PrependPath=1 Include_pip=1 Include_test=0
    Write-Host ('    安装程序退出码 ' + $LASTEXITCODE) -ForegroundColor DarkGray
    Remove-Item -LiteralPath $exe -Force -ErrorAction SilentlyContinue
}

function Install-PipPackages([string]$Py) {
    Write-Step '安装 pip 包（MaaFw==5.8.0 / opencv-python-headless）'
    & $Py -m pip --version 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host '    缺少 pip，尝试 ensurepip' -ForegroundColor Yellow
        & $Py -m ensurepip --upgrade 2>$null | Out-Null
    }
    $common = @('-m', 'pip', 'install', '--disable-pip-version-check', '--no-input')
    $specs = @('MaaFw==5.8.0', 'opencv-python-headless')
    $mirrors = @('https://pypi.tuna.tsinghua.edu.cn/simple',
                 'https://mirrors.aliyun.com/pypi/simple/')
    if ($PreferMirror) {
        Write-Host ('    使用镜像源：' + $mirrors[0]) -ForegroundColor DarkGray
        & $Py @common '-i' $mirrors[0] @specs
    } else {
        & $Py @common @specs
    }
    foreach ($mi in $mirrors) {
        if ($LASTEXITCODE -eq 0) { break }
        Write-Host ('    换源重试：' + $mi) -ForegroundColor Yellow
        & $Py @common '-i' $mi @specs
    }
    Write-Host ('    pip 退出码 ' + $LASTEXITCODE) -ForegroundColor DarkGray
}

function Install-Adb {
    Write-Step '安装 adb (platform-tools)'
    $zip = Join-Path $env:TEMP 'autobutter-platform-tools.zip'
    $official = 'https://dl.google.com/android/repository/platform-tools-latest-windows.zip'
    $mirror = 'https://mirrors.cloud.tencent.com/AndroidSDK/platform-tools-latest-windows.zip'
    if (-not (Get-RemoteFileFrom (Get-UrlOrder $official $mirror) $zip)) {
        Write-Host '    下载失败，可手动下载 platform-tools 后放进包内 platform-tools 目录' -ForegroundColor Red
        return
    }
    $tmp = Join-Path $env:TEMP ('autobutter-pt-' + [guid]::NewGuid().ToString('N'))
    try {
        Expand-Zip $zip $tmp
        $src = Join-Path $tmp 'platform-tools'
        if (-not (Test-Path (Join-Path $src 'adb.exe'))) {
            Write-Host '    压缩包内没有 platform-tools\adb.exe' -ForegroundColor Red
            return
        }
        $dst = Join-Path $PkgRoot 'platform-tools'
        if (Test-Path $dst) {
            if ($dst.StartsWith($PkgRoot, [StringComparison]::OrdinalIgnoreCase)) {
                Remove-Item -LiteralPath $dst -Recurse -Force
            } else {
                Write-Host ('    拒绝删除包外路径：' + $dst) -ForegroundColor Red
                return
            }
        }
        Move-Item -LiteralPath $src -Destination $dst
        Add-UserPath $dst
        $env:PATH = $dst + ';' + $env:PATH
        Write-Host ('    已解压到 ' + $dst + '，并加入用户 PATH') -ForegroundColor Green
    } finally {
        if (Test-Path $tmp) { Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue }
        if (Test-Path $zip) { Remove-Item -LiteralPath $zip -Force -ErrorAction SilentlyContinue }
    }
}

# ---------------------------------------------------------------- 主流程

Write-Host ''
Write-Host 'AutoButter 运行环境检查' -ForegroundColor White
Write-Host ('包目录: ' + $PkgRoot) -ForegroundColor DarkGray
if ($CheckOnly) { Write-Host '模式: 只检测 (-CheckOnly)' -ForegroundColor DarkGray }

$state = Get-DetectState
$allOk = Show-Summary $state

if ($CheckOnly -or $allOk) {
    if (-not $NoPause) { Write-Host ''; [void](Read-Host '按回车退出') }
    if ($allOk) { exit 0 } else { exit 1 }
}

$needSystem = (-not $state.dotnet.Ok) -or (-not $state.vc.Ok) -or (-not $state.python.Ok)

if ($needSystem -and -not (Test-Admin)) {
    if ($NoElevate) {
        Write-Host '需要管理员权限安装系统组件，请以管理员身份重新运行。' -ForegroundColor Red
        if (-not $NoPause) { Write-Host ''; [void](Read-Host '按回车退出') }
        exit 1
    }
    Write-Host ''
    Write-Host '需要管理员权限安装系统组件，正在请求提权…' -ForegroundColor Yellow
    $argLine = '-NoProfile -ExecutionPolicy Bypass -File "' + $PSCommandPath + '"'
    if ($SkipAdb) { $argLine = $argLine + ' -SkipAdb' }
    $self = (Get-Process -Id $PID).Path
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $self
    $psi.Arguments = $argLine
    $psi.UseShellExecute = $true
    $psi.Verb = 'runas'
    $proc = [System.Diagnostics.Process]::Start($psi)
    $proc.WaitForExit()
    exit $proc.ExitCode
}

Update-SessionPath

if (-not $state.dotnet.Ok) { Install-DotNet }
if (-not $state.vc.Ok) { Install-VC }
if (-not $state.python.Ok) {
    Install-Python
    Update-SessionPath
}

$py = Get-PythonState
if ($py.Ok) {
    $mw = Get-MaaFwState $py.Path
    if (-not $mw.Ok) { Install-PipPackages $py.Path }
}

if (-not $SkipAdb) {
    if (-not (Get-AdbState).Ok) { Install-Adb }
}

Write-Host ''
Write-Host '安装结束，重新检测…' -ForegroundColor White
$state = Get-DetectState
$allOk = Show-Summary $state
if (-not $NoPause) { Write-Host ''; [void](Read-Host '按回车退出') }
if ($allOk) { exit 0 } else { exit 1 }
