param(
    [string]$Root = (Join-Path $PSScriptRoot '..')
)

$install = Join-Path $Root 'install'

if (-not (Test-Path (Join-Path $install 'MFAAvalonia.exe'))) {
    throw "install dir not ready: $install"
}

# 资源：pipeline / image / model 全量覆盖
Copy-Item -Path (Join-Path $Root 'resource\*') -Destination (Join-Path $install 'resource') -Recurse -Force
Copy-Item -Path (Join-Path $Root 'interface.json') -Destination (Join-Path $install 'interface.json') -Force
Copy-Item -Path (Join-Path $Root 'README.md') -Destination (Join-Path $install 'README.md') -Force

# agent：自定义识别/动作的源码。MFAAvalonia 用 interface.json 的 agent 段
# 起子进程，child_args 指向 ./agent/main.py，所以这份必须同步，
# 否则 install/agent 会一直是旧版（甚至缺文件）。
$agentSrc = Join-Path $Root 'agent'
$agentDst = Join-Path $install 'agent'
if (Test-Path $agentSrc) {
    if (-not (Test-Path $agentDst)) {
        New-Item -ItemType Directory -Path $agentDst -Force | Out-Null
    }
    Copy-Item -Path (Join-Path $agentSrc '*.py') -Destination $agentDst -Force
}

Write-Output '--- synced ---'
Get-ChildItem (Join-Path $install 'resource\pipeline') -File | ForEach-Object { "pipeline/$($_.Name)" }
Get-ChildItem (Join-Path $install 'resource\image') -File | ForEach-Object { "image/$($_.Name) $($_.Length)" }
Get-ChildItem (Join-Path $install 'agent') -File -Filter '*.py' | ForEach-Object { "agent/$($_.Name)" }
