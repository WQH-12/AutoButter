param(
    [Parameter(Mandatory = $true)][string]$Name,
    [string]$Adb = 'adb'
)

$outDir = Join-Path $PSScriptRoot '..\screenshots' | Resolve-Path -ErrorAction SilentlyContinue
if (-not $outDir) {
    $outDir = Join-Path $PSScriptRoot '..\screenshots'
    New-Item -ItemType Directory -Path $outDir -Force | Out-Null
}

$target = Join-Path $outDir "$Name.png"

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $Adb
$psi.Arguments = 'exec-out screencap -p'
$psi.RedirectStandardOutput = $true
$psi.UseShellExecute = $false

$proc = [System.Diagnostics.Process]::Start($psi)
$ms = New-Object System.IO.MemoryStream
$proc.StandardOutput.BaseStream.CopyTo($ms)
$proc.WaitForExit()
[System.IO.File]::WriteAllBytes($target, $ms.ToArray())
$ms.Dispose()

$len = (Get-Item $target).Length
Write-Output "$target ($len bytes)"
