param(
    [Parameter(Mandatory = $true)][int]$X,
    [Parameter(Mandatory = $true)][int]$Y,
    [int]$WaitMs = 1200,
    [string]$Shot,
    [string]$Adb = 'adb'
)

& $Adb shell input tap $X $Y
Start-Sleep -Milliseconds $WaitMs

if ($Shot) {
    & (Join-Path $PSScriptRoot 'shot.ps1') -Name $Shot -Adb $Adb
}
else {
    Write-Output "tapped ($X, $Y)"
}
