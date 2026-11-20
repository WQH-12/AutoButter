param(
    [string]$Root = (Join-Path $PSScriptRoot '..')
)

Add-Type -AssemblyName System.Drawing

$shotDir = Join-Path $Root 'screenshots'
$imgDir = Join-Path $Root 'resource\image'
New-Item -ItemType Directory -Path $imgDir -Force | Out-Null

function Resize-To720 {
    param([string]$In, [string]$Out)
    $src = [System.Drawing.Image]::FromFile($In)
    $bmp = New-Object System.Drawing.Bitmap 720, 1560
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
    $g.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
    $g.DrawImage($src, (New-Object System.Drawing.Rectangle 0, 0, 720, 1560))
    $bmp.Save($Out, [System.Drawing.Imaging.ImageFormat]::Png)
    $g.Dispose(); $bmp.Dispose(); $src.Dispose()
    Write-Output "resize -> $Out"
}

function Crop-From720 {
    param([string]$In, [string]$Out, [int]$X, [int]$Y, [int]$W, [int]$H)
    $src = [System.Drawing.Image]::FromFile($In)
    $bmp = New-Object System.Drawing.Bitmap $W, $H
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.DrawImage($src, (New-Object System.Drawing.Rectangle 0, 0, $W, $H),
        (New-Object System.Drawing.Rectangle $X, $Y, $W, $H), [System.Drawing.GraphicsUnit]::Pixel)
    $bmp.Save($Out, [System.Drawing.Imaging.ImageFormat]::Png)
    $g.Dispose(); $bmp.Dispose(); $src.Dispose()
    Write-Output "crop -> $Out ($W x $H @ $X,$Y)"
}

Resize-To720 -In (Join-Path $shotDir '20_now.png') -Out (Join-Path $shotDir '21_main_720.png')
Resize-To720 -In (Join-Path $shotDir '06_menu_again.png') -Out (Join-Path $shotDir '22_menu_720.png')

# 原图 1080x2340 -> 720x1560，缩放系数 0.6667
# 底部导航栏「家园」：原图中心 (310,2215) -> 720 画布 (207,1477)
Crop-From720 -In (Join-Path $shotDir '21_main_720.png') -Out (Join-Path $imgDir 'nav_home.png') -X 157 -Y 1427 -W 100 -H 100

# 底部导航栏「冒险」：原图中心 (985,2215) -> (657,1477)
Crop-From720 -In (Join-Path $shotDir '21_main_720.png') -Out (Join-Path $imgDir 'nav_adventure.png') -X 607 -Y 1427 -W 100 -H 100

# 家园子菜单第一项「淫乱秘境」：只裁静态的建筑图标，避开会变的「闲置鸡妖：N」
# 图标实际比预想的宽，留足边距并排除右侧文字
Crop-From720 -In (Join-Path $shotDir '22_menu_720.png') -Out (Join-Path $imgDir 'home_menu_first.png') -X 2 -Y 1070 -W 108 -H 100

Write-Output '--- done ---'
Get-ChildItem $imgDir -Filter '*.png' | ForEach-Object { Write-Output "$($_.Name) $($_.Length) bytes" }
