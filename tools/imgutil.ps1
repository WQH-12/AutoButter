param(
    [Parameter(Mandatory = $true)][ValidateSet('resize', 'crop', 'grid')][string]$Mode,
    [Parameter(Mandatory = $true)][string]$In,
    [string]$Out,
    [int]$Width = 720,
    [int]$Height = 1560,
    [int]$X = 0,
    [int]$Y = 0,
    [int]$W = 0,
    [int]$H = 0,
    [int]$Step = 60
)

Add-Type -AssemblyName System.Drawing

if ($Mode -eq 'resize') {
    $src = [System.Drawing.Image]::FromFile($In)
    $bmp = New-Object System.Drawing.Bitmap $Width, $Height
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
    $g.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
    $g.DrawImage($src, (New-Object System.Drawing.Rectangle 0, 0, $Width, $Height))
    $bmp.Save($Out, [System.Drawing.Imaging.ImageFormat]::Png)
    $g.Dispose(); $bmp.Dispose(); $src.Dispose()
    Write-Output "resized -> $Out ($Width x $Height)"
    return
}

if ($Mode -eq 'crop') {
    if ($W -le 0 -or $H -le 0) { throw 'crop requires -W and -H' }
    $src = [System.Drawing.Image]::FromFile($In)
    $bmp = New-Object System.Drawing.Bitmap $W, $H
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $srcRect = New-Object System.Drawing.Rectangle $X, $Y, $W, $H
    $dstRect = New-Object System.Drawing.Rectangle 0, 0, $W, $H
    $g.DrawImage($src, $dstRect, $srcRect, [System.Drawing.GraphicsUnit]::Pixel)
    $bmp.Save($Out, [System.Drawing.Imaging.ImageFormat]::Png)
    $g.Dispose(); $bmp.Dispose(); $src.Dispose()
    Write-Output "cropped -> $Out ($W x $H from $X,$Y)"
    return
}

if ($Mode -eq 'grid') {
    $src = [System.Drawing.Image]::FromFile($In)
    $bmp = New-Object System.Drawing.Bitmap $src.Width, $src.Height
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.DrawImage($src, 0, 0, $src.Width, $src.Height)
    $pen = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(150, 255, 0, 0)), 1
    $font = New-Object System.Drawing.Font 'Consolas', 11
    $brush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(255, 255, 255, 0))
    for ($gx = 0; $gx -lt $src.Width; $gx += $Step) {
        $g.DrawLine($pen, $gx, 0, $gx, $src.Height)
        $g.DrawString("$gx", $font, $brush, $gx + 2, 2)
    }
    for ($gy = 0; $gy -lt $src.Height; $gy += $Step) {
        $g.DrawLine($pen, 0, $gy, $src.Width, $gy)
        $g.DrawString("$gy", $font, $brush, 2, $gy + 1)
    }
    $bmp.Save($Out, [System.Drawing.Imaging.ImageFormat]::Png)
    $pen.Dispose(); $font.Dispose(); $brush.Dispose(); $g.Dispose(); $bmp.Dispose(); $src.Dispose()
    Write-Output "grid -> $Out (step $Step)"
    return
}
