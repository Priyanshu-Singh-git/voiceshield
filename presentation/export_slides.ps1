# Export deck.pptx with PowerPoint: one 1920x1080 PNG per slide + deck.pdf
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$pptx = Join-Path $here "deck.pptx"
$slides = Join-Path $here "slides"
New-Item -ItemType Directory -Force $slides | Out-Null
Get-ChildItem $slides -Filter *.png | Remove-Item -Force

$app = New-Object -ComObject PowerPoint.Application
try {
    # open read-only, untitled, without a window
    $pres = $app.Presentations.Open($pptx, $true, $false, $false)
    foreach ($s in $pres.Slides) {
        $out = Join-Path $slides ("slide_{0:D2}.png" -f $s.SlideIndex)
        $s.Export($out, "PNG", 1920, 1080)
    }
    $n = $pres.Slides.Count
    $pres.SaveAs((Join-Path $here "deck.pdf"), 32)  # 32 = ppSaveAsPDF
    $pres.Close()
    Write-Output ("exported {0} slides + deck.pdf" -f $n)
} finally {
    $app.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($app) | Out-Null
}
