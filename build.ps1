# 声鉴打包脚本: PyInstaller onedir + 无控制台 + 裁剪无用DLL
# 用法: powershell -ExecutionPolicy Bypass -File build.ps1
# 注意: 产物目录名含中文, 脚本内不写字面量, 统一动态解析
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

Write-Host "[1/3] PyInstaller 打包..." -ForegroundColor Cyan
python -m PyInstaller build_app.spec --noconfirm --clean
if ($LASTEXITCODE -ne 0) { throw "PyInstaller 打包失败" }

Write-Host "[2/3] 裁剪无用 DLL..." -ForegroundColor Cyan
$distDir = (Get-ChildItem (Join-Path $root "dist") -Directory | Select-Object -First 1).FullName
$intDir  = Join-Path $distDir "_internal"
$qtbin = Join-Path $intDir "PyQt5\Qt5\bin"
$ort   = Join-Path $intDir "onnxruntime"
# prune list: unused Qt modules + software OpenGL renderer + DirectML (CPU-only EP)
$prune = @(
    "$qtbin\opengl32sw.dll", "$qtbin\d3dcompiler_47.dll",
    "$qtbin\Qt5Quick.dll", "$qtbin\Qt5Qml.dll", "$qtbin\Qt5QmlModels.dll",
    "$qtbin\Qt5Network.dll", "$qtbin\libGLESv2.dll", "$qtbin\libEGL.dll",
    "$qtbin\Qt5DBus.dll", "$qtbin\Qt5WebSockets.dll",
    # onnxruntime 仅 CPU 推理
    "$ort\DirectML.dll"
)
$freed = 0
foreach ($f in $prune) {
    if (Test-Path $f) {
        $freed += (Get-Item $f).Length
        Remove-Item $f -Force
        Write-Host "  removed $(Split-Path $f -Leaf)"
    }
}
Write-Host ("  freed {0:N1} MB" -f ($freed / 1MB))

Write-Host "[3/3] done" -ForegroundColor Green
$total = (Get-ChildItem $distDir -Recurse -File | Measure-Object Length -Sum).Sum
$exe = Get-ChildItem $distDir -Filter *.exe | Select-Object -First 1
Write-Host ("output: dist\{0}\{1}  size {2:N1} MB" -f (Split-Path $distDir -Leaf), $exe.Name, ($total / 1MB))
