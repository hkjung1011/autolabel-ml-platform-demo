param(
    [string]$PythonExe = ".venv\\Scripts\\python.exe",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$apiRoot = Split-Path -Parent $PSScriptRoot
$specPath = Join-Path $apiRoot "packaging\\defect_vision_research.spec"
$distDir = Join-Path $apiRoot "dist"

if (-not (Test-Path $specPath)) {
    throw "Missing PyInstaller spec: $specPath"
}

Push-Location $apiRoot
try {
    $args = @("-m", "PyInstaller", "--noconfirm")
    if ($Clean) {
        $args += "--clean"
    }
    $args += $specPath
    & $PythonExe @args
    Write-Host "Build complete. Dist directory: $distDir"
}
finally {
    Pop-Location
}
