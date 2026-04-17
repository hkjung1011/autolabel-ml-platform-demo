param(
    [string]$RepoRoot = "C:\autolabel-ml-platform-demo\apps\api",
    [string]$OutputDir = "C:\paint_defect_research\evaluations\training\raw160_baseline_p0_50ep_dev",
    [int]$Epochs = 50,
    [string]$Device = "0",
    [int]$Batch = 8,
    [int]$Imgsz = 640,
    [int]$Workers = 4
)

$ErrorActionPreference = "Stop"

$python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$trainScript = Join-Path $RepoRoot "scripts\run_baseline_p0.py"
$evalScript = Join-Path $RepoRoot "scripts\eval_test_set.py"
$weights = Join-Path $RepoRoot "yolov8n.pt"

if (-not (Test-Path $python)) {
    throw "Missing python executable: $python"
}
if (-not (Test-Path $trainScript)) {
    throw "Missing training script: $trainScript"
}
if (-not (Test-Path $evalScript)) {
    throw "Missing eval script: $evalScript"
}
if (-not (Test-Path $weights)) {
    throw "Missing weights file: $weights"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$trainLog = Join-Path $OutputDir "train_stdout.log"
$evalLog = Join-Path $OutputDir "test_eval_stdout.log"

Set-Location $RepoRoot

& $python $trainScript `
    --epochs $Epochs `
    --device $Device `
    --batch $Batch `
    --imgsz $Imgsz `
    --workers $Workers `
    --weights $weights `
    --output-dir $OutputDir *>&1 | Tee-Object -FilePath $trainLog

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $python $evalScript $OutputDir --device $Device *>&1 | Tee-Object -FilePath $evalLog
exit $LASTEXITCODE
