"""Direct P0 baseline training — bypasses the frozen EXE entirely.

Runs YOLO training from the dev venv, avoiding the PyInstaller
multiprocessing deadlock that stalls DataLoader workers in frozen mode.

Usage::

    python scripts/run_baseline_p0.py [--epochs 50] [--device 0] [--batch 8]

Results go to:
    C:/paint_defect_research/evaluations/training/raw160_baseline_p0_50ep/run/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORKSPACE = Path("C:/paint_defect_research")
DATA_YAML = WORKSPACE / "datasets" / "yolo_baseline" / "meta" / "raw160.yaml"
DEFAULT_OUTPUT_DIR = WORKSPACE / "evaluations" / "training" / "raw160_baseline_p0_50ep"
DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "yolov8n.pt"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args(argv)

    if not DATA_YAML.is_file():
        print(f"data.yaml not found: {DATA_YAML}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"P0 baseline training")
    print(f"  data:    {DATA_YAML}")
    print(f"  output:  {output_dir}")
    print(f"  epochs:  {args.epochs}")
    print(f"  device:  {args.device}")
    print(f"  batch:   {args.batch}")
    print(f"  imgsz:   {args.imgsz}")
    print(f"  weights: {args.weights}")
    print(f"  workers: {args.workers}")
    print(flush=True)

    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        print("ultralytics not installed", file=sys.stderr)
        return 1

    model = YOLO(args.weights)
    result = model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(output_dir),
        name="run",
        exist_ok=True,
        verbose=True,
        workers=args.workers,
    )

    print(f"\nTraining complete. Results: {result.save_dir}")

    # Quick metrics summary from results.csv
    import csv
    results_csv = Path(result.save_dir) / "results.csv"
    if results_csv.is_file():
        with results_csv.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            last = rows[-1]
            print(f"\nFinal epoch metrics:")
            for key in sorted(last.keys()):
                if "metrics" in key.lower() or "loss" in key.lower():
                    val = last[key].strip()
                    if val:
                        print(f"  {key.strip()}: {val}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
