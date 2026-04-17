"""Evaluate a trained YOLO model on the held-out test split.

This is the P1 evaluation step — after P0 (training) completes and produces
``best.pt``, this script runs ``model.val(split='test')`` to get real
accuracy numbers on data the model has never seen during training or
validation-loop tuning.

Usage::

    python scripts/eval_test_set.py <run_dir> [--data <data.yaml>] [--device 0]

Where ``<run_dir>`` is the training output directory containing
``run/weights/best.pt``. If ``--data`` is omitted, the script reads
``run/args.yaml`` to find the original data.yaml path.

Outputs:
    <run_dir>/test_eval/results.json   — full metrics
    <run_dir>/test_eval/summary.txt    — human-readable summary
    stdout                             — key metrics for quick inspection

Exit 0 on success, 1 on any failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def _find_best_pt(run_dir: Path) -> Path:
    candidate = run_dir / "run" / "weights" / "best.pt"
    if candidate.is_file():
        return candidate
    candidate = run_dir / "weights" / "best.pt"
    if candidate.is_file():
        return candidate
    for pt in run_dir.rglob("best.pt"):
        return pt
    raise FileNotFoundError(f"best.pt not found under {run_dir}")


def _find_data_yaml(run_dir: Path, cli_data: str | None) -> str:
    if cli_data:
        return cli_data
    args_yaml = run_dir / "run" / "args.yaml"
    if args_yaml.is_file():
        args = yaml.safe_load(args_yaml.read_text(encoding="utf-8"))
        if isinstance(args, dict) and args.get("data"):
            return str(args["data"])
    raise FileNotFoundError(
        f"Cannot determine data.yaml — no args.yaml in {run_dir} and --data not given"
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Training output directory")
    parser.add_argument("--data", default=None, help="Path to data.yaml (auto-detected from args.yaml if omitted)")
    parser.add_argument("--device", default="0", help="Device (default: 0)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold (default: 0.001)")
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold (default: 0.6)")
    args = parser.parse_args(argv)

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        print(f"eval: run_dir does not exist: {run_dir}", file=sys.stderr)
        return 1

    try:
        best_pt = _find_best_pt(run_dir)
    except FileNotFoundError as exc:
        print(f"eval: {exc}", file=sys.stderr)
        return 1

    try:
        data_yaml = _find_data_yaml(run_dir, args.data)
    except FileNotFoundError as exc:
        print(f"eval: {exc}", file=sys.stderr)
        return 1

    print(f"eval: model  = {best_pt}")
    print(f"eval: data   = {data_yaml}")
    print(f"eval: split  = test")
    print(f"eval: device = {args.device}")
    print(f"eval: imgsz  = {args.imgsz}")
    print(flush=True)

    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        print("eval: ultralytics not installed", file=sys.stderr)
        return 1

    model = YOLO(str(best_pt))

    output_dir = run_dir / "test_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = model.val(
        data=data_yaml,
        split="test",
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        project=str(output_dir),
        name="val",
        exist_ok=True,
        verbose=True,
    )

    # Extract metrics from results
    box = results.box
    metrics: dict[str, object] = {
        "split": "test",
        "model": str(best_pt),
        "data": data_yaml,
        "mAP50": round(float(box.map50), 6),
        "mAP50_95": round(float(box.map), 6),
        "precision": round(float(box.mp), 6),
        "recall": round(float(box.mr), 6),
    }

    # Per-class AP
    data_config = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    class_names = data_config.get("names", {})
    ap50_per_class = box.ap50.tolist() if hasattr(box.ap50, "tolist") else list(box.ap50)
    ap_per_class = box.ap.tolist() if hasattr(box.ap, "tolist") else list(box.ap)

    per_class: list[dict[str, object]] = []
    for i, (ap50, ap) in enumerate(zip(ap50_per_class, ap_per_class)):
        name = class_names.get(i, class_names.get(str(i), f"class_{i}"))
        per_class.append({
            "class_id": i,
            "name": name,
            "AP50": round(float(ap50), 6),
            "AP50_95": round(float(ap), 6),
        })
    metrics["per_class"] = per_class

    # Save results
    results_json = output_dir / "results.json"
    results_json.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Human-readable summary
    lines = [
        f"Test Set Evaluation Results",
        f"==========================",
        f"Model:     {best_pt.name}",
        f"Data:      {data_yaml}",
        f"Split:     test",
        f"",
        f"Overall Metrics:",
        f"  mAP@50:      {metrics['mAP50']:.4f}",
        f"  mAP@50-95:   {metrics['mAP50_95']:.4f}",
        f"  Precision:   {metrics['precision']:.4f}",
        f"  Recall:      {metrics['recall']:.4f}",
        f"",
        f"Per-Class AP@50:",
    ]
    for cls in per_class:
        lines.append(f"  {cls['name']:>20s}: {cls['AP50']:.4f}  (AP@50-95: {cls['AP50_95']:.4f})")

    summary_text = "\n".join(lines)
    summary_path = output_dir / "summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")

    print()
    print(summary_text)
    print()
    print(f"eval: results saved to {results_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
