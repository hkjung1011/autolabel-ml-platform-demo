from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import yaml

from app.core.atomic_io import atomic_write_bytes, atomic_write_text
from app.core.config import runtime_is_frozen_bundle
from app.domain.research_models import (
    AblationRunRequest,
    AblationRunResponse,
    EvaluationRunRequest,
    TestEvaluationPerClass,
    TestEvaluationRequest,
    TestEvaluationResponse,
    TrainingJobStatusResponse,
    TrainingRunHistoryItem,
    TrainingRunHistoryResponse,
    TrainingRunRequest,
    TrainingRunResponse,
)
from app.services.evaluation import evaluation_service


class TrainingService:
    def __init__(self) -> None:
        self.jobs: dict[str, dict] = {}

    def run_training(self, request: TrainingRunRequest) -> TrainingRunResponse:
        response, ultralytics_installed = self._prepare_training_response(request)
        torch_cuda_available, actual_device = self._resolve_runtime_device(request.device)
        response.torch_cuda_available = torch_cuda_available
        response.actual_device = actual_device
        response.metrics["requested_device"] = request.device
        if torch_cuda_available is not None:
            response.metrics["torch_cuda_available"] = str(torch_cuda_available)
        if actual_device:
            response.metrics["actual_device"] = actual_device

        if request.trainer_command and not request.dry_run:
            response.status = "planned"
            response.runtime_mode = "external_command"
            response.notes.append("External trainer command was supplied. Use the live training endpoint to execute it with logs.")
        elif not request.dry_run and ultralytics_installed:
            from ultralytics import YOLO  # type: ignore

            effective_weights = self._resolve_default_weights(request.arm, request.weights_path)
            staged_auxiliary_assets = self._stage_frozen_auxiliary_assets(Path(request.workspace_root))
            if staged_auxiliary_assets:
                response.notes.extend(staged_auxiliary_assets)
            model = YOLO(effective_weights)
            response.runtime_mode = "in_process_local"
            # PyInstaller-frozen EXE + Windows multiprocessing = DataLoader deadlock.
            # Force workers=0 (single-thread loading) in frozen mode to avoid it.
            dataloader_workers = 0 if runtime_is_frozen_bundle() else 8
            result = model.train(
                data=response.dataset_yaml_path,
                epochs=request.epochs,
                imgsz=request.imgsz,
                batch=request.batch,
                device=request.device,
                project=response.output_dir,
                name="run",
                exist_ok=True,
                verbose=False,
                workers=dataloader_workers,
            )
            response.status = "completed"
            response.artifact_paths["run_dir"] = str(Path(result.save_dir))
            response.metrics["save_dir"] = str(result.save_dir)
        elif not request.dry_run:
            response.status = "blocked"
            response.runtime_mode = "blocked"
            response.notes.append("Local ultralytics execution is unavailable in this environment. Use an external trainer command.")
        else:
            response.runtime_mode = "dry_run"

        self._write_run_artifacts(response)
        return response

    def start_training_job(self, request: TrainingRunRequest) -> TrainingJobStatusResponse:
        response, ultralytics_installed = self._prepare_training_response(request)
        torch_cuda_available, actual_device = self._resolve_runtime_device(request.device)
        response.torch_cuda_available = torch_cuda_available
        response.actual_device = actual_device
        job_id = f"train_{uuid4().hex[:12]}"
        started_at = datetime.now().isoformat(timespec="seconds")
        log_path = Path(response.output_dir) / "live.log"
        status_json_path = Path(response.output_dir) / "status.json"
        response.artifact_paths["status_json_path"] = str(status_json_path)
        response.artifact_paths["live_log_path"] = str(log_path)
        self._write_run_artifacts(response)

        if request.dry_run:
            job = TrainingJobStatusResponse(
                job_id=job_id,
                workspace_root=response.workspace_root,
                arm=response.arm,
                status="planned",
                dry_run=True,
                run_name=response.run_name,
                output_dir=response.output_dir,
                command_preview=response.command_preview,
                requested_device=response.requested_device,
                actual_device=response.actual_device,
                torch_cuda_available=response.torch_cuda_available,
                runtime_mode="dry_run",
                dataset_yaml_path=response.dataset_yaml_path,
                log_path=str(log_path),
                started_at=started_at,
                finished_at=started_at,
                return_code=0,
                tail_lines=["Dry run only. No external process started."],
                notes=response.notes,
                artifact_paths=response.artifact_paths,
            )
            self.jobs[job_id] = {"process": None, "status": job}
            self._persist_job(job)
            return job

        command_to_run, runtime_mode = self._resolve_live_command(
            request=request,
            response=response,
            ultralytics_installed=ultralytics_installed,
        )
        if command_to_run is None:
            job = TrainingJobStatusResponse(
                job_id=job_id,
                workspace_root=response.workspace_root,
                arm=response.arm,
                status="blocked",
                dry_run=False,
                run_name=response.run_name,
                output_dir=response.output_dir,
                command_preview=response.command_preview,
                requested_device=response.requested_device,
                actual_device=response.actual_device,
                torch_cuda_available=response.torch_cuda_available,
                runtime_mode="blocked",
                dataset_yaml_path=response.dataset_yaml_path,
                log_path=str(log_path),
                started_at=started_at,
                finished_at=started_at,
                return_code=None,
                tail_lines=[],
                notes=response.notes
                + ["No executable training backend is available. Provide `trainer_command` or install ultralytics in the runtime environment."],
                artifact_paths=response.artifact_paths,
            )
            self.jobs[job_id] = {"process": None, "status": job}
            self._persist_job(job)
            return job

        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(
                command_to_run,
                cwd=response.workspace_root,
                shell=isinstance(command_to_run, str),
                stdout=handle,
                stderr=subprocess.STDOUT,
            )

        job = TrainingJobStatusResponse(
            job_id=job_id,
            workspace_root=response.workspace_root,
            arm=response.arm,
            status="running",
            dry_run=False,
            run_name=response.run_name,
            output_dir=response.output_dir,
            command_preview=response.command_preview if isinstance(command_to_run, str) else subprocess.list2cmdline(command_to_run),
            requested_device=response.requested_device,
            actual_device=response.actual_device,
            torch_cuda_available=response.torch_cuda_available,
            runtime_mode=runtime_mode,
            dataset_yaml_path=response.dataset_yaml_path,
            log_path=str(log_path),
            started_at=started_at,
            tail_lines=[],
            notes=response.notes,
            artifact_paths=response.artifact_paths,
        )
        self.jobs[job_id] = {"process": process, "status": job}
        self._persist_job(job)
        return self.get_training_job(job_id)

    def get_training_job(self, job_id: str) -> TrainingJobStatusResponse:
        if job_id not in self.jobs:
            raise KeyError(job_id)
        return self._refresh_job(job_id)

    def list_training_runs(self, workspace_root: str) -> TrainingRunHistoryResponse:
        root = Path(workspace_root) / "evaluations" / "training"
        if not root.exists():
            raise FileNotFoundError(f"Missing training directory: {root}")
        run_dirs = [path for path in root.iterdir() if path.is_dir()]
        runs = [self._load_history_item(run_dir) for run_dir in run_dirs]
        runs.sort(key=lambda item: item.started_at or item.finished_at or "", reverse=True)
        return TrainingRunHistoryResponse(
            workspace_root=workspace_root,
            total_runs=len(runs),
            runs=runs,
        )

    def run_test_evaluation(self, request: TestEvaluationRequest) -> TestEvaluationResponse:
        run_dir = Path(request.run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"Missing training run directory: {run_dir}")

        best_pt = self._find_best_pt(run_dir)
        data_yaml_path = self._find_test_eval_data_yaml(run_dir, request.data_yaml_path)

        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise FileNotFoundError("Ultralytics is not installed in this environment.") from exc

        torch_cuda_available, actual_device = self._resolve_runtime_device(request.device)
        output_dir = run_dir / "test_eval"
        output_dir.mkdir(parents=True, exist_ok=True)

        model = YOLO(str(best_pt))
        results = model.val(
            data=data_yaml_path,
            split="test",
            device=request.device,
            imgsz=request.imgsz,
            conf=request.conf,
            iou=request.iou,
            project=str(output_dir),
            name="val",
            exist_ok=True,
            verbose=False,
        )

        box = results.box
        metrics: dict[str, float | int | str] = {
            "mAP50": round(float(box.map50), 6),
            "mAP50_95": round(float(box.map), 6),
            "precision": round(float(box.mp), 6),
            "recall": round(float(box.mr), 6),
        }

        data_config = yaml.safe_load(Path(data_yaml_path).read_text(encoding="utf-8"))
        class_names = data_config.get("names", {}) if isinstance(data_config, dict) else {}
        ap50_per_class = box.ap50.tolist() if hasattr(box.ap50, "tolist") else list(box.ap50)
        ap_per_class = box.ap.tolist() if hasattr(box.ap, "tolist") else list(box.ap)

        per_class: list[TestEvaluationPerClass] = []
        for index, (ap50, ap) in enumerate(zip(ap50_per_class, ap_per_class)):
            if isinstance(class_names, dict):
                class_name = class_names.get(index, class_names.get(str(index), f"class_{index}"))
            elif isinstance(class_names, list) and index < len(class_names):
                class_name = class_names[index]
            else:
                class_name = f"class_{index}"
            per_class.append(
                TestEvaluationPerClass(
                    class_id=index,
                    name=str(class_name),
                    ap50=round(float(ap50), 6),
                    ap50_95=round(float(ap), 6),
                )
            )

        response = TestEvaluationResponse(
            run_dir=str(run_dir),
            status="completed",
            model_path=str(best_pt),
            data_yaml_path=data_yaml_path,
            output_dir=str(output_dir),
            requested_device=request.device,
            actual_device=actual_device,
            torch_cuda_available=torch_cuda_available,
            runtime_mode="in_process_local",
            metrics=metrics,
            per_class=per_class,
            notes=["Held-out evaluation uses `split=test` from the original data.yaml."],
            results_json_path=str(output_dir / "results.json"),
            summary_text_path=str(output_dir / "summary.txt"),
        )
        self._write_test_eval_artifacts(response)
        return response

    def load_test_evaluation(self, run_dir: str) -> TestEvaluationResponse:
        root = Path(run_dir) / "test_eval"
        results_path = root / "results.json"
        if not results_path.exists():
            raise FileNotFoundError(f"Missing test evaluation report: {results_path}")
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        return TestEvaluationResponse.model_validate(payload)

    def run_ablation(self, request: AblationRunRequest) -> AblationRunResponse:
        workspace_root = Path(request.workspace_root)
        report_root = workspace_root / "evaluations" / "ablation"
        report_root.mkdir(parents=True, exist_ok=True)

        runs = [
            self.run_training(
                TrainingRunRequest(
                    workspace_root=request.workspace_root,
                    arm=arm,
                    epochs=request.epochs,
                    imgsz=request.imgsz,
                    batch=request.batch,
                    device=request.device,
                    weights_path=request.weights_path,
                    run_name=f"{arm}_ablation",
                    dry_run=request.dry_run,
                )
            )
            for arm in request.arms
        ]

        response = AblationRunResponse(
            workspace_root=str(workspace_root),
            status="planned" if all(run.status in {"planned", "blocked"} for run in runs) else "completed",
            dry_run=request.dry_run,
            runs=runs,
            report_json_path=str(report_root / "report.json"),
            report_markdown_path=str(report_root / "report.md"),
        )
        self._write_ablation_artifacts(response)
        return response

    def _prepare_training_response(self, request: TrainingRunRequest) -> tuple[TrainingRunResponse, bool]:
        workspace_root = Path(request.workspace_root)
        target = self._resolve_training_target(request, workspace_root)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = request.run_name or f"{request.arm}_{timestamp}"
        runs_root = workspace_root / "evaluations" / "training" / run_name
        runs_root.mkdir(parents=True, exist_ok=True)

        ultralytics_installed = bool(importlib.util.find_spec("ultralytics"))
        command_preview = self._build_command_preview(
            request,
            dataset_yaml_path=target["dataset_yaml_path"],
            output_dir=str(runs_root),
            task=target["task"],
            weights_path=target["weights_path"],
        )

        notes = list(target["notes"])
        artifact_paths = {
            "plan_json_path": str(runs_root / "plan.json"),
            "summary_markdown_path": str(runs_root / "summary.md"),
        }
        if target.get("source_report_path"):
            artifact_paths["source_report_path"] = target["source_report_path"]
        status = "planned"
        metrics: dict[str, float | int | str] = {
            "epochs": request.epochs,
            "imgsz": request.imgsz,
            "batch": request.batch,
            "task": target["task"],
        }

        if not ultralytics_installed:
            notes.append("`ultralytics` is not installed in this app environment.")
        elif not request.trainer_command:
            notes.append("Local Ultralytics runner is available for in-process or subprocess training.")
        if request.trainer_command:
            notes.append("External trainer command is configured for live execution.")
        notes.append(f"Training task: {target['task']}.")
        if request.dry_run:
            notes.append("Dry-run mode generated a reproducible training plan without executing YOLO.")

        response = TrainingRunResponse(
            workspace_root=str(workspace_root),
            arm=request.arm,
            status=status,
            dry_run=request.dry_run,
            dataset_yaml_path=target["dataset_yaml_path"],
            output_dir=str(runs_root),
            run_name=run_name,
            command_preview=command_preview,
            requested_device=request.device,
            metrics=metrics,
            notes=notes,
            artifact_paths=artifact_paths,
        )
        return response, ultralytics_installed

    def _build_command_preview(
        self,
        request: TrainingRunRequest,
        dataset_yaml_path: str,
        output_dir: str,
        *,
        task: str,
        weights_path: str,
    ) -> str:
        context = self._command_context(
            request=request,
            dataset_yaml_path=dataset_yaml_path,
            output_dir=output_dir,
            weights_path=weights_path,
            task=task,
        )
        template = request.trainer_command
        if template:
            return template.format(**context)
        return (
            f"yolo {context['task']} train model={context['weights_path']} data={context['dataset_yaml_path']} "
            f"epochs={context['epochs']} imgsz={context['imgsz']} batch={context['batch']} device={context['device']} "
            f"project={context['output_dir']} name=run"
        )

    def _resolve_live_command(
        self,
        request: TrainingRunRequest,
        response: TrainingRunResponse,
        ultralytics_installed: bool,
    ) -> tuple[str | list[str] | None, str]:
        if request.trainer_command:
            return response.command_preview, "external_command"
        if ultralytics_installed:
            if runtime_is_frozen_bundle():
                request_path = self._write_embedded_request(request=request, response=response)
                response.artifact_paths["embedded_request_path"] = str(request_path)
                return [sys.executable, "--embedded-train", str(request_path)], "frozen_embedded"
            runner_script = self._write_local_runner_script(request=request, response=response)
            response.artifact_paths["local_runner_script_path"] = str(runner_script)
            return [sys.executable, str(runner_script)], "python_runner_script"
        return None, "blocked"

    def _command_context(
        self,
        request: TrainingRunRequest,
        dataset_yaml_path: str,
        output_dir: str,
        *,
        weights_path: str,
        task: str,
    ) -> dict[str, str]:
        def quote(value: str) -> str:
            if any(char in value for char in " \t()[]{}&"):
                return f'"{value}"'
            return value

        return {
            "weights_path": quote(weights_path),
            "dataset_yaml_path": quote(dataset_yaml_path),
            "epochs": str(request.epochs),
            "imgsz": str(request.imgsz),
            "batch": str(request.batch),
            "device": str(request.device),
            "output_dir": quote(output_dir),
            "run_name": quote(request.run_name or ""),
            "workspace_root": quote(request.workspace_root),
            "task": task,
        }

    def _refresh_job(self, job_id: str) -> TrainingJobStatusResponse:
        record = self.jobs[job_id]
        process = record["process"]
        status: TrainingJobStatusResponse = record["status"]

        if process is not None and status.status == "running":
            return_code = process.poll()
            if return_code is not None:
                status.return_code = return_code
                status.finished_at = datetime.now().isoformat(timespec="seconds")
                status.status = "completed" if return_code == 0 else "failed"
                self._merge_completed_run_details(status)
        status.tail_lines = self._tail_lines(Path(status.log_path)) if status.log_path else []
        self._persist_job(status)
        return status

    def _load_history_item(self, run_dir: Path) -> TrainingRunHistoryItem:
        status_json = run_dir / "status.json"
        plan_json = run_dir / "plan.json"
        summary_md = run_dir / "summary.md"
        live_log = run_dir / "live.log"
        results_csv = run_dir / "results.csv"
        results_json = run_dir / "results.json"
        test_eval_json = run_dir / "test_eval" / "results.json"
        test_eval_summary = run_dir / "test_eval" / "summary.txt"

        payload: dict = {}
        if status_json.exists():
            payload = json.loads(status_json.read_text(encoding="utf-8"))
        elif plan_json.exists():
            payload = json.loads(plan_json.read_text(encoding="utf-8"))

        nested_run_dir = None
        artifact_paths = payload.get("artifact_paths", {}) if isinstance(payload, dict) else {}
        if isinstance(artifact_paths, dict):
            run_dir_hint = artifact_paths.get("run_dir")
            if run_dir_hint:
                nested_run_dir = Path(run_dir_hint)
        if nested_run_dir is None:
            candidate = run_dir / "run"
            if candidate.exists():
                nested_run_dir = candidate
        if nested_run_dir is not None:
            if not results_csv.exists():
                candidate_csv = nested_run_dir / "results.csv"
                if candidate_csv.exists():
                    results_csv = candidate_csv
            if not results_json.exists():
                candidate_json = nested_run_dir / "results.json"
                if candidate_json.exists():
                    results_json = candidate_json

        parsed_metrics = dict(payload.get("metrics", {}))
        parsed_metrics.update(self._parse_results_metrics(results_csv=results_csv, results_json=results_json))
        test_metrics = self._parse_test_eval_metrics(test_eval_json)
        parsed_metrics.update({f"test/{key}": value for key, value in test_metrics.items()})

        fallback_timestamp = datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat(timespec="seconds")
        return TrainingRunHistoryItem(
            run_name=payload.get("run_name", run_dir.name),
            arm=payload.get("arm", self._infer_arm_from_name(run_dir.name)),
            status=payload.get("status", "planned"),
            dry_run=bool(payload.get("dry_run", False)),
            output_dir=payload.get("output_dir", str(run_dir)),
            dataset_yaml_path=payload.get("dataset_yaml_path"),
            command_preview=payload.get("command_preview"),
            requested_device=payload.get("requested_device"),
            actual_device=payload.get("actual_device"),
            torch_cuda_available=payload.get("torch_cuda_available"),
            runtime_mode=payload.get("runtime_mode"),
            started_at=payload.get("started_at") or fallback_timestamp,
            finished_at=payload.get("finished_at") or fallback_timestamp,
            return_code=payload.get("return_code"),
            metrics=parsed_metrics,
            notes=list(payload.get("notes", [])),
            files={
                "plan_json_path": str(plan_json) if plan_json.exists() else "",
                "status_json_path": str(status_json) if status_json.exists() else "",
                "summary_markdown_path": str(summary_md) if summary_md.exists() else "",
                "live_log_path": str(live_log) if live_log.exists() else "",
                "results_csv_path": str(results_csv) if results_csv.exists() else "",
                "results_json_path": str(results_json) if results_json.exists() else "",
                "test_eval_results_json_path": str(test_eval_json) if test_eval_json.exists() else "",
                "test_eval_summary_path": str(test_eval_summary) if test_eval_summary.exists() else "",
            },
        )

    def _find_best_pt(self, run_dir: Path) -> Path:
        direct = run_dir / "run" / "weights" / "best.pt"
        if direct.is_file():
            return direct
        fallback = run_dir / "weights" / "best.pt"
        if fallback.is_file():
            return fallback
        for candidate in run_dir.rglob("best.pt"):
            return candidate
        raise FileNotFoundError(f"best.pt not found under {run_dir}")

    def _find_test_eval_data_yaml(self, run_dir: Path, requested_data_yaml: str | None) -> str:
        if requested_data_yaml:
            candidate = Path(requested_data_yaml)
            if not candidate.is_file():
                raise FileNotFoundError(f"Missing data.yaml: {candidate}")
            return str(candidate)

        args_yaml = run_dir / "run" / "args.yaml"
        if not args_yaml.is_file():
            raise FileNotFoundError(f"Cannot determine data.yaml for {run_dir}; args.yaml is missing.")

        payload = yaml.safe_load(args_yaml.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not payload.get("data"):
            raise FileNotFoundError(f"Cannot determine data.yaml for {run_dir}; args.yaml has no data field.")

        candidate = Path(str(payload["data"]))
        if not candidate.is_file():
            raise FileNotFoundError(f"Missing data.yaml: {candidate}")
        return str(candidate)

    def _tail_lines(self, path: Path, limit: int = 16) -> list[str]:
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        return lines[-limit:]

    def _parse_results_metrics(self, results_csv: Path, results_json: Path) -> dict[str, float | int | str]:
        metrics: dict[str, float | int | str] = {}
        if results_json.is_file():
            try:
                payload = json.loads(results_json.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    for key, value in payload.items():
                        if isinstance(value, (int, float, str)):
                            metrics[key] = value
            except json.JSONDecodeError:
                pass
        if results_csv.is_file():
            try:
                with results_csv.open("r", encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                if rows:
                    last = rows[-1]
                    for key, value in last.items():
                        if value in {"", None}:
                            continue
                        metrics[key] = self._coerce_metric_value(value)
            except OSError:
                pass
        return metrics

    def _parse_test_eval_metrics(self, results_json: Path) -> dict[str, float | int | str]:
        if not results_json.is_file():
            return {}
        try:
            payload = json.loads(results_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        if not isinstance(payload, dict):
            return {}

        metrics_payload = payload.get("metrics")
        if isinstance(metrics_payload, dict):
            return {
                str(key): value
                for key, value in metrics_payload.items()
                if isinstance(value, (int, float, str))
            }

        allowed_keys = {"mAP50", "mAP50_95", "precision", "recall"}
        return {
            str(key): value
            for key, value in payload.items()
            if key in allowed_keys and isinstance(value, (int, float, str))
        }

    def _coerce_metric_value(self, value: str) -> float | int | str:
        try:
            numeric = float(value)
        except ValueError:
            return value
        if numeric.is_integer():
            return int(numeric)
        return round(numeric, 6)

    def _infer_arm_from_name(self, run_name: str) -> str:
        for arm in ["raw160", "retinex", "mertens", "daf", "mixed_raw_forensic_wdr", "seg_bootstrap"]:
            if run_name.startswith(arm):
                return arm
        return "unknown"

    def _persist_job(self, status: TrainingJobStatusResponse) -> None:
        status_json_path = status.artifact_paths.get("status_json_path")
        if status_json_path:
            atomic_write_text(Path(status_json_path), json.dumps(status.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_local_runner_script(self, request: TrainingRunRequest, response: TrainingRunResponse) -> Path:
        script_path = Path(response.output_dir) / "run_ultralytics_train.py"
        effective_weights = self._resolve_default_weights(request.arm, request.weights_path)
        script = f"""from ultralytics import YOLO

model = YOLO(r"{effective_weights}")
model.train(
    data=r"{response.dataset_yaml_path}",
    epochs={request.epochs},
    imgsz={request.imgsz},
    batch={request.batch},
    device=r"{request.device}",
    project=r"{response.output_dir}",
    name="run",
    exist_ok=True,
    verbose=True,
)
"""
        atomic_write_text(script_path, script, encoding="utf-8")
        return script_path

    def _write_embedded_request(self, request: TrainingRunRequest, response: TrainingRunResponse) -> Path:
        request_path = Path(response.output_dir) / "embedded_training_request.json"
        payload = request.model_dump()
        payload["dry_run"] = False
        payload["trainer_command"] = None
        payload["run_name"] = response.run_name
        atomic_write_text(request_path, json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return request_path

    def _resolve_training_target(self, request: TrainingRunRequest, workspace_root: Path) -> dict[str, str | list[str]]:
        if request.arm == "seg_bootstrap":
            report_path = workspace_root / "evaluations" / "segmentation_bootstrap" / "report.json"
            if not report_path.exists():
                raise ValueError("Segmentation bootstrap dataset is not ready yet. Build it before training `seg_bootstrap`.")
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            dataset_yaml_path = payload.get("dataset_yaml_path")
            if not dataset_yaml_path or not Path(dataset_yaml_path).exists():
                raise ValueError("Segmentation bootstrap dataset yaml is missing.")
            notes = [
                f"Segmentation bootstrap source dataset: {payload.get('source_dataset_name', 'unknown')}.",
                f"Bootstrap mode: {payload.get('bootstrap_mode', 'coarse_box_mask')}.",
                "Segmentation claims still require reviewed masks or a gold subset.",
            ]
            if int(payload.get("refined_items", 0)) > 0:
                notes.append(f"SAM refined items: {payload.get('refined_items', 0)}.")
            return {
                "dataset_yaml_path": str(dataset_yaml_path),
                "task": "segment",
                "weights_path": self._resolve_default_weights(request.arm, request.weights_path),
                "notes": notes,
                "source_report_path": str(report_path),
            }

        if request.arm == "mixed_raw_forensic_wdr":
            mix = self._ensure_mixed_raw_forensic_wdr_dataset(workspace_root)
            return {
                "dataset_yaml_path": mix["dataset_yaml_path"],
                "task": "detect",
                "weights_path": self._resolve_default_weights(request.arm, request.weights_path),
                "notes": mix["notes"],
                "source_report_path": mix["report_path"],
            }

        readiness = evaluation_service.build_readiness_report(
            EvaluationRunRequest(
                workspace_root=request.workspace_root,
                include_arms=["raw160", "retinex", "mertens", "daf", "mixed_raw_forensic_wdr"],
                refresh_report=False,
            )
        )
        arm_summary = next((arm for arm in readiness.arms if arm.arm_name == request.arm), None)
        if arm_summary is None:
            raise ValueError(f"Unknown arm: {request.arm}")
        if not arm_summary.ready or not arm_summary.dataset_yaml_path:
            raise ValueError(f"Arm is not ready for training: {request.arm}")
        return {
            "dataset_yaml_path": arm_summary.dataset_yaml_path,
            "task": "detect",
            "weights_path": self._resolve_default_weights(request.arm, request.weights_path),
            "notes": list(arm_summary.notes),
        }

    def _resolve_default_weights(self, arm: str, requested_weights: str) -> str:
        if arm == "seg_bootstrap" and requested_weights in {"", "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"}:
            requested_weights = "yolov8n-seg.pt"
        # In frozen (PyInstaller) mode, ultralytics' default relative `weights/` path
        # lands under the subprocess CWD (workspace_root) and an online fallback
        # download fails when the directory is missing or unwritable. If the spec
        # bundles the weight file at _MEIPASS root, resolve it to an absolute path
        # so ultralytics skips its download path entirely.
        if runtime_is_frozen_bundle():
            meipass = getattr(sys, "_MEIPASS", None)
            if meipass:
                bundled = Path(meipass) / requested_weights
                if bundled.is_file():
                    return str(bundled)
        return requested_weights

    def _stage_frozen_auxiliary_assets(self, workspace_root: Path) -> list[str]:
        if not runtime_is_frozen_bundle():
            return []
        meipass = getattr(sys, "_MEIPASS", None)
        if not meipass:
            return []
        notes: list[str] = []
        bundle_root = Path(meipass)
        for asset_name in ("yolo26n.pt",):
            bundled = bundle_root / asset_name
            staged = workspace_root / asset_name
            if not bundled.is_file() or staged.is_file():
                continue
            atomic_write_bytes(staged, bundled.read_bytes())
            notes.append(f"Staged bundled auxiliary asset: {asset_name}")
        return notes

    def _ensure_mixed_raw_forensic_wdr_dataset(self, workspace_root: Path) -> dict[str, str | list[str]]:
        manifest_path = workspace_root / "manifests" / "labeled_pair_manifest.json"
        if not manifest_path.exists():
            raise ValueError("Labeled pair manifest is missing. Bootstrap the workspace first.")

        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        groups = payload.get("groups", [])
        dataset_root = workspace_root / "datasets" / "yolo_mixed" / "raw160_forensic_wdr"
        if dataset_root.exists():
            shutil.rmtree(dataset_root)
        for split in ["train", "val", "test"]:
            (dataset_root / "images" / split).mkdir(parents=True, exist_ok=True)
            (dataset_root / "labels" / split).mkdir(parents=True, exist_ok=True)

        class_ids: set[int] = set()
        raw_count = 0
        forensic_count = 0
        split_counts = {"train": 0, "val": 0, "test": 0}

        for group in groups:
            anchor_image_path = group.get("anchor_image_path")
            anchor_label_path = group.get("anchor_label_path")
            if not anchor_image_path or not anchor_label_path:
                continue
            source_image = Path(anchor_image_path)
            source_label = Path(anchor_label_path)
            if not source_image.exists() or not source_label.exists():
                continue

            split = group.get("frozen_split") or group.get("split") or "train"
            safe_group = str(group.get("key", "")).replace("|", "__")
            raw_suffix = source_image.suffix or ".png"
            raw_target_image = dataset_root / "images" / split / f"{safe_group}__raw160{raw_suffix}"
            raw_target_label = dataset_root / "labels" / split / f"{safe_group}__raw160.txt"
            shutil.copy2(source_image, raw_target_image)
            shutil.copy2(source_label, raw_target_label)
            self._collect_class_ids(source_label, class_ids)
            raw_count += 1
            split_counts[split] = split_counts.get(split, 0) + 1

            forensic_image = workspace_root / "variants" / "forensic_wdr" / safe_group / "fused.png"
            if forensic_image.exists():
                forensic_target_image = dataset_root / "images" / split / f"{safe_group}__forensic_wdr.png"
                forensic_target_label = dataset_root / "labels" / split / f"{safe_group}__forensic_wdr.txt"
                shutil.copy2(forensic_image, forensic_target_image)
                shutil.copy2(source_label, forensic_target_label)
                forensic_count += 1
                split_counts[split] = split_counts.get(split, 0) + 1

        if forensic_count == 0:
            raise ValueError("No forensic_wdr outputs are available yet. Run Forensic WDR before mixed training.")

        yaml_path = self._write_detection_yaml(dataset_root, "mixed_raw_forensic_wdr", sorted(class_ids))
        report_root = workspace_root / "evaluations" / "training_mixes"
        report_root.mkdir(parents=True, exist_ok=True)
        report_path = report_root / "mixed_raw_forensic_wdr.json"
        report = {
            "dataset_root": str(dataset_root),
            "dataset_yaml_path": str(yaml_path),
            "raw_items": raw_count,
            "forensic_wdr_items": forensic_count,
            "total_items": raw_count + forensic_count,
            "split_counts": split_counts,
            "class_ids": sorted(class_ids),
        }
        atomic_write_text(report_path, json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        notes = [
            f"Mixed dataset built from raw anchors ({raw_count}) and forensic_wdr variants ({forensic_count}).",
            f"Split counts: train={split_counts.get('train', 0)}, val={split_counts.get('val', 0)}, test={split_counts.get('test', 0)}.",
            "Raw and forensic WDR share the same anchor labels, so the model sees original and evidence-preserving views together.",
        ]
        return {
            "dataset_yaml_path": str(yaml_path),
            "report_path": str(report_path),
            "notes": notes,
        }

    def _collect_class_ids(self, label_path: Path, class_ids: set[int]) -> None:
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            try:
                class_ids.add(int(float(parts[0])))
            except ValueError:
                continue

    def _write_detection_yaml(self, dataset_root: Path, arm_name: str, class_ids: list[int]) -> Path:
        yaml_root = dataset_root / "meta"
        yaml_root.mkdir(parents=True, exist_ok=True)
        yaml_path = yaml_root / f"{arm_name}.yaml"
        names = [f"class_{class_id}" for class_id in class_ids] or ["class_0"]
        lines = [
            f"path: {dataset_root.as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            f"nc: {len(names)}",
            "names:",
        ]
        lines.extend([f"  {index}: {name}" for index, name in enumerate(names)])
        atomic_write_text(yaml_path, "\n".join(lines) + "\n", encoding="utf-8")
        return yaml_path

    def _write_run_artifacts(self, response: TrainingRunResponse) -> None:
        plan_json_path = Path(response.artifact_paths["plan_json_path"])
        atomic_write_text(plan_json_path, json.dumps(response.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")

        lines = [
            f"# Training Plan: {response.run_name}",
            "",
            f"- Arm: `{response.arm}`",
            f"- Status: `{response.status}`",
            f"- Dry run: `{response.dry_run}`",
            f"- Requested device: `{response.requested_device}`",
            f"- Actual device: `{response.actual_device or 'unknown'}`",
            f"- Torch CUDA available: `{response.torch_cuda_available}`",
            f"- Runtime mode: `{response.runtime_mode}`",
            f"- Dataset yaml: `{response.dataset_yaml_path}`",
            f"- Output dir: `{response.output_dir}`",
            "",
            "## Command Preview",
            "",
            f"`{response.command_preview}`",
            "",
            "## Notes",
        ]
        for note in response.notes:
            lines.append(f"- {note}")
        atomic_write_text(Path(response.artifact_paths["summary_markdown_path"]), "\n".join(lines) + "\n", encoding="utf-8")

    def _write_ablation_artifacts(self, response: AblationRunResponse) -> None:
        atomic_write_text(
            Path(response.report_json_path),
            json.dumps(response.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        lines = [
            "# Ablation Plan",
            "",
            f"- Workspace: `{response.workspace_root}`",
            f"- Status: `{response.status}`",
            f"- Dry run: `{response.dry_run}`",
            "",
            "## Runs",
        ]
        for run in response.runs:
            lines.append(f"- `{run.arm}`: status={run.status}, yaml=`{run.dataset_yaml_path}`, output=`{run.output_dir}`")
            lines.append(f"  - device: requested=`{run.requested_device}`, actual=`{run.actual_device or 'unknown'}`")
            lines.append(f"  - command: `{run.command_preview}`")
        atomic_write_text(Path(response.report_markdown_path), "\n".join(lines) + "\n", encoding="utf-8")

    def _write_test_eval_artifacts(self, response: TestEvaluationResponse) -> None:
        atomic_write_text(
            Path(response.results_json_path),
            json.dumps(response.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        lines = [
            "Test Set Evaluation Results",
            "==========================",
            f"Run dir:    {response.run_dir}",
            f"Model:      {response.model_path}",
            f"Data:       {response.data_yaml_path}",
            f"Split:      {response.split}",
            f"Device:     requested={response.requested_device}, actual={response.actual_device or 'unknown'}",
            "",
            "Overall Metrics:",
            f"  mAP@50:      {response.metrics.get('mAP50', 0):.4f}",
            f"  mAP@50-95:   {response.metrics.get('mAP50_95', 0):.4f}",
            f"  Precision:   {response.metrics.get('precision', 0):.4f}",
            f"  Recall:      {response.metrics.get('recall', 0):.4f}",
            "",
            "Per-Class AP:",
        ]
        for item in response.per_class:
            lines.append(f"  {item.name}: AP50={item.ap50:.4f}, AP50-95={item.ap50_95:.4f}")
        atomic_write_text(Path(response.summary_text_path), "\n".join(lines) + "\n", encoding="utf-8")

    def _resolve_runtime_device(self, requested_device: str) -> tuple[bool | None, str | None]:
        if not importlib.util.find_spec("torch"):
            return None, None
        try:
            import torch  # type: ignore
        except Exception:
            return None, None

        try:
            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            cuda_available = False

        normalized = (requested_device or "0").strip().lower()
        if normalized in {"cpu", "mps"}:
            return cuda_available, normalized
        if not cuda_available:
            return False, "cpu"
        if normalized.startswith("cuda"):
            return True, normalized
        if normalized.isdigit():
            return True, f"cuda:{normalized}"
        if "," in normalized:
            first = normalized.split(",", 1)[0].strip()
            if first.isdigit():
                return True, f"cuda:{first}"
        return True, "cuda:0"

    def _merge_completed_run_details(self, status: TrainingJobStatusResponse) -> None:
        plan_json_path = status.artifact_paths.get("plan_json_path")
        if not plan_json_path:
            return
        path = Path(plan_json_path)
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        status.actual_device = payload.get("actual_device", status.actual_device)
        status.torch_cuda_available = payload.get("torch_cuda_available", status.torch_cuda_available)
        status.runtime_mode = payload.get("runtime_mode", status.runtime_mode)
        notes = payload.get("notes", [])
        if isinstance(notes, list):
            status.notes = notes
        artifact_paths = payload.get("artifact_paths", {})
        if isinstance(artifact_paths, dict):
            status.artifact_paths.update({str(key): str(value) for key, value in artifact_paths.items()})


def execute_embedded_training_request(request_path: str) -> int:
    payload = json.loads(Path(request_path).read_text(encoding="utf-8"))
    request = TrainingRunRequest.model_validate(payload)
    response = training_service.run_training(request)
    print(
        json.dumps(
            {
                "run_name": response.run_name,
                "status": response.status,
                "requested_device": response.requested_device,
                "actual_device": response.actual_device,
                "torch_cuda_available": response.torch_cuda_available,
                "runtime_mode": response.runtime_mode,
                "output_dir": response.output_dir,
            },
            ensure_ascii=False,
        )
    )
    return 0 if response.status == "completed" else 1


training_service = TrainingService()
