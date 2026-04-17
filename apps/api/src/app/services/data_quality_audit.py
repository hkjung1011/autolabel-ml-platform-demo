from __future__ import annotations

import json
from pathlib import Path

from app.core.atomic_io import atomic_write_text
from app.domain.research_models import DataQualityAuditResponse, DataQualityIssue
from app.plugins.registration.verifier import iou, to_xyxy


class DataQualityAuditService:
    def build_report(self, workspace_root: str) -> DataQualityAuditResponse:
        workspace_path = Path(workspace_root)
        registration_payloads = self._load_registration_payloads(workspace_path)
        position_issues, suspect_group_ids = self._position_issues(registration_payloads)
        label_summary = self._scan_label_files(workspace_path)

        next_actions = [
            "Gold/usable/reject 3단계로 registration gate를 분리해 warning 샘플을 바로 버리지 말 것.",
            "mean_corner_error_px와 label_iou_drift가 높은 group부터 우선 재정렬하거나 review 대상으로 보낼 것.",
            "중복 bbox, 너무 작은 bbox, 범위를 벗어난 bbox가 많은 라벨 파일부터 검수 큐에 우선 투입할 것.",
            "정렬 문제가 큰 lux 조합은 mixed 학습 전에 pair subset을 다시 고정할 것.",
        ]

        report_root = workspace_path / "evaluations" / "data_quality_audit"
        report_root.mkdir(parents=True, exist_ok=True)
        response = DataQualityAuditResponse(
            workspace_root=str(workspace_path),
            registration_reports_scanned=len(registration_payloads),
            registered_groups_scanned=sum(len(payload.get("reports", [])) for payload in registration_payloads),
            position_issue_count=len(position_issues),
            severe_position_issue_count=sum(1 for issue in position_issues if issue.severity == "high"),
            label_files_scanned=label_summary["label_files_scanned"],
            label_issue_count=len(label_summary["label_issues"]),
            invalid_label_count=label_summary["invalid_label_count"],
            out_of_bounds_box_count=label_summary["out_of_bounds_box_count"],
            tiny_box_count=label_summary["tiny_box_count"],
            oversize_box_count=label_summary["oversize_box_count"],
            duplicate_box_count=label_summary["duplicate_box_count"],
            suspect_group_ids=suspect_group_ids,
            suspect_label_files=label_summary["suspect_label_files"],
            position_issues=position_issues,
            label_issues=label_summary["label_issues"],
            next_actions=next_actions,
            report_json_path=str(report_root / "report.json"),
            report_markdown_path=str(report_root / "report.md"),
        )
        self._write_report(response)
        return response

    def load_report(self, workspace_root: str) -> DataQualityAuditResponse:
        report_path = Path(workspace_root) / "evaluations" / "data_quality_audit" / "report.json"
        if not report_path.exists():
            raise FileNotFoundError(f"Missing data quality audit report: {report_path}")
        return DataQualityAuditResponse.model_validate_json(report_path.read_text(encoding="utf-8"))

    def _load_registration_payloads(self, workspace_path: Path) -> list[dict]:
        report_dir = workspace_path / "registration_reports"
        payloads: list[dict] = []
        if not report_dir.exists():
            return payloads
        for path in sorted(report_dir.glob("*.json")):
            if path.name.endswith("_accepted_manifest.json"):
                continue
            try:
                payloads.append(json.loads(path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                continue
        return payloads

    def _position_issues(self, registration_payloads: list[dict]) -> tuple[list[DataQualityIssue], list[str]]:
        issues: list[DataQualityIssue] = []
        suspect_group_ids: list[str] = []
        for payload in registration_payloads:
            for report in payload.get("reports", []):
                group_id = str(report.get("group_id", "unknown"))
                corner_error = float(report.get("mean_corner_error_px", 0.0) or 0.0)
                iou_drift = float(report.get("label_iou_drift", 0.0) or 0.0)
                similarity = float(report.get("similarity", 0.0) or 0.0)
                status = str(report.get("status", "unknown"))
                severity = None
                if status == "reject" or corner_error > 10.0 or iou_drift > 0.14:
                    severity = "high"
                elif status == "warning" or corner_error > 4.0 or iou_drift > 0.08 or similarity < 0.1:
                    severity = "medium"
                if severity is None:
                    continue
                suspect_group_ids.append(group_id)
                issues.append(
                    DataQualityIssue(
                        category="position",
                        severity=severity,
                        item_id=group_id,
                        summary=f"corner error {corner_error:.2f}px / IoU drift {iou_drift:.4f} / similarity {similarity:.4f}",
                        metrics={
                            "mean_corner_error_px": round(corner_error, 4),
                            "label_iou_drift": round(iou_drift, 4),
                            "similarity": round(similarity, 4),
                            "dx_px": int(report.get("dx_px", 0) or 0),
                            "dy_px": int(report.get("dy_px", 0) or 0),
                            "status": status,
                        },
                    )
                )
        issues.sort(key=lambda item: (0 if item.severity == "high" else 1, -float(item.metrics.get("mean_corner_error_px", 0.0))),)
        deduped_groups = list(dict.fromkeys(suspect_group_ids))
        return issues[:60], deduped_groups[:60]

    def _scan_label_files(self, workspace_path: Path) -> dict:
        label_roots = [
            workspace_path / "datasets" / "yolo_baseline" / "labels",
            workspace_path / "datasets" / "autolabel" / "labels",
        ]
        label_files = []
        seen: set[str] = set()
        for root in label_roots:
            if not root.exists():
                continue
            for path in root.rglob("*.txt"):
                normalized = str(path.resolve())
                if normalized in seen:
                    continue
                seen.add(normalized)
                label_files.append(path)

        invalid_label_count = 0
        out_of_bounds_box_count = 0
        tiny_box_count = 0
        oversize_box_count = 0
        duplicate_box_count = 0
        suspect_label_files: list[str] = []
        label_issues: list[DataQualityIssue] = []

        for label_path in sorted(label_files):
            issues_for_file: list[str] = []
            parsed: list[tuple[int, float, float, float, float]] = []
            for line_index, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    class_id = int(float(parts[0]))
                    if len(parts) < 5:
                        raise ValueError("not enough parts")
                    xc, yc, w, h = map(float, parts[1:5])
                except ValueError:
                    invalid_label_count += 1
                    issues_for_file.append(f"invalid line {line_index}")
                    continue
                parsed.append((class_id, xc, yc, w, h))
                area = w * h
                x0 = xc - (w / 2.0)
                y0 = yc - (h / 2.0)
                x1 = xc + (w / 2.0)
                y1 = yc + (h / 2.0)
                if not (0.0 <= x0 <= 1.0 and 0.0 <= y0 <= 1.0 and 0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0):
                    out_of_bounds_box_count += 1
                    issues_for_file.append(f"out-of-bounds line {line_index}")
                if area < 0.0004:
                    tiny_box_count += 1
                    issues_for_file.append(f"tiny box line {line_index}")
                if area > 0.65:
                    oversize_box_count += 1
                    issues_for_file.append(f"oversize box line {line_index}")

            for index, box_a in enumerate(parsed):
                for box_b in parsed[index + 1 :]:
                    if box_a[0] != box_b[0]:
                        continue
                    iou_value = iou(
                        to_xyxy(box_a[1], box_a[2], box_a[3], box_a[4], 1000, 1000),
                        to_xyxy(box_b[1], box_b[2], box_b[3], box_b[4], 1000, 1000),
                    )
                    if iou_value >= 0.85:
                        duplicate_box_count += 1
                        issues_for_file.append(f"duplicate box IoU {iou_value:.2f}")
                        break

            if issues_for_file:
                suspect_label_files.append(str(label_path))
                label_issues.append(
                    DataQualityIssue(
                        category="label",
                        severity="high" if any("invalid" in issue or "out-of-bounds" in issue for issue in issues_for_file) else "medium",
                        item_id=str(label_path),
                        summary=" / ".join(dict.fromkeys(issues_for_file)),
                        metrics={
                            "issue_count": len(issues_for_file),
                            "box_count": len(parsed),
                        },
                    )
                )

        return {
            "label_files_scanned": len(label_files),
            "invalid_label_count": invalid_label_count,
            "out_of_bounds_box_count": out_of_bounds_box_count,
            "tiny_box_count": tiny_box_count,
            "oversize_box_count": oversize_box_count,
            "duplicate_box_count": duplicate_box_count,
            "suspect_label_files": suspect_label_files[:60],
            "label_issues": label_issues[:60],
        }

    def _write_report(self, response: DataQualityAuditResponse) -> None:
        json_path = Path(response.report_json_path)
        markdown_path = Path(response.report_markdown_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(json_path, response.model_dump_json(indent=2), encoding="utf-8")
        markdown_lines = [
            "# Data Quality Audit",
            "",
            f"- Position issues: **{response.position_issue_count}**",
            f"- Severe position issues: **{response.severe_position_issue_count}**",
            f"- Label issues: **{response.label_issue_count}**",
            f"- Invalid labels: **{response.invalid_label_count}**",
            f"- Out-of-bounds boxes: **{response.out_of_bounds_box_count}**",
            f"- Tiny boxes: **{response.tiny_box_count}**",
            f"- Oversize boxes: **{response.oversize_box_count}**",
            f"- Duplicate boxes: **{response.duplicate_box_count}**",
            "",
            "## Next Actions",
            *(f"- {item}" for item in response.next_actions),
        ]
        atomic_write_text(markdown_path, "\n".join(markdown_lines) + "\n", encoding="utf-8")


data_quality_audit_service = DataQualityAuditService()
