from __future__ import annotations

from app.domain.defect_autolabel_models import DefectAssetRecord, DefectGroupRecord, DefectProposal
from app.plugins.registration.translation_aligner import estimate_translation, load_grayscale_array


class DefectPropagationService:
    def propagate_groups(
        self,
        *,
        groups: list[DefectGroupRecord],
        assets_by_id: dict[str, DefectAssetRecord],
        proposals_by_asset_id: dict[str, list[DefectProposal]],
    ) -> tuple[dict[str, list[DefectProposal]], list[dict[str, object]]]:
        updated = {asset_id: [proposal.model_copy() for proposal in proposals] for asset_id, proposals in proposals_by_asset_id.items()}
        reports: list[dict[str, object]] = []

        for group in groups:
            if not group.anchor_asset_id or len(group.member_asset_ids) <= 1:
                continue
            anchor_asset = assets_by_id[group.anchor_asset_id]
            anchor_gray = load_grayscale_array(anchor_asset.image_path)
            anchor_proposals = updated.get(anchor_asset.asset_id, [])
            for member_asset_id in group.member_asset_ids:
                if member_asset_id == anchor_asset.asset_id:
                    continue
                member_asset = assets_by_id[member_asset_id]
                report = {
                    "group_id": group.group_id,
                    "anchor_asset_id": anchor_asset.asset_id,
                    "target_asset_id": member_asset.asset_id,
                    "accepted": False,
                    "dx_px": 0,
                    "dy_px": 0,
                    "similarity": 0.0,
                    "reason": "registration_failed",
                }
                try:
                    member_gray = load_grayscale_array(member_asset.image_path)
                    dx_px, dy_px, similarity = estimate_translation(anchor_gray, member_gray, max_shift_px=8)
                except Exception as exc:
                    report["reason"] = f"registration_error:{exc}"
                    reports.append(report)
                    continue

                report["dx_px"] = dx_px
                report["dy_px"] = dy_px
                report["similarity"] = round(float(similarity), 4)
                mean_corner_error = (abs(dx_px) + abs(dy_px)) / 2.0
                if abs(dx_px) > 8 or abs(dy_px) > 8 or mean_corner_error > 2.5 or similarity < 0.10:
                    report["reason"] = "safety_threshold_reject"
                    reports.append(report)
                    continue

                propagated = []
                for proposal in anchor_proposals:
                    shifted_box = self._shift_box(
                        proposal.bbox_xyxy,
                        dx_px=dx_px,
                        dy_px=dy_px,
                        image_width=member_asset.width,
                        image_height=member_asset.height,
                    )
                    propagated.append(
                        proposal.model_copy(
                            update={
                                "proposal_id": "",
                                "asset_id": member_asset.asset_id,
                                "image_path": member_asset.image_path,
                                "split": member_asset.split,
                                "image_width": member_asset.width,
                                "image_height": member_asset.height,
                                "bbox_xyxy": shifted_box,
                                "bbox_yolo": self._xyxy_to_yolo(shifted_box, image_width=member_asset.width, image_height=member_asset.height),
                                "confidence": round(max(0.18, min(0.95, proposal.confidence * max(0.55, float(similarity)))), 4),
                                "source_mode": "anchor_propagated",
                                "views_supporting": sorted(set([*proposal.views_supporting, "anchor_propagated"])),
                                "review_required": True,
                                "priority": "high",
                                "quality_flags": list(dict.fromkeys([*proposal.quality_flags, "propagated_from_anchor"])),
                            }
                        )
                    )
                updated.setdefault(member_asset.asset_id, []).extend(propagated)
                report["accepted"] = True
                report["reason"] = "accepted"
                report["propagated_count"] = len(propagated)
                reports.append(report)
        return updated, reports

    def _shift_box(
        self,
        bbox_xyxy: list[float],
        *,
        dx_px: int,
        dy_px: int,
        image_width: int,
        image_height: int,
    ) -> list[float]:
        x0 = max(0.0, min(float(image_width), bbox_xyxy[0] + dx_px))
        y0 = max(0.0, min(float(image_height), bbox_xyxy[1] + dy_px))
        x1 = max(0.0, min(float(image_width), bbox_xyxy[2] + dx_px))
        y1 = max(0.0, min(float(image_height), bbox_xyxy[3] + dy_px))
        return [round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)]

    def _xyxy_to_yolo(self, bbox_xyxy: list[float], *, image_width: int, image_height: int) -> list[float]:
        x0, y0, x1, y1 = bbox_xyxy
        box_width = max(0.0, x1 - x0)
        box_height = max(0.0, y1 - y0)
        x_center = x0 + (box_width / 2.0)
        y_center = y0 + (box_height / 2.0)
        return [
            round(x_center / max(1.0, float(image_width)), 6),
            round(y_center / max(1.0, float(image_height)), 6),
            round(box_width / max(1.0, float(image_width)), 6),
            round(box_height / max(1.0, float(image_height)), 6),
        ]


defect_propagation_service = DefectPropagationService()
