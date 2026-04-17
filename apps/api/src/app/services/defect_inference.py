from __future__ import annotations

import hashlib
import numpy as np
from PIL import Image, ImageFilter, ImageOps

from app.domain.defect_autolabel_models import (
    CANONICAL_CLASS_NAMES,
    CLASS_NAME_TO_ID,
    DOMAIN_ALLOWED_CLASS_IDS,
    DefectAssetRecord,
    DefectProposal,
)
from app.services.defect_quality import defect_quality_service

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None


class DefectInferenceService:
    def detect_asset(
        self,
        *,
        asset: DefectAssetRecord,
        quality: dict[str, object],
        domain: str,
    ) -> list[DefectProposal]:
        gray = defect_quality_service.load_gray(asset.image_path)
        views = {
            "original": gray,
            "normalized_luma": defect_quality_service.build_normalized_view(gray),
            "edge_emphasis": defect_quality_service.build_edge_view(gray),
        }
        raw_candidates: list[DefectProposal] = []
        for view_name, view_gray in views.items():
            raw_candidates.extend(self._proposals_from_view(view_gray=view_gray, asset=asset, domain=domain, view_name=view_name))
        merged = self._merge_same_class_boxes(raw_candidates, iou_threshold=0.42)
        finalized: list[DefectProposal] = []
        threshold = self._threshold_for_bucket(str(quality.get("lux_bucket", "normal")))
        for proposal in merged:
            if proposal.confidence < threshold:
                continue
            quality_flags = self._quality_flags(proposal=proposal, quality=quality, image_width=asset.width, image_height=asset.height)
            review_required, priority = self._review_policy(
                proposal=proposal,
                quality=quality,
                quality_flags=quality_flags,
                split=asset.split,
                threshold=threshold,
            )
            finalized.append(
                proposal.model_copy(
                    update={
                        "proposal_id": self._proposal_id_for(proposal, suffix="direct"),
                        "review_required": review_required,
                        "priority": priority,
                        "quality_flags": quality_flags,
                        "lux_bucket": str(quality.get("lux_bucket", "normal")),
                    }
                )
            )
        return finalized

    def merge_direct_and_propagated(
        self,
        *,
        direct_proposals: list[DefectProposal],
        propagated_proposals: list[DefectProposal],
        quality: dict[str, object],
        split: str,
    ) -> list[DefectProposal]:
        proposals = [proposal.model_copy() for proposal in direct_proposals]
        for propagated in propagated_proposals:
            matched_index = None
            for index, existing in enumerate(proposals):
                if existing.class_id != propagated.class_id:
                    continue
                if self._iou(existing.bbox_xyxy, propagated.bbox_xyxy) < 0.35:
                    continue
                matched_index = index
                break
            if matched_index is None:
                proposals.append(
                    propagated.model_copy(
                        update={
                            "proposal_id": self._proposal_id_for(propagated, suffix="propagated"),
                            "review_required": True,
                            "priority": "high",
                            "quality_flags": list(dict.fromkeys([*propagated.quality_flags, "propagated_from_anchor"])),
                        }
                    )
                )
                continue
            existing = proposals[matched_index]
            merged_box = self._weighted_box([existing, propagated])
            merged_views = sorted(set(existing.views_supporting + propagated.views_supporting))
            merged_confidence = round(min(0.99, max(existing.confidence, propagated.confidence) + 0.05), 4)
            merged_flags = list(dict.fromkeys([*existing.quality_flags, *propagated.quality_flags, "propagated_from_anchor"]))
            merged = existing.model_copy(
                update={
                    "proposal_id": self._proposal_id_for(existing, suffix="direct_plus_propagated"),
                    "bbox_xyxy": merged_box,
                    "bbox_yolo": self._xyxy_to_yolo(
                        merged_box,
                        image_width=existing.image_width,
                        image_height=existing.image_height,
                    ),
                    "confidence": merged_confidence,
                    "source_mode": "direct_plus_propagated",
                    "views_supporting": merged_views,
                    "review_required": True,
                    "priority": "high",
                    "quality_flags": merged_flags,
                }
            )
            proposals[matched_index] = merged

        finalized: list[DefectProposal] = []
        threshold = self._threshold_for_bucket(str(quality.get("lux_bucket", "normal")))
        for proposal in proposals:
            quality_flags = self._quality_flags(
                proposal=proposal,
                quality=quality,
                image_width=proposal.image_width,
                image_height=proposal.image_height,
            )
            review_required, priority = self._review_policy(
                proposal=proposal,
                quality=quality,
                quality_flags=quality_flags,
                split=split,
                threshold=threshold,
            )
            if "propagated_from_anchor" in quality_flags or "propagated_from_anchor" in proposal.quality_flags:
                review_required = True
                priority = "high"
            finalized.append(
                proposal.model_copy(
                    update={
                        "quality_flags": list(dict.fromkeys([*proposal.quality_flags, *quality_flags])),
                        "review_required": review_required,
                        "priority": priority,
                    }
                )
            )
        return finalized

    def _proposals_from_view(
        self,
        *,
        view_gray: np.ndarray,
        asset: DefectAssetRecord,
        domain: str,
        view_name: str,
    ) -> list[DefectProposal]:
        allowed_class_ids = set(DOMAIN_ALLOWED_CLASS_IDS[domain])
        gray = view_gray.astype(np.uint8)
        total_pixels = float(asset.width * asset.height)
        if total_pixels <= 0:
            return []

        background = self._blur(gray, radius=7)
        diff = gray.astype(np.float32) - background.astype(np.float32)
        dark_mask = self._clean_mask((diff <= -18.0).astype(np.uint8) * 255)
        bright_mask = self._clean_mask((diff >= 18.0).astype(np.uint8) * 255)
        edge_mask = self._build_edge_mask(gray)

        proposals: list[DefectProposal] = []
        for kind, mask in (("dark", dark_mask), ("bright", bright_mask), ("edge", edge_mask)):
            for contour in self._contours_for(mask):
                x, y, box_width, box_height = self._bounding_rect(contour)
                area = max(1.0, self._contour_area(contour))
                area_ratio = area / total_pixels
                if area_ratio < 0.0007:
                    continue
                if min(box_width, box_height) < 3:
                    continue
                aspect = max(box_width / max(box_height, 1), box_height / max(box_width, 1))
                fill_ratio = area / max(1.0, float(box_width * box_height))
                near_edge = x <= 2 or y <= 2 or (x + box_width) >= (asset.width - 2) or (y + box_height) >= (asset.height - 2)
                strength = float(np.mean(np.abs(diff[y : y + box_height, x : x + box_width]))) / 42.0
                class_name = self._class_name_for(
                    kind=kind,
                    domain=domain,
                    area_ratio=area_ratio,
                    aspect=aspect,
                    fill_ratio=fill_ratio,
                    near_edge=near_edge,
                )
                class_id = CLASS_NAME_TO_ID[class_name]
                if class_id not in allowed_class_ids:
                    continue
                bbox_xyxy = [float(x), float(y), float(x + box_width), float(y + box_height)]
                bbox_yolo = self._xyxy_to_yolo(bbox_xyxy, image_width=asset.width, image_height=asset.height)
                confidence = round(float(np.clip(0.20 + (strength * 0.25) + min(area_ratio * 8.0, 0.25) + (0.04 if view_name != "original" else 0.0), 0.18, 0.92)), 4)
                proposals.append(
                    DefectProposal(
                        proposal_id="",
                        asset_id=asset.asset_id,
                        image_path=asset.image_path,
                        split=asset.split,
                        image_width=asset.width,
                        image_height=asset.height,
                        class_id=class_id,
                        class_name=CANONICAL_CLASS_NAMES[class_id],
                        confidence=confidence,
                        bbox_xyxy=bbox_xyxy,
                        bbox_yolo=bbox_yolo,
                        source_mode="direct_detect",
                        views_supporting=[view_name],
                        review_required=False,
                        priority="normal",
                        quality_flags=[],
                        lux_bucket=asset.estimated_lux_bucket,
                    )
                )
        return proposals

    def _class_name_for(
        self,
        *,
        kind: str,
        domain: str,
        area_ratio: float,
        aspect: float,
        fill_ratio: float,
        near_edge: bool,
    ) -> str:
        if kind == "edge":
            if near_edge:
                return "burr" if domain == "metal_plate_defect" else "edge_damage"
            if aspect >= 5.0:
                return "scratch" if domain == "metal_plate_defect" else "crack"
            return "weld_defect"

        if kind == "dark":
            if aspect >= 5.0 and fill_ratio <= 0.50:
                return "scratch" if domain == "metal_plate_defect" else "crack"
            if area_ratio <= 0.01 and fill_ratio >= 0.55:
                return "hole_pit"
            if area_ratio >= 0.07:
                return "dent_deformation"
            if domain == "ship_defect":
                return "corrosion" if area_ratio <= 0.045 else "coating_damage"
            return "surface_stain"

        if domain == "ship_defect":
            return "blister_bubble" if area_ratio <= 0.03 and fill_ratio >= 0.65 else "contamination"
        return "contamination" if area_ratio <= 0.03 else "surface_stain"

    def _merge_same_class_boxes(self, proposals: list[DefectProposal], *, iou_threshold: float) -> list[DefectProposal]:
        merged: list[DefectProposal] = []
        for proposal in sorted(proposals, key=lambda item: item.confidence, reverse=True):
            matched_index = None
            for index, existing in enumerate(merged):
                if existing.class_id != proposal.class_id:
                    continue
                if self._iou(existing.bbox_xyxy, proposal.bbox_xyxy) < iou_threshold:
                    continue
                matched_index = index
                break
            if matched_index is None:
                merged.append(proposal.model_copy())
                continue
            existing = merged[matched_index]
            merged_box = self._weighted_box([existing, proposal])
            merged_views = sorted(set(existing.views_supporting + proposal.views_supporting))
            merged[matched_index] = existing.model_copy(
                update={
                    "bbox_xyxy": merged_box,
                    "bbox_yolo": self._xyxy_to_yolo(
                        merged_box,
                        image_width=existing.image_width,
                        image_height=existing.image_height,
                    ),
                    "confidence": round(min(0.99, max(existing.confidence, proposal.confidence) + (0.05 * max(0, len(merged_views) - 1))), 4),
                    "views_supporting": merged_views,
                }
            )
        return merged

    def _quality_flags(
        self,
        *,
        proposal: DefectProposal,
        quality: dict[str, object],
        image_width: int,
        image_height: int,
    ) -> list[str]:
        flags: list[str] = []
        lux_bucket = str(quality.get("lux_bucket", "normal"))
        if lux_bucket in {"very_dark", "glare"}:
            flags.append(lux_bucket)
        if float(quality.get("laplacian_blur_score", 1.0)) < 0.20:
            flags.append("low_blur_score")
        x0, y0, x1, y1 = proposal.bbox_xyxy
        if x0 <= 2 or y0 <= 2 or x1 >= image_width - 2 or y1 >= image_height - 2:
            flags.append("boundary_touching")
        area_ratio = max(0.0, (x1 - x0) * (y1 - y0)) / max(1.0, float(image_width * image_height))
        if area_ratio < 0.001:
            flags.append("tiny_box")
        if len(proposal.views_supporting) == 1:
            flags.append("single_view_support")
        return flags

    def _review_policy(
        self,
        *,
        proposal: DefectProposal,
        quality: dict[str, object],
        quality_flags: list[str],
        split: str,
        threshold: float,
    ) -> tuple[bool, str]:
        lux_bucket = str(quality.get("lux_bucket", "normal"))
        if split == "test" or lux_bucket in {"very_dark", "glare"} or "propagated_from_anchor" in quality_flags:
            return True, "high"
        if proposal.source_mode != "direct_detect":
            return True, "high"
        if lux_bucket in {"dark", "bright"}:
            return True, "medium"
        if "boundary_touching" in quality_flags or "tiny_box" in quality_flags or proposal.confidence < min(0.95, threshold + 0.08):
            return True, "medium"
        return False, "normal"

    def _threshold_for_bucket(self, lux_bucket: str) -> float:
        thresholds = {
            "very_dark": 0.18,
            "dark": 0.22,
            "normal": 0.28,
            "bright": 0.30,
            "glare": 0.32,
        }
        return thresholds.get(lux_bucket, 0.28)

    def _weighted_box(self, proposals: list[DefectProposal]) -> list[float]:
        total_weight = max(1e-6, sum(proposal.confidence for proposal in proposals))
        coords = [0.0, 0.0, 0.0, 0.0]
        for proposal in proposals:
            for index, value in enumerate(proposal.bbox_xyxy):
                coords[index] += value * proposal.confidence
        return [round(value / total_weight, 4) for value in coords]

    def _proposal_id_for(self, proposal: DefectProposal, *, suffix: str) -> str:
        token = f"{proposal.asset_id}:{proposal.class_id}:{proposal.bbox_xyxy}:{suffix}"
        return f"proposal_{hashlib.sha1(token.encode('utf-8')).hexdigest()[:12]}"

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

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        if cv2 is not None:
            kernel = np.ones((3, 3), dtype=np.uint8)
            opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)  # type: ignore[union-attr]
            return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)  # type: ignore[union-attr]
        image = Image.fromarray(mask).filter(ImageFilter.MedianFilter(size=3))
        return np.asarray(image, dtype=np.uint8)

    def _build_edge_mask(self, gray: np.ndarray) -> np.ndarray:
        if cv2 is not None:
            edges = cv2.Canny(gray, 50, 150)  # type: ignore[union-attr]
            kernel = np.ones((3, 3), dtype=np.uint8)
            return cv2.dilate(edges, kernel, iterations=1)  # type: ignore[union-attr]
        image = Image.fromarray(gray).filter(ImageFilter.FIND_EDGES)
        edged = ImageOps.autocontrast(image)
        return np.asarray(edged, dtype=np.uint8)

    def _blur(self, gray: np.ndarray, *, radius: int) -> np.ndarray:
        if cv2 is not None:
            return cv2.GaussianBlur(gray, (0, 0), radius)  # type: ignore[union-attr]
        return np.asarray(Image.fromarray(gray).filter(ImageFilter.GaussianBlur(radius=max(1, radius // 2))), dtype=np.uint8)

    def _contours_for(self, mask: np.ndarray) -> list[np.ndarray]:
        if cv2 is not None:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # type: ignore[union-attr]
            return list(contours)
        positions = np.argwhere(mask > 0)
        if positions.size == 0:
            return []
        y0, x0 = positions.min(axis=0)
        y1, x1 = positions.max(axis=0)
        return [np.asarray([[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]], dtype=np.int32)]

    def _bounding_rect(self, contour: np.ndarray) -> tuple[int, int, int, int]:
        if cv2 is not None:
            return cv2.boundingRect(contour)  # type: ignore[union-attr]
        xs = contour[:, :, 0]
        ys = contour[:, :, 1]
        x0 = int(xs.min())
        y0 = int(ys.min())
        return x0, y0, int(xs.max() - x0), int(ys.max() - y0)

    def _contour_area(self, contour: np.ndarray) -> float:
        if cv2 is not None:
            return float(cv2.contourArea(contour))  # type: ignore[union-attr]
        xs = contour[:, :, 0]
        ys = contour[:, :, 1]
        return float((xs.max() - xs.min()) * (ys.max() - ys.min()))

    def _iou(self, box_a: list[float], box_b: list[float]) -> float:
        x0 = max(box_a[0], box_b[0])
        y0 = max(box_a[1], box_b[1])
        x1 = min(box_a[2], box_b[2])
        y1 = min(box_a[3], box_b[3])
        intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
        area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
        union = area_a + area_b - intersection
        if union <= 1e-9:
            return 0.0
        return intersection / union
defect_inference_service = DefectInferenceService()
