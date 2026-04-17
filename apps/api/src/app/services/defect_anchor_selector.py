from __future__ import annotations

from app.domain.defect_autolabel_models import DefectAssetRecord, DefectGroupRecord


class DefectAnchorSelectorService:
    def select_anchors(
        self,
        *,
        groups: list[DefectGroupRecord],
        assets_by_id: dict[str, DefectAssetRecord],
        quality_by_asset_id: dict[str, dict[str, object]],
    ) -> list[DefectGroupRecord]:
        selected: list[DefectGroupRecord] = []
        for group in groups:
            if len(group.member_asset_ids) <= 1:
                group.anchor_asset_id = group.member_asset_ids[0] if group.member_asset_ids else None
                selected.append(group)
                continue
            ranked = sorted(
                group.member_asset_ids,
                key=lambda asset_id: self._rank_key(
                    asset=assets_by_id[asset_id],
                    quality=quality_by_asset_id.get(asset_id, {}),
                ),
                reverse=True,
            )
            group.anchor_asset_id = ranked[0] if ranked else None
            selected.append(group)
        return selected

    def _rank_key(self, *, asset: DefectAssetRecord, quality: dict[str, object]) -> tuple[float, float, int, str]:
        vision_ready = float(quality.get("vision_ready_score", 0.0))
        glare = float(quality.get("specular_glare_score", 0.0))
        blur_score = float(quality.get("laplacian_blur_score", 0.0))
        effective_score = vision_ready - (0.35 * glare) - (0.20 * max(0.0, 1.0 - blur_score))
        reported_lux = int(asset.reported_lux or 0)
        return (
            round(effective_score, 6),
            round(-glare, 6),
            reported_lux,
            asset.image_path,
        )


defect_anchor_selector_service = DefectAnchorSelectorService()
