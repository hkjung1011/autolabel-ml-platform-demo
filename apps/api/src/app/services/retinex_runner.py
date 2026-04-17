from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from app.domain.research_models import PairGroup, RetinexJobResponse, RetinexRunRequest
from app.plugins.retinex.config import DEFAULT_MSRCR_CONFIG, MSRCRConfig
from app.plugins.retinex.provider import RetinexEnhancementProvider


class RetinexRunnerService:
    def __init__(self) -> None:
        self.jobs: dict[str, RetinexJobResponse] = {}

    def run(self, request: RetinexRunRequest) -> RetinexJobResponse:
        workspace_root = Path(request.workspace_root)
        manifest_path = workspace_root / "manifests" / "labeled_pair_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing labeled manifest: {manifest_path}")

        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        groups = [PairGroup.model_validate(item) for item in payload.get("groups", [])]
        if request.group_ids:
            group_ids = set(request.group_ids)
            groups = [group for group in groups if group.key in group_ids]

        config = self._build_config(request.params)
        provider = RetinexEnhancementProvider(config=config)
        output_root = workspace_root / "variants" / "retinex_msrcr"
        output_root.mkdir(parents=True, exist_ok=True)

        outputs = []
        errors = []
        for group in groups:
            for source_lux in request.source_luxes:
                if source_lux not in group.exposures:
                    errors.append(f"{group.key}: missing lux{source_lux} exposure")
                    continue
                try:
                    outputs.append(
                        provider.create_variant(
                            workspace_root=workspace_root,
                            group=group,
                            source_lux=source_lux,
                            overwrite=request.overwrite,
                        )
                    )
                except Exception as exc:  # pragma: no cover - surfaced in API output
                    errors.append(f"{group.key}/lux{source_lux}: {exc}")

        job = RetinexJobResponse(
            job_id=f"retinex_{uuid4().hex[:12]}",
            status="completed" if not errors else ("partial" if outputs else "failed"),
            method=request.method,
            workspace_root=str(workspace_root),
            total_groups=len(groups),
            processed_groups=len({result.group_id for result in outputs}),
            outputs=outputs,
            errors=errors,
            output_root=str(output_root),
        )
        self.jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> RetinexJobResponse:
        return self.jobs[job_id]

    def _build_config(self, params: dict[str, float | list[float] | int]) -> MSRCRConfig:
        sigma_list = params.get("sigma_list", DEFAULT_MSRCR_CONFIG.sigma_list)
        if isinstance(sigma_list, (int, float)):
            sigma_list = [float(sigma_list)]
        sigma_list = [float(value) for value in sigma_list]
        return MSRCRConfig(
            sigma_list=sigma_list or DEFAULT_MSRCR_CONFIG.sigma_list,
            gain=float(params.get("gain", DEFAULT_MSRCR_CONFIG.gain)),
            offset=float(params.get("offset", DEFAULT_MSRCR_CONFIG.offset)),
            alpha=float(params.get("alpha", DEFAULT_MSRCR_CONFIG.alpha)),
            beta=float(params.get("beta", DEFAULT_MSRCR_CONFIG.beta)),
            low_clip_percentile=float(params.get("low_clip_percentile", DEFAULT_MSRCR_CONFIG.low_clip_percentile)),
            high_clip_percentile=float(params.get("high_clip_percentile", DEFAULT_MSRCR_CONFIG.high_clip_percentile)),
        )


retinex_runner_service = RetinexRunnerService()
