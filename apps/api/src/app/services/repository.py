from __future__ import annotations

from dataclasses import dataclass, field

from app.domain.models import (
    Asset,
    AuditLog,
    EvaluationResult,
    Job,
    SubsetBenchmark,
    SyntheticAssetLink,
    SyntheticRecipe,
)


@dataclass
class InMemoryRepository:
    assets: dict[str, Asset] = field(default_factory=dict)
    jobs: dict[str, Job] = field(default_factory=dict)
    evaluations: dict[str, EvaluationResult] = field(default_factory=dict)
    benchmarks: dict[str, SubsetBenchmark] = field(default_factory=dict)
    recipes: dict[str, SyntheticRecipe] = field(default_factory=dict)
    synthetic_links: dict[str, SyntheticAssetLink] = field(default_factory=dict)
    audit_logs: list[AuditLog] = field(default_factory=list)

    def add_asset(self, asset: Asset) -> None:
        self.assets[asset.id] = asset

    def get_asset(self, asset_id: str) -> Asset:
        return self.assets[asset_id]

    def list_assets(self) -> list[Asset]:
        return sorted(self.assets.values(), key=lambda item: item.created_at, reverse=True)

    def save_job(self, job: Job) -> None:
        self.jobs[job.id] = job

    def save_evaluation(self, evaluation: EvaluationResult) -> None:
        self.evaluations[evaluation.id] = evaluation

    def save_benchmark(self, benchmark: SubsetBenchmark) -> None:
        self.benchmarks[benchmark.id] = benchmark

    def save_recipe(self, recipe: SyntheticRecipe) -> None:
        self.recipes[recipe.id] = recipe

    def save_link(self, link: SyntheticAssetLink) -> None:
        self.synthetic_links[link.id] = link

    def log(self, action: str, entity_type: str, entity_id: str, payload: dict | None = None) -> None:
        self.audit_logs.append(
            AuditLog(action=action, entity_type=entity_type, entity_id=entity_id, payload=payload or {})
        )


repository = InMemoryRepository()
