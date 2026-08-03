"""전체 파이프라인: taxonomy 로드 -> 데이터 수집 -> 밸류에이션 스코어링
-> 퀄리티 필터 -> 저평가 노드 내 우량주 후보 리스트 생성.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .data_fetcher import fetch_snapshots
from .quality_filter import QualityThresholds, filter_snapshots
from .taxonomy_manager import load_taxonomy
from .valuation import (
    NodeValuation,
    append_history,
    compute_node_valuation,
    score_undervaluation,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_TAXONOMY_PATH = BASE_DIR / "config" / "taxonomy.yaml"
DEFAULT_HISTORY_PATH = BASE_DIR / "data" / "history" / "valuation_history.csv"


@dataclass
class Candidate:
    ticker: str
    node_id: str
    node_name: str
    forward_pe: float | None
    price_to_sales: float | None
    peg_ratio: float | None
    revenue_growth: float | None
    profit_margin: float | None


@dataclass
class PipelineResult:
    node_valuations: list[NodeValuation]
    candidates: list[Candidate]


def run_pipeline(
    taxonomy_path: Path = DEFAULT_TAXONOMY_PATH,
    history_path: Path = DEFAULT_HISTORY_PATH,
    quality_thresholds: QualityThresholds = QualityThresholds(),
    top_k_nodes: int = 3,
) -> PipelineResult:
    taxonomy = load_taxonomy(taxonomy_path)

    node_valuations: list[NodeValuation] = []
    quality_snapshots_by_node: dict[str, dict] = {}

    for node in taxonomy.nodes:
        snapshots = fetch_snapshots(node.tickers)
        node_valuations.append(compute_node_valuation(node.id, node.name, snapshots))
        quality_snapshots_by_node[node.id] = filter_snapshots(snapshots, quality_thresholds)

    node_valuations = score_undervaluation(node_valuations)
    append_history(node_valuations, history_path)

    ranked_nodes = sorted(
        (nv for nv in node_valuations if nv.undervaluation_score is not None),
        key=lambda nv: nv.undervaluation_score,
    )
    undervalued_node_ids = {nv.node_id for nv in ranked_nodes[:top_k_nodes]}

    candidates: list[Candidate] = []
    for node in taxonomy.nodes:
        if node.id not in undervalued_node_ids:
            continue
        for ticker, snap in quality_snapshots_by_node[node.id].items():
            candidates.append(
                Candidate(
                    ticker=ticker,
                    node_id=node.id,
                    node_name=node.name,
                    forward_pe=snap.forward_pe,
                    price_to_sales=snap.price_to_sales,
                    peg_ratio=snap.peg_ratio,
                    revenue_growth=snap.revenue_growth,
                    profit_margin=snap.profit_margin,
                )
            )

    return PipelineResult(node_valuations=node_valuations, candidates=candidates)
