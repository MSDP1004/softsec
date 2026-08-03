"""밸류체인 노드별 밸류에이션 집계 및 상대적 저평가 스코어링.

장기투자 성향에 맞춰 단기 가격 예측이 아니라, 밸류체인을 구성하는 여러 노드
(반도체 장비, AI칩, 클라우드 등) 사이에서 "지금 상대적으로 싸게 거래되는
구간이 어디인가"를 추적하는 것이 목적이다. 노드 간 밸류에이션 배수를 비교해
percentile을 매기고, 매 실행 결과를 history CSV에 누적해 시간에 따른
자기 자신의 밸류에이션 추이도 나중에 비교할 수 있게 한다.
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .data_fetcher import TickerSnapshot

HISTORY_COLUMNS = [
    "date",
    "node_id",
    "median_forward_pe",
    "median_price_to_sales",
    "median_peg",
    "undervaluation_score",
    "n_tickers",
]


@dataclass
class NodeValuation:
    node_id: str
    node_name: str
    median_forward_pe: float | None
    median_price_to_sales: float | None
    median_peg: float | None
    n_tickers: int
    undervaluation_score: float | None = None  # 0~1, 낮을수록 밸류체인 내 상대적 저평가


def _median(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None and v > 0]
    return statistics.median(clean) if clean else None


def compute_node_valuation(
    node_id: str, node_name: str, snapshots: dict[str, TickerSnapshot]
) -> NodeValuation:
    return NodeValuation(
        node_id=node_id,
        node_name=node_name,
        median_forward_pe=_median([s.forward_pe for s in snapshots.values()]),
        median_price_to_sales=_median([s.price_to_sales for s in snapshots.values()]),
        median_peg=_median([s.peg_ratio for s in snapshots.values()]),
        n_tickers=len(snapshots),
    )


def _percentile_rank(value: float, population: list[float]) -> float:
    """population 내에서 value 이하인 비율(0~1). 낮을수록 population 대비 저평가."""
    if not population:
        return 0.5
    n_leq = sum(1 for v in population if v <= value)
    return n_leq / len(population)


def score_undervaluation(node_valuations: list[NodeValuation]) -> list[NodeValuation]:
    """밸류체인 내 노드 간 상대 비교로 저평가 스코어를 매긴다.

    forward PE, P/S, PEG 각각에 대해 노드 간 percentile을 구한 뒤 평균낸다.
    스코어가 낮을수록(0에 가까울수록) 다른 노드 대비 상대적으로 저평가된 노드다.
    """
    metrics = ["median_forward_pe", "median_price_to_sales", "median_peg"]
    populations = {
        m: [getattr(nv, m) for nv in node_valuations if getattr(nv, m) is not None]
        for m in metrics
    }

    for nv in node_valuations:
        percentiles = []
        for m in metrics:
            value = getattr(nv, m)
            if value is not None and populations[m]:
                percentiles.append(_percentile_rank(value, populations[m]))
        nv.undervaluation_score = statistics.mean(percentiles) if percentiles else None

    return node_valuations


def append_history(
    node_valuations: list[NodeValuation], history_path: Path, run_date: date | None = None
) -> None:
    run_date = run_date or date.today()
    is_new = not history_path.exists()
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_COLUMNS)
        if is_new:
            writer.writeheader()
        for nv in node_valuations:
            writer.writerow(
                {
                    "date": run_date.isoformat(),
                    "node_id": nv.node_id,
                    "median_forward_pe": nv.median_forward_pe,
                    "median_price_to_sales": nv.median_price_to_sales,
                    "median_peg": nv.median_peg,
                    "undervaluation_score": nv.undervaluation_score,
                    "n_tickers": nv.n_tickers,
                }
            )
