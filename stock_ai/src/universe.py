"""매일 추적할 종목 유니버스 선정.

노드별로 퀄리티 필터를 통과한 종목 중 PEG(성장 대비 가격)가 낮은 순으로
상위 N개를 뽑는다. 저평가 상위 노드로만 국한하지 않고 모든 노드에서 뽑는
이유는, 가중치 재적합(calibration)에 쓸 표본을 여러 노드에 걸쳐 최대한
빨리, 넓게 확보하기 위함이다. taxonomy에 노드가 늘어날수록 추적 종목 수도
자연히 늘어난다.
"""

from __future__ import annotations

from dataclasses import dataclass

from .data_fetcher import TickerSnapshot
from .valuation import NodeValuation


@dataclass
class TrackedTicker:
    ticker: str
    node_id: str
    node_name: str
    snapshot: TickerSnapshot
    node_valuation: NodeValuation | None


def select_top_n_per_node(
    node_valuations: list[NodeValuation],
    quality_snapshots_by_node: dict[str, dict[str, TickerSnapshot]],
    top_n: int = 2,
) -> list[TrackedTicker]:
    nv_by_id = {nv.node_id: nv for nv in node_valuations}
    tracked: list[TrackedTicker] = []

    for node_id, snapshots in quality_snapshots_by_node.items():
        ranked = sorted(
            snapshots.items(),
            key=lambda kv: kv[1].peg_ratio if kv[1].peg_ratio and kv[1].peg_ratio > 0 else float("inf"),
        )
        nv = nv_by_id.get(node_id)
        node_name = nv.node_name if nv else node_id
        for ticker, snap in ranked[:top_n]:
            tracked.append(
                TrackedTicker(ticker=ticker, node_id=node_id, node_name=node_name, snapshot=snap, node_valuation=nv)
            )

    return tracked
