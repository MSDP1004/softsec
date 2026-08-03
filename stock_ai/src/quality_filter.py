"""매출/이익이 실재하는 우량 기업만 남기는 퀄리티 필터.

사용자 투자성향: "이익이나 매출이 발생하지 않는 기업은 선호하지 않음"을 반영해,
매출이 없거나 순이익이 적자인 종목, 그리고 지나치게 낮은 이익률의 종목을 걸러낸다.
"""

from __future__ import annotations

from dataclasses import dataclass

from .data_fetcher import TickerSnapshot


@dataclass
class QualityThresholds:
    min_revenue: float = 0.0  # 매출이 존재해야 함
    require_positive_net_income: bool = True
    min_profit_margin: float = 0.0  # 순이익률 최소 기준 (0 = 흑자면 통과)


def passes_quality_filter(
    snapshot: TickerSnapshot, thresholds: QualityThresholds = QualityThresholds()
) -> bool:
    if snapshot.total_revenue is None or snapshot.total_revenue <= thresholds.min_revenue:
        return False

    if thresholds.require_positive_net_income:
        if snapshot.net_income is None or snapshot.net_income <= 0:
            return False

    if snapshot.profit_margin is not None and snapshot.profit_margin < thresholds.min_profit_margin:
        return False

    return True


def filter_snapshots(
    snapshots: dict[str, TickerSnapshot], thresholds: QualityThresholds = QualityThresholds()
) -> dict[str, TickerSnapshot]:
    return {
        ticker: snap
        for ticker, snap in snapshots.items()
        if passes_quality_filter(snap, thresholds)
    }
