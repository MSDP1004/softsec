"""섀도우 피처: 아직 실거래 예측식(predictor.FEATURE_NAMES)에는 안 쓰지만
매일 값만 기록해서 검증 대기 중인 후보 피처.

`config/candidate_features.yaml`에서 status가 "shadow"인 후보만, 그리고 이
레지스트리에 실제 계산 함수가 구현된 것만 매일 값이 기록된다. 표본이 충분히
쌓이고 `shadow_evaluation.py`가 기존 피처 대비 실제로 도움이 된다고 판단하기
전까지는 predictor.py의 정식 피처로 승격되지 않는다.

새 후보를 shadow로 켜려면:
1. `config/candidate_features.yaml`에 후보를 추가/검토 (LLM 리서치 결과 또는 수동 추가)
2. 여기 SHADOW_FEATURE_REGISTRY에 계산 함수를 구현
3. yaml의 해당 후보 status를 "shadow"로 변경
"""

from __future__ import annotations

from collections.abc import Callable

from .data_fetcher import TickerSnapshot


def analyst_upside(snapshot: TickerSnapshot) -> float | None:
    """애널리스트 목표주가 컨센서스 대비 상승여력. (targetMeanPrice/price - 1)"""
    target = snapshot.analyst_target_mean
    price = snapshot.price
    if not target or not price or price <= 0:
        return None
    return target / price - 1


def week52_position(snapshot: TickerSnapshot) -> float | None:
    """52주 최저~최고 구간 내에서 현재가의 상대 위치 (0=52주 최저, 1=52주 최고)."""
    low = snapshot.week52_low
    high = snapshot.week52_high
    price = snapshot.price
    if low is None or high is None or price is None or high <= low:
        return None
    return (price - low) / (high - low)


SHADOW_FEATURE_REGISTRY: dict[str, Callable[[TickerSnapshot], float | None]] = {
    "analyst_upside": analyst_upside,
    "week52_position": week52_position,
}


def compute_shadow_values(
    snapshot: TickerSnapshot, active_ids: set[str]
) -> dict[str, float]:
    """레지스트리에 구현돼 있고 candidate_features.yaml에서 shadow 상태인 피처만 계산."""
    values = {}
    for feature_id in active_ids:
        fn = SHADOW_FEATURE_REGISTRY.get(feature_id)
        if fn is None:
            continue
        value = fn(snapshot)
        if value is not None:
            values[feature_id] = value
    return values
