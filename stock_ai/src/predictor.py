"""팩터 기반 기대수익률 선형모델.

가중치는 처음엔 전부 0으로 시작한다 (아직 아무 근거도 없으니 "초과수익 0%"가
가장 정직한 출발점). 실제 실현수익률 데이터가 horizon별로 충분히 쌓이면
`calibration.py`가 회귀로 가중치를 다시 적합한다.

피처는 6개. 앞의 4개(x1~x4)는 분기 실적이 나올 때만 바뀌는 재무지표 위주라
거의 매일 값이 고정돼 있고, 뒤의 2개(x5, x6)는 가격 모멘텀이라 매일 바뀐다:
  x1 = 1 - 노드 저평가스코어  (밸류체인 내에서 더 저평가된 노드일수록 큼)
  x2 = 매출성장률
  x3 = 순이익률
  x4 = 1 / PEG  (성장 대비 저렴할수록 큼)
  x5 = 최근 5거래일(약 1주) 수익률
  x6 = 최근 21거래일(약 1개월) 수익률
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

HORIZONS = ("1d", "1w", "1m", "1y")
HORIZON_DAYS = {"1d": 1, "1w": 7, "1m": 30, "1y": 365}
FEATURE_NAMES = (
    "undervaluation",
    "revenue_growth",
    "profit_margin",
    "inv_peg",
    "momentum_5d",
    "momentum_21d",
)


def _default_weights() -> dict[str, list[float]]:
    # [intercept, w1..w6] — 피처 개수 + 1
    return {h: [0.0] * (len(FEATURE_NAMES) + 1) for h in HORIZONS}


@dataclass
class ModelWeights:
    weights: dict[str, list[float]] = field(default_factory=_default_weights)
    meta: dict[str, dict] = field(default_factory=dict)

    def predict(self, horizon: str, features: list[float]) -> float:
        w = self.weights[horizon]
        return w[0] + sum(wi * xi for wi, xi in zip(w[1:], features))


def load_weights(path: Path) -> ModelWeights:
    if not path.exists():
        return ModelWeights()
    raw = json.loads(path.read_text(encoding="utf-8"))
    expected_len = len(FEATURE_NAMES) + 1
    defaults = _default_weights()
    weights = {}
    for h in HORIZONS:
        w = raw["weights"].get(h)
        # 피처 스키마가 바뀌면(예: 새 피처 추가) 길이가 안 맞는 옛 가중치는
        # 버리고 0으로 리셋한다 — 잘못된 위치에 곱해지는 것보다 안전하다.
        weights[h] = w if w is not None and len(w) == expected_len else defaults[h]
    return ModelWeights(weights=weights, meta=raw.get("meta", {}))


def save_weights(model: ModelWeights, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"weights": model.weights, "meta": model.meta}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def feature_vector(
    undervaluation_score: float | None,
    revenue_growth: float | None,
    profit_margin: float | None,
    peg_ratio: float | None,
    momentum_5d: float | None = None,
    momentum_21d: float | None = None,
) -> list[float]:
    x1 = 1 - undervaluation_score if undervaluation_score is not None else 0.5
    x2 = clip(revenue_growth if revenue_growth is not None else 0.0, -1.0, 5.0)
    x3 = clip(profit_margin if profit_margin is not None else 0.0, -1.0, 1.0)
    peg = peg_ratio if peg_ratio and peg_ratio > 0 else None
    x4 = 1 / clip(peg, 0.05, 50.0) if peg is not None else 0.0
    x5 = clip(momentum_5d if momentum_5d is not None else 0.0, -0.5, 0.5)
    x6 = clip(momentum_21d if momentum_21d is not None else 0.0, -0.5, 0.5)
    return [x1, x2, x3, x4, x5, x6]
