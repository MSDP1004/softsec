"""섀도우 피처 값 기록.

`backtest.py`의 predictions.csv와 동일한 (predict_date, target_date, horizon,
ticker) 키로 기록해서, 나중에 evaluations.csv와 그대로 join할 수 있게 한다.
실거래 예측값(predicted_return)에는 전혀 관여하지 않는다 — 값만 기록.
"""

from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path

from .predictor import HORIZON_DAYS, HORIZONS

SHADOW_COLUMNS = [
    "predict_date",
    "target_date",
    "horizon",
    "ticker",
    "node_id",
    "feature_id",
    "value",
]


def build_shadow_rows(
    today: date, ticker: str, node_id: str, shadow_values: dict[str, float]
) -> list[dict]:
    rows = []
    for horizon in HORIZONS:
        target = today + timedelta(days=HORIZON_DAYS[horizon])
        for feature_id, value in shadow_values.items():
            rows.append(
                {
                    "predict_date": today.isoformat(),
                    "target_date": target.isoformat(),
                    "horizon": horizon,
                    "ticker": ticker,
                    "node_id": node_id,
                    "feature_id": feature_id,
                    "value": value,
                }
            )
    return rows


def append_shadow_rows(rows: list[dict], path: Path) -> None:
    is_new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SHADOW_COLUMNS)
        if is_new:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
