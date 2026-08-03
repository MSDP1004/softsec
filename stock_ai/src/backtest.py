"""예측 기록, 만기 판정, 실현수익률 평가.

매일 실행되는 파이프라인이므로 "만기"는 단순 달력일 기준이다: target_date가
오늘이거나 지난 예측은 오늘 시점의 현재가로 평가한다 (주말·휴장일에는 직전
종가가 그대로 이어지므로 며칠 정도 어긋나도 실질적인 영향은 작다).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from .predictor import HORIZON_DAYS, HORIZONS, ModelWeights

PREDICTION_COLUMNS = [
    "predict_date",
    "target_date",
    "horizon",
    "ticker",
    "node_id",
    "entry_price",
    "x1",
    "x2",
    "x3",
    "x4",
    "x5",
    "x6",
    "predicted_return",
]

EVALUATION_COLUMNS = [
    "predict_date",
    "target_date",
    "horizon",
    "ticker",
    "node_id",
    "entry_price",
    "exit_price",
    "predicted_return",
    "realized_return",
    "error",
]


@dataclass
class Prediction:
    predict_date: str
    target_date: str
    horizon: str
    ticker: str
    node_id: str
    entry_price: float
    x1: float
    x2: float
    x3: float
    x4: float
    x5: float
    x6: float
    predicted_return: float


def build_predictions(
    today: date,
    ticker: str,
    node_id: str,
    entry_price: float,
    features: list[float],
    model: ModelWeights,
) -> list[Prediction]:
    predictions = []
    for horizon in HORIZONS:
        target = today + timedelta(days=HORIZON_DAYS[horizon])
        predicted_return = model.predict(horizon, features)
        predictions.append(
            Prediction(
                predict_date=today.isoformat(),
                target_date=target.isoformat(),
                horizon=horizon,
                ticker=ticker,
                node_id=node_id,
                entry_price=entry_price,
                x1=features[0],
                x2=features[1],
                x3=features[2],
                x4=features[3],
                x5=features[4],
                x6=features[5],
                predicted_return=predicted_return,
            )
        )
    return predictions


def append_predictions(predictions: list[Prediction], path: Path) -> None:
    is_new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_COLUMNS)
        if is_new:
            writer.writeheader()
        for p in predictions:
            writer.writerow(
                {
                    "predict_date": p.predict_date,
                    "target_date": p.target_date,
                    "horizon": p.horizon,
                    "ticker": p.ticker,
                    "node_id": p.node_id,
                    "entry_price": p.entry_price,
                    "x1": p.x1,
                    "x2": p.x2,
                    "x3": p.x3,
                    "x4": p.x4,
                    "x5": p.x5,
                    "x6": p.x6,
                    "predicted_return": p.predicted_return,
                }
            )


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _evaluated_keys(evaluations_path: Path) -> set[tuple[str, str, str]]:
    return {(r["predict_date"], r["ticker"], r["horizon"]) for r in read_csv_rows(evaluations_path)}


def find_matured_predictions(
    predictions_path: Path, evaluations_path: Path, today: date
) -> list[dict]:
    """target_date가 오늘 이하이면서 아직 평가되지 않은 예측 행을 반환한다."""
    done = _evaluated_keys(evaluations_path)
    matured = []
    for row in read_csv_rows(predictions_path):
        key = (row["predict_date"], row["ticker"], row["horizon"])
        if key in done:
            continue
        if date.fromisoformat(row["target_date"]) <= today:
            matured.append(row)
    return matured


def evaluate_predictions(matured: list[dict], current_prices: dict[str, float]) -> list[dict]:
    results = []
    for row in matured:
        exit_price = current_prices.get(row["ticker"])
        if exit_price is None:
            continue
        entry_price = float(row["entry_price"])
        if entry_price <= 0:
            continue
        realized_return = exit_price / entry_price - 1
        predicted_return = float(row["predicted_return"])
        results.append(
            {
                "predict_date": row["predict_date"],
                "target_date": row["target_date"],
                "horizon": row["horizon"],
                "ticker": row["ticker"],
                "node_id": row["node_id"],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "predicted_return": predicted_return,
                "realized_return": realized_return,
                "error": realized_return - predicted_return,
            }
        )
    return results


def append_evaluations(evaluations: list[dict], path: Path) -> None:
    is_new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EVALUATION_COLUMNS)
        if is_new:
            writer.writeheader()
        for e in evaluations:
            writer.writerow(e)
