import csv
from pathlib import Path

import numpy as np

from src.backtest import EVALUATION_COLUMNS, PREDICTION_COLUMNS
from src.shadow_backtest import SHADOW_COLUMNS
from src.shadow_evaluation import evaluate_feature


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_dataset(tmp_path: Path, horizon_data: dict[str, tuple[list[float], list[float]]]):
    """horizon_data: {horizon: (shadow_values, targets)}. baseline 피처는 항상 순수 잡음."""
    pred_rows, shadow_rows, eval_rows = [], [], []
    counter = 0
    for horizon, (shadow_values, targets) in horizon_data.items():
        rng = np.random.default_rng(100 + counter)
        for i, (sv, tgt) in enumerate(zip(shadow_values, targets)):
            counter += 1
            key = {
                "predict_date": f"2026-01-01-{counter}",
                "target_date": f"2026-02-01-{counter}",
                "horizon": horizon,
                "ticker": f"T{counter}",
                "node_id": "node1",
            }
            baseline = rng.normal(0, 1, 6)
            pred_rows.append(
                {
                    **key,
                    "entry_price": 100.0,
                    "x1": baseline[0],
                    "x2": baseline[1],
                    "x3": baseline[2],
                    "x4": baseline[3],
                    "x5": baseline[4],
                    "x6": baseline[5],
                    "predicted_return": 0.0,
                }
            )
            shadow_rows.append({**key, "feature_id": "test_feature", "value": sv})
            eval_rows.append(
                {
                    **key,
                    "entry_price": 100.0,
                    "exit_price": 100.0 * (1 + tgt),
                    "predicted_return": 0.0,
                    "realized_return": tgt,
                    "error": tgt,
                }
            )

    predictions_path = tmp_path / "predictions.csv"
    shadow_path = tmp_path / "shadow_predictions.csv"
    evaluations_path = tmp_path / "evaluations.csv"
    _write_csv(predictions_path, PREDICTION_COLUMNS, pred_rows)
    _write_csv(shadow_path, SHADOW_COLUMNS, shadow_rows)
    _write_csv(evaluations_path, EVALUATION_COLUMNS, eval_rows)
    return predictions_path, shadow_path, evaluations_path


def test_insufficient_data_keeps_shadow_status(tmp_path: Path):
    predictions_path, shadow_path, evaluations_path = _build_dataset(tmp_path, {})
    result = evaluate_feature("test_feature", predictions_path, shadow_path, evaluations_path)

    assert result["verdict"] == "shadow"
    assert all(d["status"] == "insufficient_data" for d in result["per_horizon"].values())


def test_strongly_predictive_shadow_feature_is_recommended_for_promotion(tmp_path: Path):
    rng = np.random.default_rng(1)
    n = 40
    shadow_values = rng.uniform(-1, 1, n)
    targets = 0.3 * shadow_values + rng.normal(0, 0.001, n)  # 거의 결정론적 관계

    predictions_path, shadow_path, evaluations_path = _build_dataset(
        tmp_path, {"1m": (shadow_values.tolist(), targets.tolist())}
    )
    result = evaluate_feature("test_feature", predictions_path, shadow_path, evaluations_path)

    assert result["verdict"] == "recommended_for_promotion"
    assert result["per_horizon"]["1m"]["status"] == "improved"


def test_noise_shadow_feature_is_rejected_after_two_horizons(tmp_path: Path):
    rng = np.random.default_rng(2)
    n_1m, n_1y = 30, 20
    shadow_1m = rng.normal(0, 1, n_1m)
    targets_1m = rng.normal(0, 0.02, n_1m)  # shadow와 무관한 잡음
    shadow_1y = rng.normal(0, 1, n_1y)
    targets_1y = rng.normal(0, 0.02, n_1y)

    predictions_path, shadow_path, evaluations_path = _build_dataset(
        tmp_path,
        {
            "1m": (shadow_1m.tolist(), targets_1m.tolist()),
            "1y": (shadow_1y.tolist(), targets_1y.tolist()),
        },
    )
    result = evaluate_feature("test_feature", predictions_path, shadow_path, evaluations_path)

    assert result["verdict"] == "rejected"
    assert result["per_horizon"]["1m"]["status"] == "not_improved"
    assert result["per_horizon"]["1y"]["status"] == "not_improved"
