from datetime import date
from pathlib import Path

from src.backtest import (
    append_evaluations,
    append_predictions,
    build_predictions,
    evaluate_predictions,
    find_matured_predictions,
    read_csv_rows,
)
from src.predictor import ModelWeights


def test_build_predictions_creates_one_row_per_horizon_with_correct_target_dates():
    model = ModelWeights()
    today = date(2026, 1, 1)
    preds = build_predictions(
        today, "AAA", "node1", entry_price=100.0, features=[0.5, 0.1, 0.1, 1.0, 0.02, -0.01], model=model
    )

    by_horizon = {p.horizon: p for p in preds}
    assert by_horizon["1d"].target_date == "2026-01-02"
    assert by_horizon["1w"].target_date == "2026-01-08"
    assert by_horizon["1m"].target_date == "2026-01-31"
    assert by_horizon["1y"].target_date == "2027-01-01"
    assert all(p.predicted_return == 0.0 for p in preds)  # 초기 가중치는 전부 0


def test_append_and_read_predictions_roundtrip(tmp_path: Path):
    model = ModelWeights()
    today = date(2026, 1, 1)
    preds = build_predictions(today, "AAA", "node1", 100.0, [0.5, 0.1, 0.1, 1.0, 0.02, -0.01], model)

    path = tmp_path / "predictions.csv"
    append_predictions(preds, path)
    append_predictions(preds, path)  # 두 번째 append도 header 중복 없이 잘 붙는지

    rows = read_csv_rows(path)
    assert len(rows) == 8  # 4 horizons x 2 appends
    assert rows[0]["ticker"] == "AAA"


def test_find_matured_predictions_filters_by_target_date_and_evaluated_status(tmp_path: Path):
    model = ModelWeights()
    preds = build_predictions(date(2026, 1, 1), "AAA", "node1", 100.0, [0.5, 0.1, 0.1, 1.0, 0.02, -0.01], model)
    pred_path = tmp_path / "predictions.csv"
    eval_path = tmp_path / "evaluations.csv"
    append_predictions(preds, pred_path)

    # 아직 2026-01-01 시점에는 1d(target=01-02)조차 만기 전
    matured = find_matured_predictions(pred_path, eval_path, today=date(2026, 1, 1))
    assert matured == []

    # 01-02가 되면 1d만 만기
    matured = find_matured_predictions(pred_path, eval_path, today=date(2026, 1, 2))
    assert [m["horizon"] for m in matured] == ["1d"]

    # 이미 평가된 건은 다시 나오면 안 됨
    evals = evaluate_predictions(matured, {"AAA": 105.0})
    append_evaluations(evals, eval_path)
    matured_again = find_matured_predictions(pred_path, eval_path, today=date(2026, 1, 2))
    assert matured_again == []


def test_evaluate_predictions_computes_realized_return_and_error():
    rows = [
        {
            "predict_date": "2026-01-01",
            "target_date": "2026-01-02",
            "horizon": "1d",
            "ticker": "AAA",
            "node_id": "node1",
            "entry_price": "100.0",
            "predicted_return": "0.02",
        }
    ]
    results = evaluate_predictions(rows, {"AAA": 103.0})
    assert len(results) == 1
    r = results[0]
    assert abs(r["realized_return"] - 0.03) < 1e-9
    assert abs(r["error"] - 0.01) < 1e-9


def test_evaluate_predictions_skips_missing_price():
    rows = [
        {
            "predict_date": "2026-01-01",
            "target_date": "2026-01-02",
            "horizon": "1d",
            "ticker": "AAA",
            "node_id": "node1",
            "entry_price": "100.0",
            "predicted_return": "0.0",
        }
    ]
    results = evaluate_predictions(rows, {})
    assert results == []
