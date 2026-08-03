"""섀도우 피처가 기존 6개 피처 대비 실제로 예측력을 개선하는지 통계적으로 검증.

"이 피처가 좋아 보인다"는 사람/LLM의 판단이 아니라, 기존 피처만 쓴 Elastic Net과
섀도우 피처를 하나 추가한 Elastic Net의 **교차검증 MSE**를 직접 비교해서 승격
(recommended_for_promotion) 또는 기각(rejected)을 결정한다. 표본이 아직 부족하면
아무것도 바꾸지 않고 shadow 상태를 유지한다.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np

from .backtest import read_csv_rows
from .calibration import _kfold_splits, fit_elastic_net, select_alpha_by_cv
from .feature_research import CandidateFeatures
from .predictor import HORIZONS

# augmented(섀도우 포함) CV MSE가 baseline보다 최소 이 비율만큼은 낮아야 "개선"으로 인정
IMPROVEMENT_THRESHOLD = 0.02

# daily.MIN_SAMPLES와 동일한 기준 — horizon별 재적합에 필요한 최소 표본과 맞춘다
MIN_SAMPLES = {"1d": 40, "1w": 30, "1m": 25, "1y": 15}


def _cv_mse(features: list[list[float]], targets: list[float], k: int = 5) -> float:
    X = np.array(features, dtype=float)
    y = np.array(targets, dtype=float)
    n = len(y)
    k = max(2, min(k, n // 2))
    folds = _kfold_splits(n, k)

    mses = []
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        if len(train_idx) < 2 or len(test_idx) == 0:
            continue
        alpha = select_alpha_by_cv(X[train_idx].tolist(), y[train_idx].tolist())
        weights, _, _ = fit_elastic_net(X[train_idx].tolist(), y[train_idx].tolist(), alpha=alpha)
        preds = weights[0] + X[test_idx] @ np.array(weights[1:])
        mses.append(float(np.mean((y[test_idx] - preds) ** 2)))

    return float(np.mean(mses)) if mses else float("inf")


def _joined_rows(
    predictions_path: Path,
    shadow_predictions_path: Path,
    evaluations_path: Path,
    feature_id: str,
    horizon: str,
) -> tuple[list[list[float]], list[float], list[float]]:
    """(predict_date, ticker, horizon) 키로 predictions/shadow/evaluations를 join."""
    preds = {(r["predict_date"], r["ticker"], r["horizon"]): r for r in read_csv_rows(predictions_path)}
    evals = {(r["predict_date"], r["ticker"], r["horizon"]): r for r in read_csv_rows(evaluations_path)}
    shadow_rows = [
        r
        for r in read_csv_rows(shadow_predictions_path)
        if r["feature_id"] == feature_id and r["horizon"] == horizon
    ]

    baseline_features, shadow_values, targets = [], [], []
    for row in shadow_rows:
        key = (row["predict_date"], row["ticker"], row["horizon"])
        pred = preds.get(key)
        ev = evals.get(key)
        if pred is None or ev is None:
            continue
        baseline_features.append([float(pred[f"x{i}"]) for i in range(1, 7)])
        shadow_values.append(float(row["value"]))
        targets.append(float(ev["realized_return"]))

    return baseline_features, shadow_values, targets


def evaluate_feature(
    feature_id: str,
    predictions_path: Path,
    shadow_predictions_path: Path,
    evaluations_path: Path,
) -> dict:
    """이 섀도우 피처의 horizon별 검증 결과와 종합 판단(verdict)을 반환."""
    per_horizon: dict[str, dict] = {}
    any_improved = False
    n_evaluated_horizons = 0

    for horizon in HORIZONS:
        baseline_features, shadow_values, targets = _joined_rows(
            predictions_path, shadow_predictions_path, evaluations_path, feature_id, horizon
        )
        n = len(targets)
        if n < MIN_SAMPLES[horizon]:
            per_horizon[horizon] = {"n_samples": n, "status": "insufficient_data"}
            continue

        n_evaluated_horizons += 1
        augmented_features = [bf + [sv] for bf, sv in zip(baseline_features, shadow_values)]

        baseline_mse = _cv_mse(baseline_features, targets)
        augmented_mse = _cv_mse(augmented_features, targets)
        improved = augmented_mse < baseline_mse * (1 - IMPROVEMENT_THRESHOLD)

        per_horizon[horizon] = {
            "n_samples": n,
            "status": "improved" if improved else "not_improved",
            "baseline_mse": baseline_mse,
            "augmented_mse": augmented_mse,
        }
        if improved:
            any_improved = True

    if any_improved:
        verdict = "recommended_for_promotion"
    elif n_evaluated_horizons >= 2:
        verdict = "rejected"
    else:
        verdict = "shadow"

    return {"feature_id": feature_id, "verdict": verdict, "per_horizon": per_horizon}


def evaluate_all_shadow_features(
    candidates: CandidateFeatures,
    predictions_path: Path,
    shadow_predictions_path: Path,
    evaluations_path: Path,
) -> list[dict]:
    """status=shadow인 후보를 전부 평가하고, 결론 난 건(승격/기각) candidates를 갱신한다."""
    results = []
    for candidate in candidates.candidates:
        if candidate.get("status") != "shadow":
            continue
        result = evaluate_feature(
            candidate["id"], predictions_path, shadow_predictions_path, evaluations_path
        )
        if result["verdict"] in ("recommended_for_promotion", "rejected"):
            candidate["status"] = result["verdict"]
            candidate["evaluation_date"] = date.today().isoformat()
        results.append(result)
    return results
