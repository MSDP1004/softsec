"""실현수익률에 대해 가중치를 재적합(ridge regression)한다.

표본이 적을 때는 recalibrate하지 않는다 — 특히 1일/1주 horizon은 노이즈가
커서, 표본이 부족한 상태에서 매일 재적합하면 노이즈에 과적합될 위험이 크다.
horizon별 최소 표본 수(min_samples)를 넘겼을 때만 누적된 전체 데이터로
다시 적합한다.
"""

from __future__ import annotations

import numpy as np

from .predictor import ModelWeights


def fit_ridge(
    features: list[list[float]], targets: list[float], alpha: float = 1.0
) -> tuple[list[float], float, float]:
    """[intercept, w1..wk]와 R^2, MAE를 반환한다."""
    X = np.array(features, dtype=float)
    y = np.array(targets, dtype=float)
    X_aug = np.hstack([np.ones((X.shape[0], 1)), X])

    reg = alpha * np.eye(X_aug.shape[1])
    reg[0, 0] = 0.0  # 절편은 규제하지 않음

    w = np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ y)
    preds = X_aug @ w
    residuals = y - preds

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1e-9
    r2 = 1 - ss_res / ss_tot
    mae = float(np.mean(np.abs(residuals)))

    return w.tolist(), r2, mae


def calibrate_horizon(
    model: ModelWeights,
    horizon: str,
    features: list[list[float]],
    targets: list[float],
    min_samples: int,
    alpha: float = 1.0,
) -> bool:
    """표본이 충분하면 재적합 후 True, 아니면 기존 가중치를 유지하고 False."""
    if len(targets) < min_samples:
        return False

    weights, r2, mae = fit_ridge(features, targets, alpha=alpha)
    model.weights[horizon] = weights
    model.meta[horizon] = {"n_samples": len(targets), "r2": round(r2, 4), "mae": round(mae, 5)}
    return True
