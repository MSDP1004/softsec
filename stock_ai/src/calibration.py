"""실현수익률에 대해 가중치를 재적합한다 — Elastic Net(coordinate descent).

Ridge(L2)만 쓰면 예측력 없는 피처의 계수가 0에 가까워지긴 해도 정확히 0이
되지는 않는다. Elastic Net은 L1 항 덕분에 실제로 도움이 안 되는 피처의
가중치를 정확히 0으로 만든다 — "예측력 좋은 피처는 남기고 나쁜 피처는
제외"를 사람이 매번 판단하는 대신, 회귀 알고리즘이 통계적 원칙에 따라
자동으로 하게 하는 방식이다. 정규화 강도(alpha)도 고정값을 쓰지 않고
k-fold 교차검증으로 매번 데이터에서 직접 고른다.

표본이 적을 때는 recalibrate하지 않는다 — 특히 1일/1주 horizon은 노이즈가
커서, 표본이 부족한 상태에서 매일 재적합하면 노이즈에 과적합될 위험이 크다.
horizon별 최소 표본 수(min_samples)를 넘겼을 때만 누적된 전체 데이터로
다시 적합한다.
"""

from __future__ import annotations

import numpy as np

from .predictor import FEATURE_NAMES, ModelWeights

DEFAULT_ALPHAS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
DEFAULT_L1_RATIO = 0.5


def _soft_threshold(rho: float, lam: float) -> float:
    if rho > lam:
        return rho - lam
    if rho < -lam:
        return rho + lam
    return 0.0


def fit_elastic_net(
    features: list[list[float]],
    targets: list[float],
    alpha: float,
    l1_ratio: float = DEFAULT_L1_RATIO,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> tuple[list[float], float, float]:
    """[intercept, w1..wk]와 R^2, MAE를 반환한다.

    피처마다 스케일이 다르므로(x1은 0~1, x2는 -1~5 등) 표준화한 뒤 좌표하강법으로
    적합하고, 계수는 다시 원래 스케일로 되돌린다 — 그래야 규제가 스케일이 큰
    피처에만 불공평하게 강하게/약하게 걸리지 않는다.
    """
    X = np.array(features, dtype=float)
    y = np.array(targets, dtype=float)
    n, p = X.shape

    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    Xs = (X - mu) / sigma

    y_mean = y.mean()
    yc = y - y_mean

    w = np.zeros(p)
    for _ in range(max_iter):
        w_prev = w.copy()
        for j in range(p):
            residual = yc - Xs @ w + Xs[:, j] * w[j]
            rho = (Xs[:, j] @ residual) / n
            denom = (Xs[:, j] @ Xs[:, j]) / n + alpha * (1 - l1_ratio)
            w[j] = _soft_threshold(rho, alpha * l1_ratio) / denom
        if np.max(np.abs(w - w_prev)) < tol:
            break

    w_orig = w / sigma
    intercept = y_mean - mu @ w_orig

    preds = intercept + X @ w_orig
    residuals = y - preds
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1e-9
    r2 = 1 - ss_res / ss_tot
    mae = float(np.mean(np.abs(residuals)))

    return [float(intercept)] + w_orig.tolist(), r2, mae


def _kfold_splits(n: int, k: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return np.array_split(idx, k)


def select_alpha_by_cv(
    features: list[list[float]],
    targets: list[float],
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
    l1_ratio: float = DEFAULT_L1_RATIO,
    k: int = 5,
) -> float:
    """k-fold 교차검증으로 최소 평균 제곱오차를 주는 alpha를 고른다."""
    X = np.array(features, dtype=float)
    y = np.array(targets, dtype=float)
    n = len(y)
    k = max(2, min(k, n // 2))
    folds = _kfold_splits(n, k)

    best_alpha = alphas[0]
    best_mse = float("inf")

    for alpha in alphas:
        fold_mses = []
        for i in range(k):
            test_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
            if len(train_idx) < 2 or len(test_idx) == 0:
                continue
            weights, _, _ = fit_elastic_net(
                X[train_idx].tolist(), y[train_idx].tolist(), alpha=alpha, l1_ratio=l1_ratio
            )
            preds = weights[0] + X[test_idx] @ np.array(weights[1:])
            fold_mses.append(float(np.mean((y[test_idx] - preds) ** 2)))
        if fold_mses:
            avg_mse = float(np.mean(fold_mses))
            if avg_mse < best_mse:
                best_mse = avg_mse
                best_alpha = alpha

    return best_alpha


def calibrate_horizon(
    model: ModelWeights,
    horizon: str,
    features: list[list[float]],
    targets: list[float],
    min_samples: int,
    l1_ratio: float = DEFAULT_L1_RATIO,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
) -> bool:
    """표본이 충분하면 재적합 후 True, 아니면 기존 가중치를 유지하고 False."""
    if len(targets) < min_samples:
        return False

    alpha = select_alpha_by_cv(features, targets, alphas=alphas, l1_ratio=l1_ratio)
    weights, r2, mae = fit_elastic_net(features, targets, alpha=alpha, l1_ratio=l1_ratio)

    active = [name for name, wi in zip(FEATURE_NAMES, weights[1:]) if abs(wi) > 1e-9]

    model.weights[horizon] = weights
    model.meta[horizon] = {
        "n_samples": len(targets),
        "r2": round(r2, 4),
        "mae": round(mae, 5),
        "alpha": alpha,
        "l1_ratio": l1_ratio,
        "active_features": active,
    }
    return True
