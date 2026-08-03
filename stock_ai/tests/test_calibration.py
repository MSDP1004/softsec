import numpy as np

from src.calibration import calibrate_horizon, fit_elastic_net, select_alpha_by_cv
from src.predictor import ModelWeights


def _synthetic_data(n=200, seed=42):
    """y는 x1에만 진짜로 의존하고, x2는 수익률과 무관한 잡음 피처."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-1, 1, n)
    x2 = rng.normal(0, 1, n)
    noise = rng.normal(0, 0.01, n)
    y = 0.02 + 0.1 * x1 + noise
    features = np.column_stack([x1, x2]).tolist()
    targets = y.tolist()
    return features, targets


def test_fit_elastic_net_recovers_relevant_coefficient_with_low_alpha():
    features, targets = _synthetic_data()
    weights, r2, _mae = fit_elastic_net(features, targets, alpha=0.001, l1_ratio=0.5)
    intercept, w1, _w2 = weights
    assert abs(intercept - 0.02) < 0.02
    assert abs(w1 - 0.1) < 0.03
    assert r2 > 0.9


def test_fit_elastic_net_shrinks_irrelevant_feature_as_alpha_grows():
    features, targets = _synthetic_data()
    weights_low, _, _ = fit_elastic_net(features, targets, alpha=0.001, l1_ratio=0.5)
    weights_high, _, _ = fit_elastic_net(features, targets, alpha=0.5, l1_ratio=0.5)

    # 잡음 피처(x2, index 2)의 계수는 alpha가 커질수록 0에 더 가까워져야 한다
    assert abs(weights_high[2]) <= abs(weights_low[2])
    assert abs(weights_high[2]) < 0.01


def test_select_alpha_by_cv_picks_from_given_grid():
    features, targets = _synthetic_data(n=100)
    grid = (0.001, 0.01, 0.1, 1.0)
    alpha = select_alpha_by_cv(features, targets, alphas=grid, k=5)
    assert alpha in grid


def test_calibrate_horizon_skips_when_samples_below_minimum():
    model = ModelWeights()
    original = list(model.weights["1d"])

    features = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]] * 5
    targets = [0.01] * 5

    calibrated = calibrate_horizon(model, "1d", features, targets, min_samples=30)

    assert calibrated is False
    assert model.weights["1d"] == original
    assert "1d" not in model.meta


def test_calibrate_horizon_updates_weights_and_reports_active_features():
    model = ModelWeights()
    rng = np.random.default_rng(0)
    n = 60
    x1 = rng.uniform(-1, 1, n)  # "undervaluation"에 해당, 진짜 관련 있음
    others = rng.normal(0, 1, (n, 5))  # 나머지 5개 피처는 잡음
    noise = rng.normal(0, 0.01, n)
    y = 0.01 + 0.08 * x1 + noise

    features = np.column_stack([x1, others]).tolist()
    targets = y.tolist()

    calibrated = calibrate_horizon(model, "1m", features, targets, min_samples=10)

    assert calibrated is True
    assert model.weights["1m"] != [0.0] * 7
    assert model.meta["1m"]["n_samples"] == n
    assert "undervaluation" in model.meta["1m"]["active_features"]
