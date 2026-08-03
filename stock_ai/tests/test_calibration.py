from src.calibration import calibrate_horizon, fit_ridge
from src.predictor import ModelWeights


def test_fit_ridge_recovers_known_linear_relation():
    # y = 0.01 + 0.05*x1 - 0.02*x2  (노이즈 없는 합성 데이터)
    true_w = [0.01, 0.05, -0.02]
    features = [[x1, x2] for x1 in [0.0, 0.5, 1.0] for x2 in [0.0, 0.5, 1.0]]
    targets = [true_w[0] + true_w[1] * f[0] + true_w[2] * f[1] for f in features]

    w, r2, mae = fit_ridge(features, targets, alpha=1e-6)

    for est, true in zip(w, true_w):
        assert abs(est - true) < 1e-3
    assert r2 > 0.999
    assert mae < 1e-3


def test_calibrate_horizon_skips_when_samples_below_minimum():
    model = ModelWeights()
    original = list(model.weights["1d"])

    features = [[0.1, 0.2, 0.3, 0.4]] * 5
    targets = [0.01] * 5

    calibrated = calibrate_horizon(model, "1d", features, targets, min_samples=30)

    assert calibrated is False
    assert model.weights["1d"] == original
    assert "1d" not in model.meta


def test_calibrate_horizon_updates_weights_when_enough_samples():
    model = ModelWeights()
    features = [[x, 0.1, 0.1, 0.1] for x in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] * 5]
    targets = [0.02 + 0.1 * f[0] for f in features]

    calibrated = calibrate_horizon(model, "1m", features, targets, min_samples=10)

    assert calibrated is True
    assert model.weights["1m"] != [0.0, 0.0, 0.0, 0.0, 0.0]
    assert model.meta["1m"]["n_samples"] == len(targets)
