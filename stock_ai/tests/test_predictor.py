from pathlib import Path

from src.predictor import FEATURE_NAMES, HORIZONS, ModelWeights, feature_vector, load_weights, save_weights

N_FEATURES = len(FEATURE_NAMES)
ZERO_WEIGHTS = [0.0] * (N_FEATURES + 1)


def test_feature_vector_basic():
    x = feature_vector(
        undervaluation_score=0.2,
        revenue_growth=0.3,
        profit_margin=0.15,
        peg_ratio=1.0,
        momentum_5d=0.02,
        momentum_21d=-0.05,
    )
    assert len(x) == N_FEATURES
    assert x[0] == 0.8  # 1 - 0.2
    assert x[1] == 0.3
    assert x[2] == 0.15
    assert x[3] == 1.0  # 1/1.0
    assert x[4] == 0.02
    assert x[5] == -0.05


def test_feature_vector_handles_missing_values():
    x = feature_vector(
        undervaluation_score=None,
        revenue_growth=None,
        profit_margin=None,
        peg_ratio=None,
        momentum_5d=None,
        momentum_21d=None,
    )
    assert x == [0.5, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_feature_vector_clips_extreme_growth_and_margin():
    x = feature_vector(
        undervaluation_score=0.0,
        revenue_growth=50.0,
        profit_margin=-5.0,
        peg_ratio=0.0001,
        momentum_5d=2.0,
        momentum_21d=-2.0,
    )
    assert x[1] == 5.0  # clipped upper bound
    assert x[2] == -1.0  # clipped lower bound
    assert x[3] == 1 / 0.05  # peg clipped to 0.05 floor
    assert x[4] == 0.5  # momentum clipped upper bound
    assert x[5] == -0.5  # momentum clipped lower bound


def test_default_weights_predict_zero():
    model = ModelWeights()
    for h in HORIZONS:
        assert model.predict(h, [0.8, 0.3, 0.15, 1.2, 0.01, -0.02]) == 0.0


def test_predict_uses_weights_correctly():
    model = ModelWeights()
    model.weights["1d"] = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
    predicted = model.predict("1d", [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert abs(predicted - sum(model.weights["1d"])) < 1e-9


def test_save_and_load_weights_roundtrip(tmp_path: Path):
    model = ModelWeights()
    model.weights["1m"] = [0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    model.meta["1m"] = {"n_samples": 25, "r2": 0.1, "mae": 0.05}

    path = tmp_path / "weights.json"
    save_weights(model, path)
    loaded = load_weights(path)

    assert loaded.weights["1m"] == model.weights["1m"]
    assert loaded.meta["1m"]["n_samples"] == 25


def test_load_weights_missing_file_returns_defaults(tmp_path: Path):
    loaded = load_weights(tmp_path / "does_not_exist.json")
    for h in HORIZONS:
        assert loaded.weights[h] == ZERO_WEIGHTS


def test_load_weights_resets_stale_shorter_schema(tmp_path: Path):
    """피처가 추가돼 길이가 안 맞는 옛 가중치는 조용히 잘못 곱해지지 않고 리셋돼야 한다."""
    path = tmp_path / "weights.json"
    path.write_text('{"weights": {"1d": [0.1, 0.2, 0.3, 0.4, 0.5]}, "meta": {}}', encoding="utf-8")

    loaded = load_weights(path)

    assert loaded.weights["1d"] == ZERO_WEIGHTS
