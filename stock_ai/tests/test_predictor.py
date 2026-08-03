from pathlib import Path

from src.predictor import HORIZONS, ModelWeights, feature_vector, load_weights, save_weights


def test_feature_vector_basic():
    x = feature_vector(undervaluation_score=0.2, revenue_growth=0.3, profit_margin=0.15, peg_ratio=1.0)
    assert x[0] == 0.8  # 1 - 0.2
    assert x[1] == 0.3
    assert x[2] == 0.15
    assert x[3] == 1.0  # 1/1.0


def test_feature_vector_handles_missing_values():
    x = feature_vector(undervaluation_score=None, revenue_growth=None, profit_margin=None, peg_ratio=None)
    assert x == [0.5, 0.0, 0.0, 0.0]


def test_feature_vector_clips_extreme_growth_and_margin():
    x = feature_vector(undervaluation_score=0.0, revenue_growth=50.0, profit_margin=-5.0, peg_ratio=0.0001)
    assert x[1] == 5.0  # clipped upper bound
    assert x[2] == -1.0  # clipped lower bound
    assert x[3] == 1 / 0.05  # peg clipped to 0.05 floor


def test_default_weights_predict_zero():
    model = ModelWeights()
    for h in HORIZONS:
        assert model.predict(h, [0.8, 0.3, 0.15, 1.2]) == 0.0


def test_predict_uses_weights_correctly():
    model = ModelWeights()
    model.weights["1d"] = [0.01, 0.02, 0.03, 0.04, 0.05]
    predicted = model.predict("1d", [1.0, 1.0, 1.0, 1.0])
    assert predicted == 0.01 + 0.02 + 0.03 + 0.04 + 0.05


def test_save_and_load_weights_roundtrip(tmp_path: Path):
    model = ModelWeights()
    model.weights["1m"] = [0.001, 0.01, 0.02, 0.03, 0.04]
    model.meta["1m"] = {"n_samples": 25, "r2": 0.1, "mae": 0.05}

    path = tmp_path / "weights.json"
    save_weights(model, path)
    loaded = load_weights(path)

    assert loaded.weights["1m"] == model.weights["1m"]
    assert loaded.meta["1m"]["n_samples"] == 25


def test_load_weights_missing_file_returns_defaults(tmp_path: Path):
    loaded = load_weights(tmp_path / "does_not_exist.json")
    for h in HORIZONS:
        assert loaded.weights[h] == [0.0, 0.0, 0.0, 0.0, 0.0]
