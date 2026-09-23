"""
test_model.py — Unit test untuk modul src/train.py dan src/evaluate.py
Jalankan: pytest tests/test_model.py -v
"""
import sys
import os
import tempfile
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, "..")
from src.train    import get_models, split_data, train_model, save_model, load_model
from src.evaluate import compute_metrics, evaluate_all


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_data():
    """Generate data training sintetis yang deterministik."""
    np.random.seed(42)
    n = 120
    X = pd.DataFrame(
        np.random.randn(n, 6),
        columns=[f"feature_{i}" for i in range(6)],
    )
    # Target dengan sinyal nyata (bukan pure noise) agar R² > 0
    y = pd.Series(
        100_000 + 50_000 * X["feature_0"] + 30_000 * X["feature_1"] + np.random.randn(n) * 5_000,
        name="SalePrice",
    )
    return X, y


@pytest.fixture
def trained_lr(sample_data):
    """Linear Regression yang sudah dilatih, untuk dipakai di beberapa test."""
    X, y = sample_data
    X_tr, X_val, y_tr, y_val = split_data(X, y, test_size=0.2, random_state=42)
    model = train_model(get_models()["Linear Regression"], X_tr, y_tr)
    return model, X_val, y_val


# ── Test get_models ───────────────────────────────────────────────────────────

def test_get_models_returns_three():
    models = get_models()
    assert len(models) == 3

def test_get_models_keys():
    models = get_models()
    assert set(models.keys()) == {"SVR", "Random Forest", "Linear Regression"}

def test_get_models_with_config():
    cfg = {
        "random_forest":     {"n_estimators": 50, "random_state": 0, "n_jobs": 1},
        "svr":               {"kernel": "linear", "C": 0.5, "epsilon": 0.05},
        "linear_regression": {},
    }
    models = get_models(cfg)
    rf = models["Random Forest"]
    assert rf.n_estimators == 50
    assert rf.random_state == 0


# ── Test split_data ───────────────────────────────────────────────────────────

def test_split_sizes(sample_data):
    X, y = sample_data
    X_tr, X_val, y_tr, y_val = split_data(X, y, test_size=0.2)
    assert len(X_tr) == 96   # 120 * 0.8
    assert len(X_val) == 24  # 120 * 0.2

def test_split_no_overlap(sample_data):
    X, y = sample_data
    X_tr, X_val, _, _ = split_data(X, y, test_size=0.2, random_state=42)
    assert len(set(X_tr.index) & set(X_val.index)) == 0

def test_split_reproducible(sample_data):
    X, y = sample_data
    X_tr1, _, _, _ = split_data(X, y, test_size=0.2, random_state=7)
    X_tr2, _, _, _ = split_data(X, y, test_size=0.2, random_state=7)
    pd.testing.assert_frame_equal(X_tr1, X_tr2)


# ── Test train_model ──────────────────────────────────────────────────────────

def test_train_linear_regression_returns_model(sample_data):
    X, y = sample_data
    X_tr, X_val, y_tr, _ = split_data(X, y, test_size=0.2)
    model = train_model(get_models()["Linear Regression"], X_tr, y_tr)
    assert hasattr(model, "coef_")

def test_predict_shape(trained_lr):
    model, X_val, y_val = trained_lr
    preds = model.predict(X_val)
    assert preds.shape == y_val.shape

def test_linear_r2_positive(trained_lr):
    """Dengan sinyal nyata, R² harus lebih besar dari 0."""
    from sklearn.metrics import r2_score
    model, X_val, y_val = trained_lr
    preds = model.predict(X_val)
    assert r2_score(y_val, preds) > 0.0


# ── Test save_model / load_model ──────────────────────────────────────────────

def test_save_and_load_model(trained_lr, tmp_path):
    model, X_val, y_val = trained_lr
    path = str(tmp_path / "test_model.pkl")
    save_model(model, path)
    assert os.path.exists(path)
    loaded = load_model(path)
    # Prediksi harus identik
    np.testing.assert_array_almost_equal(
        model.predict(X_val),
        loaded.predict(X_val),
    )

def test_load_model_raises_if_missing():
    with pytest.raises(FileNotFoundError):
        load_model("models/nonexistent_model.pkl")


# ── Test compute_metrics ──────────────────────────────────────────────────────

def test_compute_metrics_keys(trained_lr):
    model, X_val, y_val = trained_lr
    preds = model.predict(X_val)
    result = compute_metrics(y_val, preds, "Test")
    assert set(result.keys()) == {"Model", "MAPE (%)", "MAE ($)", "R²"}

def test_compute_metrics_types(trained_lr):
    model, X_val, y_val = trained_lr
    preds = model.predict(X_val)
    result = compute_metrics(y_val, preds, "Test")
    assert isinstance(result["MAPE (%)"], float)
    assert isinstance(result["R²"], float)

def test_compute_metrics_r2_range(trained_lr):
    model, X_val, y_val = trained_lr
    preds = model.predict(X_val)
    result = compute_metrics(y_val, preds, "Test")
    assert result["R²"] <= 1.0

def test_compute_metrics_mape_positive(trained_lr):
    model, X_val, y_val = trained_lr
    preds = model.predict(X_val)
    result = compute_metrics(y_val, preds, "Test")
    assert result["MAPE (%)"] >= 0.0


# ── Test evaluate_all ─────────────────────────────────────────────────────────

def test_evaluate_all_returns_dataframe(sample_data):
    X, y = sample_data
    X_tr, X_val, y_tr, y_val = split_data(X, y, test_size=0.2, random_state=42)
    models = {"LR": get_models()["Linear Regression"]}
    preds  = {name: train_model(m, X_tr, y_tr).predict(X_val) for name, m in models.items()}
    df = evaluate_all(preds, y_val)
    assert isinstance(df, pd.DataFrame)
    assert "R²" in df.columns

def test_evaluate_all_row_count(sample_data):
    X, y = sample_data
    X_tr, X_val, y_tr, y_val = split_data(X, y, test_size=0.2, random_state=42)
    models  = {k: v for k, v in get_models().items() if k != "SVR"}  # skip SVR (lambat)
    preds   = {name: train_model(m, X_tr, y_tr).predict(X_val) for name, m in models.items()}
    df = evaluate_all(preds, y_val)
    assert len(df) == len(models)
