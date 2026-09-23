"""
train.py — Fungsi pelatihan, penyimpanan, dan pemuatan model.
"""
import os
import pickle
import joblib

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


# ── Definisi model ────────────────────────────────────────────────────────────

def get_models(config: dict | None = None) -> dict:
    """Kembalikan dictionary berisi semua model yang akan dilatih.

    Args:
        config: Blok 'models' dari config.yaml (opsional).
                Jika None, parameter default digunakan.

    Returns:
        Dict {nama_model: objek_sklearn_model}.
    """
    if config:
        rf_params  = {k: v for k, v in config.get("random_forest", {}).items() if v is not None}
        svr_params = config.get("svr", {})
        lr_params  = config.get("linear_regression", {})
    else:
        rf_params  = dict(n_estimators=200, max_depth=None,
                          min_samples_split=5, random_state=42, n_jobs=-1)
        svr_params = dict(kernel="rbf", C=1.0, epsilon=0.1)
        lr_params  = {}

    return {
        "SVR":               SVR(**svr_params),
        "Random Forest":     RandomForestRegressor(**rf_params),
        "Linear Regression": LinearRegression(**lr_params),
    }


# ── Pembagian data ────────────────────────────────────────────────────────────

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Bagi data menjadi training dan validasi.

    Returns:
        Tuple (X_train, X_valid, y_train, y_valid).
    """
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"📊 Training  : {len(X_train):,} sampel ({(1-test_size)*100:.0f}%)")
    print(f"   Validasi  : {len(X_valid):,} sampel ({test_size*100:.0f}%)")
    print(f"   Fitur     : {X.shape[1]}")
    return X_train, X_valid, y_train, y_valid


# ── Pelatihan ─────────────────────────────────────────────────────────────────

def train_model(model, X_train: pd.DataFrame, y_train: pd.Series):
    """Latih satu model.

    Args:
        model   : Objek estimator sklearn.
        X_train : Fitur training.
        y_train : Target training.

    Returns:
        Model yang sudah dilatih.
    """
    model.fit(X_train, y_train)
    return model


def train_all_models(
    models: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
):
    """Latih semua model dan kembalikan prediksi validasi.

    Returns:
        Tuple (trained_models, predictions) di mana predictions adalah
        dict {nama_model: array_prediksi}.
    """
    trained = {}
    preds   = {}
    for name, model in models.items():
        print(f"  🔄 Training: {name} …", end=" ")
        trained[name] = train_model(model, X_train, y_train)
        preds[name]   = trained[name].predict(X_valid)
        print("selesai ✅")
    return trained, preds


# ── Persistensi model ─────────────────────────────────────────────────────────

def save_model(model, path: str, use_joblib: bool = True) -> None:
    """Simpan model ke disk.

    Args:
        model      : Model sklearn yang sudah dilatih.
        path       : Path tujuan (mis. models/random_forest_v1.pkl).
        use_joblib : True = joblib (lebih efisien untuk array besar),
                     False = pickle standar.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if use_joblib:
        joblib.dump(model, path)
    else:
        with open(path, "wb") as f:
            pickle.dump(model, f)
    print(f"✅ Model disimpan : {path}")


def load_model(path: str, use_joblib: bool = True):
    """Muat model dari disk.

    Args:
        path       : Path file model.
        use_joblib : Harus sama dengan flag saat save_model dipanggil.

    Returns:
        Model sklearn yang sudah dimuat.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ File model tidak ditemukan: {path}")
    if use_joblib:
        model = joblib.load(path)
    else:
        with open(path, "rb") as f:
            model = pickle.load(f)
    print(f"✅ Model dimuat   : {path}")
    return model
