"""
data_loader.py — Fungsi memuat data mentah dan data terproses.
"""
import os
import pandas as pd
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """Muat konfigurasi dari file YAML.

    Args:
        config_path: Path ke file config.yaml.

    Returns:
        Dictionary konfigurasi.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    print(f"✅ Config dimuat dari: {config_path}")
    return cfg


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Muat dataset mentah dari file Excel.

    Args:
        filepath: Path ke file .xlsx.

    Returns:
        DataFrame dataset mentah.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"❌ File tidak ditemukan: {filepath}\n"
            "   Letakkan HousePricePrediction.xlsx di data/raw/"
        )

    df = pd.read_excel(filepath)
    print(f"✅ Dataset dimuat   : {df.shape[0]:,} baris × {df.shape[1]} kolom")
    print(f"💾 Ukuran memori   : {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    return df


def load_processed_train(train_path: str) -> pd.DataFrame:
    """Muat data training yang sudah diproses dari CSV.

    Args:
        train_path: Path ke train_cleaned.csv.

    Returns:
        DataFrame data training bersih.
    """
    df = pd.read_csv(train_path)
    print(f"✅ Training data dimuat: {df.shape[0]:,} baris × {df.shape[1]} kolom")
    return df


def load_processed_data(train_path: str, test_path: str | None = None):
    """Muat data training (dan opsional test) yang sudah diproses.

    Args:
        train_path: Path ke train_cleaned.csv.
        test_path : Path ke test_cleaned.csv (opsional).

    Returns:
        Tuple (df_train,) atau (df_train, df_test).
    """
    df_train = load_processed_train(train_path)

    if test_path and os.path.exists(test_path):
        df_test = pd.read_csv(test_path)
        print(f"✅ Test data dimuat    : {df_test.shape[0]:,} baris × {df_test.shape[1]} kolom")
        return df_train, df_test

    return (df_train,)
