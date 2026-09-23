"""
preprocessing.py — Fungsi pembersihan dan transformasi data.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


# ── 1. Utilitas umum ─────────────────────────────────────────────────────────

def drop_id_column(df: pd.DataFrame, id_col: str = "Id") -> pd.DataFrame:
    """Hapus kolom identifier yang tidak informatif untuk prediksi."""
    return df.drop(columns=[id_col], errors="ignore")


def split_by_target(df: pd.DataFrame, target: str = "SalePrice"):
    """Pisahkan DataFrame menjadi data train (punya target) dan test (tanpa target).

    Returns:
        Tuple (df_train, df_test).
    """
    df_train = df[df[target].notna()].copy()
    df_test  = df[df[target].isna()].copy()
    print(f"📊 Train : {len(df_train):,} baris | Test : {len(df_test):,} baris")
    return df_train, df_test


# ── 2. Imputasi missing values ────────────────────────────────────────────────

def fill_missing_numeric(df: pd.DataFrame, target: str = "SalePrice") -> pd.DataFrame:
    """Isi missing value numerik dengan median kolom."""
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    num_cols = [c for c in num_cols if c != target]
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def fill_missing_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Isi missing value kategorikal dengan modus kolom."""
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    return df


def fill_missing_values(df: pd.DataFrame, target: str = "SalePrice") -> pd.DataFrame:
    """Jalankan imputasi numerik dan kategorikal sekaligus."""
    df = fill_missing_numeric(df, target)
    df = fill_missing_categorical(df)
    remaining = df.isnull().sum().sum()
    print(f"✅ Missing values setelah imputasi: {remaining} (target: 0)")
    return df


# ── 3. Deteksi outlier ────────────────────────────────────────────────────────

def detect_outliers_iqr(df: pd.DataFrame, col: str = "SalePrice"):
    """Deteksi outlier menggunakan metode IQR.

    Returns:
        Tuple (df_outliers, lower_bound, upper_bound).
    """
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df_outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"📌 Batas IQR   : ${lower:,.0f} – ${upper:,.0f}")
    print(f"⚠️  Outlier {col}: {len(df_outliers)} baris ({len(df_outliers)/len(df)*100:.1f}%)")
    return df_outliers, lower, upper


def remove_outliers_iqr(df: pd.DataFrame, col: str = "SalePrice") -> pd.DataFrame:
    """Hapus baris outlier berdasarkan IQR (gunakan hanya jika diperlukan)."""
    _, lower, upper = detect_outliers_iqr(df, col)
    before = len(df)
    df = df[(df[col] >= lower) & (df[col] <= upper)].copy()
    print(f"🗑️  Baris dihapus : {before - len(df)} | Tersisa : {len(df):,}")
    return df


# ── 4. Encoding ───────────────────────────────────────────────────────────────

def encode_categorical(
    df: pd.DataFrame,
    cat_cols: list[str],
    encoder: OneHotEncoder | None = None,
    fit: bool = True,
):
    """Terapkan OneHotEncoding pada kolom kategorikal.

    Args:
        df       : DataFrame input.
        cat_cols : Daftar nama kolom kategorikal.
        encoder  : Objek encoder yang sudah di-fit (None = buat baru).
        fit      : True untuk fit+transform, False hanya transform.

    Returns:
        Tuple (df_encoded, encoder).
    """
    if encoder is None:
        encoder = OneHotEncoder(
            sparse_output=False, handle_unknown="ignore", drop="first"
        )

    if fit:
        oh_array = encoder.fit_transform(df[cat_cols])
    else:
        oh_array = encoder.transform(df[cat_cols])

    oh_df = pd.DataFrame(
        oh_array,
        columns=encoder.get_feature_names_out(cat_cols),
        index=df.index,
    )
    df_encoded = pd.concat([df.drop(columns=cat_cols), oh_df], axis=1)
    print(f"✅ Shape setelah OHE: {df_encoded.shape}")
    return df_encoded, encoder


# ── 5. Pipeline ringkas ───────────────────────────────────────────────────────

def run_preprocessing_pipeline(
    df: pd.DataFrame,
    target: str = "SalePrice",
    id_col: str = "Id",
    keep_outliers: bool = True,
):
    """Jalankan seluruh pipeline preprocessing dari awal.

    Returns:
        Tuple (df_final, encoder, cat_cols).
    """
    df = drop_id_column(df, id_col)
    df_train, _ = split_by_target(df, target)
    df_train = fill_missing_values(df_train, target)

    if not keep_outliers:
        df_train = remove_outliers_iqr(df_train, target)

    cat_cols = df_train.select_dtypes(include=["object"]).columns.tolist()
    df_final, encoder = encode_categorical(df_train, cat_cols)

    return df_final, encoder, cat_cols
