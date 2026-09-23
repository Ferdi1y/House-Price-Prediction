"""
features.py — Feature engineering untuk meningkatkan kualitas prediksi.
"""
import pandas as pd
import numpy as np


# ── Fitur turunan berbasis waktu ──────────────────────────────────────────────

def add_house_age(df: pd.DataFrame, year_col: str = "YearBuilt", ref_year: int = 2010) -> pd.DataFrame:
    """Tambahkan fitur usia rumah saat dijual.

    AgeOfHouse = ref_year - YearBuilt
    (menggunakan tahun referensi tetap agar reprodisibel)
    """
    df = df.copy()
    if year_col in df.columns:
        df["AgeOfHouse"] = ref_year - df[year_col]
        print(f"✅ Fitur 'AgeOfHouse' ditambahkan (ref_year={ref_year})")
    else:
        print(f"⚠️  Kolom '{year_col}' tidak ditemukan, fitur dilewati.")
    return df


def add_years_since_remodel(
    df: pd.DataFrame, remod_col: str = "YearRemodAdd", ref_year: int = 2010
) -> pd.DataFrame:
    """Tambahkan fitur berapa tahun sejak renovasi terakhir."""
    df = df.copy()
    if remod_col in df.columns:
        df["YearsSinceRemod"] = ref_year - df[remod_col]
        print(f"✅ Fitur 'YearsSinceRemod' ditambahkan")
    return df


# ── Fitur interaksi ───────────────────────────────────────────────────────────

def add_total_sf(df: pd.DataFrame) -> pd.DataFrame:
    """Total luas area (basement + lantai 1 + lantai 2) jika kolom tersedia."""
    df = df.copy()
    sf_cols = {"TotalBsmtSF", "1stFlrSF", "2ndFlrSF"}
    available = sf_cols.intersection(df.columns)
    if available:
        df["TotalSF"] = sum(df[c] for c in available)
        print(f"✅ Fitur 'TotalSF' dari kolom: {available}")
    return df


# ── Utilitas ──────────────────────────────────────────────────────────────────

def get_feature_columns(
    df: pd.DataFrame, target: str = "SalePrice", exclude: list[str] | None = None
) -> list[str]:
    """Kembalikan daftar kolom fitur (semua kolom kecuali target & exclude).

    Args:
        df      : DataFrame input.
        target  : Nama kolom target.
        exclude : Kolom tambahan yang ingin dikecualikan.

    Returns:
        List nama kolom fitur.
    """
    skip = {target}
    if exclude:
        skip.update(exclude)
    return [c for c in df.columns if c not in skip]


def run_feature_engineering(df: pd.DataFrame, ref_year: int = 2010) -> pd.DataFrame:
    """Jalankan seluruh pipeline feature engineering.

    Args:
        df       : DataFrame setelah preprocessing.
        ref_year : Tahun referensi untuk fitur berbasis waktu.

    Returns:
        DataFrame dengan fitur baru.
    """
    df = add_house_age(df, ref_year=ref_year)
    df = add_years_since_remodel(df, ref_year=ref_year)
    df = add_total_sf(df)
    print(f"📐 Shape setelah feature engineering: {df.shape}")
    return df
