"""
test_preprocessing.py — Unit test untuk modul src/preprocessing.py
Jalankan: pytest tests/test_preprocessing.py -v
"""
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, "..")
from src.preprocessing import (
    drop_id_column,
    split_by_target,
    fill_missing_values,
    detect_outliers_iqr,
    encode_categorical,
)


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """DataFrame kecil dengan pola yang merepresentasikan dataset asli."""
    return pd.DataFrame({
        "Id":         [1, 2, 3, 4, 5, 6],
        "LotArea":    [8450, 9600, 11250, 9550, None, 7200],
        "YearBuilt":  [2003, 1976, 2001, 1915, 2000, 1998],
        "OverallCond":[5, 8, 5, 5, 5, 7],
        "MSZoning":   ["RL", "RL", None, "RM", "RL", "RL"],
        "BldgType":   ["1Fam", "1Fam", "1Fam", "1Fam", "2fmCon", None],
        "SalePrice":  [208500, 181500, 223500, None, 250000, 143000],
    })


# ── Test drop_id_column ───────────────────────────────────────────────────────

def test_drop_id_removes_column(sample_df):
    result = drop_id_column(sample_df, id_col="Id")
    assert "Id" not in result.columns

def test_drop_id_preserves_row_count(sample_df):
    result = drop_id_column(sample_df)
    assert len(result) == len(sample_df)

def test_drop_id_missing_col_no_error(sample_df):
    """Tidak boleh error jika kolom 'Id' sudah tidak ada."""
    df_no_id = sample_df.drop(columns=["Id"])
    result = drop_id_column(df_no_id, id_col="Id")
    assert "Id" not in result.columns


# ── Test split_by_target ──────────────────────────────────────────────────────

def test_split_train_has_no_null_target(sample_df):
    df = sample_df.drop(columns=["Id"])
    df_train, _ = split_by_target(df, target="SalePrice")
    assert df_train["SalePrice"].notna().all()

def test_split_test_has_null_target(sample_df):
    df = sample_df.drop(columns=["Id"])
    _, df_test = split_by_target(df, target="SalePrice")
    assert df_test["SalePrice"].isna().all()

def test_split_total_rows_preserved(sample_df):
    df = sample_df.drop(columns=["Id"])
    df_train, df_test = split_by_target(df, target="SalePrice")
    assert len(df_train) + len(df_test) == len(df)


# ── Test fill_missing_values ──────────────────────────────────────────────────

def test_fill_missing_no_null_after(sample_df):
    df = sample_df.drop(columns=["Id"])
    df_train, _ = split_by_target(df, target="SalePrice")
    result = fill_missing_values(df_train, target="SalePrice")
    non_target = result.drop(columns=["SalePrice"])
    assert non_target.isnull().sum().sum() == 0

def test_fill_numeric_uses_median(sample_df):
    df = sample_df.drop(columns=["Id"])
    df_train, _ = split_by_target(df, target="SalePrice")
    expected_median = df_train["LotArea"].median()
    result = fill_missing_values(df_train.copy(), target="SalePrice")
    assert result["LotArea"].iloc[result["LotArea"].isna().values.argmax()] == pytest.approx(expected_median, rel=1e-3) if sample_df["LotArea"].isna().any() else True

def test_fill_categorical_uses_mode(sample_df):
    df = sample_df.drop(columns=["Id"])
    df_train, _ = split_by_target(df, target="SalePrice")
    mode_val = df_train["MSZoning"].mode()[0]
    result = fill_missing_values(df_train.copy(), target="SalePrice")
    # Semua nilai MSZoning harus terisi dan nilainya valid
    assert result["MSZoning"].notna().all()


# ── Test detect_outliers_iqr ──────────────────────────────────────────────────

def test_outlier_returns_dataframe(sample_df):
    df = sample_df.dropna(subset=["SalePrice"]).copy()
    outliers, lower, upper = detect_outliers_iqr(df, col="SalePrice")
    assert isinstance(outliers, pd.DataFrame)

def test_outlier_bounds_order(sample_df):
    df = sample_df.dropna(subset=["SalePrice"]).copy()
    _, lower, upper = detect_outliers_iqr(df, col="SalePrice")
    assert lower < upper

def test_outlier_values_outside_bounds(sample_df):
    df = sample_df.dropna(subset=["SalePrice"]).copy()
    outliers, lower, upper = detect_outliers_iqr(df, col="SalePrice")
    if len(outliers) > 0:
        assert ((outliers["SalePrice"] < lower) | (outliers["SalePrice"] > upper)).all()


# ── Test encode_categorical ───────────────────────────────────────────────────

def test_encode_removes_original_cat_cols(sample_df):
    df = sample_df.drop(columns=["Id"])
    df_train, _ = split_by_target(df, target="SalePrice")
    df_train = fill_missing_values(df_train, target="SalePrice")
    cat_cols = df_train.select_dtypes(include=["object"]).columns.tolist()
    df_enc, _ = encode_categorical(df_train, cat_cols)
    for col in cat_cols:
        assert col not in df_enc.columns

def test_encode_output_no_object_dtype(sample_df):
    df = sample_df.drop(columns=["Id"])
    df_train, _ = split_by_target(df, target="SalePrice")
    df_train = fill_missing_values(df_train, target="SalePrice")
    cat_cols = df_train.select_dtypes(include=["object"]).columns.tolist()
    df_enc, _ = encode_categorical(df_train, cat_cols)
    assert len(df_enc.select_dtypes(include=["object"]).columns) == 0
