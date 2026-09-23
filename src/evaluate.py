"""
evaluate.py — Fungsi metrik evaluasi dan visualisasi hasil model.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_absolute_error,
    r2_score,
)


# ── Metrik ────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, model_name: str) -> dict:
    """Hitung MAPE, MAE, dan R² lalu cetak hasilnya.

    Returns:
        Dict {Model, MAPE (%), MAE ($), R²}.
    """
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    print(f"{'─'*42}")
    print(f"  Model : {model_name}")
    print(f"  MAPE  : {mape:.2f}%")
    print(f"  MAE   : ${mae:,.0f}")
    print(f"  R²    : {r2:.4f}")
    print(f"{'─'*42}")

    return {
        "Model":    model_name,
        "MAPE (%)": round(mape, 2),
        "MAE ($)":  round(mae, 0),
        "R²":       round(r2, 4),
    }


def evaluate_all(preds_dict: dict, y_valid) -> pd.DataFrame:
    """Evaluasi semua model dan kembalikan DataFrame ringkasan.

    Args:
        preds_dict : Dict {nama_model: array_prediksi}.
        y_valid    : Series nilai aktual.

    Returns:
        DataFrame dengan baris = model, kolom = metrik.
    """
    results = []
    for name, y_pred in preds_dict.items():
        results.append(compute_metrics(y_valid, y_pred, name))
    df = pd.DataFrame(results).set_index("Model")
    best = df["R²"].idxmax()
    print(f"\n🏆 Model terbaik (R²): {best} — {df.loc[best, 'R²']:.4f}")
    return df


# ── Visualisasi ───────────────────────────────────────────────────────────────

def plot_model_comparison(
    results_df: pd.DataFrame,
    save_path: str | None = None,
) -> None:
    """Bar chart perbandingan MAPE, MAE, dan R² antar model."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = ["MAPE (%)", "MAE ($)", "R²"]
    colors  = ["#4C72B0", "#DD8452", "#55A868"]

    for ax, metric, color in zip(axes, metrics, colors):
        bars = ax.bar(results_df.index, results_df[metric],
                      color=color, width=0.5, edgecolor="white")
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_ylabel(metric)
        ax.set_ylim(0, results_df[metric].max() * 1.25)
        for bar, val in zip(bars, results_df[metric]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + results_df[metric].max() * 0.02,
                f"{val:.2f}", ha="center", fontsize=10, fontweight="bold",
            )
        ax.tick_params(axis="x", rotation=15)

    plt.suptitle("Perbandingan Performa Model", fontsize=15, fontweight="bold")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=120)
        print(f"📁 Gambar disimpan: {save_path}")
    plt.show()


def plot_pred_vs_actual(
    y_valid,
    preds_dict: dict,
    save_path: str | None = None,
) -> None:
    """Scatter plot prediksi vs aktual untuk setiap model."""
    n = len(preds_dict)
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, y_pred), color in zip(axes, preds_dict.items(), colors):
        ax.scatter(y_valid, y_pred, alpha=0.4, s=15, color=color, edgecolors="none")
        lims = [
            min(y_valid.min(), y_pred.min()),
            max(y_valid.max(), y_pred.max()),
        ]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Ideal (y=x)")
        ax.set_xlabel("Harga Aktual (USD)")
        ax.set_ylabel("Harga Prediksi (USD)")
        ax.set_title(
            f"{name}\nR² = {r2_score(y_valid, y_pred):.4f}",
            fontsize=11, fontweight="bold",
        )
        ax.legend(fontsize=9)

    plt.suptitle("Prediksi vs Aktual – Semua Model", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=120)
        print(f"📁 Gambar disimpan: {save_path}")
    plt.show()


def plot_feature_importance(
    model,
    feature_names: list[str],
    top_n: int = 15,
    save_path: str | None = None,
) -> None:
    """Horizontal bar chart top-N feature importance dari model tree-based."""
    if not hasattr(model, "feature_importances_"):
        print("⚠️  Model tidak memiliki attribute feature_importances_.")
        return

    importances = pd.Series(model.feature_importances_, index=feature_names)
    top = importances.nlargest(top_n).sort_values()

    plt.figure(figsize=(10, 6))
    top.plot(kind="barh", color="#4C72B0", edgecolor="white")
    plt.title(f"Top {top_n} Feature Importance (Random Forest)",
              fontsize=13, fontweight="bold")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=120)
        print(f"📁 Gambar disimpan: {save_path}")
    plt.show()
