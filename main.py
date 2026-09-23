"""
main.py — Entry point utama pipeline House Price Prediction.

Jalankan dengan:
    python main.py

Pipeline ini menjalankan langkah-langkah berikut secara berurutan:
    1. Muat config & dataset mentah
    2. Preprocessing (cleaning + encoding)
    3. Simpan data terproses ke data/processed/
    4. Split train/validasi
    5. Latih SVR, Random Forest, Linear Regression
    6. Evaluasi & tampilkan ringkasan
    7. Simpan model terbaik ke models/
"""
import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

from src.data_loader   import load_config, load_raw_data
from src.preprocessing import run_preprocessing_pipeline
from src.features      import run_feature_engineering
from src.train         import get_models, split_data, train_all_models, save_model
from src.evaluate      import evaluate_all, plot_model_comparison, plot_pred_vs_actual, plot_feature_importance


def main() -> None:
    print("\n" + "=" * 50)
    print("  🏠 House Price Prediction — Pipeline Utama")
    print("=" * 50)

    # ── 1. Config ────────────────────────────────────────────────────────────
    cfg = load_config("config.yaml")

    # ── 2. Muat data ─────────────────────────────────────────────────────────
    print("\n📂 [1/5] Memuat dataset …")
    df = load_raw_data(cfg["data"]["raw_path"])

    # ── 3. Preprocessing ─────────────────────────────────────────────────────
    print("\n⚙️  [2/5] Preprocessing …")
    df_final, encoder, cat_cols = run_preprocessing_pipeline(
        df,
        target       = cfg["data"]["target_column"],
        id_col       = cfg["data"]["id_column"],
        keep_outliers= cfg["preprocessing"]["keep_outliers"],
    )

    # ── 4. Feature engineering (opsional) ────────────────────────────────────
    # df_final = run_feature_engineering(df_final)  # aktifkan jika diperlukan

    # ── 5. Simpan data terproses ──────────────────────────────────────────────
    print("\n💾 [3/5] Menyimpan data terproses …")
    os.makedirs("data/processed", exist_ok=True)
    df_final.to_csv(cfg["data"]["train_processed"], index=False)
    print(f"   → {cfg['data']['train_processed']}  ({df_final.shape[0]:,} baris, {df_final.shape[1]} kolom)")

    # ── 6. Split data ─────────────────────────────────────────────────────────
    print("\n✂️  [4/5] Pembagian dataset …")
    target  = cfg["data"]["target_column"]
    X       = df_final.drop(columns=[target])
    y       = df_final[target]
    X_train, X_valid, y_train, y_valid = split_data(
        X, y,
        test_size    = cfg["preprocessing"]["test_size"],
        random_state = cfg["preprocessing"]["random_state"],
    )

    # ── 7. Latih model ────────────────────────────────────────────────────────
    print("\n🤖 [5/5] Pelatihan model …")
    models = get_models(cfg["models"])
    trained_models, preds = train_all_models(models, X_train, y_train, X_valid)

    # ── 8. Evaluasi ───────────────────────────────────────────────────────────
    print("\n📊 Evaluasi model:")
    results_df = evaluate_all(preds, y_valid)
    print("\n" + results_df.to_string())

    # ── 9. Visualisasi ────────────────────────────────────────────────────────
    fig_dir = cfg["output"]["figures_dir"]
    os.makedirs(fig_dir, exist_ok=True)

    plot_model_comparison(results_df,  save_path=f"{fig_dir}model_comparison.png")
    plot_pred_vs_actual(y_valid, preds, save_path=f"{fig_dir}pred_vs_actual.png")
    plot_feature_importance(
        trained_models["Random Forest"],
        feature_names = X.columns.tolist(),
        top_n         = 15,
        save_path     = f"{fig_dir}feature_importance.png",
    )

    # ── 10. Simpan model ──────────────────────────────────────────────────────
    model_path = cfg["output"]["model_path"]
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    save_model(trained_models["Random Forest"], model_path)

    # ── Selesai ───────────────────────────────────────────────────────────────
    best = results_df["R²"].idxmax()
    print("\n" + "=" * 50)
    print(f"  ✅ Pipeline selesai!")
    print(f"  🏆 Model terbaik : {best}")
    print(f"     R²            : {results_df.loc[best, 'R²']:.4f}")
    print(f"     MAPE          : {results_df.loc[best, 'MAPE (%)']:.2f}%")
    print(f"  📁 Model disimpan: {model_path}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
