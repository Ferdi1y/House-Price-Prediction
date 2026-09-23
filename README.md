# 🏠 House Price Prediction

Proyek machine learning untuk memprediksi harga jual rumah menggunakan dataset 13 fitur properti. Tiga algoritma dibandingkan: **SVR**, **Random Forest**, dan **Linear Regression**.

---

## 📁 Struktur Proyek

```
house_price_prediction/
│
├── data/
│   ├── raw/                    # Data mentah — JANGAN diubah
│   │   └── HousePricePrediction.xlsx
│   ├── processed/              # Data setelah preprocessing
│   │   ├── train_cleaned.csv
│   │   └── test_cleaned.csv
│   └── external/               # Data dari sumber luar (opsional)
│
├── notebooks/
│   ├── 01_EDA.ipynb            # Eksplorasi data
│   ├── 02_Preprocessing.ipynb  # Pembersihan data
│   ├── 03_Modeling.ipynb       # Pelatihan model
│   └── 04_Evaluation.ipynb     # Evaluasi & visualisasi
│
├── src/                        # Kode modular
│   ├── __init__.py
│   ├── data_loader.py          # Memuat data
│   ├── preprocessing.py        # Preprocessing & encoding
│   ├── features.py             # Feature engineering
│   ├── train.py                # Pelatihan model
│   └── evaluate.py             # Evaluasi & visualisasi
│
├── models/
│   └── random_forest_v1.pkl    # Model tersimpan
│
├── outputs/
│   ├── figures/                # Grafik & visualisasi
│   └── reports/                # Laporan hasil
│
├── tests/
│   ├── test_preprocessing.py
│   └── test_model.py
│
├── config.yaml                 # Konfigurasi hyperparameter
├── requirements.txt            # Daftar library
├── main.py                     # Entry point pipeline
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone & setup environment

```bash
git clone <repo-url>
cd house_price_prediction
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Siapkan data

Letakkan file dataset di:
```
data/raw/HousePricePrediction.xlsx
```

### 3. Jalankan pipeline penuh

```bash
python main.py
```

### 4. Atau eksplorasi via notebook

```bash
jupyter notebook
# Buka notebooks/ dan jalankan secara berurutan: 01 → 02 → 03 → 04
```

---

## 📊 Dataset

| Kolom | Deskripsi |
|---|---|
| `MSSubClass` | Tipe bangunan dalam transaksi |
| `MSZoning` | Klasifikasi zona properti |
| `LotArea` | Luas lahan (sqft) |
| `LotConfig` | Konfigurasi lahan |
| `BldgType` | Tipe bangunan |
| `OverallCond` | Rating kondisi keseluruhan (1–10) |
| `YearBuilt` | Tahun konstruksi |
| `YearRemodAdd` | Tahun renovasi terakhir |
| `Exterior1st` | Material eksterior utama |
| `BsmtFinSF2` | Luas basement tipe 2 (sqft) |
| `TotalBsmtSF` | Total luas basement (sqft) |
| `SalePrice` | ⭐ **Target**: harga jual (USD) |

---

## 🤖 Model

| Model | Deskripsi |
|---|---|
| `SVR` | Support Vector Regressor (kernel RBF) |
| `Random Forest` | Ensemble 200 pohon keputusan |
| `Linear Regression` | Regresi linier (baseline) |

### Metrik Evaluasi

- **MAPE** — Mean Absolute Percentage Error (lebih kecil = lebih baik)
- **MAE** — Mean Absolute Error dalam USD
- **R²** — Koefisien determinasi (lebih besar = lebih baik, maks 1.0)

---

## ⚙️ Konfigurasi

Edit `config.yaml` untuk mengubah hyperparameter tanpa menyentuh kode:

```yaml
models:
  random_forest:
    n_estimators: 200
    max_depth: null
    min_samples_split: 5
```

---

## 🧪 Testing

```bash
pytest tests/ -v
```

---

## 📈 Hasil & Temuan

- **Missing Values**: `SalePrice` (data test), `MSZoning`, `Exterior1st`, `BsmtFinSF2`, `TotalBsmtSF` — ditangani dengan median/modus.
- **Korelasi tertinggi** terhadap `SalePrice`: `TotalBsmtSF` dan `YearBuilt`.
- **Model terbaik**: Random Forest — konsisten menghasilkan MAPE terendah dan R² tertinggi.

### Rekomendasi Pengembangan

| Aspek | Rekomendasi |
|---|---|
| Feature Engineering | Tambahkan `AgeOfHouse = YearSold - YearBuilt` |
| Hyperparameter Tuning | Gunakan `GridSearchCV` atau `Optuna` |
| Model Lanjutan | Coba `XGBoost` / `LightGBM` |
| Validasi | Terapkan K-Fold Cross Validation |

---

## 👤 Author

**Ahmad Ferdy Saputra** — D3 Teknik Informatika, Politeknik Negeri Samarinda (POLNES)
