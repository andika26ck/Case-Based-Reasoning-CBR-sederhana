# Case-Based Reasoning (CBR) — Perdata Perbuatan Melawan Hukum

Sistem CBR sederhana berbasis Python untuk analisis putusan pengadilan.
**Domain:** Perdata - Perbuatan Melawan Hukum (PMH)
**Sumber data:** Direktori Putusan Mahkamah Agung RI — PN Bandung
**Mata Kuliah:** Penalaran Komputer — Semester Genap 2025/2026

---

## Anggota Tim

| Nama | NIM |
|------|-----|
| [Andika Candra Kurniawan] | [202310370311173] |
| [Ahmad Habibi] | [202310370311161] |

---

## Struktur Repository

```
.
├── 01_scraping.py          # Tahap 1: Scraping & preprocessing putusan
├── 02_representation.py    # Tahap 2: Ekstraksi metadata & feature engineering
├── 03_retrieval.py         # Tahap 3: Case Retrieval (TF-IDF + SVM/NB + BERT)
├── 04_predict.py           # Tahap 4: Case Solution Reuse
├── 05_evaluation.py        # Tahap 5: Evaluasi model (Accuracy, Precision, Recall, F1)
├── requirements.txt        # Daftar dependensi Python
├── data/
│   ├── raw/                # Teks bersih hasil scraping (case_*.txt)
│   ├── raw_extracted/      # Teks mentah sebelum cleaning
│   ├── tokens/             # Hasil tokenisasi (case_*.json)
│   ├── pdf/                # Arsip PDF putusan asli
│   ├── processed/
│   │   ├── metadata_raw.csv   # Metadata dasar dari scraping
│   │   ├── cases.csv          # Dataset terstruktur untuk model
│   │   └── cases.json         # Format JSON alternatif
│   └── eval/
│       ├── queries.json           # Query uji + ground-truth
│       ├── retrieval_metrics.csv  # Metrik evaluasi retrieval
│       ├── prediction_metrics.csv # Metrik prediksi solusi
│       ├── error_analysis.csv     # Kasus yang salah diprediksi
│       ├── confusion_matrix.png   # Visualisasi confusion matrix
│       └── model_comparison.png   # Bar chart perbandingan model
├── models/
│   ├── tfidf_vectorizer.pkl   # TF-IDF vectorizer
│   ├── svm_model.pkl          # Model SVM
│   ├── nb_model.pkl           # Model Naive Bayes
│   ├── nb_scaler.pkl          # Scaler untuk NB
│   ├── label_encoder.pkl      # Label encoder
│   ├── bert_embeddings.npy    # BERT embeddings seluruh dokumen
│   ├── train_indices.json     # Indeks data train
│   ├── test_indices.json      # Indeks data test
│   └── case_ids.json          # Urutan case_id
└── logs/
    ├── cleaning.log           # Log scraping & cleaning
    ├── representation.log     # Log tahap representasi
    ├── retrieval.log          # Log tahap retrieval
    └── evaluation.log         # Log evaluasi model
```

---

## Instalasi

### 1. Clone repository

```bash
git clone https://github.com/andika26ck/Case-Based-Reasoning-CBR-sederhana.git
cd Case-Based-Reasoning-CBR-sederhana
```

### 2. Install dependensi

```bash
pip install -r requirements.txt
```

> **Catatan:** `sentence-transformers` (untuk BERT) bersifat opsional.
> Jika tidak diinstall, pipeline tetap berjalan dengan TF-IDF + SVM/NB saja.

### 3. Prasyarat tambahan

- **Google Chrome** harus terinstall (dipakai `01_scraping.py` untuk bypass bot-protection)
- Cek versi Chrome di `chrome://settings/help`, lalu sesuaikan `version_main` di `01_scraping.py`

---

## Cara Menjalankan Pipeline End-to-End

Jalankan tiap script secara berurutan:

### Tahap 1 — Scraping & Preprocessing

```bash
python 01_scraping.py
```

Chrome akan terbuka otomatis sebentar (normal) untuk bypass bot-protection Cloudflare.
Tunggu hingga selesai (~30-60 menit untuk 90 dokumen).

**Output:** `data/raw/*.txt`, `data/processed/metadata_raw.csv`

### QA Check Tahap 1 (opsional, sangat disarankan)

```bash
python 02_qa_check.py
```

**Output:** `logs/qa_report.txt`, `data/processed/qa_flagged.csv`

### Tahap 2 — Case Representation

```bash
python 02_representation.py
```

**Output:** `data/processed/cases.csv`, `data/processed/cases.json`

### Tahap 3 — Case Retrieval

```bash
python 03_retrieval.py
```

**Output:** model di `models/`, `data/eval/queries.json`, `data/eval/retrieval_metrics.csv`

### Tahap 4 — Case Solution Reuse

```bash
python 04_predict.py
```

**Output:** `data/results/predictions.csv`

### Tahap 5 — Evaluasi Model

```bash
python 05_evaluation.py
```

**Output:** `data/eval/retrieval_metrics.csv`, `data/eval/prediction_metrics.csv`,
`data/eval/error_analysis.csv`, `data/eval/model_comparison.png`

---

## Ringkasan Metode

| Tahap | Metode | Library |
|-------|--------|---------|
| Scraping | undetected-chromedriver + requests + BeautifulSoup | `uc`, `bs4` |
| Preprocessing | pdfminer + regex cleaning + tokenisasi | `pdfminer`, `re` |
| Vectorisasi | TF-IDF (ngram 1-2, max 10k fitur) | `sklearn` |
| Klasifikasi | SVM (LinearSVC) + Naive Bayes (MultinomialNB) | `sklearn` |
| Retrieval | Cosine Similarity (TF-IDF & BERT) | `sklearn`, `sentence-transformers` |
| Embedding | all-MiniLM-L6-v2 (384 dimensi) | `sentence-transformers` |
| Evaluasi | Accuracy, Precision, Recall, F1-score | `sklearn.metrics` |

---

## Statistik Dataset

- **Domain:** Perdata - Perbuatan Melawan Hukum (PMH)
- **Pengadilan:** PN Bandung
- **Jumlah dokumen valid:** 90 putusan
- **Split:** 80% train / 20% test
- **Label (decision_type):** dikabulkan_seluruhnya, dikabulkan_sebagian, ditolak, lainnya

---

## Siklus CBR yang Diimplementasikan

```
Query Kasus Baru
      │
      ▼
[RETRIEVE] Hitung similarity dengan semua case di case base
  → TF-IDF + SVM/NB (klasifikasi)
  → TF-IDF + Cosine Similarity (retrieval)
  → BERT + Cosine Similarity (retrieval)
      │
      ▼
[REUSE] Ambil top-5 kasus paling mirip
  → Majority vote / weighted similarity pada amar_putusan
      │
      ▼
[REVISE] (opsional) Koreksi solusi jika diperlukan
      │
      ▼
[RETAIN] Simpan kasus baru ke case base jika solusinya terbukti benar
```

---

## requirements.txt

```
requests>=2.31.0
beautifulsoup4>=4.12.0
lxml>=4.9.0
pdfminer.six>=20221105
undetected-chromedriver>=3.5.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
joblib>=1.3.0
sentence-transformers>=2.2.0
```
