"""
03_retrieval.py
================
Tahap 3 (Case Retrieval) - Tugas CBR Penalaran Komputer
Domain: Perdata - Perbuatan Melawan Hukum (PMH) | Pengadilan: PN Bandung

Pendekatan yang digunakan:
  1. TF-IDF + SVM   (classification/retrieval, sesuai spesifikasi)
  2. TF-IDF + Naive Bayes (alternatif ML)
  3. TF-IDF + Cosine Similarity (retrieval langsung)
  4. BERT (all-MiniLM-L6-v2) + Cosine Similarity

Output:
  models/tfidf_vectorizer.pkl       -> TF-IDF vectorizer
  models/svm_model.pkl              -> model SVM
  models/nb_model.pkl               -> model Naive Bayes
  models/bert_embeddings.npy        -> BERT embeddings semua dokumen
  models/train_indices.json         -> indeks train split (dipakai 04_predict.py)
  models/test_indices.json          -> indeks test split
  models/case_ids.json              -> urutan case_id sesuai embeddings
  data/eval/queries.json            -> 5-10 query uji + ground-truth case_id
  data/eval/retrieval_results.csv   -> hasil retrieval semua model di test set
"""

import os
import json
import joblib
import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ── Coba import sentence-transformers (BERT), opsional ──────────────────────
try:
    from sentence_transformers import SentenceTransformer
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False

# ----------------------------------------------------------------------------
# KONFIGURASI
# ----------------------------------------------------------------------------
PROCESSED_DIR = "data/processed"
MODELS_DIR    = "models"
EVAL_DIR      = "data/eval"
LOG_DIR       = "logs"

CASES_CSV     = f"{PROCESSED_DIR}/cases.csv"
OUTPUT_QUERIES = f"{EVAL_DIR}/queries.json"
OUTPUT_RESULTS = f"{EVAL_DIR}/retrieval_results.csv"

BERT_MODEL_NAME = "all-MiniLM-L6-v2"   # ringan, cocok untuk dokumen hukum
TEST_SIZE       = 0.2                    # rasio data test (80:20)
RANDOM_STATE    = 42
TOP_K           = 5                      # jumlah kasus mirip yang dikembalikan
N_QUERY_SAMPLES = 10                     # jumlah query uji di queries.json

for d in (MODELS_DIR, EVAL_DIR, LOG_DIR):
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    filename=f"{LOG_DIR}/retrieval.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def log(msg):
    print(msg)
    logging.info(msg)


def normalize_decision_type(label: str) -> str:
    """Gabungkan label putusan yang langka agar tiap kelas punya >=2 anggota.
    Mengatasi error stratified split: 'least populated class has only 1 member'."""
    l = str(label).strip().lower()
    if l.startswith("dikabulkan"):
        return "dikabulkan"
    if l in ("ditolak", "tidak dapat diterima", "niet_ontvankelijke_verklaard"):
        return "ditolak"
    return "lainnya"


# ----------------------------------------------------------------------------
# FUNGSI RETRIEVE (sesuai signature spesifikasi tugas)
# ----------------------------------------------------------------------------
def retrieve(query: str, k: int = TOP_K,
             vectorizer=None, tfidf_matrix=None,
             bert_model=None, bert_embeddings=None,
             case_ids=None, method="tfidf") -> list:
    """
    Temukan top-k kasus paling mirip dengan query.

    Params:
        query      : teks query kasus baru
        k          : jumlah kasus yang dikembalikan
        method     : 'tfidf' atau 'bert'

    Return:
        list of case_id (top-k paling mirip)
    """
    if method == "tfidf" and vectorizer is not None and tfidf_matrix is not None:
        # 1) Pre-process query
        query_clean = query.lower().strip()
        # 2) Hitung vektor query
        query_vec = vectorizer.transform([query_clean])
        # 3) Hitung cosine similarity
        sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
        # 4) Kembalikan top-k case_id
        top_indices = np.argsort(sims)[::-1][:k]
        return [case_ids[i] for i in top_indices]

    elif method == "bert" and bert_model is not None and bert_embeddings is not None:
        # 1) Pre-process query
        query_clean = query.lower().strip()
        # 2) Hitung vektor query
        query_emb = bert_model.encode([query_clean], normalize_embeddings=True)
        # 3) Hitung cosine similarity
        sims = cosine_similarity(query_emb, bert_embeddings).flatten()
        # 4) Kembalikan top-k case_id
        top_indices = np.argsort(sims)[::-1][:k]
        return [case_ids[i] for i in top_indices]

    return []


# ----------------------------------------------------------------------------
# EVALUASI RETRIEVAL: Precision@k, Recall@k, F1@k
# (ground-truth = kasus dengan decision_type yang sama)
# ----------------------------------------------------------------------------
def evaluate_retrieval(test_indices, case_ids, labels, sim_matrix, k=TOP_K):
    """Hitung Precision@k, Recall@k, F1@k untuk retrieval berbasis similarity."""
    precisions, recalls, f1s = [], [], []

    for idx in test_indices:
        true_label = labels[idx]
        # ground-truth: semua kasus dengan label sama (di luar dirinya sendiri)
        relevant = set(
            i for i in range(len(labels))
            if labels[i] == true_label and i != idx
        )
        if not relevant:
            continue

        # retrieved: top-k berdasarkan similarity (exclude dirinya sendiri)
        sims = sim_matrix[idx].copy()
        sims[idx] = -1  # exclude self
        top_k_idx = set(np.argsort(sims)[::-1][:k])

        tp = len(top_k_idx & relevant)
        precision_k = tp / k
        recall_k = tp / len(relevant) if relevant else 0
        f1_k = (2 * precision_k * recall_k / (precision_k + recall_k)
                if (precision_k + recall_k) > 0 else 0)

        precisions.append(precision_k)
        recalls.append(recall_k)
        f1s.append(f1_k)

    return {
        f"precision@{k}": round(np.mean(precisions), 4) if precisions else 0,
        f"recall@{k}":    round(np.mean(recalls), 4) if recalls else 0,
        f"f1@{k}":        round(np.mean(f1s), 4) if f1s else 0,
    }


# ----------------------------------------------------------------------------
# BUAT queries.json (output wajib Tahap 3)
# ----------------------------------------------------------------------------
def generate_queries_json(df_test, case_ids, labels, tfidf_matrix,
                          vectorizer, bert_embeddings=None, bert_model=None):
    """
    Buat file queries.json berisi N_QUERY_SAMPLES query uji beserta:
    - teks query (diambil dari ringkasan_fakta dokumen test)
    - ground-truth case_id (dokumen itu sendiri)
    - expected_similar: case_id yang punya decision_type sama
    - top5_tfidf: hasil retrieval TF-IDF
    - top5_bert: hasil retrieval BERT (jika tersedia)
    """
    queries = []
    sample_df = df_test.sample(min(N_QUERY_SAMPLES, len(df_test)),
                                random_state=RANDOM_STATE)

    for _, row in sample_df.iterrows():
        query_text = str(row.get("ringkasan_fakta", "")) or str(row.get("text_full", ""))
        query_text = query_text[:500]  # batasi panjang query

        case_id = row["case_id"]
        label = row["decision_type"]

        # Expected similar: case_id dengan label yang sama
        expected = list(df_test[
            (df_test["decision_type"] == label) &
            (df_test["case_id"] != case_id)
        ]["case_id"])[:5]

        # Hasil retrieval TF-IDF
        top_tfidf = retrieve(
            query_text, k=TOP_K,
            vectorizer=vectorizer,
            tfidf_matrix=tfidf_matrix,
            case_ids=case_ids,
            method="tfidf"
        )

        # Hasil retrieval BERT (jika tersedia)
        top_bert = []
        if BERT_AVAILABLE and bert_model is not None and bert_embeddings is not None:
            top_bert = retrieve(
                query_text, k=TOP_K,
                bert_model=bert_model,
                bert_embeddings=bert_embeddings,
                case_ids=case_ids,
                method="bert"
            )

        queries.append({
            "query_id": f"q_{case_id[:20]}",
            "case_id": case_id,
            "no_perkara": row.get("no_perkara", ""),
            "decision_type": label,
            "query_text": query_text,
            "ground_truth_case_id": case_id,
            "expected_similar_case_ids": expected,
            "top5_tfidf": top_tfidf,
            "top5_bert": top_bert,
        })

    with open(OUTPUT_QUERIES, "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)

    log(f"[QUERIES] {len(queries)} query tersimpan -> {OUTPUT_QUERIES}")
    return queries


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    log("=== TAHAP 3: Case Retrieval ===")
    log("Domain: Perdata - Perbuatan Melawan Hukum (PMH) | Pengadilan: PN Bandung")

    # -- Load dataset --
    if not os.path.exists(CASES_CSV):
        log(f"[FATAL] {CASES_CSV} tidak ditemukan. Jalankan 02_representation.py dulu.")
        return

    df = pd.read_csv(CASES_CSV)
    log(f"Total kasus dimuat: {len(df)}")

    # -- Gabung text features untuk vectorization --
    df["text_combined"] = (
        df.get("ringkasan_fakta", "").fillna("") + " " +
        df.get("argumen_hukum", "").fillna("") + " " +
        df.get("pasal", "").fillna("") + " " +
        df.get("amar_putusan", "").fillna("")
    ).str.lower().str.strip()

    # Label untuk klasifikasi (SVM & NB)
    # FIX: gabungkan label langka agar tiap kelas punya >=2 anggota.
    # (kelas 'gugur' hanya 1 dokumen -> stratified split gagal)
    df["decision_type"] = df["decision_type"].fillna("lainnya").apply(normalize_decision_type)
    le = LabelEncoder()
    df["label_encoded"] = le.fit_transform(df["decision_type"])

    # -- Splitting Data (80:20) --
    # Guard tambahan: jika masih ada kelas <2 anggota, matikan stratify
    from collections import Counter
    label_counts = Counter(df["label_encoded"].tolist())
    stratify_arg = df["label_encoded"] if min(label_counts.values()) >= 2 else None
    if stratify_arg is None:
        log("[WARN] Masih ada kelas <2 anggota; stratify dimatikan untuk split ini.")
    idx_all = np.arange(len(df))
    idx_train, idx_test = train_test_split(
        idx_all, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=stratify_arg
    )

    df_train = df.iloc[idx_train].reset_index(drop=True)
    df_test  = df.iloc[idx_test].reset_index(drop=True)

    log(f"Split data -> train: {len(df_train)} | test: {len(df_test)}")
    log(f"Distribusi label: {dict(df['decision_type'].value_counts())}")

    # ── SIMPAN INDEKS (KRITIKAL untuk 04_predict.py) ────────────────────────
    with open(f"{MODELS_DIR}/train_indices.json", "w") as f:
        json.dump(idx_train.tolist(), f)
    with open(f"{MODELS_DIR}/test_indices.json", "w") as f:
        json.dump(idx_test.tolist(), f)

    case_ids = df["case_id"].tolist()
    with open(f"{MODELS_DIR}/case_ids.json", "w") as f:
        json.dump(case_ids, f)

    # Simpan label encoder
    joblib.dump(le, f"{MODELS_DIR}/label_encoder.pkl")
    log("[SAVED] train_indices.json, test_indices.json, case_ids.json, label_encoder.pkl")

    # ════════════════════════════════════════════════════════════════════════
    # MODEL 1: TF-IDF Vectorization
    # ════════════════════════════════════════════════════════════════════════
    log("\n--- TF-IDF Vectorization ---")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        analyzer="word",
    )
    tfidf_all    = vectorizer.fit_transform(df["text_combined"])
    tfidf_train  = tfidf_all[idx_train]
    tfidf_test   = tfidf_all[idx_test]

    joblib.dump(vectorizer, f"{MODELS_DIR}/tfidf_vectorizer.pkl")
    log(f"TF-IDF matrix: {tfidf_all.shape} | vocab: {len(vectorizer.vocabulary_)}")

    # ── MODEL 2: SVM pada TF-IDF ────────────────────────────────────────────
    log("\n--- Model: SVM (LinearSVC) ---")
    svm_model = LinearSVC(C=1.0, max_iter=2000, random_state=RANDOM_STATE)
    svm_model.fit(tfidf_train, df_train["label_encoded"])
    y_pred_svm = svm_model.predict(tfidf_test)
    y_true      = df_test["label_encoded"].values

    svm_metrics = {
        "model": "TF-IDF + SVM",
        "accuracy":  round(accuracy_score(y_true, y_pred_svm), 4),
        "precision": round(precision_score(y_true, y_pred_svm,
                           average="weighted", zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred_svm,
                           average="weighted", zero_division=0), 4),
        "f1_score":  round(f1_score(y_true, y_pred_svm,
                           average="weighted", zero_division=0), 4),
    }

    joblib.dump(svm_model, f"{MODELS_DIR}/svm_model.pkl")
    log(f"SVM Accuracy : {svm_metrics['accuracy']}")
    log(f"SVM Precision: {svm_metrics['precision']}")
    log(f"SVM Recall   : {svm_metrics['recall']}")
    log(f"SVM F1-score : {svm_metrics['f1_score']}")
    log("\nClassification Report (SVM):")
    log(classification_report(y_true, y_pred_svm,
                               target_names=le.classes_, zero_division=0))

    # ── MODEL 3: Naive Bayes pada TF-IDF ────────────────────────────────────
    log("\n--- Model: Naive Bayes (MultinomialNB) ---")
    # NB butuh nilai non-negatif; MinMaxScaler di sparse matrix
    scaler = MinMaxScaler()
    tfidf_train_nn = scaler.fit_transform(tfidf_train.toarray())
    tfidf_test_nn  = scaler.transform(tfidf_test.toarray())

    nb_model = MultinomialNB(alpha=0.1)
    nb_model.fit(tfidf_train_nn, df_train["label_encoded"])
    y_pred_nb = nb_model.predict(tfidf_test_nn)

    nb_metrics = {
        "model": "TF-IDF + Naive Bayes",
        "accuracy":  round(accuracy_score(y_true, y_pred_nb), 4),
        "precision": round(precision_score(y_true, y_pred_nb,
                           average="weighted", zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred_nb,
                           average="weighted", zero_division=0), 4),
        "f1_score":  round(f1_score(y_true, y_pred_nb,
                           average="weighted", zero_division=0), 4),
    }

    joblib.dump(nb_model, f"{MODELS_DIR}/nb_model.pkl")
    joblib.dump(scaler, f"{MODELS_DIR}/nb_scaler.pkl")
    log(f"NB Accuracy : {nb_metrics['accuracy']}")
    log(f"NB Precision: {nb_metrics['precision']}")
    log(f"NB Recall   : {nb_metrics['recall']}")
    log(f"NB F1-score : {nb_metrics['f1_score']}")
    log("\nClassification Report (Naive Bayes):")
    log(classification_report(y_true, y_pred_nb,
                               target_names=le.classes_, zero_division=0))

    # ── TF-IDF Cosine Similarity (retrieval) ────────────────────────────────
    log("\n--- TF-IDF Cosine Similarity Retrieval ---")
    labels_arr  = df["label_encoded"].values
    sim_tfidf   = cosine_similarity(tfidf_all)
    tfidf_ret   = evaluate_retrieval(idx_test, case_ids, labels_arr, sim_tfidf, k=TOP_K)

    tfidf_retrieval_metrics = {"model": "TF-IDF + Cosine", **tfidf_ret}
    log(f"TF-IDF Retrieval: {tfidf_ret}")

    # ════════════════════════════════════════════════════════════════════════
    # MODEL 4: BERT Embeddings
    # ════════════════════════════════════════════��═══════════════════════════
    bert_model       = None
    bert_embeddings  = None
    bert_metrics     = {"model": "BERT + Cosine", "precision@5": 0,
                        "recall@5": 0, "f1@5": 0}

    if BERT_AVAILABLE:
        log(f"\n--- BERT Embeddings ({BERT_MODEL_NAME}) ---")
        log("Encoding dokumen... (mungkin butuh beberapa menit)")
        bert_model = SentenceTransformer(BERT_MODEL_NAME)
        texts_to_encode = df["text_combined"].tolist()
        bert_embeddings = bert_model.encode(
            texts_to_encode,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        np.save(f"{MODELS_DIR}/bert_embeddings.npy", bert_embeddings)
        log(f"BERT embeddings shape: {bert_embeddings.shape}")

        sim_bert = cosine_similarity(bert_embeddings)
        bert_ret = evaluate_retrieval(idx_test, case_ids, labels_arr, sim_bert, k=TOP_K)
        bert_metrics = {"model": "BERT + Cosine", **bert_ret}
        log(f"BERT Retrieval: {bert_ret}")
    else:
        log("\n[WARN] sentence-transformers tidak terinstall. Skip BERT.")
        log("  Jalankan: pip install sentence-transformers")
        np.save(f"{MODELS_DIR}/bert_embeddings.npy", np.array([]))

    # ── Simpan perbandingan metrik semua model ───────────────────────────────
    all_metrics = [svm_metrics, nb_metrics, tfidf_retrieval_metrics, bert_metrics]
    df_metrics = pd.DataFrame(all_metrics)
    df_metrics.to_csv(f"{EVAL_DIR}/retrieval_metrics.csv", index=False)
    log(f"\n[SAVED] Perbandingan metrik -> {EVAL_DIR}/retrieval_metrics.csv")

    # ── Tabel ringkasan ──────────────────────────────────────────────────────
    log("\n=== RINGKASAN PERBANDINGAN MODEL ===")
    log(f"{'Model':<30} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    log("-" * 65)
    for m in all_metrics:
        acc  = m.get("accuracy", m.get(f"precision@{TOP_K}", "-"))
        prec = m.get("precision", m.get(f"precision@{TOP_K}", "-"))
        rec  = m.get("recall",  m.get(f"recall@{TOP_K}", "-"))
        f1   = m.get("f1_score", m.get(f"f1@{TOP_K}", "-"))
        log(f"{m['model']:<30} {str(acc):>10} {str(prec):>10} {str(rec):>10} {str(f1):>10}")

    # ── Generate queries.json ────────────────────────────────────────────────
    log(f"\n--- Membuat {OUTPUT_QUERIES} ---")
    generate_queries_json(
        df_test, case_ids, labels_arr,
        tfidf_all, vectorizer,
        bert_embeddings=bert_embeddings,
        bert_model=bert_model,
    )

    log("\n=== TAHAP 3 SELESAI ===")
    log(f"Model TF-IDF vectorizer -> {MODELS_DIR}/tfidf_vectorizer.pkl")
    log(f"Model SVM               -> {MODELS_DIR}/svm_model.pkl")
    log(f"Model Naive Bayes       -> {MODELS_DIR}/nb_model.pkl")
    log(f"BERT embeddings         -> {MODELS_DIR}/bert_embeddings.npy")
    log(f"Train indices           -> {MODELS_DIR}/train_indices.json  ← penting untuk 04_predict.py")
    log(f"Queries uji             -> {OUTPUT_QUERIES}")
    log(f"Metrik retrieval        -> {EVAL_DIR}/retrieval_metrics.csv")


if __name__ == "__main__":
    main()
