"""
05_evaluation.py
=================
Tahap 5 (Model Evaluation) - Tugas CBR Penalaran Komputer
Domain: Perdata - Perbuatan Melawan Hukum (PMH) | Pengadilan: PN Bandung

Sesuai spesifikasi tugas:
  i.  Evaluasi Retrieval: Accuracy, Precision, Recall, F1-score
  ii. Analisis kegagalan model (Rejection) dan rekomendasi perbaikan

Output:
  data/eval/retrieval_metrics.csv    -> metrik per model
  data/eval/prediction_metrics.csv  -> metrik prediksi solusi
  data/eval/error_analysis.csv      -> kasus yang salah diprediksi
  logs/evaluation.log               -> log lengkap
"""

import os
import json
import joblib
import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (tidak butuh display)
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ── Coba import sentence-transformers ───────────────────────────────────────
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

CASES_CSV        = f"{PROCESSED_DIR}/cases.csv"
QUERIES_JSON     = f"{EVAL_DIR}/queries.json"
METRICS_CSV      = f"{EVAL_DIR}/retrieval_metrics.csv"
PRED_METRICS_CSV = f"{EVAL_DIR}/prediction_metrics.csv"
ERROR_CSV        = f"{EVAL_DIR}/error_analysis.csv"
CONFUSION_IMG    = f"{EVAL_DIR}/confusion_matrix.png"

TOP_K        = 5
RANDOM_STATE = 42

for d in (EVAL_DIR, LOG_DIR):
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    filename=f"{LOG_DIR}/evaluation.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def log(msg):
    print(msg)
    logging.info(msg)


def normalize_decision_type(label: str) -> str:
    """Harus IDENTIK dengan fungsi di 03_retrieval.py agar label cocok dgn LabelEncoder."""
    l = str(label).strip().lower()
    if l.startswith("dikabulkan"):
        return "dikabulkan"
    if l in ("ditolak", "tidak dapat diterima", "niet_ontvankelijke_verklaard"):
        return "ditolak"
    return "lainnya"


# ----------------------------------------------------------------------------
# LOAD SEMUA ARTEFAK DARI TAHAP 3
# ----------------------------------------------------------------------------
def load_artifacts():
    required = [
        f"{MODELS_DIR}/tfidf_vectorizer.pkl",
        f"{MODELS_DIR}/svm_model.pkl",
        f"{MODELS_DIR}/nb_model.pkl",
        f"{MODELS_DIR}/label_encoder.pkl",
        f"{MODELS_DIR}/train_indices.json",
        f"{MODELS_DIR}/test_indices.json",
        f"{MODELS_DIR}/case_ids.json",
    ]
    for path in required:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"File tidak ditemukan: {path}\n"
                "Pastikan 03_retrieval.py sudah dijalankan terlebih dahulu."
            )

    vectorizer = joblib.load(f"{MODELS_DIR}/tfidf_vectorizer.pkl")
    svm_model  = joblib.load(f"{MODELS_DIR}/svm_model.pkl")
    nb_model   = joblib.load(f"{MODELS_DIR}/nb_model.pkl")
    le         = joblib.load(f"{MODELS_DIR}/label_encoder.pkl")
    scaler     = joblib.load(f"{MODELS_DIR}/nb_scaler.pkl") \
                 if os.path.exists(f"{MODELS_DIR}/nb_scaler.pkl") else None

    with open(f"{MODELS_DIR}/train_indices.json") as f:
        idx_train = json.load(f)
    with open(f"{MODELS_DIR}/test_indices.json") as f:
        idx_test = json.load(f)
    with open(f"{MODELS_DIR}/case_ids.json") as f:
        case_ids = json.load(f)

    bert_emb_path = f"{MODELS_DIR}/bert_embeddings.npy"
    bert_embeddings = None
    if os.path.exists(bert_emb_path):
        loaded = np.load(bert_emb_path, allow_pickle=True)
        if loaded.size > 0:
            bert_embeddings = loaded

    return {
        "vectorizer": vectorizer, "svm": svm_model, "nb": nb_model,
        "le": le, "scaler": scaler,
        "idx_train": idx_train, "idx_test": idx_test,
        "case_ids": case_ids, "bert_embeddings": bert_embeddings,
    }


# ----------------------------------------------------------------------------
# EVALUASI KLASIFIKASI (Accuracy, Precision, Recall, F1)
# ----------------------------------------------------------------------------
def evaluate_classification(artifacts, df):
    log("\n" + "="*70)
    log("A. EVALUASI KLASIFIKASI (SVM & Naive Bayes)")
    log("="*70)

    vectorizer = artifacts["vectorizer"]
    svm_model  = artifacts["svm"]
    nb_model   = artifacts["nb"]
    le         = artifacts["le"]
    scaler     = artifacts["scaler"]
    idx_test   = artifacts["idx_test"]

    df_test = df.iloc[idx_test].reset_index(drop=True)
    text_combined_test = (
        df_test.get("ringkasan_fakta", "").fillna("") + " " +
        df_test.get("argumen_hukum", "").fillna("") + " " +
        df_test.get("pasal", "").fillna("") + " " +
        df_test.get("amar_putusan", "").fillna("")
    ).str.lower().str.strip()

    tfidf_test = vectorizer.transform(text_combined_test)
    y_true     = le.transform(df_test["decision_type"].fillna("lainnya"))

    results = []
    error_records = []

    for model_name, model in [("SVM (LinearSVC)", svm_model),
                               ("Naive Bayes", nb_model)]:
        if model_name == "Naive Bayes" and scaler is not None:
            X_test = scaler.transform(tfidf_test.toarray())
        else:
            X_test = tfidf_test

        y_pred = model.predict(X_test)

        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        results.append({
            "model": model_name,
            "accuracy":  round(acc, 4),
            "precision": round(prec, 4),
            "recall":    round(rec, 4),
            "f1_score":  round(f1, 4),
        })

        log(f"\nModel: {model_name}")
        log(f"  Accuracy : {acc:.4f}")
        log(f"  Precision: {prec:.4f}")
        log(f"  Recall   : {rec:.4f}")
        log(f"  F1-score : {f1:.4f}")
        log(f"\n{classification_report(y_true, y_pred, target_names=le.classes_, zero_division=0)}")

        # Kumpulkan error untuk analisis kegagalan
        if model_name == "SVM (LinearSVC)":
            for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
                if yt != yp:
                    error_records.append({
                        "case_id": df_test.iloc[i]["case_id"],
                        "no_perkara": df_test.iloc[i].get("no_perkara", ""),
                        "true_label": le.inverse_transform([yt])[0],
                        "pred_label_svm": le.inverse_transform([yp])[0],
                        "jumlah_kata": df_test.iloc[i].get("jumlah_kata", 0),
                    })

            # Confusion matrix
            try:
                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots(figsize=(8, 6))
                disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
                disp.plot(ax=ax, xticks_rotation=45, colorbar=True)
                ax.set_title("Confusion Matrix - SVM")
                plt.tight_layout()
                plt.savefig(CONFUSION_IMG, dpi=120)
                plt.close()
                log(f"\n[SAVED] Confusion matrix -> {CONFUSION_IMG}")
            except Exception as e:
                log(f"[WARN] Gagal buat confusion matrix: {e}")

    return results, error_records


# ----------------------------------------------------------------------------
# EVALUASI RETRIEVAL (Precision@k, Recall@k, F1@k, Accuracy@k)
# ----------------------------------------------------------------------------
def evaluate_retrieval_models(artifacts, df):
    log("\n" + "="*70)
    log("B. EVALUASI RETRIEVAL (TF-IDF Cosine & BERT Cosine)")
    log("="*70)

    vectorizer     = artifacts["vectorizer"]
    le             = artifacts["le"]
    idx_test       = artifacts["idx_test"]
    case_ids       = artifacts["case_ids"]
    bert_embs      = artifacts["bert_embeddings"]

    df["text_combined"] = (
        df.get("ringkasan_fakta", "").fillna("") + " " +
        df.get("argumen_hukum", "").fillna("") + " " +
        df.get("pasal", "").fillna("") + " " +
        df.get("amar_putusan", "").fillna("")
    ).str.lower().str.strip()

    labels      = le.transform(df["decision_type"].fillna("lainnya"))
    tfidf_all   = vectorizer.transform(df["text_combined"])
    sim_tfidf   = cosine_similarity(tfidf_all)

    results = []

    for name, sim_matrix in [("TF-IDF + Cosine", sim_tfidf)]:
        precs, recs, f1s, accs = [], [], [], []
        for idx in idx_test:
            true_label = labels[idx]
            relevant = set(
                i for i in range(len(labels))
                if labels[i] == true_label and i != idx
            )
            if not relevant:
                continue
            sims = sim_matrix[idx].copy()
            sims[idx] = -1
            top_k = set(np.argsort(sims)[::-1][:TOP_K])
            tp = len(top_k & relevant)
            p = tp / TOP_K
            r = tp / len(relevant) if relevant else 0
            f = 2*p*r/(p+r) if (p+r) > 0 else 0
            a = 1 if tp > 0 else 0
            precs.append(p); recs.append(r); f1s.append(f); accs.append(a)

        results.append({
            "model": name,
            f"accuracy@{TOP_K}":  round(np.mean(accs), 4),
            f"precision@{TOP_K}": round(np.mean(precs), 4),
            f"recall@{TOP_K}":    round(np.mean(recs), 4),
            f"f1@{TOP_K}":        round(np.mean(f1s), 4),
        })
        log(f"\nModel: {name}")
        log(f"  Accuracy@{TOP_K} : {results[-1][f'accuracy@{TOP_K}']}")
        log(f"  Precision@{TOP_K}: {results[-1][f'precision@{TOP_K}']}")
        log(f"  Recall@{TOP_K}   : {results[-1][f'recall@{TOP_K}']}")
        log(f"  F1@{TOP_K}       : {results[-1][f'f1@{TOP_K}']}")

    # BERT jika tersedia
    if bert_embs is not None and bert_embs.shape[0] == len(df):
        sim_bert = cosine_similarity(bert_embs)
        precs, recs, f1s, accs = [], [], [], []
        for idx in idx_test:
            true_label = labels[idx]
            relevant = set(
                i for i in range(len(labels))
                if labels[i] == true_label and i != idx
            )
            if not relevant:
                continue
            sims = sim_bert[idx].copy()
            sims[idx] = -1
            top_k = set(np.argsort(sims)[::-1][:TOP_K])
            tp = len(top_k & relevant)
            p = tp / TOP_K
            r = tp / len(relevant) if relevant else 0
            f = 2*p*r/(p+r) if (p+r) > 0 else 0
            a = 1 if tp > 0 else 0
            precs.append(p); recs.append(r); f1s.append(f); accs.append(a)

        results.append({
            "model": "BERT + Cosine",
            f"accuracy@{TOP_K}":  round(np.mean(accs), 4),
            f"precision@{TOP_K}": round(np.mean(precs), 4),
            f"recall@{TOP_K}":    round(np.mean(recs), 4),
            f"f1@{TOP_K}":        round(np.mean(f1s), 4),
        })
        log(f"\nModel: BERT + Cosine")
        log(f"  Accuracy@{TOP_K} : {results[-1][f'accuracy@{TOP_K}']}")
        log(f"  Precision@{TOP_K}: {results[-1][f'precision@{TOP_K}']}")
        log(f"  Recall@{TOP_K}   : {results[-1][f'recall@{TOP_K}']}")
        log(f"  F1@{TOP_K}       : {results[-1][f'f1@{TOP_K}']}")

    return results


# ----------------------------------------------------------------------------
# EVALUASI PREDIKSI SOLUSI (dari queries.json)
# ----------------------------------------------------------------------------
def evaluate_predictions():
    log("\n" + "="*70)
    log("C. EVALUASI PREDIKSI SOLUSI (dari queries.json)")
    log("="*70)

    if not os.path.exists(QUERIES_JSON):
        log(f"[WARN] {QUERIES_JSON} tidak ditemukan, lewati evaluasi prediksi.")
        return []

    with open(QUERIES_JSON, "r", encoding="utf-8") as f:
        queries = json.load(f)

    records = []
    correct_tfidf = 0
    correct_bert  = 0
    total = len(queries)

    for q in queries:
        expected = q.get("ground_truth_case_id", "")
        top_tfidf = q.get("top5_tfidf", [])
        top_bert  = q.get("top5_bert", [])

        hit_tfidf = expected in top_tfidf
        hit_bert  = expected in top_bert

        if hit_tfidf:
            correct_tfidf += 1
        if hit_bert:
            correct_bert += 1

        records.append({
            "query_id":      q.get("query_id", ""),
            "case_id":       q.get("case_id", ""),
            "decision_type": q.get("decision_type", ""),
            "hit_tfidf":     hit_tfidf,
            "hit_bert":      hit_bert,
            "top5_tfidf":    ", ".join(top_tfidf),
            "top5_bert":     ", ".join(top_bert),
        })

    pred_metrics = [
        {"model": "TF-IDF Retrieval",
         f"hit_rate@{TOP_K}": round(correct_tfidf / total, 4) if total else 0,
         "correct": correct_tfidf, "total": total},
        {"model": "BERT Retrieval",
         f"hit_rate@{TOP_K}": round(correct_bert / total, 4) if total else 0,
         "correct": correct_bert, "total": total},
    ]

    for m in pred_metrics:
        log(f"\n{m['model']}: Hit Rate@{TOP_K} = {m[f'hit_rate@{TOP_K}']} "
            f"({m['correct']}/{m['total']} query)")

    pd.DataFrame(records).to_csv(PRED_METRICS_CSV, index=False)
    log(f"\n[SAVED] Prediksi solusi -> {PRED_METRICS_CSV}")

    return pred_metrics


# ----------------------------------------------------------------------------
# ANALISIS KEGAGALAN (Error Analysis)
# ----------------------------------------------------------------------------
def error_analysis(error_records, df):
    log("\n" + "="*70)
    log("D. ANALISIS KEGAGALAN MODEL (Error Analysis)")
    log("="*70)

    if not error_records:
        log("Tidak ada error yang ditemukan (model sempurna di test set).")
        return

    df_err = pd.DataFrame(error_records)
    df_err.to_csv(ERROR_CSV, index=False)

    log(f"\nTotal kasus yang salah diprediksi (SVM): {len(df_err)}")
    log(f"\nDistribusi true label yang sering salah:")
    log(str(df_err["true_label"].value_counts()))
    log(f"\nDistribusi pred label yang sering muncul sebagai kesalahan:")
    log(str(df_err["pred_label_svm"].value_counts()))

    # Analisis: apakah ada pola di kasus yang gagal?
    avg_kata = df_err["jumlah_kata"].mean()
    log(f"\nRata-rata jumlah kata kasus yang gagal: {avg_kata:.0f} kata")

    log("\n=== REKOMENDASI PERBAIKAN ===")
    rekomendasi = [
        "1. AUGMENTASI DATA: Kelas minoritas (gugur, NOV) sangat sedikit sehingga "
        "model sulit belajar polanya. Coba over-sampling (SMOTE) atau tambah data "
        "dari pengadilan lain.",

        "2. FEATURE ENGINEERING: Tambahkan fitur eksplisit dari pasal yang dirujuk "
        "dan identitas pihak sebagai fitur kategoris (bukan hanya teks bebas).",

        "3. FINE-TUNING BERT: Gunakan IndoBERT yang di-fine-tune pada dataset hukum "
        "Indonesia (indobenchmark/indobert-base-p1) untuk representasi yang lebih "
        "sesuai domain PMH.",

        "4. THRESHOLD REJECTION: Implementasikan mekanisme 'rejection' — jika skor "
        "similarity tertinggi < threshold (misal 0.3), sistem menyatakan 'kasus baru "
        "tidak memiliki padanan yang cukup mirip' daripada memaksakan prediksi.",

        "5. ENSEMBLE: Gabungkan prediksi TF-IDF+SVM dan BERT dengan weighted voting "
        "untuk meningkatkan robustness.",
    ]
    for r in rekomendasi:
        log(f"\n{r}")

    log(f"\n[SAVED] Error analysis -> {ERROR_CSV}")


# ----------------------------------------------------------------------------
# RINGKASAN AKHIR & VISUALISASI
# ----------------------------------------------------------------------------
def print_summary(clf_results, ret_results):
    log("\n" + "="*70)
    log("E. RINGKASAN PERBANDINGAN SEMUA MODEL")
    log("="*70)

    log(f"\n{'Model':<30} {'Accuracy':>10} {'Precision':>10} "
        f"{'Recall':>10} {'F1':>10}")
    log("-" * 65)

    all_rows = []
    for m in clf_results:
        log(f"{m['model']:<30} {m['accuracy']:>10} {m['precision']:>10} "
            f"{m['recall']:>10} {m['f1_score']:>10}")
        all_rows.append({
            "model": m["model"],
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1_score"],
            "type": "classification",
        })

    for m in ret_results:
        k = TOP_K
        acc  = m.get(f"accuracy@{k}", "-")
        prec = m.get(f"precision@{k}", "-")
        rec  = m.get(f"recall@{k}", "-")
        f1   = m.get(f"f1@{k}", "-")
        log(f"{m['model']:<30} {str(acc):>10} {str(prec):>10} "
            f"{str(rec):>10} {str(f1):>10}")
        all_rows.append({
            "model": m["model"],
            "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1,
            "type": "retrieval",
        })

    pd.DataFrame(all_rows).to_csv(METRICS_CSV, index=False)
    log(f"\n[SAVED] Metrik lengkap -> {METRICS_CSV}")

    # Bar chart
    try:
        models = [r["model"] for r in all_rows]
        f1s    = [float(r["f1"]) if r["f1"] != "-" else 0 for r in all_rows]

        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.barh(models, f1s, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
        ax.set_xlabel("F1-Score")
        ax.set_title("Perbandingan F1-Score Semua Model (CBR - PMH PN Bandung)")
        ax.set_xlim(0, 1.05)
        for bar, val in zip(bars, f1s):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=10)
        plt.tight_layout()
        chart_path = f"{EVAL_DIR}/model_comparison.png"
        plt.savefig(chart_path, dpi=120)
        plt.close()
        log(f"[SAVED] Bar chart perbandingan -> {chart_path}")
    except Exception as e:
        log(f"[WARN] Gagal buat chart: {e}")


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    log("=== TAHAP 5: Model Evaluation ===")
    log("Domain: Perdata - Perbuatan Melawan Hukum (PMH) | Pengadilan: PN Bandung")

    if not os.path.exists(CASES_CSV):
        log(f"[FATAL] {CASES_CSV} tidak ditemukan. Jalankan 02_representation.py dulu.")
        return

    df = pd.read_csv(CASES_CSV)
    # FIX: samakan label dengan yang dipakai saat training di 03_retrieval.py
    df["decision_type"] = df["decision_type"].fillna("lainnya").apply(normalize_decision_type)
    log(f"Dataset: {len(df)} kasus")

    try:
        artifacts = load_artifacts()
    except FileNotFoundError as e:
        log(f"[FATAL] {e}")
        return

    clf_results, error_records = evaluate_classification(artifacts, df)
    ret_results = evaluate_retrieval_models(artifacts, df)
    evaluate_predictions()
    error_analysis(error_records, df)
    print_summary(clf_results, ret_results)

    log("\n=== EVALUASI SELESAI ===")
    log(f"Semua output tersimpan di folder: {EVAL_DIR}/")


if __name__ == "__main__":
    main()
