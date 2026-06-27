"""
04_predict.py
==============
Tahap 4: Case Solution Reuse
Gunakan putusan lama (top-k termirip) sebagai dasar solusi kasus baru.
Strategi: majority vote / weighted similarity.

VERSI PERBAIKAN. Bug yang diperbaiki dari versi lama:
  - Path model salah ('data/models' -> seharusnya 'models', sesuai 03_retrieval.py)
  - train_indices.json dimuat dgn joblib.load -> seharusnya json.load
  - File 'bert_train_embeddings.npy' & 'bert_model.pkl' tidak pernah dibuat 03 ->
    sekarang pakai 'bert_embeddings.npy' (np.load) lalu di-slice ke train
  - Kolom solusi 'decision_summary'/'dakwaan_utama' tidak ada di cases.csv ->
    sekarang pakai 'amar_putusan' (solusi nyata), fallback ringkasan_fakta
  - Fallback query 'facts_summary' tidak ada -> pakai 'ringkasan_fakta'
  - Fallback otomatis ke TF-IDF cosine kalau BERT/embeddings tidak tersedia

Output:
  data/results/predictions.csv  (query_id, predicted_solution, top_5_case_ids)
"""

import os
import json
from collections import Counter, defaultdict

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ── Konfigurasi path (HARUS sama dgn 03_retrieval.py) ───────────────────────
PROCESSED_DIR = "data/processed"
MODELS_DIR    = "models"
EVAL_DIR      = "data/eval"
RESULTS_DIR   = "data/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

CASES_CSV    = f"{PROCESSED_DIR}/cases.csv"
QUERIES_JSON = f"{EVAL_DIR}/queries.json"
K            = 5
RANDOM_STATE = 42

# BERT opsional
try:
    from sentence_transformers import SentenceTransformer
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False


def safe_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


# ── Load data & indeks train ────────────────────────────────────────────────
cases = pd.read_csv(CASES_CSV)

with open(f"{MODELS_DIR}/train_indices.json", "r", encoding="utf-8") as f:
    train_indices = json.load(f)

train_cases    = cases.iloc[train_indices].reset_index(drop=True)
train_case_ids = train_cases["case_id"].astype(str).tolist()


# ── Solusi nyata = amar putusan (fallback: ringkasan_fakta, decision_type) ──
def extract_solution_text(row):
    for col in ("amar_putusan", "ringkasan_fakta", "decision_type"):
        val = safe_text(row.get(col, ""))
        if val:
            return val
    return ""

case_solutions = {
    safe_text(r["case_id"]): extract_solution_text(r)
    for _, r in cases.iterrows()
}


# ── Siapkan representasi retrieval (BERT kalau ada, kalau tidak TF-IDF) ──────
USE_BERT         = False
bert_model       = None
train_embeddings = None
vectorizer       = None
train_tfidf      = None

bert_path = f"{MODELS_DIR}/bert_embeddings.npy"
if BERT_AVAILABLE and os.path.exists(bert_path):
    emb = np.load(bert_path, allow_pickle=True)
    # 03 menyimpan embeddings untuk SEMUA dokumen (urutan = cases.csv)
    if emb.size > 0 and emb.shape[0] == len(cases):
        train_embeddings = emb[train_indices]
        bert_model       = SentenceTransformer("all-MiniLM-L6-v2")
        USE_BERT         = True

if not USE_BERT:
    vectorizer = joblib.load(f"{MODELS_DIR}/tfidf_vectorizer.pkl")
    text_combined = (
        train_cases.get("ringkasan_fakta", "").fillna("") + " " +
        train_cases.get("argumen_hukum", "").fillna("") + " " +
        train_cases.get("pasal", "").fillna("") + " " +
        train_cases.get("amar_putusan", "").fillna("")
    ).str.lower().str.strip()
    train_tfidf = vectorizer.transform(text_combined)
    print("[Tahap 4] BERT tidak tersedia -> memakai TF-IDF cosine untuk retrieval.")
else:
    print("[Tahap 4] Memakai BERT embeddings untuk retrieval.")


def retrieve_top_k(query: str, k: int = K):
    q = safe_text(query)
    if not q:
        return []
    if USE_BERT:
        qv   = bert_model.encode([q], normalize_embeddings=True)
        sims = cosine_similarity(qv, train_embeddings)[0]
    else:
        qv   = vectorizer.transform([q.lower()])
        sims = cosine_similarity(qv, train_tfidf)[0]
    top_idx = np.argsort(sims)[::-1][:k]
    out = []
    for i in top_idx:
        cid = train_case_ids[i]
        out.append({
            "case_id": cid,
            "similarity": float(sims[i]),
            "solution": case_solutions.get(cid, ""),
        })
    return out


def choose_solution(top_k, strategy: str = "weighted"):
    if not top_k:
        return ""

    if strategy == "majority":
        counts = Counter()
        scores = defaultdict(float)
        first  = {}
        for rank, it in enumerate(top_k):
            key = safe_text(it["solution"])
            if not key:
                continue
            counts[key] += 1
            scores[key] += it["similarity"]
            first.setdefault(key, rank)
        if not counts:
            return safe_text(top_k[0]["solution"])
        return max(counts, key=lambda k: (counts[k], scores[k], -first[k]))

    # weighted similarity
    scores = defaultdict(float)
    best   = {}
    for it in top_k:
        key = safe_text(it["solution"])
        if not key:
            continue
        scores[key] += it["similarity"]
        if key not in best or it["similarity"] > best[key]["similarity"]:
            best[key] = it
    if not scores:
        return safe_text(top_k[0]["solution"])
    return max(scores, key=lambda k: (scores[k], best[k]["similarity"]))


def predict_outcome(query: str, k: int = K, strategy: str = "weighted"):
    top_k = retrieve_top_k(query, k=k)
    predicted = choose_solution(top_k, strategy=strategy)
    top_ids   = [it["case_id"] for it in top_k]
    return predicted, top_ids, top_k


# ── Demo: pakai queries.json bila ada, kalau tidak 5 sample acak ────────────
rows = []
if os.path.exists(QUERIES_JSON):
    with open(QUERIES_JSON, "r", encoding="utf-8") as f:
        raw_queries = json.load(f)
    eval_queries = [
        {
            "query_id": q.get("query_id", q.get("case_id", "")),
            "query_text": safe_text(q.get("query_text", "")),
            "ground_truth_case_id": q.get("ground_truth_case_id", ""),
        }
        for q in raw_queries
    ]
    print("[Tahap 4] Demo prediksi dari data/eval/queries.json")
else:
    eval_queries = [
        {
            "query_id": safe_text(r["case_id"]),
            "query_text": safe_text(r.get("ringkasan_fakta", "")),
            "ground_truth_case_id": safe_text(r["case_id"]),
        }
        for _, r in cases.sample(min(5, len(cases)), random_state=RANDOM_STATE).iterrows()
    ]
    print("[Tahap 4] Demo prediksi dari 5 sample kasus")

for i, q in enumerate(eval_queries, start=1):
    predicted, top_ids, _ = predict_outcome(q["query_text"], k=K, strategy="weighted")
    gt = q.get("ground_truth_case_id", "")
    actual = case_solutions.get(gt, "") if gt else ""

    rows.append({
        "query_id": q["query_id"],
        "predicted_solution": predicted,
        "top_5_case_ids": "|".join(map(str, top_ids)),
    })

    print(f"  {i}. {q['query_id']}")
    print(f"     predicted: {predicted[:140]}")
    if actual:
        print(f"     actual   : {actual[:140]}")
    print(f"     top cases: {' | '.join(map(str, top_ids))}")

predictions_path = f"{RESULTS_DIR}/predictions.csv"
pd.DataFrame(rows).to_csv(predictions_path, index=False, encoding="utf-8-sig")
print(f"\nPredictions saved to: {predictions_path}")
print("Siap untuk Tahap 5.")
