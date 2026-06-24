"""
Tahap 4: Case Solution Reuse
Tujuan: Gunakan putusan lama sebagai dasar pencarian untuk kasus baru
Pendekatan: BERT retrieval + majority vote / weighted similarity
"""

from pathlib import Path
import json
from collections import Counter, defaultdict

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "sentence-transformers", "-q"])
    from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "data" / "models"
EVAL_DIR = BASE_DIR / "data" / "eval"
RESULTS_DIR = BASE_DIR / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

cases = pd.read_csv(DATA_DIR / "cases.csv")
train_indices = joblib.load(MODELS_DIR / "train_indices.json")
bert_train_embeddings = joblib.load(MODELS_DIR / "bert_train_embeddings.npy")

try:
    bert_model = joblib.load(MODELS_DIR / "bert_model.pkl")
except Exception:
    bert_model = SentenceTransformer("all-MiniLM-L6-v2")

train_cases = cases.iloc[train_indices].reset_index(drop=True)
train_case_ids = train_cases["case_id"].tolist()
train_embeddings = np.asarray(bert_train_embeddings)


def safe_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def extract_solution_text(row):
    for column in ["decision_summary", "dakwaan_utama", "legal_reasoning_summary"]:
        text = safe_text(getattr(row, column, ""))
        if text:
            return text
    return safe_text(getattr(row, "decision_type", ""))


case_solutions = {
    row.case_id: extract_solution_text(row)
    for row in cases.itertuples(index=False)
}


def retrieve_top_k(query: str, k: int = 5, method: str = "bert"):
    query_text = safe_text(query)
    if not query_text:
        return []

    query_embedding = bert_model.encode([query_text])[0]
    similarities = cosine_similarity([query_embedding], train_embeddings)[0]
    top_indices = np.argsort(similarities)[-k:][::-1]

    results = []
    for position in top_indices:
        case_id = train_case_ids[position]
        results.append(
            {
                "case_id": case_id,
                "similarity": float(similarities[position]),
                "solution": case_solutions.get(case_id, ""),
            }
        )
    return results


def choose_solution(top_k_cases, strategy: str = "weighted"):
    if not top_k_cases:
        return ""

    if strategy == "majority":
        counts = Counter()
        score_totals = defaultdict(float)
        first_seen = {}

        for rank, item in enumerate(top_k_cases):
            key = safe_text(item["solution"])
            if not key:
                continue
            counts[key] += 1
            score_totals[key] += item["similarity"]
            first_seen.setdefault(key, rank)

        if not counts:
            return safe_text(top_k_cases[0]["solution"])

        best_key = max(
            counts.keys(),
            key=lambda key: (counts[key], score_totals[key], -first_seen[key]),
        )
        return best_key

    score_totals = defaultdict(float)
    best_item = {}

    for item in top_k_cases:
        key = safe_text(item["solution"])
        if not key:
            continue
        score_totals[key] += item["similarity"]
        if key not in best_item or item["similarity"] > best_item[key]["similarity"]:
            best_item[key] = item

    if not score_totals:
        return safe_text(top_k_cases[0]["solution"])

    best_key = max(score_totals.keys(), key=lambda key: (score_totals[key], best_item[key]["similarity"]))
    return best_key


def predict_outcome(query: str, k: int = 5, strategy: str = "weighted", method: str = "bert"):
    top_k = retrieve_top_k(query, k=k, method=method)
    predicted_solution = choose_solution(top_k, strategy=strategy)
    top_5_case_ids = [item["case_id"] for item in top_k]
    return predicted_solution, top_5_case_ids, top_k


# Demo manual: pakai queries.json bila tersedia, kalau tidak fallback ke 5 sample kasus
queries_path = EVAL_DIR / "queries.json"
rows = []

if queries_path.exists():
    with open(queries_path, "r", encoding="utf-8") as f:
        eval_queries = json.load(f)
    print("[Tahap 4] Demo prediksi solusi dari data/eval/queries.json")
else:
    eval_queries = [
        {
            "query_id": row["case_id"],
            "query_text": safe_text(row["facts_summary"]),
            "ground_truth_case_id": row["case_id"],
        }
        for _, row in cases.sample(5, random_state=42).iterrows()
    ]
    print("[Tahap 4] Demo prediksi solusi dari 5 sample kasus")

for i, query_item in enumerate(eval_queries):
    query_id = query_item["query_id"]
    query_text = safe_text(query_item["query_text"])
    ground_truth_case_id = query_item.get("ground_truth_case_id", "")

    predicted_solution, top_case_ids, top_k = predict_outcome(query_text, k=5, strategy="weighted", method="bert")
    actual_solution = case_solutions.get(ground_truth_case_id, "") if ground_truth_case_id else ""

    rows.append(
        {
            "query_id": query_id,
            "predicted_solution": predicted_solution,
            "top_5_case_ids": "|".join(top_case_ids),
        }
    )

    print(f"  {i + 1}. {query_id}")
    if actual_solution:
        print(f"     predicted: {predicted_solution[:140]}")
        print(f"     actual:     {actual_solution[:140]}")
    else:
        print(f"     predicted: {predicted_solution[:140]}")
    print(f"     top cases:  {' | '.join(top_case_ids)}")

predictions_df = pd.DataFrame(rows)
predictions_path = RESULTS_DIR / "predictions.csv"
predictions_df.to_csv(predictions_path, index=False, encoding="utf-8-sig")

print(f"\nPredictions saved to: {predictions_path}")
print("Ready untuk Tahap 5.")


if __name__ == "__main__":
    pass
