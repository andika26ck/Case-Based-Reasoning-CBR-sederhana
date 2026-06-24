"""
Tahap 3: Case Retrieval
Tujuan: Menemukan kasus lama yang paling mirip dengan query kasus baru
Pendekatan: 
  1. TF-IDF + Cosine Similarity
  2. BERT Embedding (Lightweight)
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SETUP
# ============================================================================

DATA_DIR = Path("data/processed")
MODELS_DIR = Path("data/models")
MODELS_DIR.mkdir(exist_ok=True)

cases = pd.read_csv(DATA_DIR / "cases.csv")
print(f"[1] Loaded {len(cases)} cases")

# ============================================================================
# 1. REPRESENTASI VEKTOR (TF-IDF)
# ============================================================================

# Gabungkan text untuk vectorization
cases['combined_text'] = (
    cases['facts_summary'].fillna('') + ' ' + 
    cases['legal_reasoning_summary'].fillna('')
).str.strip()

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.8
)

X = vectorizer.fit_transform(cases['combined_text'])
y = cases['jenis_perkara']

print(f"[2] TF-IDF vectorization: {X.shape} (samples × features)")
print(f"    Vocabulary size: {len(vectorizer.get_feature_names_out())}")
print(f"    Unique case types: {y.nunique()}")

# ============================================================================
# 2. SPLITTING DATA (80:20)
# ============================================================================

indices = np.arange(len(cases))
idx_train, idx_test = train_test_split(
    indices,
    test_size=0.2,
    random_state=42
)

X_train = X[idx_train]
X_test = X[idx_test]

print(f"[3] Train-Test Split (80:20)")
print(f"    Train: {X_train.shape[0]} cases (retrieval database)")
print(f"    Test:  {X_test.shape[0]} cases (query evaluation)")

# ============================================================================
# 3. BERT EMBEDDING (Lightweight)
# ============================================================================

print("[3b] Loading BERT model (sentence-transformers)...")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("    Installing sentence-transformers...")
    import subprocess
    subprocess.check_call(["pip", "install", "sentence-transformers", "-q"])
    from sentence_transformers import SentenceTransformer

# Load model
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

print(f"    Model: all-MiniLM-L6-v2 (lightweight)")
print(f"    Embedding dimension: 384")

# Encode training cases (convert to string, handle NaN)
bert_train_texts = cases.iloc[idx_train]['facts_summary'].fillna('').astype(str).tolist()
bert_train_embeddings = bert_model.encode(bert_train_texts, show_progress_bar=False)

print(f"    BERT embeddings: {bert_train_embeddings.shape}")

# ============================================================================
# 3c. COMPUTING SIMILARITY (BOTH METHODS)
# ============================================================================

print("[4] Computing similarity metrics...")
# Pre-compute all similarities untuk efficient retrieval

# TF-IDF similarities
tfidf_similarities = cosine_similarity(X, X_train)

# BERT similarities (convert all text to string, handle NaN)
bert_all_texts = cases['facts_summary'].fillna('').astype(str).tolist()
bert_all_embeddings = bert_model.encode(bert_all_texts, show_progress_bar=False)
bert_similarities = cosine_similarity(bert_all_embeddings, bert_train_embeddings)

print(f"    TF-IDF similarity matrix: {tfidf_similarities.shape}")
print(f"    BERT similarity matrix: {bert_similarities.shape}")

# ============================================================================
# 4. EVALUASI MODEL (Retrieval Performance)
# ============================================================================

def evaluate_retrieval(similarity_matrix, method_name, k_values=[1, 3, 5]):
    """Hitung retrieval performance metrics"""
    results = {}
    
    for k in k_values:
        hit = 0
        for test_idx in idx_test:
            # Get similarity untuk test case vs semua training
            similarities = similarity_matrix[test_idx]
            top_k_train_idx = np.argsort(similarities)[-k:]
            
            # Check apakah ada same case type dalam top-k
            test_type = cases.iloc[test_idx]['jenis_perkara']
            top_k_types = cases.iloc[idx_train[top_k_train_idx]]['jenis_perkara'].values
            
            if test_type in top_k_types:
                hit += 1
        
        recall_at_k = hit / len(idx_test)
        results[f'Recall@{k}'] = recall_at_k
    
    return results

# Evaluate both methods
tfidf_results = evaluate_retrieval(tfidf_similarities, "TF-IDF", k_values=[1, 3, 5])
bert_results = evaluate_retrieval(bert_similarities, "BERT", k_values=[1, 3, 5])

print(f"[5] Retrieval Evaluation")
print(f"    TF-IDF + Cosine Similarity:")
for metric, score in tfidf_results.items():
    print(f"      {metric}: {score:.4f}")

print(f"    BERT Embedding + Cosine Similarity:")
for metric, score in bert_results.items():
    print(f"      {metric}: {score:.4f}")

# ============================================================================
# 5. SIMPAN MODEL & VECTORIZER
# ============================================================================

joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.pkl")
joblib.dump(X_train, MODELS_DIR / "tfidf_train_vectors.pkl")
joblib.dump(bert_model, MODELS_DIR / "bert_model.pkl")
joblib.dump(bert_train_embeddings, MODELS_DIR / "bert_train_embeddings.npy")

print(f"[6] Models saved:")
print(f"    - {MODELS_DIR}/tfidf_vectorizer.pkl")
print(f"    - {MODELS_DIR}/tfidf_train_vectors.pkl")
print(f"    - {MODELS_DIR}/bert_model.pkl")
print(f"    - {MODELS_DIR}/bert_train_embeddings.npy")

# ============================================================================
# 6. FUNGSI RETRIEVAL
# ============================================================================

def retrieve_similar_cases(query_text, k=5, method='tfidf', include_scores=True):
    """
    Retrieve k cases paling mirip dengan query text
    
    Args:
        query_text: Text input dari kasus baru
        k: Jumlah cases yang di-retrieve
        method: 'tfidf' atau 'bert'
        include_scores: Include similarity scores
    
    Returns:
        DataFrame dengan top-k similar cases dan similarity scores
    """
    if method == 'tfidf':
        # TF-IDF retrieval
        query_vec = vectorizer.transform([query_text])
        similarities = cosine_similarity(query_vec, X_train)[0]
    else:
        # BERT retrieval (ensure string input)
        query_text_clean = str(query_text).strip()
        query_embedding = bert_model.encode([query_text_clean])[0]
        similarities = cosine_similarity([query_embedding], bert_train_embeddings)[0]
    
    # Get top-k indices (sorted descending)
    top_indices = np.argsort(similarities)[-k:][::-1]
    
    # Map back ke original case indices
    retrieved_case_indices = idx_train[top_indices]
    
    result = cases.iloc[retrieved_case_indices][
        ['case_id', 'no_perkara', 'jenis_perkara', 'tanggal_register', 'facts_summary']
    ].copy()
    
    if include_scores:
        result['similarity_score'] = similarities[top_indices]
        result['method'] = method.upper()
    
    return result


def retrieve(query: str, k: int = 5) -> list[str]:
    """
    Retrieve top-k case_id paling mirip untuk sebuah query.
    Default memakai BERT karena dipakai juga oleh Tahap 4.
    """
    query_text = str(query).strip()
    if not query_text:
        return []

    retrieved = retrieve_similar_cases(query_text, k=k, method='bert', include_scores=False)
    return retrieved['case_id'].tolist()


def retrieve_similar_cases_from_file(case_id, k=5, method='tfidf', field='facts_summary'):
    """
    Retrieve cases mirip dengan case yang ada di database
    
    Args:
        case_id: ID dari case dalam database
        k: Jumlah cases untuk di-retrieve
        method: 'tfidf' atau 'bert'
        field: Field mana yang di-gunakan
    
    Returns:
        DataFrame dengan top-k similar cases
    """
    case = cases[cases['case_id'] == case_id]
    if case.empty:
        print(f"Case {case_id} not found")
        return pd.DataFrame()
    
    query_text = case[field].iloc[0]
    return retrieve_similar_cases(query_text, k=k, method=method, include_scores=True)


# ============================================================================
# 7. TEST RETRIEVAL
# ============================================================================

print("\n[7] Testing Retrieval Function")

# Sample query dari test data (ensure valid text)
valid_test_indices = [i for i in idx_test if isinstance(cases.iloc[i]['facts_summary'], str)]
if not valid_test_indices:
    valid_test_indices = [idx_test[0]]
    
sample_test_idx = valid_test_indices[0]
sample_case = cases.iloc[sample_test_idx]
sample_text = str(sample_case['facts_summary']).strip()

print(f"\n    Query Case: {sample_case['no_perkara']}")
print(f"    Type: {sample_case['jenis_perkara']}")
print(f"    Text preview: {sample_text[:80]}...")

# Test TF-IDF
retrieved_tfidf = retrieve_similar_cases(sample_text, k=3, method='tfidf', include_scores=True)
print(f"\n    Top 3 Similar Cases (TF-IDF):")
for idx, row in retrieved_tfidf.iterrows():
    print(f"      - {row['no_perkara']} (similarity: {row['similarity_score']:.4f})")

# Test BERT
retrieved_bert = retrieve_similar_cases(sample_text, k=3, method='bert', include_scores=True)
print(f"\n    Top 3 Similar Cases (BERT):")
for idx, row in retrieved_bert.iterrows():
    print(f"      - {row['no_perkara']} (similarity: {row['similarity_score']:.4f})")

# ============================================================================
# 8. SIMPAN METADATA
# ============================================================================

retrieval_metadata = {
    "timestamp": datetime.now().isoformat(),
    "approaches": {
        "tfidf": {
            "method": "TF-IDF + Cosine Similarity",
            "features": int(X.shape[1]),
            "metrics": tfidf_results
        },
        "bert": {
            "method": "BERT (all-MiniLM-L6-v2) + Cosine Similarity",
            "dimension": 384,
            "metrics": bert_results
        }
    },
    "total_cases": len(cases),
    "train_cases": len(idx_train),
    "test_cases": len(idx_test),
    "unique_case_types": int(y.nunique()),
    "test_case_sample": {
        "case_id": sample_case['case_id'],
        "no_perkara": sample_case['no_perkara'],
        "retrieved_top_3_tfidf": retrieved_tfidf[['no_perkara', 'similarity_score']].to_dict(orient='records'),
        "retrieved_top_3_bert": retrieved_bert[['no_perkara', 'similarity_score']].to_dict(orient='records')
    }
}

with open(DATA_DIR / "retrieval_metadata.json", "w") as f:
    json.dump(retrieval_metadata, f, indent=2, ensure_ascii=False)

print(f"\n[8] Metadata saved: data/processed/retrieval_metadata.json")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("TAHAP 3 COMPLETE: Case Retrieval (TF-IDF & BERT)")
print("="*70)

print("\n  [1] TF-IDF + Cosine Similarity")
print(f"      Features: {X.shape[1]}")
for metric, score in tfidf_results.items():
    print(f"      {metric}: {score:.4f}")

print("\n  [2] BERT Embedding (all-MiniLM-L6-v2) + Cosine Similarity")
print(f"      Dimension: 384")
for metric, score in bert_results.items():
    print(f"      {metric}: {score:.4f}")

print(f"\n  [3] Data Configuration")
print(f"      Train: {len(idx_train)} cases")
print(f"      Test: {len(idx_test)} cases (80:20 split)")
print(f"      Unique types: {int(y.nunique())}")

print(f"\n  [4] Models saved: data/models/")
print(f"      - tfidf_vectorizer.pkl")
print(f"      - tfidf_train_vectors.pkl")
print(f"      - bert_model.pkl")
print(f"      - bert_train_embeddings.npy")

print(f"\n  [5] Ready to use:")
print(f"      retrieve_similar_cases(query_text, k=5, method='tfidf')")
print(f"      retrieve_similar_cases(query_text, k=5, method='bert')")
print(f"      retrieve_similar_cases_from_file(case_id, k=5, method='bert')")

print("\n" + "="*70)
print("Ready untuk Tahap 4!")
print("="*70)
