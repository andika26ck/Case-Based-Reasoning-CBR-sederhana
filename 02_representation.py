"""
02_representation.py
====================
Tahap 2 (Case Representation) - Tugas CBR Penalaran Komputer
Domain: Perdata - Perbuatan Melawan Hukum (PMH)
Pengadilan: PN Bandung

Tujuan:
    Mengubah raw case data (metadata + teks mentah) menjadi representasi terstruktur
    yang siap untuk similarity matching di tahap 3.

Yang dikerjakan:
    1. Ekstraksi metadata dari metadata_raw.csv
    2. Ekstraksi konten kunci dari file data/raw/*.txt:
       - Bagian "DUDUK PERKARA" -> Fakta-fakta kunci kasus
       - Bagian "MENIMBANG" -> Argumen hukum & pertimbangan
       - Bagian "AMAR" -> Keputusan akhir
    3. Feature engineering:
       - Jumlah pihak (penggugat, tergugat, turut tergugat)
       - Durasi proses hukum (register date - court date)
       - Jenis putusan (Dikabulkan/Ditolak/Tidak Dapat Diterima/Sebagian)
       - Kehadiran frasa hukum penting (perbuatan melawan hukum, ganti rugi, dll)
       - Jumlah dokumen pendukung yang disebut
       - Panjang ringkasan
    4. Normalkan panjang teks untuk konsistensi

Output:
    data/processed/cases.csv -> representasi terstruktur semua kasus
    data/processed/cases.json -> format alternatif (optional)

Cara pakai:
    python 02_representation.py

"""

import os
import re
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

# ============================================================================
# KONFIGURASI
# ============================================================================
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
LOG_DIR = "logs"

METADATA_CSV = f"{PROCESSED_DIR}/metadata_raw.csv"
OUTPUT_CSV = f"{PROCESSED_DIR}/cases.csv"
OUTPUT_JSON = f"{PROCESSED_DIR}/cases.json"

# Konfigurasi ekstraksi
MAX_FACTS_LENGTH = 500          # Maksimal karakter untuk bagian "FAKTA KUNCI"
MAX_LEGAL_REASONING = 800       # Maksimal karakter untuk "ARGUMEN HUKUM"
MAX_DECISION_LENGTH = 400       # Maksimal karakter untuk "KEPUTUSAN"

# Setup logging
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=f"{LOG_DIR}/representation.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================================
# FUNGSI UTILITAS
# ============================================================================

def clean_text(text):
    """Normalisasi teks: strip, lowercase, whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    # Normalisasi multiple whitespace
    text = re.sub(r'\s+', ' ', text)
    return text


def extract_parties(text):
    """Ekstrak jumlah pihak dari teks."""
    penggugat = len(re.findall(r'\bpenggugat\s+[0-9]', text, re.IGNORECASE))
    tergugat = len(re.findall(r'\btergugat\s+[0-9]', text, re.IGNORECASE))
    turut_tergugat = len(re.findall(r'\bturut\s+tergugat\s+[0-9]', text, re.IGNORECASE))
    
    # Jika pattern ketat tidak ketemu, gunakan heuristik sederhana
    if penggugat == 0:
        penggugat = 1  # Minimal ada 1 penggugat
    if tergugat == 0:
        tergugat = 1   # Minimal ada 1 tergugat
    
    # Batasi maksimal 20 pihak per kategori (untuk menghindari overcounting)
    penggugat = min(penggugat, 20)
    tergugat = min(tergugat, 20)
    turut_tergugat = min(turut_tergugat, 10)
    
    return {
        "num_penggugat": penggugat,
        "num_tergugat": tergugat,
        "num_turut_tergugat": turut_tergugat,
        "total_pihak": penggugat + tergugat + turut_tergugat,
    }


def extract_decision_type(text):
    """
    Ekstrak tipe putusan dari teks amar.
    Return: "Dikabulkan", "Ditolak", "Tidak Dapat Diterima", "Sebagian", atau "Lainnya"
    """
    text_clean = text.lower()
    
    # Cek tipe putusan
    if re.search(r'tidak dapat diterima|niet ontvankelijke', text_clean):
        return "Tidak Dapat Diterima"
    elif re.search(r'dikabulkan', text_clean):
        if re.search(r'sebagian', text_clean):
            return "Dikabulkan Sebagian"
        return "Dikabulkan"
    elif re.search(r'ditolak', text_clean):
        return "Ditolak"
    elif re.search(r'sebagian', text_clean):
        return "Sebagian"
    else:
        return "Lainnya"


def extract_legal_keywords(text):
    """
    Deteksi kehadiran frasa hukum penting yang relevan dengan PMH.
    Return: dict dengan boolean flags dan frekuensi
    """
    text_clean = text.lower()
    
    keywords = {
        "has_perbuatan_melawan_hukum": bool(re.search(r'perbuatan melawan hukum|onrechmatiged', text_clean)),
        "has_ganti_rugi": bool(re.search(r'ganti rugi|ganti\s+kerugian', text_clean)),
        "has_tanah": bool(re.search(r'tanah|sertifikat|hak milik', text_clean)),
        "has_kontrak": bool(re.search(r'kontrak|perjanjian|akta|notaris', text_clean)),
        "has_perbuatan_tercela": bool(re.search(r'perbuatan tercela|onzedelijk', text_clean)),
        "has_dokumen": bool(re.search(r'dokumen|surat|bukti', text_clean)),
        "has_mediasi": bool(re.search(r'mediasi|perdamaian|perma', text_clean)),
    }
    
    return keywords


def extract_important_pasal(text):
    """
    Ekstrak nomor pasal yang direferensikan dalam putusan.
    Return: list pasal yang paling sering disebut
    """
    text_clean = text.lower()
    # Pattern: "Pasal 123", "pasal 123 KUHPerdata", dll
    pasal_matches = re.findall(r'pasal\s+(\d+(?:\s+[a-z\.]+)?)', text_clean)
    
    # Ambil yang unik dan hitung frekuensi
    pasal_freq = {}
    for p in pasal_matches:
        pasal_freq[p] = pasal_freq.get(p, 0) + 1
    
    # Sort by frequency dan ambil top 5
    top_pasal = sorted(pasal_freq.items(), key=lambda x: x[1], reverse=True)[:5]
    return [p[0] for p in top_pasal]


def extract_key_facts(text):
    """
    Ekstrak fakta kunci: barang bukti, dakwaan, objek sengketa.
    Return: structured dict dengan fakta-fakta penting
    """
    text_clean = text.lower()
    
    facts = {
        "bukti_utama": [],
        "objek_sengketa": "",
        "dakwaan_utama": "",
        "kerugian_dituntut": "",
    }
    
    # ---- Ekstrak bukti ----
    bukti_patterns = [
        r'(?:bukti|dokumen|surat|akta|sertifikat)[^.!?]*?(?:nomor|no\.?)\s*[\d\/\w\-]+',
        r'(?:sertifikat hak milik|shm|ahu)[^.!?]*?(?:nomor|no\.?)\s*[\d\/]+',
        r'(?:akta|notaris|ppat)[^.!?]*?(?:nomor|no\.?)\s*[\d\/\w\-]+',
    ]
    
    for pattern in bukti_patterns:
        matches = re.findall(pattern, text_clean)
        facts["bukti_utama"].extend(matches[:3])  # Ambil max 3 per pattern
    
    # ---- Ekstrak objek sengketa ----
    if re.search(r'tanah.*?seluas\s+[\d.]+\s*(?:m2|ha)', text_clean):
        match = re.search(r'(tanah[^.!?]*?seluas\s+[\d.]+\s*(?:m2|ha)[^.!?]*)', text_clean)
        if match:
            facts["objek_sengketa"] = match.group(1)[:200]
    
    # ---- Ekstrak dakwaan utama ----
    dakwaan_patterns = [
        r'(?:penggugat|pemohon)[^.]*?(?:mendalilkan|mengajukan gugatan|menggugat)[^.]*?bahwa[^.]*?[.!?]',
        r'(?:perbuatan melawan hukum|perlanggarah|pelanggaran)[^.]*?[.!?]',
    ]
    
    for pattern in dakwaan_patterns:
        match = re.search(pattern, text_clean)
        if match:
            facts["dakwaan_utama"] = match.group(0)[:250]
            break
    
    # ---- Ekstrak kerugian yang dituntut ----
    kerugian_match = re.search(
        r'(?:kerugian|ganti rugi|penggantian)[^.]*?(?:rp\.?\s*[\d.,]+[^.]*?)?(?:miliar|juta|ribu)?',
        text_clean
    )
    if kerugian_match:
        facts["kerugian_dituntut"] = kerugian_match.group(0)[:200]
    
    return facts


def generate_qa_pairs(case_data):
    """
    Generate simple QA pairs dari case representation.
    Ini berguna untuk information retrieval nantinya.
    
    Return: list of dict dengan "question" dan "answer"
    """
    qa_pairs = []
    
    # QA 1: Siapa pihak-pihak yang terlibat?
    qa_pairs.append({
        "question": "siapa saja pihak yang terlibat dalam perkara ini?",
        "answer": f"{case_data['num_penggugat']} penggugat, {case_data['num_tergugat']} tergugat, "
                  f"{case_data['num_turut_tergugat']} turut tergugat"
    })
    
    # QA 2: Berapa lama proses perkara?
    if case_data.get('process_duration_days'):
        days = case_data['process_duration_days']
        months = round(days / 30, 1)
        qa_pairs.append({
            "question": "berapa lama durasi proses perkara ini?",
            "answer": f"{days} hari ({months} bulan)"
        })
    
    # QA 3: Apa jenis putusan?
    qa_pairs.append({
        "question": "bagaimana hasil putusan dalam perkara ini?",
        "answer": case_data.get('decision_type', 'Tidak diketahui')
    })
    
    # QA 4: Apakah melibatkan objek tanah?
    qa_pairs.append({
        "question": "apakah perkara ini melibatkan masalah pertanahan?",
        "answer": "ya" if case_data.get('has_tanah') else "tidak"
    })
    
    # QA 5: Apakah melibatkan ganti rugi?
    qa_pairs.append({
        "question": "apakah ada tuntutan ganti rugi dalam perkara ini?",
        "answer": "ya" if case_data.get('has_ganti_rugi') else "tidak"
    })
    
    # QA 6: Nomor perkara?
    qa_pairs.append({
        "question": "berapa nomor perkara ini?",
        "answer": case_data.get('no_perkara', '')
    })
    
    return qa_pairs


def extract_section(text, section_name):
    """
    Ekstrak bagian tertentu dari teks putusan.
    
    section_name: "duduk_perkara", "menimbang", "amar"
    Return: teks bagian (dengan panjang terbatas)
    """
    text_lower = text.lower()
    
    # Tentukan pattern awal dan akhir section
    if section_name == "duduk_perkara":
        start_pattern = r'(?:tentang\s+)?duduk\s+perkara|posita'
        end_pattern = r'(?:menimbang|pertimbangan|pengadilan negeri|berdasarkan perma)'
        max_len = MAX_FACTS_LENGTH
    elif section_name == "menimbang":
        start_pattern = r'menimbang|pertimbangan|pengadilan negeri tersebut'
        end_pattern = r'(?:amar|oleh karena itu|putusan|mengadili)'
        max_len = MAX_LEGAL_REASONING
    elif section_name == "amar":
        start_pattern = r'(?:amar|mengadili|p\s*u\s*t\s*u\s*s\s*a\s*n|dalam perkara|dalam konvensi)'
        end_pattern = r'(?:atas segala penetapan|demikian putusan|itulah putusannya|biaya perkara|rp\.)'
        max_len = MAX_DECISION_LENGTH
    else:
        return ""
    
    # Cari posisi awal section
    match_start = re.search(start_pattern, text_lower)
    if not match_start:
        return ""
    
    start_idx = match_start.start()
    
    # Cari posisi akhir section
    match_end = re.search(end_pattern, text_lower[match_start.end():])
    if match_end:
        end_idx = match_start.end() + match_end.start()
    else:
        end_idx = len(text)
    
    section_text = text[start_idx:end_idx].strip()
    
    # Bersihkan dan potong
    section_text = clean_text(section_text)
    section_text = section_text[:max_len]
    
    return section_text


def count_referenced_documents(text):
    """Hitung jumlah dokumen yang direferensikan dalam putusan."""
    # Pattern untuk referensi dokumen (nomor, tanggal, nama)
    patterns = [
        r'\bno\.?\s*\d+',           # No. 123
        r'\bsurat\s+\w+',           # Surat [nama dokumen]
        r'\bvide\.',                # Vide (latin untuk "lihat")
        r'\bpasal\s+\d+',           # Pasal 123
        r'\bpp\s+no\.?\s*\d+',      # PP No. 123
        r'\bperma\s+no\.?\s*\d+',   # Perma No. 123
        r'\byurisprudensi\s+',      # Yurisprudensi
    ]
    
    count = 0
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        count += len(matches)
    
    # Batasi dan normalisasi (jangan terlalu tinggi karena ada repetisi)
    return min(count // 5, 20)  # Bagi 5 karena ada banyak repetisi


def parse_date(date_str):
    """Parse tanggal dari berbagai format."""
    if not date_str or pd.isna(date_str):
        return None
    
    date_str = str(date_str).strip()
    
    # List format yang umum
    formats = [
        "%d %B %Y",        # 15 April 2026
        "%d %b %Y",        # 15 Apr 2026
        "%d-%m-%Y",        # 15-04-2026
        "%Y-%m-%d",        # 2026-04-15
        "%d/%m/%Y",        # 15/04/2026
        "%d Januari %Y",
        "%d Februari %Y",
        "%d Maret %Y",
        "%d April %Y",
        "%d Mei %Y",
        "%d Juni %Y",
        "%d Juli %Y",
        "%d Agustus %Y",
        "%d September %Y",
        "%d Oktober %Y",
        "%d Nopember %Y",
        "%d November %Y",
        "%d Desember %Y",
    ]
    
    for fmt in formats:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except:
            continue
    
    # Fallback: coba parsing otomatis pandas
    try:
        return pd.to_datetime(date_str)
    except:
        return None


def calculate_process_duration(tanggal_register, tanggal_musyawarah):
    """Hitung durasi proses (hari) antara tanggal register dan musyawarah."""
    try:
        date_reg = parse_date(tanggal_register)
        date_mus = parse_date(tanggal_musyawarah)
        
        if date_reg and date_mus:
            duration = (date_mus - date_reg).days
            return max(duration, 0)  # Tidak boleh negatif
    except:
        pass
    
    return None


def extract_case_representation(case_id, row, case_text):
    """
    Extract full case representation untuk satu kasus.
    
    Args:
        case_id: ID kasus
        row: Row dari metadata_raw.csv (pandas Series)
        case_text: Teks mentah kasus dari file .txt
    
    Returns:
        dict dengan representasi terstruktur kasus
    """
    
    # ---- Ekstrak metadata dasar ----
    representation = {
        "case_id": case_id,
        "no_perkara": row.get("no_perkara", ""),
        "jenis_perkara": row.get("jenis_perkara", ""),
        "tanggal_register": row.get("tanggal_register", ""),
        "tanggal_musyawarah": row.get("tanggal_musyawarah", ""),
    }
    
    # ---- Ekstrak bagian-bagian kunci ----
    facts = extract_section(case_text, "duduk_perkara")
    legal_reasoning = extract_section(case_text, "menimbang")
    decision = extract_section(case_text, "amar")
    
    representation.update({
        "facts_summary": facts[:MAX_FACTS_LENGTH],
        "legal_reasoning_summary": legal_reasoning[:MAX_LEGAL_REASONING],
        "decision_summary": decision[:MAX_DECISION_LENGTH],
    })
    
    # ---- Feature engineering: Ekstrak fakta kunci ----
    key_facts = extract_key_facts(case_text)
    representation["key_facts_bukti"] = "|".join(key_facts["bukti_utama"][:3]) if key_facts["bukti_utama"] else ""
    representation["objek_sengketa"] = key_facts["objek_sengketa"][:300]
    representation["dakwaan_utama"] = key_facts["dakwaan_utama"][:300]
    
    # ---- Feature engineering: Ekstrak pasal-pasal penting ----
    top_pasal = extract_important_pasal(case_text)
    representation["top_pasal_referenced"] = "|".join(top_pasal) if top_pasal else ""
    
    # ---- Feature engineering: Pihak ----
    party_features = extract_parties(case_text)
    representation.update(party_features)
    
    # ---- Feature engineering: Jenis putusan ----
    decision_type = extract_decision_type(decision)
    representation["decision_type"] = decision_type
    
    # ---- Feature engineering: Keyword hukum ----
    keywords = extract_legal_keywords(case_text)
    representation.update(keywords)
    
    # ---- Feature engineering: Jumlah dokumen ----
    representation["num_referenced_docs"] = count_referenced_documents(case_text)
    
    # ---- Feature engineering: Durasi proses ----
    duration = calculate_process_duration(
        row.get("tanggal_register"),
        row.get("tanggal_musyawarah")
    )
    representation["process_duration_days"] = duration
    
    # ---- Feature engineering: Panjang dokumen ----
    representation["doc_length_words"] = row.get("jumlah_kata", 0)
    representation["doc_length_tokens"] = row.get("jumlah_token", 0)
    representation["text_completeness"] = row.get("keutuhan_teks", 0.0)
    
    # ---- Feature engineering: QA Pairs ----
    qa_pairs = generate_qa_pairs(representation)
    representation["qa_pairs_json"] = json.dumps(qa_pairs, ensure_ascii=False)
    
    # ---- Metadata tambahan ----
    representation["pengadilan"] = row.get("pengadilan", "PN Bandung")
    representation["hakim_ketua"] = row.get("hakim_ketua", "")
    representation["klasifikasi"] = row.get("klasifikasi", "")
    representation["kata_kunci"] = row.get("kata_kunci", "")
    
    return representation


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def main():
    logger.info("=" * 80)
    logger.info("Tahap 2: Case Representation - START")
    logger.info("=" * 80)
    
    # ---- Baca metadata ----
    print("[1] Membaca metadata_raw.csv...")
    try:
        metadata_df = pd.read_csv(METADATA_CSV)
        logger.info(f"Metadata loaded: {len(metadata_df)} kasus")
        print(f"    ✓ Loaded {len(metadata_df)} cases")
    except Exception as e:
        logger.error(f"Error loading metadata: {e}")
        print(f"    ✗ Error: {e}")
        return
    
    # ---- Proses setiap kasus ----
    print(f"[2] Memproses representasi kasus ({len(metadata_df)} kasus)...")
    
    representations = []
    errors = []
    
    for idx, (_, row) in enumerate(metadata_df.iterrows()):
        case_id = row.get("case_id", f"case_{idx}")
        
        # Baca file teks kasus
        case_file = os.path.join(RAW_DIR, f"{case_id}.txt")
        
        if not os.path.exists(case_file):
            msg = f"Case file not found: {case_file}"
            logger.warning(msg)
            errors.append({"case_id": case_id, "error": msg})
            continue
        
        try:
            with open(case_file, 'r', encoding='utf-8') as f:
                case_text = f.read()
        except Exception as e:
            msg = f"Error reading case file {case_id}: {e}"
            logger.error(msg)
            errors.append({"case_id": case_id, "error": msg})
            continue
        
        # Extract representasi
        try:
            rep = extract_case_representation(case_id, row, case_text)
            representations.append(rep)
        except Exception as e:
            msg = f"Error extracting representation for {case_id}: {e}"
            logger.error(msg)
            errors.append({"case_id": case_id, "error": msg})
            continue
        
        # Progress
        if (idx + 1) % 10 == 0:
            print(f"    Processed {idx + 1}/{len(metadata_df)} cases...")
    
    if not representations:
        print("    ✗ No cases successfully processed!")
        logger.error("No cases successfully processed")
        return
    
    print(f"    ✓ Successfully processed {len(representations)} cases")
    
    # ---- Simpan ke CSV ----
    print(f"[3] Menyimpan ke {OUTPUT_CSV}...")
    try:
        df_cases = pd.DataFrame(representations)
        df_cases.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
        logger.info(f"Cases representation saved to {OUTPUT_CSV}")
        print(f"    ✓ Saved {len(df_cases)} cases to CSV")
    except Exception as e:
        logger.error(f"Error saving CSV: {e}")
        print(f"    ✗ Error: {e}")
        return
    
    # ---- Simpan ke JSON (optional) ----
    print(f"[4] Menyimpan ke {OUTPUT_JSON}...")
    try:
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(representations, f, ensure_ascii=False, indent=2)
        logger.info(f"Cases representation saved to {OUTPUT_JSON}")
        print(f"    ✓ Saved {len(representations)} cases to JSON")
    except Exception as e:
        logger.error(f"Error saving JSON: {e}")
        print(f"    ✗ Error: {e}")
    
    # ---- Summary dan error report ----
    print(f"\n[5] Summary:")
    print(f"    Total cases processed: {len(representations)}")
    print(f"    Total errors: {len(errors)}")
    
    if errors:
        print(f"\n[6] Error cases ({len(errors)}):")
        for err in errors[:5]:  # Tampilkan 5 error pertama
            print(f"    - {err['case_id']}: {err['error']}")
        if len(errors) > 5:
            print(f"    ... and {len(errors) - 5} more errors")
    
    # ---- Statistics ----
    print(f"\n[7] Representation Statistics:")
    print(f"    Average doc length: {df_cases['doc_length_words'].mean():.0f} words")
    print(f"    Average parties: {df_cases['total_pihak'].mean():.1f} people")
    print(f"    Average process duration: {df_cases['process_duration_days'].mean():.0f} days")
    print(f"    Decision types:")
    for dec_type, count in df_cases['decision_type'].value_counts().items():
        print(f"      - {dec_type}: {count}")
    
    # ---- Kolom baru yang ditambahkan ----
    print(f"\n[8] Enhanced Features:")
    print(f"    ✓ Key facts extracted (bukti utama, objek sengketa, dakwaan)")
    print(f"    ✓ Top pasal referenced: {df_cases['top_pasal_referenced'].str.len().mean():.0f} chars avg")
    print(f"    ✓ QA pairs generated: {df_cases['qa_pairs_json'].str.len().mean():.0f} chars avg")
    
    # ---- Sample QA pairs untuk 1 kasus ----
    if len(df_cases) > 0:
        sample_qa = json.loads(df_cases.iloc[0]['qa_pairs_json'])
        print(f"\n[9] Sample QA Pairs (Case 1):")
        for i, qa in enumerate(sample_qa[:3], 1):
            print(f"    Q{i}: {qa['question']}")
            print(f"    A{i}: {qa['answer'][:80]}...")
    
    logger.info("=" * 80)
    logger.info("Tahap 2: Case Representation - COMPLETED")
    logger.info("=" * 80)
    print("\n✓ Case Representation tahap selesai!")


if __name__ == "__main__":
    main()
