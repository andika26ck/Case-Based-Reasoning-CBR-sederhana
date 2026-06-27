"""
02_representation.py
=====================
Tahap 2 (Case Representation) - Tugas CBR Penalaran Komputer
Domain: Perdata - Perbuatan Melawan Hukum (PMH) | Pengadilan: PN Bandung

Perbaikan dari versi sebelumnya:
- Tambah ekstraksi NAMA penggugat & tergugat (bukan hanya jumlahnya)
- Tambah kolom text_full (sesuai spesifikasi tugas)
- Kolom pasal diberi nama eksplisit sesuai spesifikasi
- Perbaikan ekstraksi ringkasan fakta agar lebih representatif

Output:
    data/processed/cases.csv    -> dataset terstruktur siap dipakai model
    data/processed/cases.json   -> format alternatif JSON
"""

import os
import re
import json
import logging

import pandas as pd

# ----------------------------------------------------------------------------
# KONFIGURASI
# ----------------------------------------------------------------------------
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
LOG_DIR = "logs"
METADATA_CSV = f"{PROCESSED_DIR}/metadata_raw.csv"
OUTPUT_CSV = f"{PROCESSED_DIR}/cases.csv"
OUTPUT_JSON = f"{PROCESSED_DIR}/cases.json"

MAX_TEXT_FULL_CHARS = 5000   # karakter maksimum untuk kolom text_full (preview)
                              # teks lengkap tetap bisa dibaca dari data/raw/*.txt

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=f"{LOG_DIR}/representation.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def log(msg):
    print(msg)
    logging.info(msg)


# ----------------------------------------------------------------------------
# HELPER: BACA TEKS RAW
# ----------------------------------------------------------------------------
def read_raw_text(case_id: str) -> str:
    path = os.path.join(RAW_DIR, f"{case_id}.txt")
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ----------------------------------------------------------------------------
# EKSTRAKSI NAMA PIHAK (penggugat & tergugat)
# Sesuai spesifikasi tugas: "Nama pihak Penggugat & Tergugat"
# ----------------------------------------------------------------------------
def extract_pihak(text: str) -> dict:
    """Ekstrak nama penggugat dan tergugat dari teks putusan.

    Pola yang dicari (sesuai format umum putusan perdata MA RI):
    - "PT. XXX sebagai Penggugat"
    - "melawan / dengan / lawan"  <-- pemisah
    - "CV. YYY sebagai Tergugat"

    Return dict: {
        'penggugat': 'nama penggugat',
        'tergugat': 'nama tergugat',
        'num_penggugat': int,
        'num_tergugat': int,
    }
    """
    result = {
        "penggugat": "",
        "tergugat": "",
        "num_penggugat": 0,
        "num_tergugat": 0,
    }

    text_lower = text.lower()

    # --- Cari blok identitas pihak (biasanya di awal putusan) ---
    # Pola: nama diikuti "selanjutnya disebut penggugat" atau "sebagai penggugat"
    # atau sebelum kata "melawan"/"lawan"
    pola_penggugat = [
        r"([A-Z][A-Za-z0-9\s\.,\-/]+?)\s*,?\s*selanjutnya\s+disebut\s+(?:sebagai\s+)?penggugat",
        r"([A-Z][A-Za-z0-9\s\.,\-/]+?)\s*(?:dalam hal ini diwakili\s+oleh\s+.+?)?\s*sebagai\s+penggugat",
        r"penggugat\s*(?:adalah|:)?\s*([A-Z][A-Za-z0-9\s\.,\-/]{3,60})",
    ]
    pola_tergugat = [
        r"([A-Z][A-Za-z0-9\s\.,\-/]+?)\s*,?\s*selanjutnya\s+disebut\s+(?:sebagai\s+)?tergugat",
        r"([A-Z][A-Za-z0-9\s\.,\-/]+?)\s*(?:dalam hal ini diwakili\s+oleh\s+.+?)?\s*sebagai\s+tergugat",
        r"tergugat\s*(?:adalah|:)?\s*([A-Z][A-Za-z0-9\s\.,\-/]{3,60})",
    ]

    # Cari nama penggugat (case-insensitive search, nama diambil dari match group)
    penggugat_names = []
    for pola in pola_penggugat:
        matches = re.findall(pola, text, re.IGNORECASE)
        penggugat_names.extend([m.strip() for m in matches if len(m.strip()) > 3])

    tergugat_names = []
    for pola in pola_tergugat:
        matches = re.findall(pola, text, re.IGNORECASE)
        tergugat_names.extend([m.strip() for m in matches if len(m.strip()) > 3])

    # Deduplikasi dan batasi panjang nama
    penggugat_names = list(dict.fromkeys(penggugat_names))[:5]
    tergugat_names = list(dict.fromkeys(tergugat_names))[:5]

    # Fallback: coba cari dari pola singkat jika tidak ditemukan
    if not penggugat_names:
        m = re.search(r"(?:penggugat|pemohon)\s*[:\-]?\s*([A-Z][A-Za-z\s\.]{3,50})",
                      text, re.IGNORECASE)
        if m:
            penggugat_names = [m.group(1).strip()]

    if not tergugat_names:
        m = re.search(r"(?:tergugat|termohon)\s*[:\-]?\s*([A-Z][A-Za-z\s\.]{3,50})",
                      text, re.IGNORECASE)
        if m:
            tergugat_names = [m.group(1).strip()]

    # Hitung jumlah (penggugat I, II, III dst atau berdasarkan temuan)
    n_peng = text_lower.count("penggugat") - text_lower.count("penggugat:")
    n_terg = text_lower.count("tergugat") - text_lower.count("tergugat:")
    n_peng = max(len(penggugat_names), min(n_peng // 3, 10))
    n_terg = max(len(tergugat_names), min(n_terg // 3, 10))

    result["penggugat"] = "; ".join(penggugat_names) if penggugat_names else "tidak terdeteksi"
    result["tergugat"] = "; ".join(tergugat_names) if tergugat_names else "tidak terdeteksi"
    result["num_penggugat"] = max(1, n_peng) if penggugat_names else 0
    result["num_tergugat"] = max(1, n_terg) if tergugat_names else 0

    return result


# ----------------------------------------------------------------------------
# EKSTRAKSI PASAL YANG DIRUJUK
# Sesuai spesifikasi tugas: kolom "pasal"
# ----------------------------------------------------------------------------
def extract_pasal(text: str) -> str:
    """Ekstrak pasal-pasal hukum yang dirujuk dalam putusan.
    Return: string dipisah koma, misal 'Pasal 1365 KUHPerdata, Pasal 1243 KUHPerdata'
    """
    patterns = [
        r"pasal\s+\d+(?:\s+ayat\s+\(\d+\))?\s+(?:jo\.?\s+pasal\s+\d+\s+)?[A-Za-z\.]+(?:\s+\d{4})?",
        r"pasal\s+\d+(?:\s*[,/]\s*\d+)*\s+[A-Za-z\.]+",
        r"pasal\s+\d+\s+huruf\s+[a-z]\s+[A-Za-z\.]+",
    ]

    found = []
    for pat in patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        found.extend([m.strip() for m in matches])

    # Normalisasi dan deduplikasi
    found_normalized = []
    seen = set()
    for p in found:
        key = re.sub(r"\s+", " ", p.lower()).strip()
        if key not in seen and len(key) > 5:
            seen.add(key)
            found_normalized.append(p.strip())

    return ", ".join(found_normalized[:10]) if found_normalized else ""


# ----------------------------------------------------------------------------
# EKSTRAKSI RINGKASAN FAKTA
# ----------------------------------------------------------------------------
def extract_ringkasan_fakta(text: str, max_chars: int = 800) -> str:
    """Ekstrak ringkasan fakta dari bagian 'duduk perkara' atau 'menimbang'.

    Prioritas:
    1. Paragraf setelah 'duduk perkara' atau 'posita'
    2. Paragraf pertama dari section 'menimbang'
    3. Fallback: 3 paragraf pertama teks setelah nomor putusan
    """
    # Coba cari section duduk perkara
    pola_sections = [
        r"duduk\s+perkara\s*\n+((?:.+\n?){1,15})",
        r"posita\s*\n+((?:.+\n?){1,15})",
        r"tentang\s+duduk\s+perkara\s*\n+((?:.+\n?){1,15})",
        r"kronologi\s*\n+((?:.+\n?){1,10})",
    ]

    for pola in pola_sections:
        m = re.search(pola, text, re.IGNORECASE | re.MULTILINE)
        if m:
            ringkasan = m.group(1).strip()
            if len(ringkasan.split()) > 30:
                return ringkasan[:max_chars]

    # Fallback: ambil dari 'menimbang' pertama
    m = re.search(r"menimbang\s*[:,]?\s*(.*?)(?=menimbang|mengadili|memutus)",
                  text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()[:max_chars]

    # Fallback terakhir: ambil paragraf setelah header putusan
    lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 50]
    return " ".join(lines[2:7])[:max_chars] if lines else text[:max_chars]


# ----------------------------------------------------------------------------
# EKSTRAKSI ARGUMEN HUKUM UTAMA
# ----------------------------------------------------------------------------
def extract_argumen_hukum(text: str, max_chars: int = 800) -> str:
    """Ekstrak argumen hukum utama dari section 'menimbang' atau 'pertimbangan hukum'."""
    pola_sections = [
        r"pertimbangan\s+hukum\s*\n+((?:.+\n?){1,20})",
        r"pertimbangan\s+majelis\s*\n+((?:.+\n?){1,20})",
    ]
    for pola in pola_sections:
        m = re.search(pola, text, re.IGNORECASE | re.MULTILINE)
        if m:
            argumen = m.group(1).strip()
            if len(argumen.split()) > 20:
                return argumen[:max_chars]

    # Kumpulkan semua kalimat 'menimbang'
    menimbang_parts = re.findall(
        r"menimbang\s*,?\s*bahwa\s+(.+?)(?=menimbang|mengadili|;|\n\n)",
        text, re.IGNORECASE | re.DOTALL
    )
    if menimbang_parts:
        combined = " | ".join([p.strip() for p in menimbang_parts[:3]])
        return combined[:max_chars]

    return ""


# ----------------------------------------------------------------------------
# EKSTRAKSI JENIS PUTUSAN (label untuk model)
# ----------------------------------------------------------------------------
def extract_decision_type(amar: str, text: str) -> str:
    """Klasifikasi jenis putusan dari kolom amar atau teks."""
    amar_lower = (amar or "").lower()
    text_lower = text.lower()

    sources = [amar_lower] + [text_lower]
    for src in sources:
        if any(k in src for k in ["dikabulkan untuk seluruhnya", "dikabulkan seluruhnya"]):
            return "dikabulkan_seluruhnya"
        if any(k in src for k in ["dikabulkan untuk sebagian", "dikabulkan sebagian"]):
            return "dikabulkan_sebagian"
        if any(k in src for k in ["ditolak", "tidak dapat diterima"]):
            return "ditolak"
        if "gugatan tidak dapat diterima" in src:
            return "niet_ontvankelijke_verklaard"
        if "gugur" in src:
            return "gugur"
        if "dikabulkan" in src:
            return "dikabulkan"

    return "lainnya"


# ----------------------------------------------------------------------------
# MAIN: PROSES SEMUA CASE
# ----------------------------------------------------------------------------
def main():
    log("=== TAHAP 2: Case Representation ===")
    log("Domain: Perdata - Perbuatan Melawan Hukum (PMH) | Pengadilan: PN Bandung")

    # Load metadata
    if not os.path.exists(METADATA_CSV):
        log(f"[FATAL] {METADATA_CSV} tidak ditemukan. Jalankan 01_scraping.py dulu.")
        return

    df_meta = pd.read_csv(METADATA_CSV)
    # Hanya proses dokumen valid
    df_valid = df_meta[df_meta["valid_min_200_kata"] == True].copy()
    log(f"Total dokumen valid yang akan diproses: {len(df_valid)}")

    records = []
    for i, row in enumerate(df_valid.itertuples(), start=1):
        case_id = row.case_id
        raw_path = os.path.join(RAW_DIR, f"{case_id}.txt")
        text = read_raw_text(case_id)

        if not text:
            log(f"  [{i}/{len(df_valid)}] SKIP {case_id}: file raw tidak ditemukan")
            continue

        log(f"  [{i}/{len(df_valid)}] memproses {case_id}...")

        # Ekstraksi pihak (nama penggugat & tergugat)
        pihak = extract_pihak(text)

        # Ekstraksi konten kunci
        ringkasan_fakta = extract_ringkasan_fakta(text)
        argumen_hukum = extract_argumen_hukum(text)
        pasal = extract_pasal(text)

        # Jenis putusan (label untuk model CBR)
        amar = str(row.amar) if hasattr(row, "amar") else ""
        decision_type = extract_decision_type(amar, text)

        # text_full: preview teks bersih (sesuai spesifikasi kolom)
        # Teks lengkap tetap ada di data/raw/{case_id}.txt
        text_full_preview = text[:MAX_TEXT_FULL_CHARS].replace("\n", " ")

        record = {
            # --- Identitas & metadata dasar ---
            "case_id": case_id,
            "no_perkara": getattr(row, "no_perkara", ""),
            "tanggal": getattr(row, "tanggal_register", ""),
            "jenis_perkara": getattr(row, "jenis_perkara", ""),
            "pengadilan": getattr(row, "pengadilan", ""),
            "hakim_ketua": getattr(row, "hakim_ketua", ""),
            "klasifikasi": getattr(row, "klasifikasi", ""),

            # --- Pihak (NAMA, bukan hanya jumlah) ---
            "penggugat": pihak["penggugat"],
            "tergugat": pihak["tergugat"],
            "num_penggugat": pihak["num_penggugat"],
            "num_tergugat": pihak["num_tergugat"],

            # --- Konten kunci ---
            "ringkasan_fakta": ringkasan_fakta,
            "argumen_hukum": argumen_hukum,
            "pasal": pasal,               # kolom eksplisit sesuai spesifikasi
            "amar_putusan": amar,
            "decision_type": decision_type,  # label untuk model

            # --- Feature engineering ---
            "jumlah_kata": getattr(row, "jumlah_kata", len(text.split())),
            "jumlah_token": getattr(row, "jumlah_token", 0),
            "panjang_ringkasan_fakta": len(ringkasan_fakta.split()),
            "panjang_argumen_hukum": len(argumen_hukum.split()),
            "jumlah_pasal_dirujuk": len([p for p in pasal.split(",") if p.strip()]),

            # --- text_full (sesuai spesifikasi kolom di tugas) ---
            "text_full": text_full_preview,
            "raw_file_path": raw_path,     # path ke teks lengkap
        }

        records.append(record)

    df_out = pd.DataFrame(records)

    # Simpan ke CSV dan JSON
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    df_out.to_json(OUTPUT_JSON, orient="records", force_ascii=False, indent=2)

    log(f"\n=== SELESAI: {len(df_out)} case berhasil direpresentasikan ===")
    log(f"Output CSV  -> {OUTPUT_CSV}")
    log(f"Output JSON -> {OUTPUT_JSON}")

    # Ringkasan distribusi decision_type (label)
    if "decision_type" in df_out.columns:
        log("\nDistribusi decision_type (label untuk model):")
        for dtype, count in df_out["decision_type"].value_counts().items():
            log(f"  {dtype}: {count} dokumen")

    # Cek kelengkapan pihak
    n_pihak_ok = (df_out["penggugat"] != "tidak terdeteksi").sum()
    log(f"\nNama penggugat berhasil diekstrak: {n_pihak_ok}/{len(df_out)} dokumen")
    n_tergugat_ok = (df_out["tergugat"] != "tidak terdeteksi").sum()
    log(f"Nama tergugat berhasil diekstrak : {n_tergugat_ok}/{len(df_out)} dokumen")


if __name__ == "__main__":
    main()
