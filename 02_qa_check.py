"""
02_qa_check.py
===============
QA / Sanity-Check hasil Tahap 1 (Membangun Case Base) - Tugas CBR Penalaran Komputer
Domain: Perdata - Perbuatan Melawan Hukum (PMH) | Pengadilan: PN Bandung

Tujuan:
    Verifikasi kualitas & kelengkapan hasil 01_scraping.py SEBELUM lanjut ke
    Tahap 2 (Case Representation). Script ini TIDAK scraping apa pun, cuma
    membaca file-file yang sudah ada di folder data/.

Yang dicek:
    1. Jumlah dokumen valid vs syarat minimum tugas (>=90 target, >=30 minimum tugas)
    2. Konsistensi antar folder (pdf vs raw vs raw_extracted vs tokens vs csv)
    3. Distribusi panjang dokumen (jumlah kata) -> deteksi outlier/dokumen kosong
    4. Distribusi skor keutuhan teks (validasi 80%)
    5. Duplikasi (no_perkara yang sama muncul lebih dari sekali)
    6. Kelengkapan metadata (kolom yang sering kosong)
    7. Sample acak buat dibaca manual (spot-check kualitas cleaning)

Cara pakai:
    python 02_qa_check.py

Output:
    - Ringkasan QA dicetak ke terminal
    - logs/qa_report.txt -> laporan lengkap (bisa dilampirkan ke laporan tugas)
    - data/processed/qa_flagged.csv -> daftar case_id yang butuh dicek manual
"""

import os
import re
import csv
import json
import random

import pandas as pd

# ----------------------------------------------------------------------------
# KONFIGURASI (samakan dengan 01_scraping.py)
# ----------------------------------------------------------------------------
RAW_DIR = "data/raw"
RAW_EXTRACTED_DIR = "data/raw_extracted"
TOKENS_DIR = "data/tokens"
PDF_DIR = "data/pdf"
PROCESSED_DIR = "data/processed"
LOG_DIR = "logs"
METADATA_CSV = f"{PROCESSED_DIR}/metadata_raw.csv"

MIN_DOCS_REQUIRED = 90      # target dokumen valid sesuai 01_scraping.py (TARGET_DOCS)
MIN_DOCS_TUGAS = 30         # syarat minimum tugas dari soal (tidak boleh kurang dari ini)
MIN_WORDS = 100             # disesuaikan dengan MIN_WORDS_VALID di 01_scraping.py
MIN_COMPLETENESS = 0.5      # disesuaikan dengan MIN_COMPLETENESS di 01_scraping.py
SAMPLE_SIZE = 5             # jumlah dokumen yang ditampilkan cuplikannya buat spot-check

DOMAIN = "Perdata - Perbuatan Melawan Hukum (PMH)"   # ← DIUBAH
PENGADILAN = "PN Bandung"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

report_lines = []


def log(msg=""):
    """Cetak ke terminal sekaligus simpan ke buffer laporan."""
    print(msg)
    report_lines.append(msg)


def section(title):
    log()
    log("=" * 78)
    log(title)
    log("=" * 78)


# ----------------------------------------------------------------------------
# 0. CEK FILE METADATA UTAMA
# ----------------------------------------------------------------------------
def load_metadata():
    if not os.path.exists(METADATA_CSV):
        log(f"[FATAL] {METADATA_CSV} tidak ditemukan. Jalankan 01_scraping.py dulu.")
        raise SystemExit(1)
    df = pd.read_csv(METADATA_CSV)
    return df


# ----------------------------------------------------------------------------
# 1. RINGKASAN JUMLAH DOKUMEN
# ----------------------------------------------------------------------------
def check_doc_count(df):
    section("1. JUMLAH DOKUMEN")
    total = len(df)
    n_valid = int(df["valid_min_200_kata"].sum()) if "valid_min_200_kata" in df else 0
    n_invalid = total - n_valid

    log(f"Domain         : {DOMAIN}")
    log(f"Pengadilan     : {PENGADILAN}")
    log(f"Total baris di metadata_raw.csv : {total}")
    log(f"Dokumen valid (>={MIN_WORDS} kata & keutuhan>={MIN_COMPLETENESS:.0%}) : {n_valid}")
    log(f"Dokumen tidak valid                                        : {n_invalid}")

    # cek terhadap target scraping (90)
    if n_valid >= MIN_DOCS_REQUIRED:
        log(f"[OK] Jumlah dokumen valid ({n_valid}) >= target scraping ({MIN_DOCS_REQUIRED}).")
    elif n_valid >= MIN_DOCS_TUGAS:
        log(f"[WARN] Dokumen valid ({n_valid}) sudah memenuhi syarat minimum tugas "
            f"({MIN_DOCS_TUGAS}), tapi belum mencapai target scraping "
            f"({MIN_DOCS_REQUIRED}). Pertimbangkan jalankan ulang 01_scraping.py.")
    else:
        log(f"[GAGAL] Dokumen valid ({n_valid}) BELUM memenuhi syarat minimum tugas "
            f"({MIN_DOCS_TUGAS}). Jalankan ulang 01_scraping.py (RESUME aktif, "
            "jadi aman dilanjut) atau naikkan MAX_LIST_PAGES.")

    return n_valid


# ----------------------------------------------------------------------------
# 2. KONSISTENSI ANTAR FOLDER
# ----------------------------------------------------------------------------
def check_folder_consistency(df):
    section("2. KONSISTENSI ANTAR FOLDER")

    case_ids = set(df["case_id"].astype(str)) if "case_id" in df else set()

    def files_in(folder, ext):
        if not os.path.isdir(folder):
            return set()
        return {os.path.splitext(f)[0] for f in os.listdir(folder) if f.endswith(ext)}

    raw_files = files_in(RAW_DIR, ".txt")
    raw_extracted_files = files_in(RAW_EXTRACTED_DIR, ".txt")
    token_files = files_in(TOKENS_DIR, ".json")
    pdf_files = files_in(PDF_DIR, ".pdf")

    log(f"case_id di metadata        : {len(case_ids)}")
    log(f"file di {RAW_DIR:<22}: {len(raw_files)}")
    log(f"file di {RAW_EXTRACTED_DIR:<22}: {len(raw_extracted_files)}")
    log(f"file di {TOKENS_DIR:<22}: {len(token_files)}")
    log(f"file di {PDF_DIR:<22}: {len(pdf_files)}")

    valid_ids = set(df.loc[df.get("valid_min_200_kata", False) == True, "case_id"].astype(str)) \
        if "valid_min_200_kata" in df else set()

    missing_raw = valid_ids - raw_files
    missing_tokens = valid_ids - token_files
    missing_pdf = valid_ids - pdf_files

    if missing_raw:
        log(f"[WARN] {len(missing_raw)} case_id valid di CSV tapi TIDAK punya file .txt di {RAW_DIR}: "
            f"{sorted(missing_raw)[:10]}{' ...' if len(missing_raw) > 10 else ''}")
    else:
        log(f"[OK] Semua dokumen valid punya file .txt di {RAW_DIR}.")

    if missing_tokens:
        log(f"[WARN] {len(missing_tokens)} case_id valid tidak punya file token di {TOKENS_DIR}.")
    else:
        log(f"[OK] Semua dokumen valid punya file token di {TOKENS_DIR}.")

    if missing_pdf:
        log(f"[INFO] {len(missing_pdf)} case_id valid tidak punya arsip PDF (mungkin terhapus manual).")
    else:
        log(f"[OK] Semua dokumen valid punya arsip PDF.")

    return missing_raw, missing_tokens


# ----------------------------------------------------------------------------
# 3. DISTRIBUSI PANJANG DOKUMEN
# ----------------------------------------------------------------------------
def check_length_distribution(df):
    section("3. DISTRIBUSI PANJANG DOKUMEN (JUMLAH KATA)")

    if "jumlah_kata" not in df:
        log("[WARN] kolom 'jumlah_kata' tidak ada di CSV, lewati pengecekan ini.")
        return []

    lengths = df["jumlah_kata"].dropna()
    if len(lengths) == 0:
        log("[WARN] tidak ada data panjang dokumen.")
        return []

    log(f"Rata-rata jumlah kata : {lengths.mean():.0f}")
    log(f"Median                : {lengths.median():.0f}")
    log(f"Minimum               : {lengths.min():.0f}")
    log(f"Maksimum              : {lengths.max():.0f}")
    log(f"Std deviasi           : {lengths.std():.0f}")

    very_short = df[(df["jumlah_kata"] > 0) & (df["jumlah_kata"] < MIN_WORDS)]
    empty_docs = df[df["jumlah_kata"] == 0]

    if len(empty_docs):
        log(f"[WARN] {len(empty_docs)} dokumen dengan 0 kata (gagal ekstrak/download). "
            f"case_id: {list(empty_docs['case_id'])[:10]}")
    if len(very_short):
        log(f"[INFO] {len(very_short)} dokumen di bawah ambang {MIN_WORDS} kata (otomatis ditandai tidak valid).")

    return list(empty_docs["case_id"]) if len(empty_docs) else []


# ----------------------------------------------------------------------------
# 4. DISTRIBUSI SKOR KEUTUHAN
# ----------------------------------------------------------------------------
def check_completeness_distribution(df):
    section("4. DISTRIBUSI SKOR KEUTUHAN TEKS (VALIDASI >=80%)")

    if "keutuhan_teks" not in df:
        log("[WARN] kolom 'keutuhan_teks' tidak ada di CSV. Lewati pengecekan ini.")
        return []

    comp = df["keutuhan_teks"].dropna()
    if len(comp) == 0:
        log("[WARN] tidak ada data keutuhan teks.")
        return []

    log(f"Rata-rata keutuhan : {comp.mean():.1%}")
    log(f"Median             : {comp.median():.1%}")
    log(f"Minimum            : {comp.min():.1%}")

    below_threshold = df[(df["keutuhan_teks"] > 0) & (df["keutuhan_teks"] < MIN_COMPLETENESS)]
    if len(below_threshold):
        log(f"[INFO] {len(below_threshold)} dokumen di bawah ambang keutuhan {MIN_COMPLETENESS:.0%} "
            "(otomatis ditandai tidak valid, kemungkinan proses cleaning kebablasan "
            "atau PDF aslinya memang banyak watermark/noise).")

    return list(below_threshold["case_id"]) if len(below_threshold) else []


# ----------------------------------------------------------------------------
# 5. DUPLIKASI NOMOR PERKARA
# ----------------------------------------------------------------------------
def check_duplicates(df):
    section("5. DUPLIKASI NOMOR PERKARA")

    if "no_perkara" not in df:
        log("[WARN] kolom 'no_perkara' tidak ada, lewati pengecekan duplikasi.")
        return []

    valid_no = df[df["no_perkara"].notna() & (df["no_perkara"] != "")]
    dup_mask = valid_no["no_perkara"].duplicated(keep=False)
    dups = valid_no[dup_mask]

    if len(dups):
        log(f"[WARN] ditemukan {len(dups)} baris dengan no_perkara duplikat:")
        for no, group in dups.groupby("no_perkara"):
            log(f"  - '{no}' muncul di case_id: {list(group['case_id'])}")
    else:
        log("[OK] Tidak ada duplikasi nomor perkara.")

    return list(dups["case_id"]) if len(dups) else []


# ----------------------------------------------------------------------------
# 6. KELENGKAPAN METADATA
# ----------------------------------------------------------------------------
def check_metadata_completeness(df):
    section("6. KELENGKAPAN METADATA")

    cols_to_check = [
        "no_perkara", "tanggal_register", "klasifikasi",
        "amar", "hakim_ketua", "judul",
    ]
    cols_to_check = [c for c in cols_to_check if c in df.columns]

    if not cols_to_check:
        log("[WARN] tidak ada kolom metadata standar yang ditemukan.")
        return

    total = len(df)
    for col in cols_to_check:
        n_empty = df[col].isna().sum() + (df[col] == "").sum()
        pct = (n_empty / total * 100) if total else 0
        status = "[OK]" if pct < 10 else "[WARN]"
        log(f"{status} kolom '{col}': {n_empty}/{total} ({pct:.0f}%) kosong")


# ----------------------------------------------------------------------------
# 7. CEK KONSISTENSI DOMAIN (validasi jenis_perkara di CSV)
# ----------------------------------------------------------------------------
def check_domain_consistency(df):
    section("7. KONSISTENSI DOMAIN / JENIS PERKARA")

    if "jenis_perkara" not in df.columns:
        log("[WARN] kolom 'jenis_perkara' tidak ditemukan di CSV.")
        return

    domain_counts = df["jenis_perkara"].value_counts()
    log("Distribusi jenis_perkara di metadata:")
    for domain, count in domain_counts.items():
        log(f"  - '{domain}': {count} dokumen")

    # deteksi kalau masih ada record dari domain lama (Senjata Api)
    old_domain_mask = df["jenis_perkara"].str.contains("Senjata Api", na=False, case=False)
    n_old = old_domain_mask.sum()
    if n_old > 0:
        log(f"[WARN] Ditemukan {n_old} record dengan domain lama 'Senjata Api'. "
            "Ini kemungkinan sisa dari run sebelumnya. "
            "Set RESUME=False di 01_scraping.py dan jalankan ulang, "
            "atau hapus manual dari metadata_raw.csv dan folder data/.")
    else:
        log(f"[OK] Semua record sudah menggunakan domain baru: '{DOMAIN}'.")


# ----------------------------------------------------------------------------
# 8. SAMPLE ACAK UNTUK SPOT-CHECK MANUAL
# ----------------------------------------------------------------------------
def show_random_samples(df, n=SAMPLE_SIZE):
    section(f"8. SAMPLE ACAK ({n} DOKUMEN) UNTUK DIBACA MANUAL")

    valid_df = df[df.get("valid_min_200_kata", False) == True] if "valid_min_200_kata" in df else df
    if len(valid_df) == 0:
        log("[WARN] tidak ada dokumen valid untuk di-sample.")
        return

    sample_ids = random.sample(list(valid_df["case_id"]), min(n, len(valid_df)))

    for cid in sample_ids:
        path = f"{RAW_DIR}/{cid}.txt"
        log(f"\n--- {cid} ---")
        if not os.path.exists(path):
            log(f"  [WARN] file {path} tidak ditemukan!")
            continue
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        preview = text[:300].replace("\n", " ")
        log(f"  panjang: {len(text.split())} kata")
        log(f"  cuplikan awal: {preview}...")

    log("\n[ACTION REQUIRED] Baca ulang cuplikan di atas secara manual:")
    log("  - Apakah masih ada header/footer/watermark yang ketinggalan?")
    log("  - Apakah isi teks relevan dengan domain PMH (Perbuatan Melawan Hukum)?")
    log("  - Apakah ada karakter aneh/rusak akibat OCR atau ekstraksi PDF?")


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    log("=== QA / SANITY-CHECK HASIL TAHAP 1 (Membangun Case Base) ===")
    log(f"Domain: {DOMAIN} | Pengadilan: {PENGADILAN}")

    df = load_metadata()

    n_valid = check_doc_count(df)
    missing_raw, missing_tokens = check_folder_consistency(df)
    empty_doc_ids = check_length_distribution(df)
    low_completeness_ids = check_completeness_distribution(df)
    dup_ids = check_duplicates(df)
    check_metadata_completeness(df)
    check_domain_consistency(df)   # ← pengecekan baru: konsistensi domain
    show_random_samples(df)

    # ------------------------------------------------------------------
    # KESIMPULAN AKHIR
    # ------------------------------------------------------------------
    section("KESIMPULAN")
    issues = []
    if n_valid < MIN_DOCS_TUGAS:
        issues.append(f"Dokumen valid kurang dari syarat minimum tugas ({n_valid}/{MIN_DOCS_TUGAS})")
    elif n_valid < MIN_DOCS_REQUIRED:
        issues.append(f"Dokumen valid sudah memenuhi syarat tugas tapi belum mencapai "
                      f"target scraping ({n_valid}/{MIN_DOCS_REQUIRED})")
    if missing_raw:
        issues.append(f"{len(missing_raw)} dokumen valid kehilangan file .txt")
    if missing_tokens:
        issues.append(f"{len(missing_tokens)} dokumen valid kehilangan file token")
    if dup_ids:
        issues.append(f"{len(dup_ids)} baris dengan nomor perkara duplikat")

    # cek apakah ada sisa record domain lama
    if "jenis_perkara" in df.columns:
        n_old_domain = df["jenis_perkara"].str.contains("Senjata Api", na=False, case=False).sum()
        if n_old_domain > 0:
            issues.append(f"{n_old_domain} record masih menggunakan domain lama (Senjata Api)")

    if issues:
        log("[STATUS] Tahap 1 PERLU PERHATIAN sebelum lanjut ke Tahap 2. Catatan:")
        for i in issues:
            log(f"  - {i}")
        if n_valid >= MIN_DOCS_TUGAS:
            log("[INFO] Jumlah dokumen sudah memenuhi syarat minimum tugas (>=30). "
                "Bisa dilanjutkan ke Tahap 2, tapi disarankan selesaikan catatan di atas.")
    else:
        log("[STATUS] Tahap 1 tampak siap dilanjutkan ke Tahap 2 (Case Representation).")
        log("         Tetap disarankan baca beberapa sample di atas secara manual sebelum lanjut.")

    # ------------------------------------------------------------------
    # SIMPAN LAPORAN & DAFTAR FLAGGED CASE_ID
    # ------------------------------------------------------------------
    report_path = f"{LOG_DIR}/qa_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    flagged_ids = sorted(set(
        list(missing_raw) + list(missing_tokens) +
        empty_doc_ids + low_completeness_ids + dup_ids
    ))
    flagged_path = f"{PROCESSED_DIR}/qa_flagged.csv"
    with open(flagged_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "alasan_perlu_dicek"])
        for cid in flagged_ids:
            alasan = []
            if cid in missing_raw:
                alasan.append("file_raw_hilang")
            if cid in missing_tokens:
                alasan.append("file_token_hilang")
            if cid in empty_doc_ids:
                alasan.append("dokumen_kosong")
            if cid in low_completeness_ids:
                alasan.append("keutuhan_di_bawah_80persen")
            if cid in dup_ids:
                alasan.append("nomor_perkara_duplikat")
            writer.writerow([cid, ";".join(alasan)])

    print(f"\nLaporan lengkap tersimpan -> {report_path}")
    print(f"Daftar case_id yang perlu dicek manual -> {flagged_path}")


if __name__ == "__main__":
    main()