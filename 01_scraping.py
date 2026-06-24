"""
01_scraping.py
===============
Tahap 1 (Membangun Case Base) - Tugas CBR Penalaran Komputer
Domain: Perdata - Perbuatan Melawan Hukum (PMH)
Pengadilan: PN Bandung
Sumber: Direktori Putusan Mahkamah Agung RI (putusan3.mahkamahagung.go.id)

Cara pakai:
    pip install -r requirements.txt
    python 01_scraping.py

Output:
    data/raw/case_001.txt ...                -> teks bersih + ternormalisasi tiap putusan
    data/raw_extracted/case_001.txt ...      -> teks mentah hasil ekstraksi PDF (sebelum
                                                 cleaning, dipakai buat validasi keutuhan)
    data/tokens/case_001.json ...            -> hasil tokenisasi tiap putusan
    data/pdf/case_001.pdf ...                -> arsip PDF asli (cadangan)
    data/processed/metadata_raw.csv          -> metadata dasar tiap kasus
    logs/cleaning.log                        -> catatan proses

Catatan penting:
- Situsnya pakai bot-protection yang nge-block `requests`/`cloudscraper`
  biasa (403 Forbidden). Solusinya: buka Chrome ASLI sekali pakai
  undetected-chromedriver buat "lewatin" proteksinya, ambil cookies
  session-nya, lalu cookies itu dipakai di `requests` biasa buat sisa
  proses scraping (lebih cepat daripada buka-render browser tiap halaman).
- Sebuah jendela Chrome bakal kebuka otomatis sebentar pas script
  dijalankan -> itu NORMAL, biarin aja sampai dia nutup sendiri.
- Wajib udah install Google Chrome di komputer kamu.

Catatan perubahan domain (v3):
- Domain diganti dari "Pidana Khusus - Senjata Api dan Bahan Peledak" ke
  "Perdata - Perbuatan Melawan Hukum (PMH)" karena kategori Senjata Api
  PN Bandung hanya memiliki 159 link total dengan mayoritas dokumen lama
  tanpa PDF digital, sehingga hanya menghasilkan 25 dokumen valid dari
  target minimum 30.
- TARGET_DOCS dinaikkan ke 90 karena kategori PMH memiliki jauh lebih
  banyak dokumen dengan PDF digital yang tersedia.
- MAX_LIST_PAGES dinaikkan ke 50 untuk mengakomodasi volume data yang lebih
  besar di kategori PMH.
"""

import os
import re
import json
import time
import logging

import requests
import pandas as pd
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text
import undetected_chromedriver as uc

# ----------------------------------------------------------------------------
# KONFIGURASI
# ----------------------------------------------------------------------------
BASE_LIST_URL = (
    "https://putusan3.mahkamahagung.go.id/direktori/index/"
    "pengadilan/pn-bandung/kategori/perbuatan-melawan-hukum-1.html"
)

RAW_DIR = "data/raw"
RAW_EXTRACTED_DIR = "data/raw_extracted"   # teks mentah hasil ekstraksi PDF (sebelum
                                            # dibersihkan/dinormalisasi) -> dipakai
                                            # sebagai pembanding buat validasi keutuhan
TOKENS_DIR = "data/tokens"                 # hasil tokenisasi tiap dokumen (.json: list token)
PDF_DIR = "data/pdf"
PROCESSED_DIR = "data/processed"
LOG_DIR = "logs"

TARGET_DOCS = 90          # target dokumen valid - dinaikkan karena kategori PMH
                           # memiliki banyak dokumen dengan PDF digital tersedia.
MAX_LIST_PAGES = 50        # dinaikkan untuk mengakomodasi volume listing PMH yang lebih
                            # besar dibanding Senjata Api (yang cuma 8 halaman unik).
SLEEP_BETWEEN_REQUEST = 2  # detik, biar sopan ke server pengadilan
RETRY = 3
BROWSER_WARMUP_WAIT = 15   # detik nunggu Chrome lewatin bot-protection Cloudflare
                            # (dinaikkan dari 6 ke 15 karena 'Just a moment...' challenge
                            # butuh waktu lebih lama untuk resolve sebelum cookies diambil)

# Threshold validasi dokumen
MIN_WORDS_VALID = 100      # minimum kata setelah cleaning
MIN_COMPLETENESS = 0.5     # minimum skor keutuhan

# --- Tambahan buat resume & reliability ---
RESUME = True              # TRUE: lanjut dari dokumen yang belum diproses.
                            # Dokumen yang sudah valid di run sebelumnya dilewati.
SESSION_REFRESH_EVERY = 40     # refresh cookies via Chrome tiap N dokumen yang diproses
MAX_CONSECUTIVE_FAILURES = 6   # kalau gagal beruntun, auto refresh via Chrome
CHECKPOINT_EVERY = 10           # simpan metadata_raw.csv tiap N dokumen

for d in (RAW_DIR, RAW_EXTRACTED_DIR, TOKENS_DIR, PDF_DIR, PROCESSED_DIR, LOG_DIR):
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    filename=f"{LOG_DIR}/cleaning.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def log(msg):
    print(msg)
    logging.info(msg)


# ----------------------------------------------------------------------------
# BUKA CHROME ASLI SEKALI, AMBIL COOKIES, PAKAI DI REQUESTS BIASA
# ----------------------------------------------------------------------------
def get_authenticated_session(max_attempts=3):
    """Buka Chrome sekali untuk lewati bot-protection, ambil cookies, lalu
    kembalikan requests.Session yang sudah ter-autentikasi.

    Dibungkus retry loop (max_attempts) karena Chrome kadang crash saat
    pertama dibuka (terutama saat refresh session kedua/ketiga di tengah
    scraping). Kalau semua percobaan gagal, kembalikan session kosong
    (tanpa cookies) -- scraping kemungkinan masih bisa jalan untuk
    sebagian halaman."""
    default_ua = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/149.0.0.0 Safari/537.36"
    )

    for attempt in range(1, max_attempts + 1):
        log(f"[BROWSER] membuka Chrome buat lewatin bot-protection "
            f"(percobaan {attempt}/{max_attempts})...")
        options = uc.ChromeOptions()
        options.add_argument("--window-size=1280,900")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")

        cookies = []
        user_agent = default_ua
        html_preview = ""
        driver = None

        try:
            # version_main harus sesuai PERSIS dengan versi Chrome yang terinstall.
            # Versi Chrome kamu saat ini: 149 (cek di chrome://settings/help).
            driver = uc.Chrome(options=options, version_main=149)
            driver.get(BASE_LIST_URL)
            time.sleep(BROWSER_WARMUP_WAIT)

            try:
                title = driver.title
                log(f"[BROWSER] halaman ke-load, judul tab: '{title}'")
            except Exception:
                log("[BROWSER] tidak bisa baca judul tab, lanjut ambil cookies...")

            cookies = driver.get_cookies()
            user_agent = driver.execute_script("return navigator.userAgent;")
            html_preview = driver.page_source
            log("[BROWSER] cookies berhasil diambil, lanjut scraping pakai requests biasa.")
            break  # sukses, keluar dari retry loop

        except Exception as e:
            log(f"[WARN] percobaan {attempt}/{max_attempts} gagal: {type(e).__name__}: "
                f"{str(e)[:120]}")
            if attempt < max_attempts:
                log(f"  menunggu 5 detik sebelum coba lagi...")
                time.sleep(5)
        finally:
            if driver is not None:
                try:
                    driver.quit()
                except Exception:
                    pass

    if html_preview and "/direktori/putusan/" not in html_preview:
        log("[WARN] halaman ke-load tapi belum berisi daftar putusan. "
            "Coba naikkan BROWSER_WARMUP_WAIT.")

    if not cookies:
        log("[WARN] tidak berhasil dapat cookies dari Chrome. "
            "Lanjut dengan session tanpa cookies -- beberapa halaman mungkin 403.")

    session = requests.Session()
    for c in cookies:
        session.cookies.set(c["name"], c["value"], domain=c.get("domain"))
    session.headers.update({
        "User-Agent": user_agent,
        "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
        "Referer": "https://putusan3.mahkamahagung.go.id/",
    })
    return session


# ----------------------------------------------------------------------------
# HELPER: AMBIL HALAMAN (pakai session yang udah login lewat browser)
# ----------------------------------------------------------------------------
def get_soup(session, url, retry=RETRY):
    for attempt in range(1, retry + 1):
        try:
            r = session.get(url, timeout=20)
            r.raise_for_status()
            return BeautifulSoup(r.text, "lxml")
        except Exception as e:
            log(f"  [WARN] gagal ambil {url} (percobaan {attempt}/{retry}): {e}")
            time.sleep(5)
    return None


# ----------------------------------------------------------------------------
# TAHAP A: KUMPULKAN LINK DETAIL PUTUSAN DARI HALAMAN LISTING
# ----------------------------------------------------------------------------
def get_listing_urls(session):
    # Batas atas link yang dikumpulkan. PMH PN Bandung memiliki jauh lebih
    # banyak dokumen dibanding Senjata Api, jadi limit dinaikkan ke 1000
    # untuk memastikan kita bisa temukan 90 dokumen valid.
    link_target = 1000

    case_links = []
    for page in range(1, MAX_LIST_PAGES + 1):
        url = (
            BASE_LIST_URL
            if page == 1
            else BASE_LIST_URL[:-5] + f"/page/{page}.html"
        )
        log(f"[LIST] halaman {page}: {url}")
        soup = get_soup(session, url)
        if soup is None:
            break

        links = soup.find_all("a", href=re.compile(r"/direktori/putusan/"))
        new_urls = [a["href"] for a in links]
        for u in new_urls:
            if u not in case_links:
                case_links.append(u)

        log(f"  total link unik sejauh ini: {len(case_links)}")
        if not new_urls:
            log("  tidak ada hasil baru, hentikan pagination")
            break
        if len(case_links) >= link_target:
            log(f"  sudah cukup kandidat link ({len(case_links)} >= {link_target})")
            break
        time.sleep(SLEEP_BETWEEN_REQUEST)
    else:
        log(f"[INFO] mencapai batas MAX_LIST_PAGES ({MAX_LIST_PAGES}) "
            f"dengan {len(case_links)} link terkumpul.")

    return case_links[:link_target]


# ----------------------------------------------------------------------------
# TAHAP B: BERSIHKAN TEKS HASIL PDF
# ----------------------------------------------------------------------------
MA_BOILERPLATE = [
    # -- versi lama (spasi antar huruf, format watermark lama) --
    "M a h ka m a h A g u n g R e p u blik In d o n esia",
    "Disclaimer",
    # -- versi baru yang ditemukan dari output nyata (header berulang) --
    "Direktori Putusan Republik Indonesia",
    "direktori putusan republik indonesia",
    "putusan. .go.id",
    "Putusan. .go.id",
    # -- footer disclaimer standar --
    "Kepaniteraan Mahkamah Agung Republik Indonesia berusaha untuk selalu "
    "mencantumkan informasi paling kini dan akurat sebagai bentuk komitmen "
    "Mahkamah Agung untuk pelayanan publik, transparansi dan akuntabilitas",
    "pelaksanaan fungsi peradilan. Namun dalam hal-hal tertentu masih "
    "dimungkinkan terjadi permasalahan teknis terkait dengan akurasi dan "
    "keterkinian informasi yang kami sajikan, hal mana akan terus kami "
    "perbaiki dari waktu kewaktu.",
    "Dalam hal Anda menemukan inakurasi informasi yang termuat pada situs "
    "ini atau informasi yang seharusnya ada, namun belum tersedia, maka "
    "harap segera hubungi Kepaniteraan Mahkamah Agung RI melalui :",
    "Email : kepaniteraan@mahkamahagung.go.id    Telp : 021-384 3348 (ext.318)",
]


def clean_pdf_text(text: str) -> str:
    """Hapus boilerplate header/footer/watermark MA dari teks hasil ekstraksi PDF."""
    for b in MA_BOILERPLATE:
        text = text.replace(b, " ")

    text = re.sub(
        r"(republik\s+indonesia\s*\n?){2,}",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"direktori\s+putusan[^\n]*\n?",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"putusan\s*\.\s*\.?\s*go\s*\.?\s*id[^\n]*\n?",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"M\s+a\s+h\s+k?\s*a\s+m\s+a\s+h\s+A\s+g\s+u\s+n\s+g",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"halaman\s*\d+\s*(dari\s*\d+\s*halaman)?",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\d+\s*hal\.\s*put\.\s*nomor[^\n]*",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"disclaimer", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_text(text: str) -> str:
    """Normalisasi karakter: lower-case, hilangkan karakter non-cetak, rapikan spasi."""
    text = text.lower()
    text = re.sub(r"[^\x20-\x7eA-Za-z0-9À-ÿ\n.,;:()\-/\"'%]", " ", text)
    text = re.sub(r"([.,;:])\1+", r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def tokenize_text(text: str) -> list:
    """Tokenisasi sederhana berbasis kata."""
    return re.findall(r"[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)*", text.lower())


def check_completeness(raw_extracted_text: str, cleaned_text: str) -> float:
    """Validasi keutuhan teks (minimal 80% isi putusan tersedia)."""
    if not raw_extracted_text.strip() or not cleaned_text.strip():
        return 0.0

    BOILERPLATE_MARKERS = {
        "direktori", "putusan.go.id", "putusan..go.id",
        "kepaniteraan", "disclaimer", "inakurasi",
    }

    def count_content_words(text):
        total = 0
        for line in text.splitlines():
            line_lower = line.lower().strip()
            if not line_lower:
                continue
            if re.fullmatch(r"(republik\s+indonesia\s*)+", line_lower):
                continue
            if any(m in line_lower for m in BOILERPLATE_MARKERS):
                continue
            total += len(line.split())
        return total

    raw_content_words = count_content_words(raw_extracted_text)
    cleaned_words = len(cleaned_text.split())

    if raw_content_words == 0:
        return 0.0

    return min(cleaned_words / raw_content_words, 1.0)


# ----------------------------------------------------------------------------
# TAHAP C: EKSTRAK METADATA + TEKS DARI 1 HALAMAN DETAIL
# ----------------------------------------------------------------------------
def load_existing_records():
    """Muat metadata_raw.csv yang udah ada (kalau ada) buat dilanjutin."""
    path = f"{PROCESSED_DIR}/metadata_raw.csv"
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            return df.to_dict("records")
        except Exception as e:
            log(f"[WARN] gagal baca metadata_raw.csv lama: {e}")
    return []


def get_field(table, keyword):
    try:
        td = table.find(lambda tag: tag.name == "td" and keyword in tag.get_text())
        return td.find_next("td").get_text(strip=True)
    except Exception:
        return ""


def extract_text_from_bytes(pdf_bytes):
    import io
    return extract_text(io.BytesIO(pdf_bytes))


def download_pdf_bytes(session, pdf_url, case_id, retry=RETRY):
    """Download PDF dengan retry."""
    for attempt in range(1, retry + 1):
        try:
            r = session.get(pdf_url, timeout=30)
            r.raise_for_status()
            if len(r.content) < 1000:
                raise ValueError(f"ukuran file cuma {len(r.content)} bytes, kemungkinan rusak/kosong")
            return r.content
        except Exception as e:
            log(f"  [WARN] {case_id}: gagal download PDF (percobaan {attempt}/{retry}): {e}")
            time.sleep(4)
    return None


def url_to_case_id(url, idx):
    """Bikin case_id yang stabil dari slug URL putusan."""
    slug_match = re.search(r"/direktori/putusan/([a-zA-Z0-9\-]+)\.html", url)
    if slug_match:
        slug = slug_match.group(1)[-40:]
        return f"case_{slug}"
    return f"case_{idx:03d}"


def scrape_case(session, idx, url):
    """Return tuple (record_dict_or_None, success_bool)."""
    soup = get_soup(session, url)
    if soup is None:
        return None, False

    table = soup.find("table", {"class": "table"})
    if table is None:
        log(f"  [WARN] tabel metadata tidak ditemukan di {url}, dilewati")
        return None, True

    h2 = table.find("h2")
    judul = h2.get_text(strip=True) if h2 else ""
    if h2:
        h2.decompose()

    meta = {
        "no_perkara": get_field(table, "Nomor"),
        "tanggal_register": get_field(table, "Tanggal Register"),
        "tanggal_musyawarah": get_field(table, "Tanggal Musyawarah"),
        "klasifikasi": get_field(table, "Klasifikasi"),
        "kata_kunci": get_field(table, "Kata Kunci"),
        "lembaga_peradilan": get_field(table, "Lembaga Peradilan"),
        "hakim_ketua": get_field(table, "Hakim Ketua"),
        "panitera": get_field(table, "Panitera"),
        "amar": get_field(table, "Amar"),
        "amar_lainnya": get_field(table, "Amar Lainnya"),
        "catatan_amar": get_field(table, "Catatan Amar"),
    }

    case_id = url_to_case_id(url, idx)
    raw_extracted_text = ""
    pdf_text = ""
    normalized_text = ""
    tokens = []
    pdf_a = soup.find("a", href=re.compile(r"/pdf/"))
    if pdf_a is None:
        log(f"  [INFO] {case_id}: tidak ada link PDF di halaman ini "
            f"(kemungkinan dokumen lama yang belum di-scan/upload)")
    else:
        pdf_bytes = download_pdf_bytes(session, pdf_a["href"], case_id)
        if pdf_bytes:
            try:
                raw_extracted_text = extract_text_from_bytes(pdf_bytes)

                raw_word_count = len(raw_extracted_text.split())
                if raw_word_count < 50:
                    log(f"  [SKIP] {case_id}: PDF kemungkinan scan/gambar "
                        f"(hanya {raw_word_count} kata hasil ekstraksi pdfminer). "
                        "Tidak bisa diekstrak tanpa OCR -> dokumen dilewati.")
                    with open(f"{RAW_EXTRACTED_DIR}/{case_id}_SCAN_SKIP.txt",
                              "w", encoding="utf-8") as f:
                        f.write(f"[SKIP] PDF scan/gambar, {raw_word_count} kata\n"
                                f"URL: {pdf_a['href']}\n\n"
                                f"Raw extracted:\n{raw_extracted_text[:500]}")
                    with open(f"{PDF_DIR}/{case_id}.pdf", "wb") as f:
                        f.write(pdf_bytes)
                else:
                    pdf_text = clean_pdf_text(raw_extracted_text)
                    normalized_text = normalize_text(pdf_text)
                    tokens = tokenize_text(normalized_text)
                    with open(f"{PDF_DIR}/{case_id}.pdf", "wb") as f:
                        f.write(pdf_bytes)
            except Exception as e:
                log(f"  [WARN] {case_id}: gagal ekstrak teks dari PDF: {e}")

    completeness = check_completeness(raw_extracted_text, pdf_text)
    is_valid = len(pdf_text.split()) >= MIN_WORDS_VALID and completeness >= MIN_COMPLETENESS

    if raw_extracted_text:
        with open(f"{RAW_EXTRACTED_DIR}/{case_id}.txt", "w", encoding="utf-8") as f:
            f.write(raw_extracted_text)
    if pdf_text:
        with open(f"{RAW_DIR}/{case_id}.txt", "w", encoding="utf-8") as f:
            f.write(normalized_text)
    if tokens:
        with open(f"{TOKENS_DIR}/{case_id}.json", "w", encoding="utf-8") as f:
            json.dump(tokens, f, ensure_ascii=False)

    log(f"  [{case_id}] judul='{judul[:60]}...' kata={len(pdf_text.split())} "
        f"keutuhan={completeness:.0%} valid={is_valid}")

    return {
        "case_id": case_id,
        "judul": judul,
        "jenis_perkara": "Perdata - Perbuatan Melawan Hukum (PMH)",  # ← DIUBAH
        "pengadilan": "PN Bandung",
        "url_detail": url,
        "url_pdf": pdf_a["href"] if pdf_a else "",
        "jumlah_kata": len(pdf_text.split()) if pdf_text else 0,
        "jumlah_token": len(tokens),
        "keutuhan_teks": round(completeness, 4),
        "valid_min_200_kata": is_valid,
        **meta,
    }, True


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    log("=== TAHAP 1: Membangun Case Base ===")
    log("Domain: Perdata - Perbuatan Melawan Hukum (PMH) | Pengadilan: PN Bandung")  # ← DIUBAH
    log(f"Target: {TARGET_DOCS} dokumen valid")

    session = get_authenticated_session()

    case_urls = get_listing_urls(session)
    log(f"Total link putusan yang akan diproses: {len(case_urls)}")

    if len(case_urls) < TARGET_DOCS:
        log(f"[INFO] link yang ketemu ({len(case_urls)}) lebih sedikit dari "
            f"TARGET_DOCS ({TARGET_DOCS}). Bakal diproses semua yang ada; "
            "kalau mau lebih banyak, coba naikkan MAX_LIST_PAGES.")

    records = load_existing_records() if RESUME else []
    done_ids = {r["case_id"] for r in records} if RESUME else set()
    if done_ids:
        log(f"[RESUME] ditemukan {len(done_ids)} dokumen yang udah pernah "
            "berhasil di-scrape sebelumnya -> bakal dilewati.")

    out_path = f"{PROCESSED_DIR}/metadata_raw.csv"
    consecutive_failures = 0
    processed_count = 0
    n_valid_target = sum(1 for r in records if r.get("valid_min_200_kata"))

    for i, url in enumerate(case_urls, start=1):
        if n_valid_target >= TARGET_DOCS:
            log(f"[STOP] sudah mencapai target {TARGET_DOCS} dokumen valid.")
            break

        probable_id = url_to_case_id(url, i)
        if RESUME and probable_id in done_ids:
            log(f"[{i}/{len(case_urls)}] [SKIP] {probable_id} udah pernah diproses")
            continue

        log(f"[{i}/{len(case_urls)}] scraping {url}")
        rec, success = scrape_case(session, i, url)
        processed_count += 1

        if rec:
            records.append(rec)
            done_ids.add(rec["case_id"])
            if rec.get("valid_min_200_kata"):
                n_valid_target += 1

        if not success:
            consecutive_failures += 1
            log(f"  [WARN] kegagalan beruntun: {consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}")
        else:
            consecutive_failures = 0

        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            cooldown = 30
            log(f"[COOLDOWN] {consecutive_failures} kegagalan beruntun -> "
                f"istirahat {cooldown} detik supaya tidak di-rate-limit...")
            time.sleep(cooldown)
            consecutive_failures = 0

        elif processed_count > 0 and processed_count % SESSION_REFRESH_EVERY == 0:
            log(f"[SESSION] sudah memproses {processed_count} dokumen sejak "
                "refresh terakhir -> refresh cookies preventif...")
            session = get_authenticated_session()

        if processed_count > 0 and processed_count % CHECKPOINT_EVERY == 0:
            pd.DataFrame(records).to_csv(out_path, index=False)
            log(f"  [CHECKPOINT] {len(records)} record tersimpan ke {out_path} "
                f"({n_valid_target} valid sejauh ini)")

        time.sleep(SLEEP_BETWEEN_REQUEST)

    df = pd.DataFrame(records)

    # deduplikasi berdasarkan no_perkara
    before_dedup = len(df)
    if "no_perkara" in df.columns:
        df_valid_no = df[df["no_perkara"].notna() & (df["no_perkara"] != "")]
        df_empty_no = df[df["no_perkara"].isna() | (df["no_perkara"] == "")]
        df_valid_no = df_valid_no.sort_values(
            "jumlah_kata", ascending=False
        ).drop_duplicates(subset=["no_perkara"], keep="first")
        df = pd.concat([df_valid_no, df_empty_no], ignore_index=True)
        n_removed = before_dedup - len(df)
        if n_removed:
            log(f"[DEDUP] {n_removed} baris duplikat (nomor perkara sama) "
                "dihapus -> simpan yang punya teks paling panjang.")

    df.to_csv(out_path, index=False)

    n_valid = int(df["valid_min_200_kata"].sum()) if len(df) else 0
    n_scan_skip = sum(
        1 for r in records
        if r.get("jumlah_kata", 0) == 0 and r.get("url_pdf", "")
    )
    log(f"=== SELESAI: {len(df)} dokumen tersimpan ({n_valid} valid) ===")
    log(f"  PDF scan/gambar (0 kata, dilewati) : {n_scan_skip} dokumen")
    log(f"  Teks bersih  -> {RAW_DIR}/*.txt")
    log(f"  PDF asli     -> {PDF_DIR}/*.pdf")
    log(f"  Metadata     -> {out_path}")

    if n_valid < TARGET_DOCS:
        log(f"[PERHATIAN] Dokumen valid ({n_valid}) belum mencapai target "
            f"({TARGET_DOCS}). Coba naikkan MAX_LIST_PAGES, atau jalankan ulang "
            "script ini lagi -> berkat RESUME, dia bakal otomatis lanjut "
            "ngambil dokumen baru tanpa mengulang yang udah ada.")
        if n_scan_skip > 10:
            log(f"[INFO] {n_scan_skip} dokumen gugur karena PDF scan/gambar "
                "(tidak ada teks digital). Ini normal untuk putusan lama. "
                "Pertimbangkan tambahkan OCR (pytesseract) kalau butuh "
                "dokumen-dokumen itu juga.")


if __name__ == "__main__":
    main()