"""
Bulk PDF Downloader (Metadata-Based, Manifest-Free, Improved Resolvers)
-----------------------------------------------------------------------
Reads:
    reports/combined/<TOPIC>/<TOPIC>_metadata_combined_seminal.csv

For each row:
  - Cleans/validates DOI again.
  - Tries resolvers in this order:
        1. Unpaywall
        2. OpenAlex
        3. Semantic Scholar
        4. CORE
  - Downloads PDF if any resolver returns a URL.
  - Accepts file if it is a real PDF (header + size check).
  - Skips tiny/corrupt PDFs (<30KB).

Logs:
  - Stage 0/1/1.5/2.5 counts from RPYS summary
  - Download outcomes & sources
  - Manual review list
  - Final summary with runtime
"""

import os, re, csv, time, zipfile, threading, urllib.parse, random
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib import robotparser
from collections import Counter
from pathlib import Path

import requests
import pandas as pd

# ------------------------------------------------
# CONFIG
# ------------------------------------------------
TOPIC = "lung_transplant"  # change per script if needed
CONTACT_EMAIL = "vm2725@nyu.edu"

MAX_WORKERS = 8
TIMEOUT = 60
ZIP_BATCH_SIZE = 3000
USER_AGENT = f"capstone-downloader/1.0 ({CONTACT_EMAIL})"
SLEEP_BETWEEN_REQS = 0.15

# ------------------------------------------------
# PATHS
# ------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]

# Weighted-union output (metadata-level) – direct input
INPUT_FILE = (
    ROOT_DIR
    / "reports"
    / "combined"
    / TOPIC
    / f"{TOPIC}_metadata_combined_seminal.csv"
)

# RPYS summary for Stage 0/1/1.5 counts
RPYS_DIR = ROOT_DIR / "reports" / "rpys"

# Output dirs
COMBINED_DIR = ROOT_DIR / "reports" / "combined" / TOPIC
COMBINED_DIR.mkdir(parents=True, exist_ok=True)

PAPERS_DIR = COMBINED_DIR / f"papers_{TOPIC}"
PAPERS_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = COMBINED_DIR / f"download_log_{TOPIC}.csv"
MANUAL_REVIEW_FILE = COMBINED_DIR / f"manual_review_{TOPIC}.csv"
SUMMARY_FILE = COMBINED_DIR / f"summary_report_{TOPIC}.txt"

# ------------------------------------------------
# DOI CLEANER (DOWNLOADER-SIDE SAFETY NET)
# ------------------------------------------------
def extract_clean_doi(s: str):
    """
    Re-clean and validate DOI locally before hitting APIs.
    Returns a strict DOI or None.
    """
    if not isinstance(s, str):
        return None

    s = s.strip().lower()
    if s in ("", "nan", "none", "no doi", "null", "0", ".", "-"):
        return None

    # Remove URL prefix
    if "doi.org/" in s:
        s = s.split("doi.org/")[-1].strip()

    # Remove 'doi:' prefix
    s = re.sub(r"^doi[:\s]+", "", s)

    # Strip trailing punctuation/whitespace
    s = s.strip(" ;,.\n\t\r")

    # If the string contains more than just the DOI, try to extract it
    m = re.search(r"(10\.\d{4,9}/[^\s]+)", s)
    if m:
        s = m.group(1)

    # Remove internal whitespace and obvious junk chars
    s = s.replace(" ", "").replace("\\", "").replace('"', "")

    # Final strict validation
    if re.match(r"^10\.\d{4,9}/\S+$", s):
        return s

    return None

# ------------------------------------------------
# PDF VALIDATION
# ------------------------------------------------
def is_valid_pdf(file_path: str) -> bool:
    if not os.path.exists(file_path):
        return False

    size = os.path.getsize(file_path)
    # Avoid 0–16KB corrupt PDFs; 30KB is safe minimum for real papers
    if size < 30_000:
        return False

    try:
        with open(file_path, "rb") as f:
            header = f.read(5)
        return header == b"%PDF-"
    except:
        return False

# ------------------------------------------------
# Robots.txt caching
# ------------------------------------------------
ROBOTS_CACHE = {}
ROBOTS_LOCK = threading.Lock()

def robots_allowed(url: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        with ROBOTS_LOCK:
            rp = ROBOTS_CACHE.get(base)

            if rp is None:
                rp = robotparser.RobotFileParser()
                rp.set_url(urllib.parse.urljoin(base, "/robots.txt"))
                try:
                    rp.read()
                except Exception:
                    ROBOTS_CACHE[base] = rp
                    return True
                ROBOTS_CACHE[base] = rp

        try:
            return rp.can_fetch(USER_AGENT, url) if rp.default_entry else True
        except Exception:
            return True
    except Exception:
        return True

# ------------------------------------------------
# Download helpers
# ------------------------------------------------
def doi_to_filename(doi: str) -> str:
    clean = re.sub(r"[^\w\-]+", "_", doi.strip())
    return clean[:180] + ".pdf"

def stream_download(url: str, dest: str, session: requests.Session) -> bool:
    """
    Download URL blindly, then validate PDF by header+size.
    We no longer rely on Content-Type or URL suffix to decide.
    """
    for attempt in range(3):
        try:
            with session.get(url, stream=True, timeout=TIMEOUT) as r:
                r.raise_for_status()

                with open(dest, "wb") as f:
                    for chunk in r.iter_content(16384):
                        if chunk:
                            f.write(chunk)

            if is_valid_pdf(dest):
                return True

            # Not a valid PDF -> delete and retry / fail
            os.remove(dest)
            raise Exception("Corrupted or non-PDF content")
        except:
            if attempt < 2:
                time.sleep(random.uniform(1.0, 3.0))
                continue
            return False
    return False

# ------------------------------------------------
# Resolvers (no Sci-Hub)
# ------------------------------------------------
def resolve_unpaywall(doi, session):
    api = f"https://api.unpaywall.org/v2/{doi}?email={CONTACT_EMAIL}"
    time.sleep(SLEEP_BETWEEN_REQS)
    r = session.get(api, timeout=TIMEOUT)
    if r.status_code != 200:
        return None
    data = r.json()
    loc = data.get("best_oa_location") or {}
    return loc.get("url_for_pdf") or loc.get("url")

def resolve_openalex(doi, session):
    api = f"https://api.openalex.org/works/https://doi.org/{doi}"
    time.sleep(SLEEP_BETWEEN_REQS)
    r = session.get(api, timeout=TIMEOUT)
    if r.status_code != 200:
        return None
    d = r.json()
    primary = d.get("primary_location") or {}
    return primary.get("pdf_url")

def resolve_semanticscholar(doi, session):
    api = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=openAccessPdf"
    time.sleep(SLEEP_BETWEEN_REQS)
    r = session.get(api, timeout=TIMEOUT)
    if r.status_code != 200:
        return None
    d = r.json()
    pdf = (d.get("openAccessPdf") or {}).get("url")
    return pdf

def resolve_core(doi, session):
    api = f"https://api.core.ac.uk/internal/works?doi={doi}"
    time.sleep(SLEEP_BETWEEN_REQS)
    r = session.get(api, timeout=TIMEOUT)
    if r.status_code != 200:
        return None
    try:
        results = r.json().get("results", [])
    except Exception:
        return None
    for it in results:
        return it.get("downloadUrl") or it.get("fullTextLink")
    return None

# Ordered, Sci-Hub excluded
RESOLVER_CHAIN = [
    ("resolve_unpaywall", resolve_unpaywall),
    ("resolve_openalex", resolve_openalex),
    ("resolve_semanticscholar", resolve_semanticscholar),
    ("resolve_core", resolve_core),
]

# ------------------------------------------------
# Stage Counts (Stage 0, 1, 1.5, 2.5)
# ------------------------------------------------
def load_stage_counts():
    """Reads RPYS summary for Stage 0/1/1.5 counts."""
    summary_file = RPYS_DIR / f"{TOPIC}_rpys_summary.txt"
    stage0 = stage1 = stage15 = 0

    if summary_file.exists():
        with open(summary_file, "r") as f:
            txt = f.read()

        m = re.findall(r"Total metadata\s*:\s*(\d+)", txt)
        stage0 = int(m[0]) if m else 0

        m = re.findall(r"Seminal references\s*:\s*(\d+)", txt)
        stage1 = int(m[0]) if m else 0

        m = re.findall(r"Metadata papers scored\s*:\s*(\d+)", txt)
        stage15 = int(m[0]) if m else 0

    # Weighted union count (Stage 2.5)
    wu_file = INPUT_FILE
    if wu_file.exists():
        df = pd.read_csv(wu_file)
        stage25 = len(df)
    else:
        stage25 = 0

    return stage0, stage1, stage15, stage25

# ------------------------------------------------
# MAIN PER-ROW PROCESSING
# ------------------------------------------------
def process_row(idx, row, session):
    raw_doi = row.get("doi", "")
    doi = extract_clean_doi(str(raw_doi))

    if not doi:
        return {"idx": idx, "doi": str(raw_doi), "source": "", "status": "no_doi"}

    dest = str(PAPERS_DIR / doi_to_filename(doi))

    # Cached valid PDF?
    if os.path.exists(dest) and is_valid_pdf(dest):
        return {"idx": idx, "doi": doi, "source": "cache", "status": "success"}

    # Try resolvers in strict order
    for name, resolver in RESOLVER_CHAIN:
        try:
            url = resolver(doi, session)
            if not url:
                continue
            if not robots_allowed(url):
                continue
            if stream_download(url, dest, session):
                return {"idx": idx, "doi": doi, "source": name, "status": "success"}
        except Exception:
            continue

    return {"idx": idx, "doi": doi, "source": "", "status": "failed"}

# ------------------------------------------------
# MAIN
# ------------------------------------------------
def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Weighted-union file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    if "doi" not in df.columns:
        raise ValueError("Weighted-union file must contain a 'doi' column.")

    stage0, stage1, stage15, stage25 = load_stage_counts()

    print(f"\n=== Starting PDF download for {TOPIC} ===")
    print(f"Total weighted-union metadata papers (Stage 2.5): {stage25}\n")

    start_time = time.time()
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    results = []

    # Parallel requests
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [
            ex.submit(process_row, i, row, session)
            for i, row in df.iterrows()
        ]
        total = len(futures)
        for j, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            results.append(res)

            if j % 250 == 0 or j == total:
                elapsed = time.time() - start_time
                print(f"[Progress] {j}/{total} PDFs attempted")

    # Save logs
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "doi", "source", "status"])
        w.writeheader()
        for r in results:
            w.writerow(r)

    failed = [r for r in results if r["status"] == "failed"]
    pd.DataFrame(failed).to_csv(MANUAL_REVIEW_FILE, index=False)

    counts = Counter(r["status"] for r in results)
    by_source = Counter(r["source"] for r in results if r["status"] == "success")

    runtime_min = (time.time() - start_time) / 60

    # Write summary
    with open(SUMMARY_FILE, "w") as f:
        f.write("========== FINAL DOWNLOAD SUMMARY ==========\n")
        f.write(f"TOPIC: {TOPIC}\n\n")

        f.write("Stage 0 – Metadata Input\n")
        f.write(f"  Total metadata papers: {stage0}\n\n")

        f.write("Stage 1 – RPYS Reference-Level\n")
        f.write(f"  Seminal references   : {stage1}\n\n")

        f.write("Stage 1.5 – Metadata RPYS Influence\n")
        f.write(f"  Metadata papers scored: {stage15}\n\n")

        f.write("Stage 2.5 – Weighted Union Selection\n")
        f.write(f"  Selected metadata papers: {stage25}\n\n")

        f.write("Stage 4 – PDF Downloads\n")
        for k, v in counts.items():
            f.write(f"  {k:>10}: {v}\n")

        f.write("\nSuccessful sources breakdown:\n")
        for s, c in by_source.items():
            f.write(f"  {s}: {c}\n")

        f.write(f"\nManual review list: {MANUAL_REVIEW_FILE}\n")
        f.write(f"Total runtime: {runtime_min:.2f} minutes\n")
        f.write("============================================\n")

    print("\n✔ Download complete!")
    print(f"Manual review: {MANUAL_REVIEW_FILE}")
    print(f"Summary saved: {SUMMARY_FILE}")

if __name__ == "__main__":
    main()
