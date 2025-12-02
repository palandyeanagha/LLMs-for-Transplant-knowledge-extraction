# -*- coding: utf-8 -*-
"""
Bulk PDF Downloader (Unified – WoS + PubMed Aware, Improved v5)
---------------------------------------------------------------
New in v5 (on top of v4):
- Scholarly host allow-list in robots_allowed():
    * Ignore robots.txt for big publishers when on VPN (Elsevier, Wiley, Springer, etc.)
- New Crossref resolver (resolve_crossref):
    * Uses Crossref "link" entries with content-type=application/pdf
- Keeps:
    * PMC + PubMed full-text logic
    * Unpaywall, CORE, OpenAlex, Semantic Scholar
    * Publisher DOI resolver with heuristics
    * Same directory + logging structure
"""

import os, re, csv, time, threading, urllib.parse, random
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib import robotparser
from collections import Counter
from pathlib import Path

import requests
import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------
TOPIC = "heart_transplant"
CONTACT_EMAIL = "vm2725@nyu.edu"

MAX_WORKERS = 8
TIMEOUT = 180
SLEEP_BETWEEN_REQS = 0.20
MIN_VALID_PDF_BYTES = 16_000

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:117.0) "
    "Gecko/20100101 Firefox/117.0"
)

# Known-open test PDFs for access checking
TEST_URLS = {
    "ScienceDirect": "https://www.sciencedirect.com/science/article/pii/S0140673618329934/pdfft",
    "Wiley": "https://onlinelibrary.wiley.com/doi/pdfdirect/10.1002/adma.201606842",
    "Springer": "https://link.springer.com/content/pdf/10.1007/s00134-020-06294-x.pdf",
}

# Scholarly hosts where, under NYU VPN, we safely ignore robots.txt
SCHOLARLY_HOST_ALLOWLIST = [
    "sciencedirect.com",
    "elsevier.com",
    "onlinelibrary.wiley.com",
    "wiley.com",
    "link.springer.com",
    "springer.com",
    "nature.com",
    "nejm.org",
    "thelancet.com",
    "cell.com",
    "jamanetwork.com",
    "ahajournals.org",
    "acc.org",
    "karger.com",
    "lww.com",
    "tandfonline.com",
    "bmj.com",
    "ersjournals.com",
    "atsjournals.org",
    "annalsthoracicsurgery.org",
    "sagepub.com",
    "oup.com",
]

# -----------------------------
# PATHS
# -----------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]

COMBINED_DIR = ROOT_DIR / "reports" / "combined" / TOPIC
COMBINED_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = COMBINED_DIR / f"{TOPIC}_metadata_combined_seminal.csv"

PAPERS_DIR = COMBINED_DIR / f"papers_{TOPIC}"
PAPERS_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = COMBINED_DIR / f"download_log_{TOPIC}.csv"
MANUAL_REVIEW_FILE = COMBINED_DIR / f"manual_review_{TOPIC}.csv"
SUMMARY_FILE = COMBINED_DIR / f"summary_report_{TOPIC}.txt"

RPYS_DIR = ROOT_DIR / "reports" / "rpys"

# -----------------------------
# UTILITIES
# -----------------------------
ROBOTS_CACHE = {}
ROBOTS_LOCK = threading.Lock()

def sanitize_name(s: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^\w\-]+", "_", str(s).strip())).strip("_")[:180]

def doi_to_filename(doi: str, fallback: str = "") -> str:
    base = doi or fallback or "no_doi"
    return sanitize_name(base) + ".pdf"

def extract_clean_doi(val):
    if pd.isna(val):
        return None
    if not isinstance(val, str):
        val = str(val)
    s = val.strip().replace(" ", "")

    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^doi[:\s]+", "", s, flags=re.IGNORECASE)

    m = re.search(r"(10\.\d{4,9}/\S+)", s)
    if m:
        s = m.group(1)

    s = s.strip(".,;")
    if re.match(r"^10\.\d{4,9}/\S+$", s):
        return s
    return None

def normalize_pmcid(pmcid):
    if not pmcid or not isinstance(pmcid, str):
        return None
    pmcid = re.sub(r"[^A-Za-z0-9]", "", pmcid).upper()
    return pmcid if pmcid.startswith("PMC") else None

def robots_allowed(url):
    """
    Relaxed robots:
    - If host is in SCHOLARLY_HOST_ALLOWLIST, allow regardless of robots.txt
    - Else, use standard robots.txt parsing
    """
    try:
        parsed = urllib.parse.urlparse(url)
        host = parsed.netloc.lower()

        # If it's a big scholarly host and you're on VPN, we allow it.
        if any(h in host for h in SCHOLARLY_HOST_ALLOWLIST):
            return True

        base = f"{parsed.scheme}://{parsed.netloc}"
        with ROBOTS_LOCK:
            rp = ROBOTS_CACHE.get(base)
            if rp is None:
                rp = robotparser.RobotFileParser()
                rp.set_url(base + "/robots.txt")
                try:
                    rp.read()
                except:
                    ROBOTS_CACHE[base] = rp
                    return True
                ROBOTS_CACHE[base] = rp
        return rp.can_fetch(USER_AGENT, url)
    except:
        return True

def is_pdf_response(resp, url):
    ctype = (resp.headers.get("Content-Type") or "").lower()
    if "pdf" in ctype:
        return True
    if url.lower().endswith(".pdf") or "/pdf" in url.lower():
        return True
    cd = resp.headers.get("Content-Disposition") or ""
    if "filename=" in cd.lower() and ".pdf" in cd.lower():
        return True
    return False

def is_valid_pdf(path: Path):
    if not path.exists() or path.stat().st_size < MIN_VALID_PDF_BYTES:
        return False
    try:
        with path.open("rb") as f:
            return f.read(5) == b"%PDF-"
    except:
        return False

def stream_download(url, dest: Path, session):
    for attempt in range(3):
        try:
            try:
                session.head(url, timeout=TIMEOUT, allow_redirects=True)
            except:
                pass

            with session.get(url, stream=True, allow_redirects=True, timeout=TIMEOUT) as r:
                r.raise_for_status()
                with dest.open("wb") as f:
                    for chunk in r.iter_content(16384):
                        if chunk:
                            f.write(chunk)

            if is_valid_pdf(dest):
                return True
            dest.unlink(missing_ok=True)
        except:
            if attempt < 2:
                time.sleep(random.uniform(1, 3))
                continue
            return False
    return False

# -------- Access check utilities --------
def check_single_access(name, url, session):
    try:
        r = session.get(url, timeout=min(60, TIMEOUT), allow_redirects=True)
        if r.status_code == 200 and r.content[:5] == b"%PDF-":
            msg = f"✔ {name}: PDF access OK"
        else:
            msg = f"⚠ {name}: non-PDF or blocked (status {r.status_code})"
    except Exception as e:
        msg = f"⚠ {name}: access check failed ({e})"
    print(msg)
    return name, msg

def check_multi_access(session):
    print("=== Running NYU/VPN access checks (publisher PDFs) ===")
    results = {}
    for name, url in TEST_URLS.items():
        k, msg = check_single_access(name, url, session)
        results[k] = msg
    print("=== Access check completed ===\n")
    return results

# -----------------------------
# HTML → PDF EXTRACTION
# -----------------------------
def extract_pdf_links_from_html(html, base):
    links = []

    # citation_pdf_url
    for m in re.finditer(
        r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]*content=["\']([^"\']+)["\']',
        html, flags=re.IGNORECASE):
        links.append(urllib.parse.urljoin(base, m.group(1)))

    # href
    for m in re.finditer(r'href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE):
        href = m.group(1)
        if ".pdf" in href.lower() or "/pdf" in href.lower():
            links.append(urllib.parse.urljoin(base, href))

    # embed + iframe
    for tag in ["embed", "iframe"]:
        for m in re.finditer(fr'<{tag}[^>]+src=["\']([^"\']+)["\']', html, flags=re.IGNORECASE):
            src = m.group(1)
            if ".pdf" in src.lower() or "/pdf" in src.lower():
                links.append(urllib.parse.urljoin(base, src))

    return list(dict.fromkeys(links))

# -----------------------------
# RESOLVERS
# -----------------------------
def resolve_unpaywall(doi, session):
    api = f"https://api.unpaywall.org/v2/{doi}?email={CONTACT_EMAIL}"
    time.sleep(SLEEP_BETWEEN_REQS)
    try:
        r = session.get(api, timeout=TIMEOUT)
    except:
        return (None, None)
    if r.status_code != 200:
        return (None, None)
    try:
        data = r.json()
    except:
        return (None, None)

    urls = []
    best = data.get("best_oa_location") or {}
    for key in ("url_for_pdf", "url"):
        if best.get(key):
            urls.append(best[key])

    for loc in data.get("oa_locations") or []:
        for key in ("url_for_pdf", "url"):
            if loc.get(key):
                urls.append(loc[key])

    for u in urls:
        if robots_allowed(u):
            return ("unpaywall", u)
    return (None, None)

def resolve_crossref(doi, session):
    """
    New: Crossref resolver.
    Uses Crossref's 'link' array to find content-type=application/pdf URLs.
    """
    api = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"
    time.sleep(SLEEP_BETWEEN_REQS)
    try:
        r = session.get(api, timeout=TIMEOUT)
    except:
        return (None, None)
    if r.status_code != 200:
        return (None, None)
    try:
        js = r.json()
    except:
        return (None, None)

    msg = js.get("message") or {}
    links = msg.get("link") or []
    for ln in links:
        url = ln.get("URL")
        ctype = (ln.get("content-type") or "").lower()
        if url and "pdf" in ctype and robots_allowed(url):
            return ("crossref", url)
    return (None, None)

def resolve_core(doi, session):
    api = f"https://api.core.ac.uk/internal/works?doi={urllib.parse.quote(doi)}"
    time.sleep(SLEEP_BETWEEN_REQS)
    try:
        r = session.get(api, timeout=TIMEOUT)
    except:
        return (None, None)
    if r.status_code != 200:
        return (None, None)
    try:
        js = r.json()
    except:
        return (None, None)
    for it in js.get("results", []) or []:
        pdf = it.get("downloadUrl") or it.get("fullTextLink")
        if pdf and robots_allowed(pdf):
            return ("core", pdf)
    return (None, None)

def resolve_openalex(doi, session):
    api = f"https://api.openalex.org/works/https://doi.org/{urllib.parse.quote(doi)}"
    time.sleep(SLEEP_BETWEEN_REQS)
    try:
        r = session.get(api, timeout=TIMEOUT)
    except:
        return (None, None)
    if r.status_code != 200:
        return (None, None)
    try:
        data = r.json()
    except:
        return (None, None)

    urls = []

    loc = data.get("primary_location") or {}
    if loc.get("pdf_url"):
        urls.append(loc["pdf_url"])

    for e in data.get("locations") or []:
        if e.get("pdf_url"):
            urls.append(e["pdf_url"])
        src = e.get("source") or {}
        u = src.get("url")
        if u and (u.endswith(".pdf") or "/pdf" in u.lower()):
            urls.append(u)

    for u in urls:
        if robots_allowed(u):
            return ("openalex", u)
    return (None, None)

def resolve_semanticscholar(doi, session):
    api = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=openAccessPdf"
    time.sleep(SLEEP_BETWEEN_REQS)
    try:
        r = session.get(api, timeout=TIMEOUT)
    except:
        return (None, None)
    if r.status_code != 200:
        return (None, None)
    try:
        data = r.json()
    except:
        return (None, None)

    pdf = (data.get("openAccessPdf") or {}).get("url")
    if pdf and robots_allowed(pdf):
        return ("semanticscholar", pdf)
    return (None, None)

def _publisher_specific_heuristics(final_url):
    urls = []
    lower = final_url.lower().rstrip("/")

    # ScienceDirect
    if "sciencedirect.com" in lower:
        if not lower.endswith("pdfft"):
            urls.append(lower + "/pdfft")
        urls.append(lower + "?download=1")
        urls.append(lower + "/pdfft?isDTMRedir=true&download=1")

    # Wiley
    if "onlinelibrary.wiley.com" in lower and "pdfdirect" not in lower:
        urls.append(lower.replace("/doi/", "/doi/pdfdirect/"))

    # SpringerLink
    if "link.springer.com" in lower:
        if "/content/pdf" not in lower:
            urls.append(lower.replace("/article/", "/content/pdf/") + ".pdf")

    return urls

def resolve_publisher_doi(doi, session):
    doi_url = f"https://doi.org/{doi}"
    time.sleep(SLEEP_BETWEEN_REQS)

    try:
        r = session.get(doi_url, timeout=TIMEOUT, allow_redirects=True)
    except:
        return (None, None)

    final = r.url

    if is_pdf_response(r, final) and robots_allowed(final):
        return ("publisher_direct", final)

    try:
        html = r.text
    except:
        html = ""

    for cand in extract_pdf_links_from_html(html, final):
        if robots_allowed(cand):
            return ("publisher_html_pdf", cand)

    base = final.rstrip("/")
    for cand in [base + ".pdf", base + "/pdf"]:
        if robots_allowed(cand):
            return ("publisher_pdf_fallback", cand)

    for cand in _publisher_specific_heuristics(final):
        if robots_allowed(cand):
            return ("publisher_pdf_heuristic", cand)

    return (None, None)

def resolve_pmc(pmcid, session):
    pmcid = normalize_pmcid(pmcid)
    if not pmcid:
        return (None, None)
    url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/pdf/"
    return ("pmc", url)

def resolve_pubmed_fulltext(pmid, session):
    if not pmid:
        return (None, None)
    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    time.sleep(SLEEP_BETWEEN_REQS)
    try:
        r = session.get(url, timeout=TIMEOUT)
    except:
        return (None, None)
    if r.status_code != 200:
        return (None, None)

    html = r.text

    for m in re.finditer(
        r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]*content=["\']([^"\']+)["\']',
        html, flags=re.IGNORECASE):
        cand = urllib.parse.urljoin(url, m.group(1))
        if robots_allowed(cand):
            return ("pubmed_meta_pdf", cand)

    for cand in extract_pdf_links_from_html(html, url):
        if robots_allowed(cand):
            return ("pubmed_html_pdf", cand)

    return (None, None)

# Resolver order: publisher → Unpaywall → Crossref → CORE → OpenAlex → S2
RESOLVERS = [
    resolve_publisher_doi,
    resolve_unpaywall,
    resolve_crossref,
    resolve_core,
    resolve_openalex,
    resolve_semanticscholar,
]

# -----------------------------
# COUNTS
# -----------------------------
def load_stage_counts():
    summary = RPYS_DIR / f"{TOPIC}_rpys_summary.txt"
    stage0 = stage1 = stage15 = 0
    if summary.exists():
        txt = summary.read_text()
        m = re.search(r"WoS metadata:\s*(\d+)", txt)
        wos_n = int(m.group(1)) if m else 0

        m = re.search(r"PubMed metadata:\s*(\d+)", txt)
        pm_n = int(m.group(1)) if m else 0

        stage0 = wos_n + pm_n

        m = re.search(r"Seminal refs:\s*(\d+)", txt)
        stage1 = int(m.group(1)) if m else 0

        stage15 = stage0

    stage25 = len(pd.read_csv(INPUT_FILE)) if INPUT_FILE.exists() else 0
    return stage0, stage1, stage15, stage25

# -----------------------------
# PROCESS EACH PAPER
# -----------------------------
def process_row(idx, row, session):
    doi = extract_clean_doi(row.get("doi", ""))
    pmid = str(row.get("pmid", "")).strip() if "pmid" in row else ""
    pmcid = normalize_pmcid(str(row.get("pmcid", "")).strip())
    paper_id = str(row.get("paper_id", "")).strip()

    base_id = doi or pmcid or pmid or paper_id or f"row_{idx}"
    dest = PAPERS_DIR / doi_to_filename(doi, base_id)

    if dest.exists() and is_valid_pdf(dest):
        return {"idx": idx, "paper_id": paper_id, "doi": doi,
                "pmid": pmid, "pmcid": pmcid, "source": "cache", "status": "success"}

    attempted = False

    # 1) PMC
    if pmcid:
        attempted = True
        src, url = resolve_pmc(pmcid, session)
        if url and stream_download(url, dest, session):
            return {"idx": idx, "paper_id": paper_id, "doi": doi,
                    "pmid": pmid, "pmcid": pmcid, "source": src, "status": "success"}

    # 2) PubMed via pmid
    if pmid:
        attempted = True
        src, url = resolve_pubmed_fulltext(pmid, session)
        if url and stream_download(url, dest, session):
            return {"idx": idx, "paper_id": paper_id, "doi": doi,
                    "pmid": pmid, "pmcid": pmcid, "source": src, "status": "success"}

    # 3) DOI chain
    if doi:
        attempted = True
        for resolver in RESOLVERS:
            src, url = resolver(doi, session)
            if url and stream_download(url, dest, session):
                return {"idx": idx, "paper_id": paper_id, "doi": doi,
                        "pmid": pmid, "pmcid": pmcid, "source": src, "status": "success"}

    if not doi and not pmcid and not pmid:
        status = "no_doi"
    elif attempted:
        status = "failed"
    else:
        status = "no_doi"

    return {"idx": idx, "paper_id": paper_id, "doi": doi or "",
            "pmid": pmid, "pmcid": pmcid or "", "source": "", "status": status}

# -----------------------------
# MAIN
# -----------------------------
def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)
    stage0, stage1, stage15, stage25 = load_stage_counts()

    print(f"=== Starting bulk download for {TOPIC} ===")
    print(f"Stage0: {stage0}, Stage1: {stage1}, Stage25: {stage25}\n")

    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://doi.org",
        "Upgrade-Insecure-Requests": "1",
        "DNT": "1",
    })

    # Access check
    access_results = check_multi_access(session)

    start = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(process_row, i, row, session): i for i, row in df.iterrows()}
        total = len(futs)
        for j, fut in enumerate(as_completed(futs), 1):
            results.append(fut.result())
            if j % 100 == 0 or j == total:
                rate = j / (time.time() - start)
                print(f"[Progress] {j}/{total}  ({rate:.2f} papers/sec)")

    # Log file
    with LOG_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["idx", "paper_id", "doi", "pmid", "pmcid", "source", "status"],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    failed = [r for r in results if r["status"] != "success"]
    pd.DataFrame(failed).to_csv(MANUAL_REVIEW_FILE, index=False)

    counts = Counter(r["status"] for r in results)
    src_counts = Counter(r["source"] for r in results if r["status"] == "success")

    runtime = (time.time() - start) / 60

    with SUMMARY_FILE.open("w", encoding="utf-8") as f:
        f.write("========== FINAL DOWNLOAD SUMMARY ==========\n")
        f.write(f"Topic: {TOPIC}\n\n")
        f.write(f"Stage0 metadata: {stage0}\n")
        f.write(f"Stage1 seminal refs: {stage1}\n")
        f.write(f"Stage25 selected: {stage25}\n\n")

        f.write("Access check results (NYU/VPN):\n")
        for name, msg in access_results.items():
            f.write(f"  {name:12} : {msg}\n")
        f.write("\n")

        f.write("Download outcomes:\n")
        for k, v in counts.items():
            f.write(f"  {k:10} : {v}\n")
        f.write("\nSuccessful sources:\n")
        for k, v in src_counts.items():
            f.write(f"  {k or 'unknown'} : {v}\n")
        f.write(f"\nManual review: {MANUAL_REVIEW_FILE}\n")
        f.write(f"Runtime: {runtime:.2f} minutes\n")
        f.write("============================================\n")

    print("\n✔ DONE")
    print(f"Summary : {SUMMARY_FILE}")
    print(f"Log     : {LOG_FILE}")
    print(f"Review  : {MANUAL_REVIEW_FILE}")

if __name__ == "__main__":
    main()
