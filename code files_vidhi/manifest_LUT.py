"""
Manifest Builder (Merged Sources)
---------------------------------
Combines PubMed + WoS combined seminal article lists for a topic
and builds a single Unpaywall manifest (metadata only).

This version adds:
 - ETA progress printing
 - Total runtime printing

It does NOT compute any final consolidated summaries
(successful download counts belong in the bulk_downloader script).
"""

import pandas as pd
import requests
import os
import time
from time import sleep
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
TOPIC = "lung_transplant"
CONTACT_EMAIL = "vm2725@nyu.edu"
ROLLBACK_DELAY = 1.0

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Combined bibliometric output directory
COMBINED_DIR = os.path.join(BASE_DIR, "..", "reports", "combined", TOPIC)
os.makedirs(COMBINED_DIR, exist_ok=True)

WOS_FILE = os.path.join(COMBINED_DIR, f"{TOPIC}_wos_combined_seminal_articles.csv")
PUBMED_FILE = os.path.join(COMBINED_DIR, f"{TOPIC}_pubmed_combined_seminal_articles.csv")

OUTPUT_FILE  = os.path.join(COMBINED_DIR, f"manifest_{TOPIC}_merged.csv")

print(f"=== Building merged manifest for {TOPIC.replace('_',' ').title()} ===")

# -----------------------------
# LOAD + MERGE SOURCES
# -----------------------------
dfs = []
for src, path in [("WoS", WOS_FILE), ("PubMed", PUBMED_FILE)]:
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["source"] = src
        dfs.append(df)
    else:
        print(f"[WARN] Missing: {path}")

if not dfs:
    raise SystemExit("No input combined seminal article lists found.")

merged = pd.concat(dfs, ignore_index=True)

# Detect DOI column
doi_col = None
for c in ["ref_doi", "doi"]:
    if c in merged.columns:
        doi_col = c
        break
if not doi_col:
    raise ValueError("No DOI column found in inputs.")

# Deduplicate
merged = merged.dropna(subset=[doi_col])
merged = merged.drop_duplicates(subset=[doi_col])

total_entries = len(merged)
print(f"Total papers to process: {total_entries}\n")

merged["oa_status"] = ""
merged["pdf_url"] = ""

# -----------------------------
# FUNCTION: Unpaywall fetch
# -----------------------------
def fetch_unpaywall(doi):
    url = f"https://api.unpaywall.org/v2/{doi}?email={CONTACT_EMAIL}"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            data = r.json()
            oa_status = data.get("oa_status", "unknown")
            best = data.get("best_oa_location")
            pdf_url = best.get("url_for_pdf") if best else None
            return oa_status, pdf_url
        else:
            return "error", None
    except:
        return "error", None

# -----------------------------
# MAIN LOOP WITH ETA
# -----------------------------
start_time = time.time()

for idx, row in merged.iterrows():
    doi = str(row[doi_col]).strip()
    if not doi or doi.lower() == "nan":
        continue

    # Calculate ETA
    elapsed = time.time() - start_time
    progress = (idx + 1) / total_entries
    eta = elapsed / progress - elapsed if progress > 0 else 0

    print(f"[{idx+1}/{total_entries}] DOI: {doi} | ETA: {eta/60:.2f} min")

    oa_status, pdf_url = fetch_unpaywall(doi)
    merged.at[idx, "oa_status"] = oa_status
    merged.at[idx, "pdf_url"] = pdf_url if pdf_url else ""

    sleep(ROLLBACK_DELAY)

end_time = time.time()
runtime_minutes = (end_time - start_time) / 60

# -----------------------------
# SAVE OUTPUT
# -----------------------------
merged.to_csv(OUTPUT_FILE, index=False)

print(f"\n✅ Manifest saved to: {OUTPUT_FILE}")
print(f"⏱ Total runtime: {runtime_minutes:.2f} minutes")
print("Use this manifest as input to bulk_downloader.py.\n")
