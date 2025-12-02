"""
Manifest Builder (Metadata-Level)
---------------------------------
Takes the metadata-level weighted-union output:
    <topic>_metadata_combined_seminal.csv

For each DOI:
 - Queries Unpaywall for OA status + best pdf link
 - Outputs manifest_<topic>_merged.csv into the same combined directory

Adds:
 - ETA progress printing
 - Total runtime printing
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
TOPIC = "liver_transplant"
CONTACT_EMAIL = "vm2725@nyu.edu"
ROLLBACK_DELAY = 1.0

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Combined bibliometric output directory
COMBINED_DIR = os.path.join(BASE_DIR, "..", "reports", "combined", TOPIC)
os.makedirs(COMBINED_DIR, exist_ok=True)

# NEW unified metadata-level input file
COMBINED_FILE = os.path.join(
    COMBINED_DIR,
    f"{TOPIC}_metadata_combined_seminal.csv"
)

# Manifest output file
OUTPUT_FILE = os.path.join(
    COMBINED_DIR,
    f"manifest_{TOPIC}_merged.csv"
)

print(f"=== Building unified metadata manifest for {TOPIC.replace('_',' ').title()} ===")

# -----------------------------
# LOAD SOURCE
# -----------------------------
if not os.path.exists(COMBINED_FILE):
    raise SystemExit(f"Combined metadata file not found: {COMBINED_FILE}")

merged = pd.read_csv(COMBINED_FILE)

# Detect DOI column
doi_col = None
for c in ["doi", "ref_doi"]:
    if c in merged.columns:
        doi_col = c
        break

if not doi_col:
    raise ValueError("No DOI column found in metadata combined file.")

# Deduplicate DOIs & drop missing
merged = merged.dropna(subset=[doi_col])
merged = merged.drop_duplicates(subset=[doi_col])

total_entries = len(merged)
print(f"Total metadata papers to process: {total_entries}\n")

merged["oa_status"] = ""
merged["pdf_url"]  = ""

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

    # ETA calculation
    elapsed = time.time() - start_time
    progress = (idx + 1) / total_entries
    eta = elapsed / progress - elapsed if progress > 0 else 0

    print(f"[{idx+1}/{total_entries}] DOI: {doi} | ETA: {eta/60:.2f} min")

    oa_status, pdf_url = fetch_unpaywall(doi)
    merged.at[idx, "oa_status"] = oa_status
    merged.at[idx, "pdf_url"]  = pdf_url if pdf_url else ""

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
