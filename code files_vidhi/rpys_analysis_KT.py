"""
RPYS Analysis (Updated)
-----------------------
This performs reference publication year spectroscopy on:
 - Web of Science .bib metadata
 - PubMed metadata via reference_edges

NEW ADDITION:
 - Computes metadata-level RPYS influence scores:
       how strongly each metadata paper cites RPYS-peak references.

Outputs:
  reports/rpys/<topic>_rpys_reference_peaks.csv
  reports/rpys/<topic>_metadata_rpys_scores.csv
  reports/rpys/<topic>_rpys_summary.txt
"""

import os, re, glob
import pandas as pd
from pathlib import Path
from collections import defaultdict
from statistics import median

import bibtexparser

# -----------------------------
# CONFIG
# -----------------------------
TOPIC = "kidney_transplant"

ROOT_DIR = Path(__file__).resolve().parents[1]
RPYS_DIR = ROOT_DIR / "reports" / "rpys"
RPYS_DIR.mkdir(parents=True, exist_ok=True)

# WoS paths
WOS_DIR = ROOT_DIR / "data" / "WoS" / TOPIC
WOS_SUBDIRS = [
    f"{TOPIC} AND surgical management",
    f"{TOPIC} AND natural history",
    f"{TOPIC} AND pathophysiology",
]

# PubMed paths
PUBMED_DIR = ROOT_DIR / "data" / "pubmed" / TOPIC
PUBMED_EDGES = PUBMED_DIR / "pubmed_references_edges.csv"
PUBMED_MASTER = PUBMED_DIR / "pubmed_master.csv"

# -----------------------------
# Helpers
# -----------------------------
YEAR_RE = re.compile(r"\b(18|19|20)\d{2}\b")
DOI_RE  = re.compile(r"\b10\.\d{4,9}/\S+\b", re.IGNORECASE)

def extract_year(text):
    m = YEAR_RE.search(text)
    return int(m.group(0)) if m else None

def extract_doi(text):
    m = DOI_RE.search(text)
    return m.group(0).rstrip(".,;") if m else None

def normalize_ref(text):
    t = text.upper()
    t = re.sub(r"[^A-Z0-9 ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:300]

# -----------------------------
# WoS Reference Extraction
# -----------------------------
def run_wos_rpys():
    parser = bibtexparser.bparser.BibTexParser(common_strings=True)

    bib_files = []
    for sub in WOS_SUBDIRS:
        bib_files.extend(glob.glob(str((WOS_DIR / sub / "*.bib").resolve())))

    entries = []
    for p in bib_files:
        with open(p, "r", encoding="utf-8", errors="ignore") as fh:
            db = bibtexparser.load(fh, parser=parser)
            entries.extend(db.entries)

    # For metadata-level scoring
    wos_metadata = []

    ref_records = []

    def find_cr(e):
        for k, v in e.items():
            if k.lower() in {"cr", "cited-references", "cited_references", "citedreferences"}:
                return v
        return None

    def split_cr(text):
        return [p.strip() for p in re.split(r"[\n;]\s*", text) if p.strip()]

    for e in entries:
        ut = e.get("ut") or e.get("ID") or e.get("title", "")[:80]

        wos_metadata.append({
            "paper_id": ut,
            "doi": e.get("doi", "").lower().strip()
        })

        cr_blob = find_cr(e)
        if not cr_blob:
            continue

        for ref in split_cr(cr_blob):
            year = extract_year(ref)
            if not year:
                continue
            doi = extract_doi(ref)
            key = doi if doi else normalize_ref(ref)

            ref_records.append({
                "citing_id": ut,
                "ref_year": year,
                "ref_key": key,
                "ref_doi": doi
            })

    return pd.DataFrame(ref_records), pd.DataFrame(wos_metadata)

# -----------------------------
# PubMed Reference Extraction
# -----------------------------
def run_pubmed_rpys():
    if not PUBMED_EDGES.exists():
        return pd.DataFrame(), pd.DataFrame()

    edges = pd.read_csv(PUBMED_EDGES, dtype=str)
    master = pd.read_csv(PUBMED_MASTER, dtype=str) if PUBMED_MASTER.exists() else pd.DataFrame()

    # PubMed metadata-level paper IDs
    pm_metadata = []
    if not master.empty:
        for _, row in master.iterrows():
            pm_metadata.append({
                "paper_id": str(row["pmid"]),
                "doi": str(row.get("doi", "")).lower().strip()
            })

    records = []
    for _, row in edges.iterrows():
        citing = row["focal_pmid"]
        cited  = row["ref_pmid"]

        ref_year = None  # year unavailable in pubmed edges
        records.append({
            "citing_id": citing,
            "ref_year": None,   # will merge with RPYS peaks later
            "ref_key": cited,
            "ref_doi": cited if cited.startswith("10.") else None
        })

    return pd.DataFrame(records), pd.DataFrame(pm_metadata)

# -----------------------------
# Compute RPYS peaks
# -----------------------------
def detect_peak_years(ref_df):
    counts = ref_df[ref_df["ref_year"].notna()].groupby("ref_year").size()

    diffs = counts - counts.rolling(5, center=True, min_periods=1).median()
    z = (diffs - diffs.mean()) / diffs.std(ddof=0)
    peak_years = set(z[z > 1.0].index)

    return counts, diffs, z, peak_years

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    # --- Collect reference-level RPYS input ---
    wos_refs, wos_meta = run_wos_rpys()
    pm_refs,  pm_meta  = run_pubmed_rpys()

    refs = pd.concat([wos_refs, pm_refs], ignore_index=True)
    metadata = pd.concat([wos_meta, pm_meta], ignore_index=True)
        # ==========================================================
    # DOI CLEANING + DOI RECOVERY (NEW STEP)
    # ==========================================================

    import re
    import requests

    def extract_clean_doi(s):
        """
        Extracts, normalizes, and validates a DOI from a messy string.
        Returns clean DOI or None.
        """

        if not isinstance(s, str):
            return None

        s = s.strip().lower()

        # Empty or obvious junk
        if s in ("", "nan", "none", "no doi", "null", "0", ".", "-"):
            return None

        # If DOI is inside a URL
        if "doi.org/" in s:
            s = s.split("doi.org/")[-1].strip()

        # Remove prefixes like "doi:", "doi "
        s = re.sub(r"^doi[:\s]+", "", s)

        # Remove trailing punctuation
        s = s.strip(" ;,.\n\t")

        # Extract embedded DOI if the string contains more than one word
        m = re.search(r"(10\.\d{4,9}/[^\s]+)", s)
        if m:
            s = m.group(1)

        # Clean whitespace & illegal characters
        s = s.replace(" ", "").replace("\\", "").replace('"', "")

        # Strict DOI validation
        if re.match(r"^10\.\d{4,9}/\S+$", s):
            return s

        return None


    # ----------------------------------------------------------
    # DOI RECOVERY HELPERS
    # ----------------------------------------------------------

    def recover_from_pubmed(pmid):
        """Fetch DOI from PubMed ESummary."""
        if not str(pmid).isdigit():
            return None
        try:
            url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={pmid}&retmode=json"
            r = requests.get(url, timeout=10)
            d = r.json()
            for item in d["result"].values():
                if isinstance(item, dict) and "elocationid" in item:
                    el = item["elocationid"]
                    m = re.search(r"(10\.\d{4,9}/\S+)", el)
                    if m:
                        return m.group(1)
        except:
            return None


    def recover_from_crossref(title):
        """Query Crossref using the title."""
        if not isinstance(title, str) or len(title) < 5:
            return None
        try:
            url = f"https://api.crossref.org/works?query.title={title}&rows=1"
            r = requests.get(url, timeout=10)
            items = r.json()["message"]["items"]
            if items:
                doi = items[0].get("DOI")
                if doi and extract_clean_doi(doi):
                    return extract_clean_doi(doi)
        except:
            return None


    def recover_from_openalex(pmid):
        """Recover DOI from PMIDs using OpenAlex."""
        if not str(pmid).isdigit():
            return None
        try:
            url = f"https://api.openalex.org/works/pmid:{pmid}"
            r = requests.get(url, timeout=10)
            data = r.json()
            doi = data.get("doi")
            if doi and extract_clean_doi(doi):
                return extract_clean_doi(doi)
        except:
            return None


    # ----------------------------------------------------------
    # APPLY CLEANING + RECOVERY
    # ----------------------------------------------------------

    print("\n=== Cleaning and recovering DOIs... ===")

    cleaned_dois = []
    missing_doi_count = 0
    recovered_count = 0

    for i, row in metadata.iterrows():
        raw = row.get("doi", "")
        clean = extract_clean_doi(raw)

        if clean:
            cleaned_dois.append(clean)
            continue

        # Try DOI recovery using PMIDs
        pmid = row.get("paper_id")
        recovered = recover_from_openalex(pmid) or recover_from_pubmed(pmid)

        # Try title-based Crossref search if metadata has titles
        if not recovered and "title" in row:
            recovered = recover_from_crossref(row["title"])

        if recovered:
            cleaned_dois.append(recovered)
            recovered_count += 1
        else:
            cleaned_dois.append(None)
            missing_doi_count += 1

    metadata["clean_doi"] = cleaned_dois

    print(f"Clean DOIs fixed:     {len(metadata) - missing_doi_count}")
    print(f"Recovered DOIs:       {recovered_count}")
    print(f"Still missing DOIs:   {missing_doi_count}")
    print("========================================\n")

    # Replace original DOI with cleaned DOI for downstream usage
    metadata["doi"] = metadata["clean_doi"]
    metadata.drop(columns=["clean_doi"], inplace=True)


    # SAVE Stage-0 counts
    stage0_wos = len(wos_meta)
    stage0_pm  = len(pm_meta)

    # --- Run RPYS on references ---
    counts, diffs, z, peak_years = detect_peak_years(refs)

    # Save reference-level peaks (unchanged)
    peaks_path = RPYS_DIR / f"{TOPIC}_rpys_reference_peaks.csv"
    counts.to_frame("count").to_csv(peaks_path)

    # --- Identify peak-year references ---
    peak_refs = refs[refs["ref_year"].isin(peak_years)]

    # Compute attributable citations = counts of ref_key in peak years
    seminal_refs = (
        peak_refs.groupby(["ref_key", "ref_doi"], dropna=False)
        .size()
        .reset_index(name="attributable_citations")
    )

    # Save reference-level seminal list (unchanged)
    seminal_path = RPYS_DIR / f"{TOPIC}_rpys_reference_seminal.csv"
    seminal_refs.to_csv(seminal_path, index=False)

    # -----------------------------
    # NEW: metadata-level RPYS influence scoring
    # -----------------------------
    ref_weights = dict(zip(seminal_refs["ref_key"], seminal_refs["attributable_citations"]))

    # Score each metadata paper
    influence = defaultdict(float)

    for _, row in refs.iterrows():
        if row["ref_key"] in ref_weights:
            influence[row["citing_id"]] += ref_weights[row["ref_key"]]

    mdf = metadata.copy()
    mdf["rpys_raw_score"] = mdf["paper_id"].map(influence).fillna(0.0)

    # normalize
    if mdf["rpys_raw_score"].max() > 0:
        mdf["rpys_norm"] = mdf["rpys_raw_score"] / mdf["rpys_raw_score"].max()
    else:
        mdf["rpys_norm"] = 0.0

    # Save metadata-level RPYS scores
    meta_scores_path = RPYS_DIR / f"{TOPIC}_metadata_rpys_scores.csv"
    mdf.to_csv(meta_scores_path, index=False)

    # -----------------------------
    # SAVE SUMMARY
    # -----------------------------
    summary_file = RPYS_DIR / f"{TOPIC}_rpys_summary.txt"
    with open(summary_file, "w") as f:
        f.write(f"RPYS Summary for {TOPIC}\n\n")
        f.write(f"Stage 0 - Metadata counts:\n")
        f.write(f"  WoS metadata papers   : {stage0_wos}\n")
        f.write(f"  PubMed metadata papers: {stage0_pm}\n")
        f.write(f"  Total metadata        : {stage0_wos + stage0_pm}\n\n")
        f.write(f"Stage 1 - Reference-level RPYS:\n")
        f.write(f"  Total cited refs      : {len(refs)}\n")
        f.write(f"  Peak years detected   : {sorted(peak_years)}\n")
        f.write(f"  Seminal references    : {len(seminal_refs)}\n\n")
        f.write(f"Stage 1.5 - Metadata RPYS influence:\n")
        f.write(f"  Metadata papers scored: {len(mdf)}\n")
        f.write(f"  Max rpys_raw_score    : {mdf['rpys_raw_score'].max()}\n")

    print("\n✔ RPYS complete.")
    print(f"  Reference peaks saved to: {peaks_path}")
    print(f"  Metadata RPYS scores   : {meta_scores_path}")
    print(f"  Summary                : {summary_file}\n")
