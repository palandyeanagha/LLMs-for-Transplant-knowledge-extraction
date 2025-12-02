# -*- coding: utf-8 -*-
"""
PubMed → References → RPYS pipeline (CPU-only).
Runs 3 PubMed queries, downloads metadata, builds reference graph (via ELink),
extracts reference publication years, and prepares RPYS input (year,count).

Outputs (saved to OUTPUT_DIR):
  - pubmed_query_q1_natural_history.csv
  - pubmed_query_q2_surgical_management.csv
  - pubmed_query_q3_pathophysiology.csv
  - pubmed_master.csv
  - pubmed_references_edges.csv        (focal_pmid, ref_pmid)
  - pubmed_reference_years.csv         (ref_pmid, year)
  - pubmed_rpys_input.csv              (year, count)
"""

import os
import time
import math
import csv
from collections import Counter
from typing import List, Dict, Tuple, Iterable

import pandas as pd

# Biopython Entrez/Medline
from Bio import Entrez, Medline
from urllib.error import HTTPError
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
TOPIC = "heart_transplant"       # or "kidney_transplant"

# dynamic path: go one level up from "code files/"
ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "data" / "pubmed" / TOPIC
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EMAIL = "vm2725@nyu.edu"         # required by NCBI
TOOL = "capstone_rpys_pubmed"
API_SLEEP = 0.5                  # seconds between NCBI calls (safe & slow)
MAX_RETRIES = 3                  # retry logic
BATCH_SIZE = 200                 # efetch batching for speed

# Exact queries for the current topic
QUERY_LIST: List[Tuple[str, str]] = [
    ("q1_natural_history",      f'"{TOPIC.replace("_", " ")}" AND "natural history"'),
    ("q2_surgical_management",  f'"{TOPIC.replace("_", " ")}" AND "surgical management"'),
    ("q3_pathophysiology",      f'"{TOPIC.replace("_", " ")}" AND "pathophysiology"'),
]

# -----------------------------
# SETUP
# -----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
Entrez.email = EMAIL
Entrez.tool = TOOL

def _sleep():
    time.sleep(API_SLEEP)

def _with_retries(func, *args, **kwargs):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
                wait = API_SLEEP * (2 ** (attempt - 1))
                print(f"[WARN] HTTP {e.code}. Retry {attempt}/{MAX_RETRIES} after {wait:.1f}s...")
                time.sleep(wait)
            else:
                raise

# -----------------------------
# NCBI HELPERS
# -----------------------------
def esearch_pmids(term, retmax=100000, usehistory="y"):
    """
    PubMed ESearch wrapper (robust to transient NCBI backend errors)
    Returns only the IdList for consistency across topics.
    """
    print(f"🔍 Running ESearch for: {term}")
    from urllib.error import HTTPError
    import time

    last_err = None
    for attempt in range(3):
        try:
            with Entrez.esearch(db="pubmed",
                                term=term,
                                retmode="xml",
                                retmax=retmax,
                                usehistory=usehistory) as handle:
                try:
                    record = Entrez.read(handle)
                    # ✅ extract only IdList even if dict returned
                    pmids = record.get("IdList", []) if isinstance(record, dict) else record
                    if not pmids and "IdList" in record:
                        pmids = record["IdList"]
                    return pmids
                except RuntimeError as e:
                    if "Search Backend failed" in str(e):
                        print(f"⚠️ PubMed backend failed (attempt {attempt+1}/3). Retrying...")
                        time.sleep(2)
                        continue
                    else:
                        raise
        except HTTPError as e:
            last_err = e
            if e.code >= 500:
                print(f"⚠️ HTTP {e.code} from NCBI (attempt {attempt+1}/3). Retrying...")
                time.sleep(2)
                continue
            else:
                raise

        time.sleep(0.5)
    # final lightweight fallback
    try:
        print("⚠️ All retries failed, trying one last lightweight ESearch...")
        with Entrez.esearch(db="pubmed", term=term, retmode="xml", retmax=min(1000, retmax)) as handle:
            record = Entrez.read(handle)
        pmids = record.get("IdList", []) if isinstance(record, dict) else record
        if not pmids and "IdList" in record:
            pmids = record["IdList"]
        return pmids
    except Exception:
        raise RuntimeError("❌ PubMed ESearch failed after all attempts — try again later.") from last_err


def efetch_medline(pmids, batch_size=200):
    from Bio import Entrez
    from urllib.error import HTTPError
    import time

    pmids = [str(p).strip() for p in pmids if str(p).strip()]
    if not pmids:
        print("⚠️ No valid PMIDs to fetch.")
        return []

    records = []
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i+batch_size]
        joined = ",".join(batch)
        for attempt in range(3):
            try:
                with Entrez.efetch(db="pubmed", id=joined,
                                   rettype="medline", retmode="text") as handle:
                    data = handle.read()
                records.append(data)
                time.sleep(0.5)
                break
            except HTTPError as e:
                if e.code in (400,) or (500 <= e.code < 600):
                    print(f"⚠️ EFetch HTTP {e.code} for batch starting at {i}, attempt {attempt+1}/3. Retrying...")
                    time.sleep(2)
                    continue
                raise
        else:
            print(f"❌ Skipping batch starting at {i} after repeated failures.")
    return records


def elink_references(pmid: str) -> List[str]:
    """Return PMIDs of references cited by the focal PubMed article (via ELink)."""
    _sleep()
    handle = _with_retries(Entrez.elink, dbfrom="pubmed", id=pmid, linkname="pubmed_pubmed_refs")
    record = Entrez.read(handle)
    links = record[0].get("LinkSetDb", [])
    if links:
        return [link["Id"] for link in links[0]["Link"]]
    return []

# -----------------------------
# PARSERS
# -----------------------------
def parse_medline_record(rec: dict) -> dict:
    """Extract core fields from a MEDLINE record."""
    pmid = rec.get("PMID", "")
    title = rec.get("TI", "")
    journal = rec.get("JT", "")
    abstract = rec.get("AB", "")
    authors = "; ".join(rec.get("AU", []))

    # Year extraction
    year = ""
    dp = rec.get("DP", "")
    if dp and len(dp) >= 4 and dp[:4].isdigit():
        year = dp[:4]
    elif "PHST" in rec:
        for stamp in rec["PHST"]:
            if len(stamp) >= 4 and stamp[:4].isdigit():
                year = stamp[:4]
                break

    doi = ""
    aids = rec.get("AID", [])
    if isinstance(aids, list):
        for a in aids:
            if a.endswith(" [doi]"):
                doi = a.replace(" [doi]", "")
                break
    elif isinstance(aids, str) and aids.endswith(" [doi]"):
        doi = aids.replace(" [doi]", "")

    pmcid = rec.get("PMC", "")
    if not pmcid and isinstance(aids, list):
        for a in aids:
            if a.startswith("PMCID:"):
                pmcid = a.split(":", 1)[1]
                break

    return {
        "pmid": pmid,
        "doi": doi,
        "pmcid": pmcid,
        "title": title,
        "journal": journal,
        "year": year,
        "authors": authors,
        "abstract": abstract,
    }

# -----------------------------
# PIPELINE
# -----------------------------
def run_query_save(term_key: str, term: str) -> pd.DataFrame:
    print(f"\n[Stage 0] Searching PubMed for: {term_key} -> {term}")
    pmids = esearch_pmids(term)
    print(f"  Found {len(pmids)} PMIDs.")

    if not pmids:
        print(f"⚠️ No PMIDs found for {term_key}. Skipping.")
        return pd.DataFrame()

    print("  Fetching MEDLINE metadata...")
    recs = efetch_medline(pmids)
    # recs are text blocks, need parsing
    parsed = []
    for block in recs:
        for rec in Medline.parse(block.splitlines(True)):
            parsed.append(parse_medline_record(rec))

    df = pd.DataFrame(parsed)
    out_csv = os.path.join(OUTPUT_DIR, f"pubmed_query_{term_key}.csv")
    df.to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"  Saved: {out_csv} ({len(df)} rows)")
    return df

def build_master(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    print("\n[Stage 0B] Merging + de-duplicating...")
    master = pd.concat(dfs, ignore_index=True)
    before = len(master)
    master = master.sort_values(["doi", "pmid"]).drop_duplicates(subset=["doi", "pmid"], keep="first")
    after = len(master)
    print(f"  Master merged: {before} -> {after} unique rows.")
    out_csv = os.path.join(OUTPUT_DIR, "pubmed_master.csv")
    master.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv}")
    return master

def fetch_reference_edges(focal_pmids: List[str]) -> pd.DataFrame:
    print("\n[Stage 1] Retrieving cited-reference PMIDs via ELink...")
    edges: List[Tuple[str, str]] = []
    for i, pmid in enumerate(focal_pmids, 1):
        try:
            refs = elink_references(pmid)
            edges.extend([(pmid, r) for r in refs])
        except Exception as e:
            print(f"  [ELink error] PMID {pmid}: {e}")
        if i % 50 == 0:
            print(f"  Processed {i}/{len(focal_pmids)} focal articles...")
    df_edges = pd.DataFrame(edges, columns=["focal_pmid", "ref_pmid"])
    out_csv = os.path.join(OUTPUT_DIR, "pubmed_references_edges.csv")
    df_edges.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv} ({len(df_edges)} edges)")
    return df_edges

def fetch_reference_years(unique_ref_pmids: List[str]) -> pd.DataFrame:
    print("\n[Stage 2] Fetching publication years for all reference PMIDs...")
    total_refs = len(unique_ref_pmids)
    recs = efetch_medline(unique_ref_pmids)
    rows = []
    missing_year = 0
    parsed_count = 0

    # each entry in recs is MEDLINE text; parse it first
    for block in recs:
        for rec in Medline.parse(block.splitlines(True)):
            parsed_count += 1
            pmid = rec.get("PMID", "")
            year = ""
            dp = rec.get("DP", "")
            if dp and len(dp) >= 4 and dp[:4].isdigit():
                year = dp[:4]
            elif "PHST" in rec:
                for stamp in rec["PHST"]:
                    if len(stamp) >= 4 and stamp[:4].isdigit():
                        year = stamp[:4]
                        break
            if pmid and year:
                rows.append({"ref_pmid": pmid, "year": int(year)})
            else:
                missing_year += 1

    df_ref_years = pd.DataFrame(rows).drop_duplicates(subset=["ref_pmid"])
    out_csv = os.path.join(OUTPUT_DIR, "pubmed_reference_years.csv")
    df_ref_years.to_csv(out_csv, index=False)

    # summary log
    print(f"  Parsed {parsed_count} MEDLINE records from {total_refs} reference PMIDs.")
    print(f"  → Found {len(df_ref_years)} refs with valid publication year.")
    print(f"  → Missing year for {missing_year} refs ({missing_year / max(parsed_count,1):.1%}).")
    print(f"  Saved: {out_csv} ({len(df_ref_years)} refs with year)")
    return df_ref_years


    df_ref_years = pd.DataFrame(rows).drop_duplicates(subset=["ref_pmid"])
    out_csv = os.path.join(OUTPUT_DIR, "pubmed_reference_years.csv")
    df_ref_years.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv} ({len(df_ref_years)} refs with year)")
    return df_ref_years

def build_rpys_input(edges: pd.DataFrame, ref_years: pd.DataFrame) -> pd.DataFrame:
    print("\n[Stage 3] Building RPYS input (year,count)...")
    merged = edges.merge(ref_years, left_on="ref_pmid", right_on="ref_pmid", how="inner")
    year_counts = Counter(merged["year"].astype(int))
    df_counts = pd.DataFrame(sorted(year_counts.items()), columns=["year", "count"])
    out_csv = os.path.join(OUTPUT_DIR, "pubmed_rpys_input.csv")
    df_counts.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv} ({len(df_counts)} years)")
    return df_counts

def main():
    print("=== PubMed → References → RPYS pipeline (safe-throttled) ===")

    dfs = []
    for key, term in QUERY_LIST:
        df = run_query_save(key, term)
        dfs.append(df)

    master = build_master([d for d in dfs if not d.empty])

    focal_pmids = [p for p in master["pmid"].astype(str) if p.strip()]
    edges = fetch_reference_edges(focal_pmids)

    if edges.empty:
        print("\n[NOTE] No reference edges found. RPYS will be empty.")
        return

    unique_ref_pmids = sorted(set(edges["ref_pmid"].astype(str)))
    ref_years = fetch_reference_years(unique_ref_pmids)

    if ref_years.empty:
        print("\n[NOTE] No reference publication years could be resolved.")
        return

    _ = build_rpys_input(edges, ref_years)

    print("\nDone. Next: run RPYS plotting on pubmed_rpys_input.csv.")
    print("Tip: use a 5-year rolling median for seminal-year peaks.")

if __name__ == "__main__":
    main()
