# -*- coding: utf-8 -*-
"""
Unified RPYS for WoS + PubMed WITH DOI CLEANING (HT trials)
-----------------------------------------------------------
- Uses:
    data/WoS/heart_transplant/**/*.bib
    data/pubmed/heart_transplant/pubmed_master.csv
    data/pubmed/heart_transplant/pubmed_references_edges.csv
    data/pubmed/heart_transplant/pubmed_reference_years.csv

- Outputs:
    reports/rpys/heart_transplant_rpys_reference_peaks.csv
    reports/rpys/heart_transplant_metadata_rpys_scores.csv
    reports/rpys/heart_transplant_rpys_summary.txt
"""

import re
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
import bibtexparser

# -----------------------------
# CONFIG
# -----------------------------
TOPIC = "heart_transplant"
ROOT_DIR = Path(__file__).resolve().parents[1]

WOS_DIR = ROOT_DIR / "data" / "WoS" / TOPIC
PUB_DIR = ROOT_DIR / "data" / "pubmed" / TOPIC

OUT_DIR = ROOT_DIR / "reports" / "rpys"
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEAR_RE = re.compile(r"\b(18|19|20)\d{2}\b")
DOI_RE  = re.compile(r"(10\.\d{4,9}/\S+)", re.IGNORECASE)


# -----------------------------
# Helpers
# -----------------------------
def normalize_ref_key(text: str) -> str:
    t = (text or "").upper()
    t = re.sub(r"[^A-Z0-9 ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:300]


def normalize_doi(raw) -> str | None:
    """String-only DOI cleaner; no external API calls."""
    if pd.isna(raw):
        return None
    s = str(raw).strip()
    if not s or s.lower() in {"nan", "none", "no doi", "null"}:
        return None

    # strip URL prefix and 'doi:' prefix
    s = s.replace(" ", "")
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^doi[:\s]+", "", s, flags=re.IGNORECASE)
    s = s.strip(" ;,.\n\t\r")

    # pull out embedded DOI if needed
    m = DOI_RE.search(s)
    if m:
        s = m.group(1)

    # fix 10_ → 10.
    s = s.replace("10_", "10.")

    # final trim
    s = s.strip().strip(".,;")

    if re.match(r"^10\.\d{4,9}/\S+$", s):
        return s
    return None


def clean_dois(metadata: pd.DataFrame) -> pd.DataFrame:
    """Clean and (lightly) recover DOIs in the metadata frame."""
    if "doi" not in metadata.columns:
        metadata["doi"] = np.nan

    cleaned = []
    recovered_n = 0
    missing_n = 0

    for _, row in metadata.iterrows():
        raw = row.get("doi", "")
        cd = normalize_doi(raw)

        if not cd:
            # light recovery from paper_id if it looks like a DOI
            pid = row.get("paper_id", "")
            rec = normalize_doi(pid)
            if rec:
                cd = rec
                recovered_n += 1

        if not cd:
            missing_n += 1

        cleaned.append(cd)

    metadata["doi"] = cleaned

    fixed_n = len(metadata) - missing_n

    print("\n=== Cleaning DOIs (string-based only) ===")
    print(f"Clean DOIs fixed:     {fixed_n}")
    print(f"Recovered from IDs:   {recovered_n}")
    print(f"Still missing DOIs:   {missing_n}")
    print("========================================\n")

    return metadata


# -----------------------------
# Load WoS metadata + refs
# -----------------------------
def load_wos():
    parser = bibtexparser.bparser.BibTexParser(common_strings=True)

    entries = []
    for bib in WOS_DIR.glob("**/*.bib"):
        try:
            with open(bib, "r", encoding="utf-8", errors="ignore") as f:
                db = bibtexparser.load(f, parser=parser)
                entries.extend(db.entries)
        except Exception:
            continue

    wos_meta = []
    wos_refs = []

    for e in entries:
        # paper_id MUST match what you'll use for BC/PR: ut → ID → title snippet
        pid = e.get("ut") or e.get("ID") or e.get("title", "")[:80]
        year_raw = e.get("year")
        try:
            year = int(year_raw)
        except Exception:
            year = None

        doi = e.get("doi", "")
        title = e.get("title", "")
        journal = e.get("journal", "") or e.get("journal-iso", "")

        wos_meta.append({
            "paper_id": str(pid),
            "year": year,
            "source": "wos",
            "doi": doi,
            "pmid": np.nan,
            "pmcid": np.nan,
            "title": title,
            "journal": journal,
        })

        cr_raw = (
            e.get("cr")
            or e.get("cited-references")
            or e.get("cited_references")
            or e.get("citedreferences")
            or ""
        )
        if not cr_raw:
            continue

        for ref in re.split(r"[\n;]+", cr_raw):
            ref = ref.strip()
            if not ref:
                continue
            my = YEAR_RE.search(ref)
            ref_year = int(my.group(0)) if my else None
            md = DOI_RE.search(ref)
            ref_doi = md.group(0).rstrip(".,;") if md else None
            key = ref_doi if ref_doi else normalize_ref_key(ref)
            wos_refs.append({
                "citing_id": str(pid),
                "ref_key": key,
                "ref_year": ref_year
            })

    return pd.DataFrame(wos_meta), pd.DataFrame(wos_refs)


# -----------------------------
# Load PubMed metadata + refs
# -----------------------------
def load_pubmed():
    meta_path = PUB_DIR / "pubmed_master.csv"
    edges_path = PUB_DIR / "pubmed_references_edges.csv"
    yrs_path   = PUB_DIR / "pubmed_reference_years.csv"

    if not meta_path.exists():
        return pd.DataFrame(), pd.DataFrame()

    meta = pd.read_csv(meta_path, dtype=str)

    # Basic columns
    meta["paper_id"] = meta["pmid"].astype(str)
    meta["year"] = pd.to_numeric(meta.get("year"), errors="coerce")
    meta["source"] = "pubmed"

    if "doi" not in meta.columns:
        meta["doi"] = np.nan
    if "pmcid" not in meta.columns:
        meta["pmcid"] = np.nan
    if "title" not in meta.columns:
        meta["title"] = ""
    if "journal" not in meta.columns:
        meta["journal"] = ""

    pm_meta = meta[["paper_id", "year", "source", "doi", "pmid", "pmcid", "title", "journal"]].copy()

    if not edges_path.exists() or not yrs_path.exists():
        return pm_meta, pd.DataFrame()

    edges = pd.read_csv(edges_path, dtype=str)
    yrs   = pd.read_csv(yrs_path, dtype=str)

    yrmap = {}
    for _, r in yrs.iterrows():
        ref_pmid = str(r["ref_pmid"])
        y = r.get("year")
        try:
            yrmap[ref_pmid] = int(y)
        except Exception:
            yrmap[ref_pmid] = None

    pm_refs = []
    for _, r in edges.iterrows():
        cited = str(r["ref_pmid"])
        y = yrmap.get(cited)
        pm_refs.append({
            "citing_id": str(r["focal_pmid"]),
            "ref_key": cited,
            "ref_year": y,
        })

    return pm_meta, pd.DataFrame(pm_refs)


# -----------------------------
# RPYS core
# -----------------------------
def run_rpys():
    # 1) Load metadata + refs
    wos_meta, wos_refs = load_wos()
    pm_meta, pm_refs   = load_pubmed()

    metadata = pd.concat([wos_meta, pm_meta], ignore_index=True)
    refs     = pd.concat([wos_refs, pm_refs], ignore_index=True)

    # 2) Clean DOIs at metadata level
    metadata = clean_dois(metadata)

    stage0_wos = len(wos_meta)
    stage0_pm  = len(pm_meta)

    # 3) Year counts for ref-year spectroscopy
    valid_refs = refs[refs["ref_year"].notna()].copy()
    if valid_refs.empty:
        raise RuntimeError("No valid reference years found for RPYS.")

    valid_refs["ref_year"] = valid_refs["ref_year"].astype(int)
    counts = valid_refs["ref_year"].value_counts().sort_index()

    # RPYS diff against rolling mean, then simple threshold
    diff = counts - counts.rolling(5, center=True, min_periods=1).mean()
    threshold = diff.mean() + diff.std(ddof=0)
    peak_years = set(diff[diff > threshold].index)

    # 4) Seminal references = refs in peak years
    seminal_refs = valid_refs[valid_refs["ref_year"].isin(peak_years)].copy()
    ref_weights = Counter(seminal_refs["ref_key"])

    # 5) Metadata-level RPYS influence
    influence = defaultdict(float)
    for _, r in valid_refs.iterrows():
        w = ref_weights.get(r["ref_key"])
        if w:
            influence[str(r["citing_id"])] += float(w)

    metadata["paper_id"] = metadata["paper_id"].astype(str)
    metadata["rpys_raw_score"] = metadata["paper_id"].map(lambda pid: influence.get(pid, 0.0)).astype(float)

    # 6) Save reference-level peaks (by year)
    peaks_path = OUT_DIR / f"{TOPIC}_rpys_reference_peaks.csv"
    counts.to_frame("count").to_csv(peaks_path)

    # 7) Save metadata-level scores (this feeds weighted union)
    meta_scores_path = OUT_DIR / f"{TOPIC}_metadata_rpys_scores.csv"
    metadata.to_csv(meta_scores_path, index=False)

    # 8) Summary (compatible with downstream parsers)
    summary_file = OUT_DIR / f"{TOPIC}_rpys_summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"Unified RPYS summary for {TOPIC}\n\n")
        f.write(f"WoS metadata:    {stage0_wos}\n")
        f.write(f"PubMed metadata: {stage0_pm}\n")
        f.write(f"Total metadata:  {len(metadata)}\n\n")
        f.write(f"Peak years:      {sorted(peak_years)}\n")
        f.write(f"Seminal refs:    {len(seminal_refs)}\n")
        f.write(f"Metadata scored: {len(metadata)}\n")

    print("\n✔ RPYS complete for", TOPIC)
    print(f"  Reference peaks (year counts): {peaks_path}")
    print(f"  Metadata RPYS scores         : {meta_scores_path}")
    print(f"  Summary                      : {summary_file}\n")


if __name__ == "__main__":
    run_rpys()
