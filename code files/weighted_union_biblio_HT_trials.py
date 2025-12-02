# -*- coding: utf-8 -*-
"""
Weighted Union Ranking (HT trials, 30% cutoff, metadata-preserving)
-------------------------------------------------------------------
Combines:
  - RPYS influence (rpys_raw_score from metadata_rpys_scores.csv)
  - Bibliographic Coupling (BC) on WoS papers
  - PageRank on BC graph

Outputs:
  reports/combined/heart_transplant/heart_transplant_metadata_combined_seminal.csv
"""

import re
from pathlib import Path
from itertools import combinations
from collections import defaultdict

import pandas as pd
import numpy as np
import bibtexparser

TOPIC = "heart_transplant"
ROOT  = Path(__file__).resolve().parents[1]

RPYS_DIR = ROOT / "reports" / "rpys"
OUT_DIR  = ROOT / "reports" / "combined" / TOPIC
OUT_DIR.mkdir(parents=True, exist_ok=True)

WOS_DIR = ROOT / "data" / "WoS" / TOPIC

TOP_PERCENT = 0.30  # you asked to keep top 30%


# -----------------------------
# Helpers
# -----------------------------
def minmax(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    mn, mx = float(s.min()), float(s.max())
    if mn == mx:
        return pd.Series(0.0, index=s.index)
    return (s - mn) / (mx - mn)


def pagerank(adj: dict, d: float = 0.85, max_iter: int = 50, tol: float = 1e-6) -> dict:
    nodes = sorted(adj.keys())
    if not nodes:
        return {}

    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    pr = np.ones(n) / n

    outw = np.zeros(n)
    for u in nodes:
        outw[idx[u]] = sum(adj[u].values())

    for _ in range(max_iter):
        prev = pr.copy()
        pr = np.ones(n) * (1 - d) / n
        dangling = prev[outw == 0].sum()

        for u in nodes:
            j = idx[u]
            if outw[j] == 0:
                continue
            for v, w in adj[u].items():
                i = idx[v]
                pr[i] += d * prev[j] * w / outw[j]

        pr += d * dangling / n
        if np.abs(prev - pr).sum() < tol:
            break

    return {n: float(pr[idx[n]]) for n in nodes}


# -----------------------------
# Build WoS BC graph
# -----------------------------
def build_wos_bc(meta: pd.DataFrame) -> dict:
    """
    Build a bibliographic coupling adjacency among WoS papers.

    NOTE: For consistency, WOS paper_id in RPYS and here must both be:
       ut → ID → title[:80]
    (which is how rpys_analysis_HT_trials.py defines paper_id for WoS).
    """
    candidate = set(meta.loc[meta["source"] == "wos", "paper_id"].astype(str))

    parser = bibtexparser.bparser.BibTexParser(common_strings=True)
    entries = []
    for bib in WOS_DIR.glob("**/*.bib"):
        try:
            with open(bib, "r", encoding="utf-8", errors="ignore") as f:
                db = bibtexparser.load(f, parser=parser)
                entries.extend(db.entries)
        except Exception:
            continue

    adj = defaultdict(lambda: defaultdict(float))

    for e in entries:
        pid = e.get("ut") or e.get("ID") or e.get("title", "")[:80]
        pid = str(pid)
        if pid not in candidate:
            continue

        cr_raw = (
            e.get("cr")
            or e.get("cited-references")
            or e.get("cited_references")
            or e.get("citedreferences")
            or ""
        )
        if not cr_raw:
            continue

        ref_keys = set()
        for p in re.split(r"[\n;]+", cr_raw):
            p = p.strip()
            if not p:
                continue
            t = p.upper()
            t = re.sub(r"[^A-Z0-9 ]+", " ", t)
            t = re.sub(r"\s+", " ", t).strip()
            ref_keys.add(t[:300])

        ks = sorted(ref_keys)
        for a, b in combinations(ks, 2):
            adj[a][b] += 1.0
            adj[b][a] += 1.0

    # ensure every candidate node appears (even if degree zero)
    for pid in candidate:
        _ = adj[pid]  # triggers dict creation if missing

    return adj


# -----------------------------
# MAIN
# -----------------------------
def main():
    meta_file = RPYS_DIR / f"{TOPIC}_metadata_rpys_scores.csv"
    df = pd.read_csv(meta_file, dtype=str)

    # Ensure core columns exist
    if "paper_id" not in df.columns:
        raise ValueError("metadata_rpys_scores must contain 'paper_id'.")
    if "rpys_raw_score" not in df.columns:
        raise ValueError("metadata_rpys_scores must contain 'rpys_raw_score'.")

    df["paper_id"] = df["paper_id"].astype(str)

    # --- Build BC graph on WoS subset ---
    adj = build_wos_bc(df)

    # BC weighted degree
    bc_score = {k: sum(adj[k].values()) for k in adj}
    df["bc_raw"] = df["paper_id"].map(lambda x: bc_score.get(x, 0.0))

    # PageRank over BC graph
    pr_score = pagerank(adj)
    df["pr_raw"] = df["paper_id"].map(lambda x: pr_score.get(x, 0.0))

    # --- Normalize all three components ---
    df["rpys_norm"] = minmax(df["rpys_raw_score"])
    df["bc_norm"]   = minmax(df["bc_raw"])
    df["pr_norm"]   = minmax(df["pr_raw"])

    # Weighted union (equal weights for now)
    W_RPYS = 1.0
    W_BC   = 1.0
    W_PR   = 1.0

    df["combined_score"] = (
        W_RPYS * df["rpys_norm"] +
        W_BC   * df["bc_norm"]   +
        W_PR   * df["pr_norm"]
    ) / (W_RPYS + W_BC + W_PR)

    # --- Rank + top 30% cutoff ---
    df = df.sort_values("combined_score", ascending=False).reset_index(drop=True)
    keep_n = int(len(df) * TOP_PERCENT)
    if keep_n < 1:
        keep_n = len(df)

    df_top = df.head(keep_n).copy()
    df_top.insert(0, "rank", df_top.index + 1)

    out_path = OUT_DIR / f"{TOPIC}_metadata_combined_seminal.csv"
    df_top.to_csv(out_path, index=False)

    print(f"\n✔ Weighted union + TOP {int(TOP_PERCENT*100)}% complete")
    print(f"  Saved {keep_n} papers to:")
    print(f"    {out_path}\n")


if __name__ == "__main__":
    main()
