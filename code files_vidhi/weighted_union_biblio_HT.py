"""
Weighted Union (Updated)
-----------------------
Combines metadata-level:
 - RPYS influence (from rpys_analysis.py)
 - Bibliographic Coupling (BC)
 - PageRank

Outputs:
    reports/combined/<topic>/<topic>_metadata_combined_seminal.csv
"""

import os, re, glob
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import pandas as pd
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
TOPIC = "heart_transplant"
TOP_PERCENT = 0.30     # keep top 30%

ROOT_DIR = Path(__file__).resolve().parents[1]

# Paths
RPYS_DIR = ROOT_DIR / "reports" / "rpys"
COMBINED_DIR = ROOT_DIR / "reports" / "combined" / TOPIC
COMBINED_DIR.mkdir(parents=True, exist_ok=True)

# Metadata input
META_RPYS_FILE = RPYS_DIR / f"{TOPIC}_metadata_rpys_scores.csv"

# WoS metadata for BC & PR
WOS_DIR = ROOT_DIR / "data" / "WoS" / TOPIC
WOS_SUBDIRS = [
    f"{TOPIC} AND surgical management",
    f"{TOPIC} AND natural history",
    f"{TOPIC} AND pathophysiology",
]

# -----------------------------
# Bibliographic Coupling graph
# -----------------------------
def build_bc_graph():
    parser = None
    try:
        import bibtexparser
        parser = bibtexparser.bparser.BibTexParser(common_strings=True)
    except:
        raise ImportError("bibtexparser required")

    citing_refs = defaultdict(set)   # paper → set of cited keys
    all_refs = set()

    for sub in WOS_SUBDIRS:
        for p in glob.glob(str((WOS_DIR / sub / "*.bib").resolve())):
            with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                db = bibtexparser.load(fh, parser=parser)
                for e in db.entries:
                    pid = e.get("ut") or e.get("ID") or e.get("title","")[:80]

                    # extract CR
                    cr = None
                    for k,v in e.items():
                        if k.lower() in {"cr", "cited-references","citedreferences","cited_references"}:
                            cr = v
                            break
                    if not cr:
                        continue

                    parts = re.split(r"[\n;]+\s*", cr)
                    for ref in parts:
                        ref = ref.strip()
                        if not ref:
                            continue
                        key = ref.upper()[:300]
                        citing_refs[pid].add(key)
                        all_refs.add(key)

    # Build BC weights
    adj = defaultdict(lambda: defaultdict(float))
    pids = list(citing_refs.keys())

    for i, a in enumerate(pids):
        for b in pids[i+1:]:
            w = len(citing_refs[a].intersection(citing_refs[b]))
            if w > 0:
                adj[a][b] = w
                adj[b][a] = w

    return adj, pids

# -----------------------------
# PageRank
# -----------------------------
def pagerank(adj, damping=0.85, max_iter=50, tol=1e-6):
    nodes = sorted(adj.keys())
    if not nodes: 
        return {}

    n = len(nodes)
    idx = {node:i for i,node in enumerate(nodes)}

    outweight = np.zeros(n)
    for u in nodes:
        j = idx[u]
        outweight[j] = sum(adj[u].values())

    pr = np.full(n, 1.0/n)

    for _ in range(max_iter):
        prev = pr.copy()
        pr = np.full(n, (1-damping)/n)

        dangling = prev[outweight==0].sum()

        for u in nodes:
            j = idx[u]
            if outweight[j] == 0:
                continue
            share = damping * prev[j] / outweight[j]
            for v,w_uv in adj[u].items():
                pr[idx[v]] += share*w_uv

        pr += damping*dangling/n

        if np.abs(pr-prev).sum() < tol:
            break

    return {node: float(score) for node,score in zip(nodes,pr)}

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    # --- Load metadata RPYS scores ---
    df = pd.read_csv(META_RPYS_FILE)

    # We want metadata papers that have some ID
    df["paper_id"] = df["paper_id"].astype(str)

    # --- Build BC graph ---
    adj, pids = build_bc_graph()

    # Map BC weighted degree
    bc_score = {pid: sum(adj[pid].values()) for pid in pids}
    df["bc_raw"] = df["paper_id"].map(lambda x: bc_score.get(x,0.0))

    # --- PageRank ---
    pr_score = pagerank(adj)
    df["pr_raw"] = df["paper_id"].map(lambda x: pr_score.get(x,0.0))

    # --- Normalize ---
    def norm(s):
        s = s.astype(float)
        if s.max() == 0:
            return s*0
        return s/s.max()

    df["rpys_norm"] = norm(df["rpys_raw_score"])
    df["bc_norm"]   = norm(df["bc_raw"])
    df["pr_norm"]   = norm(df["pr_raw"])

    # Weighted union
    W_RPYS = 1.0
    W_BC   = 1.0
    W_PR   = 1.0

    df["combined_score"] = (
        W_RPYS*df["rpys_norm"] +
        W_BC*df["bc_norm"] +
        W_PR*df["pr_norm"]
    ) / (W_RPYS + W_BC + W_PR)

    # Rank and threshold
    df = df.sort_values("combined_score", ascending=False)
    keep_n = int(len(df) * TOP_PERCENT)
    keep_df = df.head(keep_n)

    out_path = COMBINED_DIR / f"{TOPIC}_metadata_combined_seminal.csv"
    keep_df.to_csv(out_path, index=False)

    print("\n✔ Weighted union complete.")
    print(f"Saved top {TOP_PERCENT*100:.0f}% ({keep_n} papers) to:")
    print(f"  {out_path}\n")
