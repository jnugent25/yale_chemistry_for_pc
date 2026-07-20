"""Build an unsupervised pretraining corpus from the literature NMRexp parquet.

Parses the ``NMR_processed`` column into the pipeline's peak-dict format and keeps
molecules that have BOTH 1H and 13C spectra, writing a peak-list DataFrame
(``smiles``, ``h_nmr_peaks``, ``c_nmr_peaks``) — the same schema ``load_dataset``
yields — for use as an NMF pretraining set (no quantum targets needed).

    NMR_processed formats:
      13C: [(shift, mult, J), ...]                      -> {'delta (ppm)': shift, 'intensity': 1.0}
      1H : [(mult, [J], 'nH', shift_hi, shift_lo), ...] -> {'delta': (hi+lo)/2, 'nH': int}

Example:
    python build_nmrexp_corpus.py \
        --parquet ~/Downloads/NMRexp_10to24_1_1004.parquet \
        --out Datasets/nmrexp_100k.pkl --n 100000
"""
from __future__ import annotations

import argparse
import ast
import random
import re
from pathlib import Path

import pandas as pd

_NH = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*H")


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def parse_13c(s: str):
    """[(shift, mult, J), ...] -> [{'delta (ppm)', 'intensity'}], 13C-range only."""
    try:
        peaks = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return None
    out = []
    for p in peaks:
        v = _num(p[0])
        if v is not None and -50 <= v <= 260:
            out.append({"delta (ppm)": v, "intensity": 1.0})
    return out or None


def parse_1h(s: str):
    """[(mult, [J], 'nH', hi, lo), ...] -> [{'delta', 'nH'}], 1H-range only."""
    try:
        peaks = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return None
    out = []
    for p in peaks:
        hi, lo = _num(p[3]), _num(p[4])
        if hi is None or lo is None:
            continue
        delta = 0.5 * (hi + lo)
        if not (-2 <= delta <= 16):
            continue
        m = _NH.search(str(p[2])) if len(p) > 2 else None
        out.append({"delta": delta, "nH": float(m.group(1)) if m else 1.0})
    return out or None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=100000, help="target number of molecules")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    raw = pd.read_parquet(args.parquet, columns=["SMILES", "NMR_type", "NMR_processed"])
    raw = raw[raw["NMR_type"].isin(["1H NMR", "13C NMR"])]
    s1 = set(raw.loc[raw["NMR_type"] == "1H NMR", "SMILES"])
    s13 = set(raw.loc[raw["NMR_type"] == "13C NMR", "SMILES"])
    both = sorted(s1 & s13)
    print(f"molecules with both 1H and 13C: {len(both)}")

    random.seed(args.seed)
    # oversample candidates (some rows fail to parse to non-empty peaks)
    cand = set(random.sample(both, min(len(both), int(args.n * 1.3))))
    sub = raw[raw["SMILES"].isin(cand)]
    h_first = sub[sub["NMR_type"] == "1H NMR"].drop_duplicates("SMILES").set_index("SMILES")["NMR_processed"]
    c_first = sub[sub["NMR_type"] == "13C NMR"].drop_duplicates("SMILES").set_index("SMILES")["NMR_processed"]

    rows = []
    for smi in cand:
        if smi in h_first.index and smi in c_first.index:
            hp, cp = parse_1h(h_first[smi]), parse_13c(c_first[smi])
            if hp and cp:
                rows.append({"smiles": smi, "h_nmr_peaks": hp, "c_nmr_peaks": cp})
        if len(rows) >= args.n:
            break

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(args.out)
    print(f"wrote {len(df)} molecules -> {args.out}")


if __name__ == "__main__":
    main()
