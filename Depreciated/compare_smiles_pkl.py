#!/usr/bin/env python3
import json
import pickle
from collections import Counter
from pathlib import Path

import pandas as pd

FILES = {
    "gaussian_1_1k": Path("/Users/jacknugent/Downloads/gaussian_1_1k.pkl"),
    "ids_nmr_1k": Path("/Users/jacknugent/Downloads/ids_nmr_1k.pkl"),
    "ids_nmr_10k": Path("/Users/jacknugent/Downloads/ids_nmr_10k.pkl"),
    "ids_nmr_100k": Path("/Users/jacknugent/Downloads/ids_nmr_100k.pkl"),
}


def norm_smiles(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    return None if s in ("", "nan", "None") else s


def load_df(path: Path) -> pd.DataFrame:
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"{path} is {type(obj)}, expected DataFrame")
    return obj


def smiles_series(df: pd.DataFrame) -> pd.Series:
    col = next(c for c in df.columns if "smile" in str(c).lower())
    return df[col].map(norm_smiles)


def compare(g_list, o_list, other_name: str) -> dict:
    g_set = {x for x in g_list if x}
    o_set = {x for x in o_list if x}
    overlap = g_set & o_set
    n = len(g_list)
    g_norm = [norm_smiles(x) for x in g_list]
    prefix = [norm_smiles(x) for x in o_list[:n]]
    return {
        "other": other_name,
        "gaussian_unique": len(g_set),
        "other_unique": len(o_set),
        "overlap_count": len(overlap),
        "pct_gaussian_in_other": round(100 * len(overlap) / len(g_set), 2) if g_set else 0,
        "pct_other_in_gaussian": round(100 * len(overlap) / len(o_set), 2) if o_set else 0,
        "gaussian_subset_of_other": g_set <= o_set,
        "other_subset_of_gaussian": o_set <= g_set,
        "gaussian_only_count": len(g_set - o_set),
        "other_only_count": len(o_set - g_set),
        "is_gaussian_first_N_rows_of_other": prefix == g_norm,
        "first_N_same_multiset_as_gaussian": Counter(prefix) == Counter(g_norm),
        "N_compared": n,
    }


def main() -> None:
    result: dict = {"per_file": {}, "comparisons": {}}
    smiles_by_name: dict[str, list] = {}

    for name, path in FILES.items():
        info = {"path": str(path), "exists": path.is_file()}
        if not path.is_file():
            result["per_file"][name] = info
            continue
        df = load_df(path)
        s = smiles_series(df)
        info.update(
            {
                "shape": list(df.shape),
                "columns": list(df.columns),
                "unique_smiles": int(s.dropna().nunique()),
                "duplicate_rows": int(s.duplicated().sum()),
            }
        )
        result["per_file"][name] = info
        smiles_by_name[name] = s.tolist()

    if "gaussian_1_1k" in smiles_by_name:
        gdf = load_df(FILES["gaussian_1_1k"])
        cols = list(gdf.columns)
        result["gaussian_columns"] = cols
        result["gaussian_homo_lumo_gap_columns"] = [
            c
            for c in cols
            if any(k in str(c).lower() for k in ("homo", "lumo", "gap", "orbital", "energy"))
        ]
        g_list = smiles_by_name["gaussian_1_1k"]
        for other in ("ids_nmr_1k", "ids_nmr_10k", "ids_nmr_100k"):
            if other in smiles_by_name:
                result["comparisons"][other] = compare(g_list, smiles_by_name[other], other)

    out = Path(__file__).with_name("smiles_compare_results.json")
    out.write_text(json.dumps(result, indent=2))
    print(out.read_text())


if __name__ == "__main__":
    main()
