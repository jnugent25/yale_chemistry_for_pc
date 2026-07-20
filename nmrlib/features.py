"""Feature-set definitions and the featurization pipeline.

The named feature sets here are the single source of truth for what the
notebooks previously each defined inline (and let drift — e.g. the NMF code
count was hardcoded as 115 in one notebook and 90 in another). NMF code
columns are now *derived* from the dataframe, so any featurized dataset works
without editing counts.

Column-name generators are reused from the existing pipeline modules:
``an2014_nmr`` for the An et al. (2014) 1H descriptors and ``add_c_nmr_bins``
for the 13C bins.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from add_c_nmr_bins import process_row as c13_process_row
from an2014_nmr import (
    an2014_features_from_peaks,
    full_descriptor_columns,
    mlr_feature_columns,
)

NMF_CODE_PREFIX = "nmr_dict_code_"
NMF_ERROR_COLS = [
    "nmr_h_reconstruction_error",
    "nmr_c_reconstruction_error",
    "nmr_total_reconstruction_error",
]

# Comprehensive, symmetric peak-list summary statistics. Built programmatically
# so the 1H/13C blocks stay in lockstep. The original 8 columns (num/max/mean/
# skew/kurt _delta_ 13C/1H) remain a subset of this list, so older references and
# featurized frames keep resolving.
_STAT_NAMES = ["num", "min", "max", "range", "mean", "median", "std", "skew", "kurt"]
_DIST_STATS_COLS = [f"{s}_delta_{nuc}" for nuc in ("13C", "1H") for s in _STAT_NAMES]

# Chemical-shift regions (ppm) as fractions of each nucleus's peaks — cheap proxies
# for aromaticity / heteroatom substitution / carbonyl content (drives gap_ev).
H_SHIFT_REGIONS = {"aliph_0_3": (0.0, 3.0), "oxy_3_5": (3.0, 5.0),
                   "arom_6_85": (6.0, 8.5), "down_85_12": (8.5, 12.0)}
C_SHIFT_REGIONS = {"aliph_0_50": (0.0, 50.0), "oxy_50_100": (50.0, 100.0),
                   "arom_100_150": (100.0, 150.0), "carbonyl_150_210": (150.0, 210.0)}
_REGION_COLS = ([f"h_frac_{k}" for k in H_SHIFT_REGIONS]
                + [f"c_frac_{k}" for k in C_SHIFT_REGIONS])

# Integration / mass (what a balanced-W1 shape-only NMF discards), intensity-weighted
# shift centroids, peak crowding, cross-nucleus ratio, and mean 13C linewidth.
_EXTRA_COLS = [
    "total_nH_1H", "total_integral_13C", "total_intensity_13C",
    "wmean_delta_1H", "wmean_delta_13C",
    "density_1H", "density_13C", "c_to_h_peak_ratio", "mean_width_13C",
]

NMR_STATS_COLS = _DIST_STATS_COLS + _REGION_COLS + _EXTRA_COLS


def c_bin_cols(bin_width: int) -> list[str]:
    """13C integration-bin columns for a given width (matches add_c_nmr_bins
    naming), plus broadness and total-nC terms."""
    return [f"c_x{e}" for e in range(bin_width, 220 + bin_width, bin_width)] + [
        "c_b1", "c_b2", "c_b3", "c_total_nC",
    ]


def nmf_cols(df: pd.DataFrame) -> list[str]:
    """NMF dictionary-code columns present in ``df`` (prefix-derived, so the
    component count never needs hardcoding), plus reconstruction errors."""
    codes = sorted(c for c in df.columns if c.startswith(NMF_CODE_PREFIX))
    errors = [c for c in NMF_ERROR_COLS if c in df.columns]
    return codes + errors


def feature_sets(df: pd.DataFrame, warn_missing: bool = True) -> dict[str, list[str]]:
    """Named feature sets available in ``df``.

    Sets with missing columns are dropped (with a warning) rather than raising,
    so the same notebook runs on partially-featurized datasets.
    """
    an14_full = full_descriptor_columns()
    candidates: dict[str, list[str]] = {
        "an14": mlr_feature_columns(),
        "an14_full": an14_full,
        "nmr_stats": NMR_STATS_COLS,
        "NMF": nmf_cols(df),
        "c_bins_10": c_bin_cols(10),
        "c_bins_5": c_bin_cols(5),
    }
    candidates["an14_full_with_c_bins_10"] = an14_full + candidates["c_bins_10"]
    candidates["an14_full_with_c_bins_5"] = an14_full + candidates["c_bins_5"]
    if candidates["NMF"]:
        candidates["NMF_with_nmr_stats"] = candidates["NMF"] + NMR_STATS_COLS
        candidates["NMF_with_c_bins_5"] = candidates["NMF"] + candidates["c_bins_5"]
    candidates["an14_full_with_nmr_stats"] = an14_full + NMR_STATS_COLS

    available: dict[str, list[str]] = {}
    for name, cols in candidates.items():
        missing = [c for c in cols if c not in df.columns]
        if not cols or missing:
            if warn_missing:
                detail = f"missing {len(missing)} columns e.g. {missing[:3]}" if missing else "no columns"
                warnings.warn(f"feature set {name!r} unavailable ({detail})")
            continue
        available[name] = cols
    return available


def _dist_stats(vals) -> list[float]:
    """[num, min, max, range, mean, median, std, skew, kurt] over a shift list.

    Higher moments need enough points to be defined (skew>=3, kurt>=4); below that
    they are NaN (HGB/trees handle NaN natively). Order matches ``_STAT_NAMES``.
    """
    v = np.asarray(vals, dtype=float)
    if v.size == 0:
        return [0.0] + [np.nan] * 8
    return [
        float(v.size), v.min(), v.max(), v.max() - v.min(), v.mean(), float(np.median(v)),
        v.std() if v.size >= 2 else 0.0,
        skew(v) if v.size >= 3 else np.nan,
        kurtosis(v) if v.size >= 4 else np.nan,
    ]


def _region_fracs(deltas, regions: dict) -> list[float]:
    """Fraction of peaks whose shift falls in each [lo, hi) region."""
    d = np.asarray(deltas, dtype=float)
    n = d.size
    if n == 0:
        return [np.nan] * len(regions)
    return [float(((d >= lo) & (d < hi)).sum()) / n for lo, hi in regions.values()]


def add_nmr_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Comprehensive, symmetric summary statistics of the raw 1H/13C peak lists.

    Full distributional stats for both nuclei, chemical-shift region fractions,
    integration/mass totals, intensity-weighted centroids, peak crowding, the
    C:H peak ratio, and mean 13C linewidth. Columns are ``NMR_STATS_COLS``.
    """
    rows: list[list[float]] = []
    for h_peaks, c_peaks in zip(df["h_nmr_peaks"], df["c_nmr_peaks"]):
        H = list(h_peaks) if h_peaks is not None else []
        C = list(c_peaks) if c_peaks is not None else []
        hd = [p["delta"] for p in H]
        cd = [p["delta (ppm)"] for p in C]
        nH = np.array([p.get("nH", 0) for p in H], dtype=float)
        integ = np.array([p.get("integral", 0.0) for p in C], dtype=float)
        inten = np.array([p.get("intensity", 0.0) for p in C], dtype=float)
        width = np.array([p.get("width (ppm)", 0.0) for p in C], dtype=float)

        vals = _dist_stats(cd) + _dist_stats(hd)                       # 13C then 1H
        vals += _region_fracs(hd, H_SHIFT_REGIONS) + _region_fracs(cd, C_SHIFT_REGIONS)
        rng_h = (max(hd) - min(hd)) if hd else 0.0
        rng_c = (max(cd) - min(cd)) if cd else 0.0
        vals += [
            float(nH.sum()), float(integ.sum()), float(inten.sum()),
            float(np.average(hd, weights=nH)) if (nH.sum() > 0 and hd) else np.nan,
            float(np.average(cd, weights=inten)) if (inten.sum() > 0 and cd) else np.nan,
            len(hd) / rng_h if rng_h else np.nan,
            len(cd) / rng_c if rng_c else np.nan,
            len(cd) / len(hd) if hd else np.nan,
            float(width.mean()) if width.size else np.nan,
        ]
        rows.append(vals)
    return pd.DataFrame(rows, columns=NMR_STATS_COLS, index=df.index)


def add_an14_features(df: pd.DataFrame) -> pd.DataFrame:
    """An et al. (2014) 1H descriptors: 24 integration bins + broadness + nH."""
    return pd.DataFrame(
        df["h_nmr_peaks"].apply(an2014_features_from_peaks).tolist(), index=df.index
    )


def add_c13_features(df: pd.DataFrame) -> pd.DataFrame:
    """13C integration bins (10 and 5 ppm), broadness terms, and total nC."""
    return pd.DataFrame(df.apply(c13_process_row, axis=1).tolist(), index=df.index)


def add_nmf_codes(df: pd.DataFrame, dictionary_path: str | Path) -> pd.DataFrame:
    """Transform spectra into a pretrained NMF dictionary's code space and
    attach reconstruction-error features (the pattern from build_dict_codes)."""
    from create_nmr_dictionary_features import (
        _apply_transform,
        _invert_transform,
        build_soft_peak_matrix,
    )

    with open(dictionary_path, "rb") as f:
        artifact = pickle.load(f)
    nmf_model = artifact["nmf_model"]
    config = artifact["config"]

    h_matrix = build_soft_peak_matrix(
        df["h_nmr_peaks"], artifact["h_grid_ppm"], config["h_sigma_ppm"], modality="h",
        width_scale=config["h_width_scale"], use_peak_width=config["use_peak_width"],
    )
    c_matrix = build_soft_peak_matrix(
        df["c_nmr_peaks"], artifact["c_grid_ppm"], config["c_sigma_ppm"], modality="c",
        width_scale=config["c_width_scale"], use_peak_width=config["use_peak_width"],
    )
    h_t = _apply_transform(h_matrix, config["intensity_transform"])
    c_t = _apply_transform(c_matrix, config["intensity_transform"])
    x = np.concatenate(
        [config["h_modality_weight"] * h_t, config["c_modality_weight"] * c_t], axis=1
    )
    codes = nmf_model.transform(x)
    out = pd.DataFrame(codes, columns=artifact["feature_columns"], index=df.index)

    h_recon = _invert_transform(codes @ artifact["h_components"], config["intensity_transform"])
    c_recon = _invert_transform(codes @ artifact["c_components"], config["intensity_transform"])
    out["nmr_h_reconstruction_error"] = np.linalg.norm(h_matrix - h_recon, axis=1)
    out["nmr_c_reconstruction_error"] = np.linalg.norm(c_matrix - c_recon, axis=1)
    out["nmr_total_reconstruction_error"] = (
        out["nmr_h_reconstruction_error"] + out["nmr_c_reconstruction_error"]
    )
    return out


def featurize(df: pd.DataFrame, dictionary_path: str | Path | None = None) -> pd.DataFrame:
    """Attach all NMR feature blocks to ``df`` and return the combined frame.

    Blocks: An 2014 1H descriptors, 13C bins, peak summary stats, and (when a
    dictionary artifact path is given) NMF codes + reconstruction errors.
    Pre-existing columns with the same names are replaced, so re-featurizing
    an already-featurized frame is safe.
    """
    df = df.reset_index(drop=True)
    blocks = [add_an14_features(df), add_c13_features(df), add_nmr_stats(df)]
    if dictionary_path is not None:
        blocks.append(add_nmf_codes(df, dictionary_path))
    new_cols = pd.concat(blocks, axis=1)
    base = df.drop(columns=[c for c in new_cols.columns if c in df.columns])
    return pd.concat([base, new_cols], axis=1)
