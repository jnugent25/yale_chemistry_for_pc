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

NMR_STATS_COLS = [
    "num_delta_13C", "num_delta_1H",
    "max_delta_13C", "max_delta_1H",
    "mean_delta_13C",
    "skew_delta_13C", "skew_delta_1H",
    "kurt_delta_13C",
]


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


def _safe_stat(peaks, key: str, func):
    vals = [p[key] for p in peaks] if peaks is not None and len(peaks) > 0 else []
    if len(vals) == 0:
        return np.nan
    return func(vals)


def add_nmr_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Summary statistics of the raw peak lists (Mueller-style descriptors)."""
    out = pd.DataFrame(index=df.index)
    out["num_delta_13C"] = df["c_nmr_peaks"].apply(len)
    out["num_delta_1H"] = df["h_nmr_peaks"].apply(len)
    out["max_delta_13C"] = df["c_nmr_peaks"].apply(lambda p: _safe_stat(p, "delta (ppm)", np.max))
    out["max_delta_1H"] = df["h_nmr_peaks"].apply(lambda p: _safe_stat(p, "delta", np.max))
    out["mean_delta_13C"] = df["c_nmr_peaks"].apply(lambda p: _safe_stat(p, "delta (ppm)", np.mean))
    out["skew_delta_13C"] = df["c_nmr_peaks"].apply(
        lambda p: _safe_stat(p, "delta (ppm)", lambda v: skew(v) if len(v) >= 3 else np.nan))
    out["skew_delta_1H"] = df["h_nmr_peaks"].apply(
        lambda p: _safe_stat(p, "delta", lambda v: skew(v) if len(v) >= 3 else np.nan))
    out["kurt_delta_13C"] = df["c_nmr_peaks"].apply(
        lambda p: _safe_stat(p, "delta (ppm)", lambda v: kurtosis(v) if len(v) >= 4 else np.nan))
    return out


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
