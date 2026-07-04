import pickle
import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("/Users/jacknugent/Downloads/ids_nmr_1k_with_10k_features.pkl")

# c nmr usually from 0 to 220
C_MIN_PPM = 0.0
C_MAX_PPM = 220.0

# threshold to broad peak
C_WIDE_PEAK_MIN_PPM = 0.02

# split spectra into bins
def bin_peaks(peaks, bin_width):
    n_bins = int((C_MAX_PPM - C_MIN_PPM) / bin_width)
    integ = np.zeros(n_bins, dtype=np.float64)
    
    if peaks is None or (isinstance(peaks, float) and pd.isna(peaks)):
        peaks = []
        
    for peak in peaks:
        delta = peak.get("delta (ppm)")
        if delta is None:
            continue
        delta = float(delta)
        if delta < C_MIN_PPM or delta >= C_MAX_PPM:
            continue
        idx = min(int((delta - C_MIN_PPM) / bin_width), n_bins - 1)
        integ[idx] += 1.0 
        
    out = {}
    for i in range(n_bins):
        upper_edge = C_MIN_PPM + (i + 1) * bin_width
        edge_str = f"{int(upper_edge)}" if upper_edge.is_integer() else f"{upper_edge:g}"
        out[f"c_x{edge_str}"] = float(integ[i])
    return out

# get broadness features
def get_broadness_features(peaks):
    broad_widths = []
    
    if peaks is None or (isinstance(peaks, float) and pd.isna(peaks)):
        peaks = []
        
    for peak in peaks:
        width = peak.get("width (ppm)")
        if width is not None:
            width = float(width)
            if width >= C_WIDE_PEAK_MIN_PPM:
                broad_widths.append(width)
                
    broad_widths = sorted(broad_widths, reverse=True)[:3]
    out = {}
    for j in range(1, 4):
        out[f"c_b{j}"] = float(broad_widths[j - 1]) if j <= len(broad_widths) else 0.0
    return out

# create all features required nicely
def process_row(row):
    peaks = row["c_nmr_peaks"]
    
    # 10 ppm bins
    features_10 = bin_peaks(peaks, bin_width=10.0)
    # 5 ppm bins
    features_5 = bin_peaks(peaks, bin_width=5.0)
    # Broadness
    broadness = get_broadness_features(peaks)
    
    # Total peaks (nC)
    total_nC = float(len(peaks)) if hasattr(peaks, "__len__") else 0.0
    
    # Merge all
    merged = {}
    merged.update(features_10)
    merged.update(features_5)
    merged.update(broadness)
    merged["c_total_nC"] = total_nC
    return merged
