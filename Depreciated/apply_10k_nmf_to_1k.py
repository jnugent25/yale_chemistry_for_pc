import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
MODEL_PATH = Path("/Users/jacknugent/Downloads/ids_nmr_1k_dictionary_model.pkl")
DATA_1K_PATH = Path("/Users/jacknugent/Downloads/ids_nmr_1k_all_features.pkl")
OUTPUT_PATH = Path("/Users/jacknugent/Downloads/ids_nmr_1k_with_10k_features.pkl")

def apply_nmf(model_path, data_path, output_path):
    print("Loading model and dataset...")
    with open(model_path, "rb") as f:
        model_artifact = pickle.load(f)
        
    df = pd.read_pickle(data_path)
    
    nmf = model_artifact["nmf_model"]
    config = model_artifact["config"]
    h_grid = model_artifact["h_grid_ppm"]
    c_grid = model_artifact["c_grid_ppm"]
    
    # We need to build the soft peak matrices for the 1k dataset to transform them
    # Since build_soft_peak_matrix isn't directly importable unless we add the path,
    # let's just import it from the local script.
    import sys
    sys.path.append("/Users/jacknugent/Desktop/yale_chemistry_project")
    from create_nmr_dictionary_features import build_soft_peak_matrix, _apply_transform, _invert_transform
    
    print("Building soft peak matrices...")
    h_matrix = build_soft_peak_matrix(
        df["h_nmr_peaks"], h_grid, config["h_sigma_ppm"], modality="h", 
        width_scale=config["h_width_scale"], use_peak_width=config["use_peak_width"]
    )
    c_matrix = build_soft_peak_matrix(
        df["c_nmr_peaks"], c_grid, config["c_sigma_ppm"], modality="c", 
        width_scale=config["c_width_scale"], use_peak_width=config["use_peak_width"]
    )
    
    h_transformed = _apply_transform(h_matrix, config["intensity_transform"])
    c_transformed = _apply_transform(c_matrix, config["intensity_transform"])
    
    x = np.concatenate([
        config["h_modality_weight"] * h_transformed,
        config["c_modality_weight"] * c_transformed
    ], axis=1)
    
    print("Transforming dataset (NO LEAKAGE)...")
    # ONLY TRANSFORM! We do NOT fit.
    codes = nmf.transform(x)
    
    code_columns = model_artifact["feature_columns"]
    features_df = pd.DataFrame(codes, columns=code_columns, index=df.index)
    
    # Calculate reconstruction errors
    h_components = model_artifact["h_components"]
    c_components = model_artifact["c_components"]
    
    h_reconstruction = _invert_transform(codes @ h_components, config["intensity_transform"])
    c_reconstruction = _invert_transform(codes @ c_components, config["intensity_transform"])
    
    h_error = np.linalg.norm(h_matrix - h_reconstruction, axis=1)
    c_error = np.linalg.norm(c_matrix - c_reconstruction, axis=1)
    
    features_df["nmr_h_reconstruction_error"] = h_error
    features_df["nmr_c_reconstruction_error"] = c_error
    features_df["nmr_total_reconstruction_error"] = h_error + c_error
    
    # Remove old nmr_dict_code columns if they exist
    old_cols = [c for c in df.columns if c.startswith("nmr_dict_code") or "reconstruction_error" in c]
    df = df.drop(columns=old_cols)
    
    # Concatenate the new leak-free features
    final_df = pd.concat([df, features_df], axis=1)
    final_df.to_pickle(output_path)
    
    print(f"Success! Output saved to: {output_path}")
    print(f"Final shape: {final_df.shape}")
    print(f"Average 1H recon error: {h_error.mean():.4f}")
    print(f"Average 13C recon error: {c_error.mean():.4f}")

if __name__ == "__main__":
    apply_nmf(MODEL_PATH, DATA_1K_PATH, OUTPUT_PATH)
