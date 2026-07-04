import pandas as pd

gaussian_input = pd.read_pickle("/Users/jacknugent/Downloads/gaussian_1_1k.pkl")

ids = pd.read_pickle('/Users/jacknugent/Downloads/ids_100k.pkl')

# add a temp name column
def build_gaussian_name(smiles, formula):
    smiles_tail = str(smiles)[-5:]
    return sanitize_gaussian_name(f"{formula}_{smiles_tail}")

gaussian_input["name"] = gaussian_input.apply(
    lambda row: build_gaussian_name(row["smiles"], row["molecular_formula"]),
    axis=1,
)