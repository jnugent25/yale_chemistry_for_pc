from rdkit import Chem
from rdkit.Chem import Draw
import pandas as pd
from rdkit.Chem import Descriptors
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)
from rdkit.Chem import AllChem
import numpy as np

def rdkit_worker(mol_smiles):

    if not isinstance(mol_smiles, str) or not mol_smiles:
        return None

    mol = Chem.MolFromSmiles(mol_smiles)

    if mol is None:
        return None
        
    # 4. Do the heavy lifting safely
    try:
        # --- YOUR CUSTOM RDKIT LOGIC GOES HERE ---
        # Example A: Calculate Molecular Weight
        # result = Descriptors.MolWt(mol)
        
        # Example B: Generate a 2048-bit Morgan Fingerprint
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        result = np.array(fp) # Convert RDKit object to standard array
        
        return result
        
    except Exception as e:
        return None



if __name__ == "__main__":
    df = pd.read_csv("/Users/jacknugent/Downloads/ids_10k.csv")   

    #create new columns
    df['morgan_fps'] = df['smiles'].parallel_apply(rdkit_worker)