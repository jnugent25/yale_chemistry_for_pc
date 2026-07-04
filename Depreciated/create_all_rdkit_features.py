import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

# Example usage loading the data
ids_100k = pd.read_pickle('/Users/jacknugent/Downloads/ids_100k.pkl')

def get_rdkit_props(df, smiles_col):
    """
    df: the pandas dataframe
    smiles_col: the name of the column containing the SMILES strings
    
    Returns a new dataframe with all RDKit 2D descriptors appended as columns.
    """
    # create a copy of the dataframe to avoid modifying the original
    working_df = df.copy()
    
    # List to store the dictionary of descriptors for each molecule
    all_descriptors = []
    
    # loop through the SMILES strings and calculate the properties
    for idx, smi in enumerate(working_df[smiles_col]):
        print(idx/len(working_df), end='\r') # Progress indicator
        mol = None
        if pd.notna(smi):
            try:
                mol = Chem.MolFromSmiles(str(smi))
            except Exception:
                pass
                
        if mol is None:
            print(f'Error making the rdkit object for said smile: {smi}')
            all_descriptors.append({}) # Append empty dict for failed molecules
            continue
            
        try:
            # CalcMolDescriptors returns a dictionary of ~200+ 2D molecular descriptors
            # including MolWt, MolLogP, TPSA, NumRotatableBonds, etc.
            desc_dict = Descriptors.CalcMolDescriptors(mol)
            all_descriptors.append(desc_dict)
        except Exception as e:
            print(f'Error calculating descriptors for smile {smi}: {e}')
            all_descriptors.append({})
            
    # Convert the list of dictionaries into a new DataFrame
    desc_df = pd.DataFrame(all_descriptors)
    
    # Concatenate the original dataframe with the new descriptors dataframe
    result_df = pd.concat([working_df.reset_index(drop=True), desc_df], axis=1)

    return result_df

# If you were to run it:
df_with_features = get_rdkit_props(ids_100k, 'smiles') # replace 'smiles' with actual column name

print(df_with_features.head())

#save new df
df_with_features.to_pickle('/Users/jacknugent/Downloads/ids_100k_with_rdkit_features_with_3d.pkl')