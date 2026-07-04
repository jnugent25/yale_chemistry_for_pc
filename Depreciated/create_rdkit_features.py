import pandas as pd
ids_1k = pd.read_pickle('/Users/jacknugent/Desktop/yale_chemistry_project/gaussian_with_nmr.pkl')
from rdkit import Chem
from rdkit.Chem import Descriptors

def get_rdkit_props(df, smiles_col):
    """
    df: the pandas dataframe
    smiles_col: the name of the column containing the SMILES strings
    """
    # create a copy of the dataframe to avoid modifying the original
    working_df = df.copy()
    # create empty lists to store the properties
    mol_weights = []
    log_p = []
    ring_counts = []
    rotatable_bonds = []
    tpsa = []
    h_acc = []
    h_donors = []
    c_ct = []
    n_ct = []
    o_ct = []
    f_ct = []
    cl_ct = []
    s_ct = []
    
    # loop through the SMILES strings and calculate the properties
    for smi in working_df[smiles_col]: # loop through the SMILES strings
        mol = Chem.MolFromSmiles(smi) # convert the SMILES string to a molecule object
        if mol is None:
            print(f'error making the rdkit object for said smile:{smi}')
        mol_weights.append(Descriptors.MolWt(mol)) # now get the molecular weight
        log_p.append(Descriptors.MolLogP(mol)) # get the logP
        ring_counts.append(Descriptors.RingCount(mol))
        rotatable_bonds.append(Descriptors.NumRotatableBonds(mol))
        tpsa.append(Descriptors.TPSA(mol))
        h_acc.append(Descriptors.NumHAcceptors(mol))
        h_donors.append(Descriptors.NumHDonors(mol))
        c_ct.append(Descriptors.NumCarbonAtoms(mol))
        n_ct.append(Descriptors.NumNitrogenAtoms(mol))
        o_ct.append(Descriptors.NumOxygenAtoms(mol))
        f_ct.append(Descriptors.NumFluorineAtoms(mol))
        cl_ct.append(Descriptors.NumChlorineAtoms(mol))
        s_ct.append(Descriptors.NumSulfurAtoms(mol))


        # add other properties below here

    # add the properties to the dataframe
    working_df['mol_weight'] = mol_weights
    working_df['log_p'] = log_p
    working_df['ring_count'] = ring_counts
    working_df['rotatable_bonds'] = rotatable_bonds
    working_df['tpsa'] = tpsa
    working_df['h_acc'] = h_acc
    working_df['h_donors'] = h_donors
    working_df['c_ct'] = c_ct
    working_df['n_ct'] = n_ct
    working_df['o_ct'] = o_ct
    working_df['f_ct'] = f_ct
    working_df['cl_ct'] = cl_ct
    working_df['s_ct'] = s_ct

    return working_df

# now run this and see what we get
ids_1k_props = get_rdkit_props(ids_1k, 'smiles')
print(ids_1k_props.head(5))
#pd description of the dataframe
print(ids_1k.describe())