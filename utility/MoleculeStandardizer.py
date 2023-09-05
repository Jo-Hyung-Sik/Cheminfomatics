from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize

def Standardizer(smiles):
    mol = Chem.MolFromSmiles(smiles, sanitize=True)

    # removeHs, disconnect metal atoms, normalize the molecule, reionize the molecule
    clean_mol = rdMolStandardize.Cleanup(mol) 
        
    # try to neutralize molecule
    uncharger = rdMolStandardize.Uncharger() # annoying, but necessary as no convenience method exists
    uncharged_parent_clean_mol = uncharger.uncharge(clean_mol)
    
    # te = rdMolStandardize.TautomerEnumerator() # idem
    # taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)
    
    # remove salt
    remover = SaltRemover()
    salt_stripped = remover.StripMol(uncharged_parent_clean_mol)
    
    # remove stereo
    # Chem.RemoveStereochemistry(salt_stripped)
    
    standard_smiles = Chem.MolToSmiles(salt_stripped)
    
    return standard_smiles