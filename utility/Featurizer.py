import numpy as np
from rdkit import Chem

LIST_SYMBOLS = ['C', 'N', 'O', 'S', 'F', 'H', 'Si', 'P', 'Cl', 'Br',
            'Li', 'Na', 'K', 'Mg', 'Ca', 'Fe', 'As', 'Al', 'I', 'B',
            'V', 'Tl', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
            'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'Mn', 'Cr', 'Pt', 'Hg', 'Pb']
MAX_LEN = 100

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [int(x == s) for s in allowable_set]

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [int(x == s) for s in allowable_set]

def atom_feature(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(), LIST_SYMBOLS) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]) +
                    one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                    one_of_k_encoding(int(atom.GetIsAromatic()), [0, 1]))    # (40, 7, 5, 6, 2)
   
def mol2graph(smi):
    mol = Chem.MolFromSmiles(smi)
    
    X = np.zeros((MAX_LEN, 60), dtype=np.uint8)
    A = np.zeros((MAX_LEN, MAX_LEN), dtype=np.uint8)

    temp_A = Chem.rdmolops.GetAdjacencyMatrix(mol).astype(np.uint8, copy=False)[:MAX_LEN, :MAX_LEN]
    print(temp_A)
    num_atom = temp_A.shape[0]
    A[:num_atom, :num_atom] = temp_A + np.eye(temp_A.shape[0], dtype=np.uint8)

    for i, atom in enumerate(mol.GetAtoms()):
        feature = atom_feature(atom)
        X[i, :] = feature
        if i + 1 >= num_atom: break

    return X, A