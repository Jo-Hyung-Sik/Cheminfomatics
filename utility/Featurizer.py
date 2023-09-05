import numpy as np
from rdkit import Chem
import torch
from torch_geometric.data import Data

LIST_SYMBOLS = ['C', 'N', 'O', 'S', 'F', 'H', 'Si', 'P', 'Cl', 'Br',
            'Li', 'Na', 'K', 'Mg', 'Ca', 'Fe', 'As', 'Al', 'I', 'B',
            'V', 'Tl', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
            'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'Mn', 'Cr', 'Pt', 'Hg', 'Pb', 'unknown']
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
   
def bond_feature(bond, use_chirality=True):
    bt = bond.GetBondType()
    
    bond_feats = [
        int(bt == Chem.rdchem.BondType.SINGLE), int(bt == Chem.rdchem.BondType.DOUBLE),
        int(bt == Chem.rdchem.BondType.TRIPLE), int(bt == Chem.rdchem.BondType.AROMATIC),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing())
    ]
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
    
    return np.array(bond_feats)
   
def mol2graph(smi):
    # 최대 사이즈 fix 된 부분 torch dataset 에서 조정 
    mol = Chem.MolFromSmiles(smi)
    
    X = np.zeros((MAX_LEN, 61), dtype=np.uint8)
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

def new_mol2graph(smiles_list, y_list):
    data_list = []
    for smi, y in zip(smiles_list, y_list):
        mol = Chem.MolFromSmiles(smi)
        
        # atom feature
        X = np.zeros((mol.GetNumAtoms(), 61))
        for i, atom in enumerate(mol.GetAtoms()):
            feature = atom_feature(atom)
            X[i, :] = feature
        X = torch.tensor(X, dtype = torch.float)
        
        # bond index
        (rows, cols) = np.nonzero(Chem.GetAdjacencyMatrix(mol))
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        E = torch.stack([torch_rows, torch_cols], dim = 0)
        
        # bond feature
        EF = np.zeros((2*mol.GetNumBonds(), 10))
        for (k, (i,j)) in enumerate(zip(rows, cols)):
            EF[k] = bond_feature(mol.GetBondBetweenAtoms(int(i),int(j)))
        EF = torch.tensor(EF, dtype = torch.float)
        
        data = Data(x=X, edge_index=E, edge_attr=EF, y =y)
        
        data_list.append(data)
        
    return data_list