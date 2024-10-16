from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import Descriptors, MACCSkeys, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors

from padelpy import from_smiles

import numpy as np
import pandas as pd
import operator

class Caculate_descriptor:
    def __init__(self):
        return
    
    def calc_rdkit_desc(self, df):
        calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
        desc_names = calc.GetDescriptorNames()
        print(desc_names)
        mol_desc = []
        for idx, row in df.iterrows():
            smiles = row['smiles']
            try:
                mol = Chem.MolFromSmiles(smiles)
                mol = Chem.AddHs(mol)
            except:
                print('error id :', idx, ', smiles :', smiles )
                continue
                
            desc = calc.CalcDescriptors(mol)
            mol_desc.append(desc)
            
        df2 = pd.DataFrame(mol_desc, columns=desc_names)
        df3 = pd.concat([df, df2], axis=1)
        
        return df3
    
    def calc_padel_desc(self, df):
        smi_list = df.smiles.tolist()
        mol_desc = []
        for idx, row in df.iterrows():
            print(idx)
            smiles = row['smiles']
            try:
                desc = from_smiles(smiles, timeout=600)
                mol_desc.append([ desc[i] for i in desc ])
            except:
                print('error id :', idx, ', smiles :', smiles)
                mol_desc.append([ '' for i in range(len_mol_desc)])
                continue
            
            if idx == 0:
                desc_names = [ i for i in desc ]
                len_mol_desc = len(mol_desc)
        
        df2 = pd.DataFrame(mol_desc, columns=desc_names)
        df3 = pd.concat([df, df2], axis=1)
        
        return df3
    
    def calc_ecfp(self, df):
        finger_names = [ 'bit_' + str(i+1) for i in range(1024)] 
        mol_finger = []
        for idx, row in df.iterrows():
            smiles = row['smiles']
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)
            
            finger = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=1024)
            mol_finger.append(list(finger))

        df2 = pd.DataFrame(mol_finger, columns=finger_names)
        df3 = pd.concat([df, df2], axis=1)
        
        return df3
    
    def calc_maccs(self, df):
        finger_names = [ MACCSkeys.smartsPatts[i][0] for i in MACCSkeys.smartsPatts ] 
        mol_finger = []
        for idx, row in df.iterrows():
            smiles = row['smiles']
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)
            
            finger = list(MACCSkeys.GenMACCSKeys(mol))
            mol_finger.append(finger[1:])

        df2 = pd.DataFrame(mol_finger, columns=finger_names)
        df3 = pd.concat([df, df2], axis=1)
            
        return df3

class Preprocessing:
    def Column_filtering(self, X, y):
        # Remove missing value
        for item in X:
            try:
                X[item].astype('float')
            except:
                X[item] = X[item].replace(r'^\s*$', np.NaN, regex=True)

        X = X.dropna(axis=1).astype('float')

        # standard deviation; SD < 0.01 
        for item in X:
            if np.std(X[item]) < 0.01:
                X = X.drop(columns=item, axis=1)

        # R2 filter; Xs vs. Y <= 0.01
        Xs_y_score_dict = {}
        for item in X:
            score = np.corrcoef(X[item], y)[0,1]
            if score <= 0.01:
                X = X.drop(columns=item, axis=1)
                
            else:
                Xs_y_score_dict[item] = score
        
        sorted_score_dict = sorted(Xs_y_score_dict.items(), key=operator.itemgetter(1), reverse=True)
        sorted_list = [ i[0] for i in sorted_score_dict]

        # Colinear filter; Xs vs. Xs > 0.9 
        sorted_score_dict = sorted(Xs_y_score_dict.items(), key=operator.itemgetter(1), reverse=True)
        sorted_list = [ i[0] for i in sorted_score_dict]

        already_column = []
        remove_column = []
        for item1 in sorted_list:
            already_column.append(item1)    
            for item2 in sorted_list:   
                if item1 in already_column:
                    continue
                
                if item2 in remove_column:
                    continue
                
                if np.corrcoef(X[item1], X[item2])[0,1] > 0.9 and item1 != item2:
                    print(item1, item2)
                    X = X.drop(columns=item2, axis=1)
                    remove_column.append(item2)
                    
        return X
    
    