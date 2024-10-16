import pandas as pd
import numpy as np

import math

from rdkit import Chem

from utils.utils import Caculate_descriptor, Standardizer

df = pd.read_csv('./MBT compounds (SMILES)_0910.csv')
df2 = df.filter(items=['ID', 'Chemical Structure (SMILES)'])

std = Standardizer()

for ix, row in df2.iterrows():
    smi = std.calc_standard(row['Chemical Structure (SMILES)'])
    
    df2['Chemical Structure (SMILES)'][ix] = smi

df2.rename(columns={'Chemical Structure (SMILES)': 'smiles'}, inplace=True)
print(df2)

calc = Caculate_descriptor()

padel_desc_df = calc.calc_padel_desc(df2)

padel_desc_df.to_csv('./MBT_all_0910_padel.csv')