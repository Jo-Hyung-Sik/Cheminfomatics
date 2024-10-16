from ..utility import Featurizer
from experiment import *
from model import GCN

from torch_geometric.loader import DataLoader
from torch import nn
from torch_geometric.nn import summary
from sklearn.metrics import * 

import pandas as pd

df = pd.read_excel('./data/basic_paper_with_choi.xlsx', header=1)
df['Label'] = [ 0 if value >= 10 else 1 for value in df['IC50 (μM)']]

blocker = df.loc[df['Label'] == 1]
non_blocker = df.loc[df['Label'] == 0]

print(len(blocker), len(non_blocker))

blocker_train = blocker.groupby('Label').sample(frac=.8, random_state=950228)
blocker_val = blocker_train.groupby('Label').sample(frac=.2, random_state=950228)
blocker_train = blocker_train.loc[blocker_train.index.difference(blocker_val.index)]
blocker_test = blocker.loc[blocker.index.difference(blocker_train.index)]

non_blocker_train = non_blocker.groupby('Label').sample(frac=.8, random_state=950228)
non_blocker_val = non_blocker_train.groupby('Label').sample(frac=.2, random_state=950228)
non_blocker_train = non_blocker_train.loc[non_blocker_train.index.difference(non_blocker_val.index)]
non_blocker_test = non_blocker.loc[non_blocker.index.difference(non_blocker_train.index)]

# print(len(blocker_train), len(blocker_test), len(non_blocker_train), len(non_blocker_test))

train = pd.concat([blocker_train, non_blocker_train])
val = pd.concat([blocker_val, non_blocker_val])
test = pd.concat([blocker_test, non_blocker_test])

print(len(train), len(val), len(test))

train_smiles_list = train['SMILES'].tolist()
train_y_list = train['Label'].tolist()

val_smiles_list = val['SMILES'].tolist()
val_y_list = val['Label'].tolist()

test_smiles_list = test['SMILES'].tolist()
test_y_list = test['Label'].tolist()

train_data_list = new_mol2graph(train_smiles_list, train_y_list)
val_data_list = new_mol2graph(val_smiles_list, val_y_list)
test_data_list = new_mol2graph(test_smiles_list, test_y_list)

train_dataloader = DataLoader(train_data_list, batch_size=256, shuffle=True)
val_dataloader = DataLoader(val_data_list, batch_size=256)
test_dataloader = DataLoader(test_data_list, batch_size=256)

model = GCN(61, 128)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# summary
sample_smiles_list = ['CCC']
sample_y_list = [0]
sample_list = new_mol2graph(sample_smiles_list, sample_y_list)
sample_dataloader = DataLoader(sample_list, batch_size=1)
for batch in sample_dataloader:
    print(summary(model, batch))

# early stopping
# best_loss = 10**9
best_auc = 0
patience_check = 0
patience_limit = 500
for epoch in range(5000):
    print("epoch : ", epoch)
    model, loss = training(train_dataloader, model, optimizer, criterion)
    val_auc, val_loss = eval(val_dataloader, model, criterion)
    test_auc, test_loss = eval(test_dataloader, model, criterion)
    
    if best_auc > val_auc:
        patience_check += 1
        
        if patience_check >= patience_limit:
            print(best_auc)
            break
    
    else:
        best_auc = val_auc
        patience_check = 0
    
    # if val_loss > best_loss:
    #     patience_check += 1
        
    #     if patience_check >= patience_limit:
    #         break
    # else:
    #     best_loss = val_loss
    #     patience_check = 0
