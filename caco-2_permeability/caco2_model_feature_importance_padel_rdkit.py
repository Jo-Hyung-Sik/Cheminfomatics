from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

from rdkit import Chem, SimDivFilters
from rdkit.Chem import rdMolDescriptors

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

import operator
import pandas as pd
import numpy as np
import math

from utils.utils import Preprocessing

# ext_rdkit_df = pd.read_csv('./MBT_rdkit_desc2.csv', index_col='Unnamed: 0')
# ext_padel_df = pd.read_csv('./MBT_padel_desc2.csv', index_col='Unnamed: 0')
# ext_merge_df = pd.merge(ext_padel_df, ext_rdkit_df, on='smiles')

# ext_merge_df = ext_merge_df.drop('name_y', axis=1)
# ext_merge_df = ext_merge_df.drop('logPapp_y', axis=1)
# ext_merge_df.rename(columns={'logPapp_x': 'logPapp'}, inplace=True)


rdkit_df = pd.read_csv('./data/2_1_all_rdkit_desc.csv', index_col='Unnamed: 0')
padel_df = pd.read_csv('./data/2_1_all_padel_desc.csv', index_col='Unnamed: 0')
merge_df = pd.merge(padel_df, rdkit_df, on='smiles')

merge_df = merge_df.drop('name_y', axis=1)
merge_df = merge_df.drop('logPapp_y', axis=1)
merge_df.rename(columns={'logPapp_x': 'logPapp'}, inplace=True)

mask2 = (merge_df.logPapp > -2) & (merge_df.logPapp < 2) 
df2 = merge_df.loc[mask2, :]

nciFps = []
for smi in df2['smiles']:
    mol = Chem.MolFromSmiles(smi)
    nciFps.append(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2))
    
mmp = SimDivFilters.MaxMinPicker()
picks = mmp.LazyBitVectorPick(nciFps, len(nciFps), int(len(df2) * 0.7), seed=709)

y = df2['logPapp']
X = df2.iloc[:,3:]

y_ext = ext_merge_df['logPapp']
X_ext = ext_merge_df.iloc[:,3:]

## Preprocessing -- already filtering file
# Remove missing value
for item in X:
    try:
        X[item].astype('float')
    except:
        X[item] = X[item].replace(r'^\s*$', np.NaN, regex=True)

X = X.dropna(axis=1).astype('float')

print('remove missing value : ', len(X.columns))

# standard deviation; SD < 0.01 
for item in X:
    if np.std(X[item]) < 0.01:
        X = X.drop(columns=item, axis=1)

print('standard deviation : ', len(X.columns))

# R2 filter; Xs vs. Y <= 0.01
Xs_y_score_dict = {}
for item in X:
    score = np.corrcoef(X[item], y)[0,1]
    if score <= 0.01:
        X = X.drop(columns=item, axis=1)
        
    else:
        Xs_y_score_dict[item] = score

print('Xs vs. Y : ', len(X.columns))

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

print('Colinear : ', len(X.columns))

X_ext = X.filter(items=list(X.columns))

# # ## train_test split
X_train = X.iloc[list(picks)]
y_train = y.iloc[list(picks)]

X_test = X.loc[set(X.index) - set(list(picks))]
y_test = y.loc[set(y.index) - set(list(picks))]

estimator = ExtraTreesRegressor(random_state=709)
rfecv = RFECV(estimator, min_features_to_select=10, cv=5, scoring='neg_mean_absolute_error')

rfecv.fit(X_train, y_train)

# rfecv_df = pd.DataFrame(rfecv.cv_results_)
# name = file_name.replace('.csv', '.xlsx')
# rfecv_df.to_excel(excel_writer=name)

print( 'n features : ', rfecv.n_features_)
print( 'features : ', rfecv.get_feature_names_out)

X_train_fit = rfecv.transform(X_train)
X_test_fit = rfecv.transform(X_test)
X_ext_fit = rfecv.transform(X_ext)

model = LGBMRegressor()
param_grid = [{
        "num_leaves": [127],
        "max_depth": [-1],
        "subsample": [0.8],
        "colsample_bytree": [1.0]
    }]

gscv = GridSearchCV(estimator=model, param_grid= param_grid, scoring=['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error'], cv=5, refit='neg_root_mean_squared_error', n_jobs=1, verbose=2)
gscv.fit(X_train_fit, y_train)

print('SVR rbf params : ', gscv.best_params_)
print('SVR rbf score : ', gscv.best_score_)

pred_y_train = gscv.predict(X_train_fit)

# model.fit(X_train_fit, y_train)
# pred_y_train = model.predict(X_train_fit)

train_r2 = r2_score(y_train, pred_y_train)
train_rmse = math.sqrt(mean_squared_error(y_train, pred_y_train))
train_mae = mean_absolute_error(y_train, pred_y_train)

print('train R2 :', train_r2, 'train RMSE :', train_rmse, 'train_MAE :', train_mae)

pred_y_test = gscv.predict(X_test_fit)
test_r2 = r2_score(y_test, pred_y_test)
test_rmse = math.sqrt(mean_squared_error(y_test, pred_y_test))
test_mae = mean_absolute_error(y_test, pred_y_test)

print('test R2 :', test_r2, 'test RMSE :', test_rmse, 'test_MAE :', test_mae)

# pred_y_ext = gscv.predict(X_ext_fit)
# ext_r2 = r2_score(y_ext, pred_y_ext)
# ext_rmse = math.sqrt(mean_squared_error(y_ext, pred_y_ext))
# ext_mae = mean_absolute_error(y_ext, pred_y_ext)

# print('test R2 :', ext_r2, 'test RMSE :', ext_rmse, 'test_MAE :', ext_mae)

# result_list = [['rdkit_padel_merge', type(model).__name__, gscv.best_params_, train_r2, train_rmse, train_mae, test_r2, test_rmse, test_mae]]

# result_df = pd.DataFrame(result_list, columns=['data', 'model', 'parameter', 'trainR2', 'trainRMSE', 'trainMAE', 'testR2', 'testRMSE', 'testMAE'])
# result_df.to_excel('./results/LightGBM_rdkit_padel_merge.xlsx')