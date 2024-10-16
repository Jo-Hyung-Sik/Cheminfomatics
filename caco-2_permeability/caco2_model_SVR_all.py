from sklearn.svm import SVR
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

import operator
import pandas as pd
import numpy as np
import math

from utils.utils import Preprocessing

file_list = ['2_1_all_rdkit_desc.csv', '2_1_all_padel_desc.csv', '3_1_all_morgan.csv', '3_1_all_rdkit_maccs.csv']
# file_list = ['3_1_all_morgan.csv', '3_1_all_rdkit_maccs.csv']

result_list = []
for file_name in file_list:
    # ## Data Read
    file_name2 = './data/' + file_name
    
    print('this file :: ', file_name2)
    df = pd.read_csv(file_name2, index_col='Unnamed: 0')

    mask2 = (df.logPapp > -2) & (df.logPapp < 2) 
    df2 = df.loc[mask2, :]

    nciFps = []
    for smi in df2['smiles']:
        mol = Chem.MolFromSmiles(smi)
        nciFps.append(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2))
        
    mmp = SimDivFilters.MaxMinPicker()
    picks = mmp.LazyBitVectorPick(nciFps, len(nciFps), int(len(df2) * 0.7), seed=709)

    y = df2['logPapp']
    X = df2.iloc[:,3:]

    # ## Preprocessing
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

    # ## train_test split
    X_train = X.iloc[list(picks)]
    y_train = y.iloc[list(picks)]

    X_test = X.loc[set(X.index) - set(list(picks))]
    y_test = y.loc[set(y.index) - set(list(picks))]

    estimator = ExtraTreesRegressor(random_state=709)
    rfecv = RFECV(estimator, min_features_to_select=10, cv=5, scoring='neg_mean_absolute_error')

    rfecv.fit(X_train, y_train)
    
    rfecv_df = pd.DataFrame(rfecv.cv_results_)
    name = file_name.replace('.csv', '.xlsx')
    rfecv_df.to_excel(excel_writer=name)
    
    print( 'n features : ', rfecv.n_features_)
    print( 'features : ', rfecv.get_feature_names_out)
    
    X_train_fit = rfecv.transform(X_train)
    X_test_fit = rfecv.transform(X_test)

    model = SVR()
    param_grid = [
    {'C': [0.01, 0.1, 1, 10, 100], 'gamma': [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001], 'kernel': ['rbf']},
    ]

    gscv_svr = GridSearchCV(estimator=model, param_grid= param_grid, scoring=['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error'], cv=5, refit='neg_root_mean_squared_error', n_jobs=1, verbose=2)
    gscv_svr.fit(X_train_fit, y_train)

    print('SVR rbf params : ', gscv_svr.best_params_)
    print('SVR rbf score : ', gscv_svr.best_score_)

    pred_y_train = gscv_svr.predict(X_train_fit)
    train_r2 = r2_score(y_train, pred_y_train)
    train_rmse = math.sqrt(mean_squared_error(y_train, pred_y_train))
    train_mae = mean_absolute_error(y_train, pred_y_train)
    
    print('train R2 :', train_r2, 'train RMSE :', train_rmse, 'train_MAE :', train_mae)

    pred_y_test = gscv_svr.predict(X_test_fit)
    test_r2 = r2_score(y_test, pred_y_test)
    test_rmse = math.sqrt(mean_squared_error(y_test, pred_y_test))
    test_mae = mean_absolute_error(y_test, pred_y_test)
    
    print('test R2 :', test_r2, 'test RMSE :', test_rmse, 'test_MAE :', test_mae)

    result_list.append([file_name, type(model).__name__, gscv_svr.get_feature_names_out, train_r2, train_rmse, train_mae, test_r2, test_rmse, test_mae])
    
result_df = pd.DataFrame(result_list, columns=['data', 'model', 'parameter', 'trainR2', 'trainRMSE', 'trainMAE', 'testR2', 'testRMSE', 'testMAE'])
result_df.to_excel('./results/SVR_result.xlsx')