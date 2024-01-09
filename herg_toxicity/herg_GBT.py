from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
# from sklearn.externals import joblib

smiles = 'CC'

m = Chem.MolFromSmiles(smiles)
desc_list = [n[0] for n in Descriptors._descList]
phc_desc = [i for i in desc_list if not i.startswith('fr_')]
print(len(phc_desc))
calc = MoleculeDescriptors.MolecularDescriptorCalculator(phc_desc)

file_name = '../data/basic_paper_with_choi.xlsx'
df = pd.read_excel(file_name, header=1)

df['Label'] = [ 0 if value >= 10 else 1 for value in df['IC50 (μM)']]
for desc in phc_desc:
    df[desc] = np.NaN

for i, smiles in enumerate(df['SMILES']):
    m = Chem.MolFromSmiles(smiles)
    desc = calc.CalcDescriptors(m)    
    for j, de in enumerate(desc):
        df.loc[i, phc_desc[j]] = de

blocker = df.loc[df['Label'] == 1]
non_blocker = df.loc[df['Label'] == 0]

print(len(blocker), len(non_blocker))

blocker_train = blocker.groupby('Label').sample(frac=.8, random_state=950228)
# blocker_val = blocker_train.groupby('Label').sample(frac=.2, random_state=950228)
# blocker_train = blocker_train.loc[blocker_train.index.difference(blocker_val.index)]
blocker_test = blocker.loc[blocker.index.difference(blocker_train.index)]

non_blocker_train = non_blocker.groupby('Label').sample(frac=.8, random_state=950228)
# non_blocker_val = non_blocker_train.groupby('Label').sample(frac=.2, random_state=950228)
# non_blocker_train = non_blocker_train.loc[non_blocker_train.index.difference(non_blocker_val.index)]
non_blocker_test = non_blocker.loc[non_blocker.index.difference(non_blocker_train.index)]

print(len(blocker_train), len(blocker_test), len(non_blocker_train), len(non_blocker_test))

train = pd.concat([blocker_train, non_blocker_train])
# val = pd.concat([blocker_val, non_blocker_val])
test = pd.concat([blocker_test, non_blocker_test])

train_X = train.loc[:, phc_desc]
train_y = train.Label

# val_X = val.loc[:, phc_desc]
# val_y = val.Label

test_X = test.loc[:, phc_desc]
test_y = test.Label

print(train_X.shape, train_y.shape)

param_grid = {
    "loss":["deviance"],
    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    "min_samples_split": np.linspace(0.1, 0.5, 12),
    "min_samples_leaf": np.linspace(0.1, 0.5, 12),
    "max_depth":[3,5,8],
    "max_features":["log2","sqrt"],
    "criterion": ["squared_error",  "absolute_error"],
    "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
    "n_estimators":[10]
}

clf = GradientBoostingClassifier()
grid_search = GridSearchCV(estimator=clf, param_grid= param_grid, cv=5, n_jobs=5, verbose=2)

grid_search.fit(train_X, train_y)

print(grid_search.best_params_)

# clf = GradientBoostingClassifier(
#             loss='deviance', ## ‘deviance’, ‘exponential’
#             criterion='squared_error', ## 개별 트리의 불순도 측도
#             n_estimators=5000, ## 반복수 또는 base_estimator 개수
#             min_samples_leaf=5, ## 개별 트리 최소 끝마디 샘플 수
#             max_depth=3, ## 개별트리 최대 깊이
#             learning_rate=0.01, ## 스텝 사이즈
#             subsample = 0.8,
#             random_state=950228
#         ).fit(train_X, train_y)

# # print(clf.predict(X)[:3]) 
 
# ## 변수 중요도
# # for i, col in enumerate(train_X.columns):
# #     print(f'{col} 중요도 : {clf.feature_importances_[i]}')
 
# print(clf.get_params()) ## GradientBoostingClassifier 클래스 인자 설정 정보
# print('정확도 : ', clf.score(train_X,train_y)) ## 성능 평가 점수(Accuracy)
# print('정확도 : ', clf.score(test_X,test_y)) ## 성능 평가 점수(Accuracy)

# joblib.dump(clf, '../model/GBT.pkl')