
import deepchem as dc
from deepchem.models import AttentiveFPModel
from deepchem.models.callbacks import ValidationCallback
from deepchem.models import KerasModel
import tensorflow as tf
from silence_tensorflow import silence_tensorflow

import time
import pandas as pd
import numpy as np
from sklearn import metrics
from rdkit import DataStructs, Chem
from rdkit.Chem import AllChem

from Cheminfomatics.utility.Featurizer import *

silence_tensorflow()
pd.set_option('display.max_columns', None)

featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)

class hERG_model:
    def __init__(self):
        return

    def run_model(self):

        # feature & label generation - ConvMolFeaturizer

        file_name = './Cheminfomatics/data/basic_paper_with_choi.xlsx'
        df = pd.read_excel(file_name, header=1)

        df['Label'] = [ 0 if value >= 10 else 1 for value in df['IC50 (μM)']]
        df['feature'] = [ featurizer.featurize(smiles)[0] for smiles in df['SMILES'] ] 
        
        # file_name = './deeplearning/data/herg_central.tab'
        # df = pd.read_csv(file_name, delimiter='\t')
        
        # df['Label'] = df['hERG_inhib']
        # df['feature'] = [ featurizer.featurize(smiles)[0] for smiles in df['X'] ] 

        # Data split - train / test
        blocker = df.loc[df['Label'] == 1]
        non_blocker = df.loc[df['Label'] == 0]

        print(len(blocker), len(non_blocker))

        blocker_train = blocker.groupby('Label').sample(frac=.8, random_state=950228)
        blocker_test = blocker.loc[blocker.index.difference(blocker_train.index)]

        non_blocker_train = non_blocker.groupby('Label').sample(frac=.8, random_state=950228)
        non_blocker_test = non_blocker.loc[non_blocker.index.difference(non_blocker_train.index)]

        print(len(blocker_train), len(blocker_test), len(non_blocker_train), len(non_blocker_test))

        train = pd.concat([blocker_train, non_blocker_train])
        test = pd.concat([blocker_test, non_blocker_test])

        # Data split - train / validation ( 5-fold cross validation )
        train_dataset = dc.data.NumpyDataset(train['feature'].to_numpy(), train['Label'].to_numpy())
        test_dataset = dc.data.NumpyDataset(test['feature'].to_numpy(), test['Label'].to_numpy())
        print(train_dataset, test_dataset)

        splitter = dc.splits.RandomSplitter()
        k_splitter = splitter.k_fold_split(train_dataset, 5)
        print(len(k_splitter), k_splitter)
        # train, val = splitter.train_test_split(train_dataset, seed=950228)

        # model define
        metric = dc.metrics.Metric(dc.metrics.roc_auc_score, mode='classification')
        # metric = dc.metrics.Metric(dc.metrics.auc, mode='classification')

        # model running
        k = 1
        for train, val in k_splitter:
            start = time.time()
            save_dir = './attive_' + str(k)
            model = AttentiveFPModel(1, batch_size=64, mode='classification', dropout=0.5, learning_rate=0.0001) # best parameter
            callback = ValidationCallback(val, 1000, metrics=[metric], save_dir=save_dir)

            # model.fit(train, nb_epoch=5000, callbacks=callback)
            model.fit(train, nb_epoch=5000, callbacks=callback)
            print('training : ', model.evaluate(train, [metric]), 'validate : ', model.evaluate(val, [metric]), 'test : ', model.evaluate(test_dataset, [metric]))

            k += 1

            end = time.time() - start
            print('finish 1 model time : ', end)
        
        # whole model
        # save_dir = './deeplearning/model/whole'
        # model = GraphConvModel(1, batch_size=64, mode='classification', dropout=0.5, learning_rate=0.0001) # best parameter
        # callback = ValidationCallback(test_dataset, 1000, metrics=[metric], save_dir=save_dir)
        
        # model.fit(train_dataset, nb_epoch=5000, callbacks=callback)
        
        # print(model.evaluate(train_dataset), model.evaluate(test_dataset))
        
        
        # load test
        # for train, val in k_splitter:
        #     print('training : ', model.evaluate(train, [metric]), 'validate : ', model.evaluate(val, [metric]), 'test : ', model.evaluate(test_dataset, [metric]))
        #     break
        
        # save_dir = './deeplearning/model/test_1'
        # model = GraphConvModel(1, batch_size=64, mode='classification', dropout=0.5, learning_rate=0.0001, model_dir=save_dir) # best parameter
        # callback = ValidationCallback(val, 1000, metrics=[metric])
        # model.fit(train, nb_epoch=500, callbacks=callback)
        # print('training : ', model.evaluate(train, [metric]), 'validate : ', model.evaluate(val, [metric]), 'test : ', model.evaluate(test_dataset, [metric]))

        # model = GraphConvModel(1, mode='classification', model_dir='deeplearning/data/1')
        # model.restore()
        # print('training : ', model.evaluate(train, [metric]), 'validate : ', model.evaluate(val, [metric]), 'test : ', model.evaluate(test_dataset, [metric]))

    def load_model(self):
        file_name = './deeplearning/data/hERG_8-14-2023_frac.xlsx'
        df = pd.read_excel(file_name, header=0)
        
        print(df)
        
        df['feature'] = [ featurizer.featurize(smiles)[0] for smiles in df['smiles'] ]
        df['warhead_feature'] = [ featurizer.featurize(smiles)[0] for smiles in df['warhead'] ]
        df['e3_feature'] = [ featurizer.featurize(smiles)[0] for smiles in df['e3ligand'] ]
        linker_feature = [ featurizer.featurize(smiles)[0] for smiles in df['linker'] if str(smiles) !='nan' ]
        
        test_dataset = dc.data.NumpyDataset(df['feature'].to_numpy(), df['label'].to_numpy())
        test_warhead = dc.data.NumpyDataset(df['warhead_feature'].to_numpy())
        test_e3 = dc.data.NumpyDataset(df['e3_feature'].to_numpy())
        test_linker = dc.data.NumpyDataset(linker_feature)
        
        print(test_dataset)        
        metric = dc.metrics.Metric(dc.metrics.roc_auc_score, mode='classification')
        all_performance = 0
        all_list = []
        warhead_list = []
        e3_list = []
        linker_list = []
        for i in range(1):
            # path = './deeplearning/model/mod_' + str(3)
            path = './deeplearning/model/central_1'
            model = AttentiveFPModel(1, mode='classification', batch_size=64, model_dir=path)
            model.restore()
            
            performance = float(model.evaluate(test_dataset, [metric])['roc_auc_score'])
            all_performance = all_performance + performance
            
            prediction = model.predict(test_dataset)
            
            print(performance)
            
            all_list = self.result_check(all_list, prediction)
            
            warhead_pred = model.predict(test_warhead)
            e3_pred = model.predict(test_e3)
            linker_pred = model.predict(test_linker)
            
            warhead_list = self.result_check(warhead_list, warhead_pred)
            e3_list = self.result_check(e3_list, e3_pred)
            linker_list = self.result_check(linker_list, linker_pred)
            
        print(df['label'].tolist())
        print(all_list)
        print(warhead_list)
        print(e3_list)
        print(linker_list)
        
        # result_dict = {'experiment': df['label'].tolist(),
        #                'protac_predict': all_list,
        #                'warhead_predict': warhead_list,
        #                'e3_predict': e3_list}
        
        # result_df = pd.DataFrame(result_dict, columns=['experiment', 'protac_predict', 'warhead_predict', 'e3_predict'])
        # result_df.to_csv('./test_result.csv', header=True)
        
        # all_performance = all_performance/5
        # print(all_performance)
        
        # model2 = KerasModel(model=tf.keras.Model, loss=metric, batch_size=64, learning_rate=0.0001, model_dir='./deeplearning/data/1')
        # model3 = model2.get_checkpoints()
        # print(model3)
        # model4 = model2.restore(checkpoint=model3, model_dir='deeplearning/data/1')
        # print(model4)
        
    def result_check(self, result_list, prediction):
        # result = []
        for j, pred in enumerate(prediction):
            prob_0 = pred[0][0]
            prob_1 = pred[0][1]
            
            if prob_0 > prob_1:
                label = 0
            else:
                label = 1
            
            # if len(result_list) > 0:
            #     result.append(result_list[-1][j] + label)
            # else:
            #     result.append(label)
            
            result_list.append(label)
        # result_list.append(result)
    
        return result_list

    def calc_similarity(self):
        file_name = './deeplearning/data/data.xlsx'
        origin_df = pd.read_excel(file_name, header=1)
        
        file_name = './deeplearning/data/hERG_8-14-2023_frac.xlsx'
        df = pd.read_excel(file_name, header=0)
        
        fpgen = AllChem.GetRDKitFPGenerator()
        # fps = [ fpgen.GetFingerprint(Chem.MolFromSmiles(x)) for x in origin_df['SMILES'] ]
        fps = [ fpgen.GetFingerprint(Chem.MolFromSmiles(x)) for x in df['smiles'] ]
        
        protac = fpgen.GetFingerprint(Chem.MolFromSmiles(df['smiles'][0]))
        warhead = fpgen.GetFingerprint(Chem.MolFromSmiles(df['warhead'][0]))
        e3 = fpgen.GetFingerprint(Chem.MolFromSmiles(df['e3ligand'][0]))
        linker = fpgen.GetFingerprint(Chem.MolFromSmiles(df['linker'][0]))
        
        protac_sim = 0
        warhead_sim = 0
        e3_sim = 0
        linker_sim = 0
        
        for fp in fps:
            score = DataStructs.TanimotoSimilarity(fp, protac)
            print(score)    
            if score > 0.85:
                protac_sim += 1
            
            # score = DataStructs.TanimotoSimilarity(fp, warhead)
            # if score > 0.7:
            #     warhead_sim += 1
                
            # score = DataStructs.TanimotoSimilarity(fp, e3)
            # if score > 0.7:
            #     print(score)
            #     e3_sim += 1
                
            # score = DataStructs.TanimotoSimilarity(fp, linker)
            # if score > 0.5:
            #     linker_sim += 1
            
        print(protac_sim, warhead_sim, e3_sim, linker_sim)
        
    def merge_data(self):
        file_1 = './deeplearning/data/data.xlsx' # base paper
        file_2 = './deeplearning/data/ToxTree.csv' # toxtree
        file_3 = './deeplearning/data/train_validation_cardio_tox_data.csv' # cardiotox
        
        file_4 = './deeplearning/data/herg_central.tab'
        df = pd.read_csv(file_4, delimiter='\t')
        
        
        file_name = './deeplearning/data/hERG_8-14-2023_frac.xlsx'
        origin_df = pd.read_excel(file_name, header=0)
        
        print(df)
        
        # fpgen = AllChem.GetRDKitFPGenerator()
        # fps = [ fpgen.GetFingerprint(Chem.MolFromSmiles(SingleStandardizer(smiles))) for smiles in df['X'] ]
        
        # print(len(fps))
        
        # protac = fpgen.GetFingerprint(Chem.MolFromSmiles(origin_df['smiles'][0]))
        # warhead = fpgen.GetFingerprint(Chem.MolFromSmiles(origin_df['warhead'][0]))
        # e3 = fpgen.GetFingerprint(Chem.MolFromSmiles(origin_df['e3ligand'][0]))
        # linker = fpgen.GetFingerprint(Chem.MolFromSmiles(origin_df['linker'][0]))
        
        # protac_sim = 0
        # warhead_sim = 0
        # e3_sim = 0
        # linker_sim = 0
        
        # for fp in fps:
        #     score = DataStructs.TanimotoSimilarity(fp, protac)
        #     if score > 0.6:
        #         protac_sim += 1
            
        #     score = DataStructs.TanimotoSimilarity(fp, warhead)
        #     if score > 0.5:
        #         warhead_sim += 1
                
        #     score = DataStructs.TanimotoSimilarity(fp, e3)
        #     if score > 0.5:
        #         print(score)
        #         e3_sim += 1
                
        #     score = DataStructs.TanimotoSimilarity(fp, linker)
        #     if score > 0.5:
        #         linker_sim += 1
            
        # print(protac_sim, warhead_sim, e3_sim, linker_sim)
        
        # df_1 = pd.read_excel(file_1, header=1)
        # df_2 = pd.read_csv(file_2, header=0)
        # df_3 = pd.read_csv(file_3, header=0)
        
        # df_1_smiles = [ SingleStandardizer(smiles) for smiles in df_1['SMILES']]
        # df_1_label = [ 0 if value >= 10 else 1 for value in df_1['IC50 (μM)']]
        # df_1_ids = df_1['No.'].tolist()
        
        
        # new_df_1 = pd.DataFrame()
    
        # df_2_smiles = [ SingleStandardizer(smiles) for smiles in df_2['SMILES']]
        # df_3_smiles = [ SingleStandardizer(smiles) for smiles in df_3['smiles']]
        
hERG_model().run_model()
