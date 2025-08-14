from typing import List

import numpy
import numpy as np
import pandas as pd
import sklearn
import torch
from scipy.stats import yeojohnson
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from .multiple_processes import test_get_predict
from util import create_adj



def auto_minmax_scaler(df, use_percentile=True, lower=0.01, upper=0.99, feature_range=(0, 1)):
    df = df.copy()
    a, b = feature_range
    if use_percentile:
        q_min = df.quantile(lower)
        q_max = df.quantile(upper)
    else:
        q_min = df.min()
        q_max = df.max()
    denominator = (q_max - q_min).replace(0, 1e-9)
    df_clipped = df.clip(lower=q_min, upper=q_max, axis=1)
    scaled_df = (df_clipped - q_min) / denominator * (b - a) + a

    return scaled_df


class load_Dataset_ellipticPlusPlus(Dataset):
    def __init__(self,tr_path='data/txs_label.csv',edge_path='data/txs_edgelist.csv'):
        self.data_df = pd.read_csv(tr_path, dtype=float)
        self.edge_df = pd.read_csv(edge_path, dtype=float)
        self.data_df = self.data_df[self.data_df["class"].isin([1, 2])]

        self.data_timeList = List


    def get_Sliceoftimeofdatadf(self, Sliceoftime):
        self.data = self.data_df[self.data_df["Time step"].eq(Sliceoftime)]
        self.feature = self.data.iloc[:, 2:-1]
        self.label = self.data.iloc[:, -1]
        self.features_index = self.data.iloc[:, 0]

def co_association(matrix: numpy, cluster_result: numpy) -> numpy:

    num = len(matrix)

    cluster_matrix = cluster_result[:, None] == cluster_result[None, :]

    matrix += cluster_matrix.astype(np.float32)

    return matrix


def dataload_preprocess(semi_flag=False,tr_path='data/txs_label.csv',edge_path='data/txs_edgelist.csv'):

    ellipticplusplus = load_Dataset_ellipticPlusPlus(tr_path,edge_path)

    if semi_flag == True:
        test_get_predict()

    featureList = [None]
    semi_resultList = [None]
    fakelabelList = [None]
    lableLsit = [None]
    adjList = [None]
    ca_weightsList = [None]
    ca_matrixList = [None]
    ca_featureList = [None]
    print('loading data............')
    for timenum in [[i] for i in range(1, 50)]:
        path = './CMOS/semi/semi_supervised_results'
        AdaBoost = pd.read_csv(path + f'/AdaBoost/AdaBoost{timenum}.csv')
        DecisionTree = pd.read_csv(path + f'/DecisionTree/DecisionTree{timenum}.csv')
        RandomForest = pd.read_csv(path + f'/RandomForest/RandomForest{timenum}.csv')
        xgboost = pd.read_csv(path + f'/xgboost/xgboost{timenum}.csv')
        gbBoost = pd.read_csv(path + f'/gbBoost/gbBoost{timenum}.csv')
        SVC = pd.read_csv(path + f'/SVC/SVC{timenum}.csv')

        data = AdaBoost
        data.columns.values[0] = 'txId'
        data.columns.values[1] = 'AdaBoost'

        data["DecisionTree"] = DecisionTree['predict']
        data["gbBoost"] = gbBoost['predict']
        data["RandomForest"] = RandomForest['predict']
        data["xgboost"] = xgboost['predict']
        data["SVC"] = SVC['predict']

        data_num = len(data['txId'])
        ca_matrix = np.zeros((data_num, data_num))
        # Compute the Co-association matrix

        ca_matrix = co_association(ca_matrix, data["AdaBoost"].to_numpy())
        ca_matrix = co_association(ca_matrix, data["DecisionTree"].to_numpy())
        ca_matrix = co_association(ca_matrix, data["gbBoost"].to_numpy())
        ca_matrix = co_association(ca_matrix, data["xgboost"].to_numpy())
        ca_matrix = co_association(ca_matrix, data["RandomForest"].to_numpy())
        ca_matrix = co_association(ca_matrix, data["SVC"].to_numpy())
        ca_matrix /= 2
        ca_matrix = ca_matrix
        ca_matrixList.append(ca_matrix)

        fakelabe = data.iloc[:, 1:].values.sum(axis=1) / len(ca_matrix[0])
        fakelabelList.append(torch.tensor(fakelabe, dtype=torch.float32).cuda())

        semi_resultList.append(torch.tensor(data.iloc[:, 1:].values,dtype=torch.float32).cuda())

        ca_weights = []
        adj, _ = create_adj(pd.DataFrame(data["txId"]),edge_df=ellipticplusplus.edge_df)
        adjList.append(adj.cuda())

        # Edge cocoweighting Iterates over each edge and extracts the corresponding weight
        for edge_i in range(len(adj[0])):
            node1, node2 = adj[0][edge_i], adj[1][edge_i]
            ca_weights.append(ca_matrix[node1][node2])
        ca_weightsList.append(torch.tensor(ca_weights, dtype=torch.float).cuda())


        ellipticplusplus.get_Sliceoftimeofdatadf(timenum[0])
        feature_oftimeseg = ellipticplusplus.feature

        # Feature scaling
        feature_oftimeseg = auto_minmax_scaler(feature_oftimeseg, use_percentile=True)

        feature = np.concatenate((feature_oftimeseg, data.iloc[:, 1:].values * 0.9), axis=1)
        feature = torch.tensor(feature, dtype=torch.float32).cuda()
        featureList.append(feature)

        def ca_feature_computation(ca_matrix,feature_oftimeseg):
            ca_feature = ca_matrix @ feature_oftimeseg
            ca_feature = (1 + 1e-8 - auto_minmax_scaler(ca_feature, use_percentile=True)) * feature_oftimeseg.values
            ca_feature = feature_oftimeseg.values + ca_feature.values

            pca = PCA(n_components=1)
            ca_feature = pca.fit_transform(ca_feature)
            ca_feature = np.concatenate((ca_feature, data.iloc[:, 1:].values * 0.9), axis=1)
            ca_feature = torch.tensor(ca_feature, dtype=torch.float32).cuda()
            return ca_feature

        # Co - association feature aggregation
        ca_feature = ca_feature_computation(ca_matrix,feature_oftimeseg)
        ca_featureList.append(ca_feature)

        lable = ellipticplusplus.label - 1
        lable = torch.tensor(lable.values, dtype=torch.long)
        lableLsit.append(lable.cuda())

    return (featureList, adjList, lableLsit, ca_matrixList, ca_weightsList,semi_resultList, fakelabelList,ca_featureList)
