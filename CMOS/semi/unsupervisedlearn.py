import numpy
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from scipy.stats import yeojohnson
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def ClusterData_preprocessing(data: numpy) -> numpy:

    data = StandardScaler().fit_transform(data)

    data = sklearn.preprocessing.normalize(data, norm='l2')

    data = np.apply_along_axis(lambda arr: yeojohnson(arr)[0], 0, data)

    return data


def slefTrain_DecisionTree(data_df, time):
    data_df = data_df.reset_index(drop=True)

    data = data_df.copy()
    data_true = data.copy()
    data["class"] = data["class"] - 1

    data_0 = data[data["class"] == 0]
    data_1 = data[data["class"] == 1]
    data_0_mask = data_0.sample(n=int(len(data_0) * 0.7), random_state=0)
    data_1_mask = data_1.sample(n=int(len(data_1) * 0.7), random_state=0)
    data_mask = pd.concat((data_0_mask, data_1_mask), axis=0)

    for n in data_mask.index.tolist():
        data.loc[n, "class"] = -1

    x, y = data.iloc[:, 2:-1], data.iloc[:, -1]

    x = ClusterData_preprocessing(x)

    base_classifier = DecisionTreeClassifier(
        criterion='gini',
        max_depth=8,
        min_samples_split=2
    )

    st = SelfTrainingClassifier(base_classifier,
                                threshold=0.85,
                                criterion="threshold",
                                k_best=50,
                                max_iter=1000,
                                ).fit(x, y)

    y_predict = st.transduction_
    y_index = data_true.iloc[:, 0]

    i = np.where(y_predict == -1)
    y_predict[i] = 1


    df = pd.DataFrame({'txId': y_index, 'predict': y_predict})
    df.to_csv(path_or_buf=f'.\semi_supervised_results\DecisionTree\DecisionTree{time}.csv', sep=',', index=False)


    print(f'DecisionTree{time}.csv saved')


    return 0


def slefTrain_SVC(data_df, time):
    data_df = data_df.reset_index(drop=True)
    data = data_df.copy()
    data_true = data.copy()
    data["class"] = data["class"] - 1
    data_0 = data[data["class"] == 0]
    data_1 = data[data["class"] == 1]
    data_0_mask = data_0.sample(n=int(len(data_0) * 0.7), random_state=0)
    data_1_mask = data_1.sample(n=int(len(data_1) * 0.7), random_state=0)
    data_mask = pd.concat((data_0_mask, data_1_mask), axis=0)
    for n in data_mask.index.tolist():
        data.loc[n, "class"] = -1

    x, y = data.iloc[:, 2:-1], data.iloc[:, -1]

    x = ClusterData_preprocessing(x)

    base_classifier = SVC(kernel="rbf", gamma=5, probability=True, class_weight={0: 0.8, 1: 0.2}, tol=1e-5)

    st = SelfTrainingClassifier(base_classifier,
                                threshold=0.85,
                                criterion="threshold",
                                k_best=50,
                                max_iter=1000,
                                ).fit(x, y)

    y_predict = st.transduction_
    y_index = data_true.iloc[:, 0]
    i = np.where(y_predict == -1)
    y_predict[i] = 1
    df = pd.DataFrame({'txId': y_index, 'predict': y_predict})
    df.to_csv(path_or_buf=f'.\semi_supervised_results\SVC\SVC{time}.csv', sep=',', index=False)
    print(f'SVC{time}.csv saved')
    return 0


def slefTrain_RandomForest(data_df, time):
    data_df = data_df.reset_index(drop=True)

    data = data_df.copy()
    data_true = data.copy()
    data["class"] = data["class"] - 1

    data_0 = data[data["class"] == 0]
    data_1 = data[data["class"] == 1]
    data_0_mask = data_0.sample(n=int(len(data_0) * 0.7), random_state=0)
    data_1_mask = data_1.sample(n=int(len(data_1) * 0.7), random_state=0)
    data_mask = pd.concat((data_0_mask, data_1_mask), axis=0)

    for n in data_mask.index.tolist():
        data.loc[n, "class"] = -1

    x, y = data.iloc[:, 2:-1], data.iloc[:, -1]

    x = ClusterData_preprocessing(x)

    base_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    st = SelfTrainingClassifier(base_classifier,
                                threshold=0.85,
                                criterion="threshold",
                                k_best=50,
                                max_iter=1000,
                                ).fit(x, y)

    y_true = data_true.iloc[:, -1]

    y_predict = st.transduction_
    y_index = data_true.iloc[:, 0]

    i = np.where(y_predict == -1)
    y_predict[i] = 1

    df = pd.DataFrame({'txId': y_index, 'predict': y_predict})
    df.to_csv(path_or_buf=f'.\semi_supervised_results\RandomForest\RandomForest{time}.csv', sep=',', index=False)


    print(f'RandomForest{time}.csv saved')

    return 0


def slefTrain_AdaBoost(data_df, time):
    data_df = data_df.reset_index(drop=True)

    data = data_df.copy()
    data_true = data.copy()
    data["class"] = data["class"] - 1

    data_0 = data[data["class"] == 0]
    data_1 = data[data["class"] == 1]
    data_0_mask = data_0.sample(n=int(len(data_0) * 0.7), random_state=0)
    data_1_mask = data_1.sample(n=int(len(data_1) * 0.7), random_state=0)
    data_mask = pd.concat((data_0_mask, data_1_mask), axis=0)

    for n in data_mask.index.tolist():
        data.loc[n, "class"] = -1

    x, y = data.iloc[:, 2:-1], data.iloc[:, -1]

    x = ClusterData_preprocessing(x)

    base_classifier = AdaBoostClassifier(n_estimators=50, random_state=42)

    st = SelfTrainingClassifier(base_classifier,
                                threshold=0.85,
                                criterion="threshold",
                                k_best=50,
                                max_iter=1000,
                                ).fit(x, y)

    y_predict = st.transduction_
    y_index = data_true.iloc[:, 0]

    i = np.where(y_predict == -1)
    y_predict[i] = 1

    df = pd.DataFrame({'txId': y_index, 'predict': y_predict})
    df.to_csv(path_or_buf=f'.\semi_supervised_results\AdaBoost\AdaBoost{time}.csv', sep=',', index=False)

    print(f'AdaBoost{time}.csv saved')

    return 0


def slefTrain_gbBoost(data_df, time):
    data_df = data_df.reset_index(drop=True)

    data = data_df.copy()
    data_true = data.copy()
    data["class"] = data["class"] - 1

    data_0 = data[data["class"] == 0]
    data_1 = data[data["class"] == 1]
    data_0_mask = data_0.sample(n=int(len(data_0) * 0.7), random_state=0)
    data_1_mask = data_1.sample(n=int(len(data_1) * 0.7), random_state=0)
    data_mask = pd.concat((data_0_mask, data_1_mask), axis=0)

    for n in data_mask.index.tolist():
        data.loc[n, "class"] = -1

    x, y = data.iloc[:, 2:-1], data.iloc[:, -1]

    x = ClusterData_preprocessing(x)

    base_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)

    st = SelfTrainingClassifier(base_classifier,
                                threshold=0.85,
                                criterion="threshold",
                                k_best=50,
                                max_iter=1000,
                                ).fit(x, y)

    y_true = data_true.iloc[:, -1]

    y_predict = st.transduction_
    y_index = data_true.iloc[:, 0]

    i = np.where(y_predict == -1)
    y_predict[i] = 1

    df = pd.DataFrame({'txId': y_index, 'predict': y_predict})
    df.to_csv(path_or_buf=f'.\semi_supervised_results\gbBoost\gbBoost{time}.csv', sep=',', index=False)

    print(f'gbBoost{time}.csv saved')

    return 0


def slefTrain_GaussianProcess(data_df, time):
    data_df = data_df.reset_index(drop=True)

    data = data_df.copy()
    data_true = data.copy()
    data["class"] = data["class"] - 1

    data_0 = data[data["class"] == 0]
    data_1 = data[data["class"] == 1]
    data_0_mask = data_0.sample(n=int(len(data_0) * 0.7), random_state=0)
    data_1_mask = data_1.sample(n=int(len(data_1) * 0.7), random_state=0)
    data_mask = pd.concat((data_0_mask, data_1_mask), axis=0)

    for n in data_mask.index.tolist():
        data.loc[n, "class"] = -1

    x, y = data.iloc[:, 2:-1], data.iloc[:, -1]

    x = ClusterData_preprocessing(x)

    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))

    base_classifier = GaussianProcessClassifier(kernel=kernel)

    st = SelfTrainingClassifier(base_classifier,
                                threshold=0.85,
                                criterion="threshold",
                                k_best=50,
                                max_iter=1000,
                                ).fit(x, y)

    y_true = data_true.iloc[:, -1]
    y_true = y_true - 1
    y_predict = st.transduction_
    y_index = data_true.iloc[:, 0]

    i = np.where(y_predict == -1)
    y_predict[i] = 1


    df = pd.DataFrame({'txId': y_index, 'predict': y_predict})
    df.to_csv(path_or_buf=f'slef_train\slef_train{time}.csv', sep=',', index=False)

    print(f'slef_train{time}.csv saved')

    return 0


def slefTrain_xgboost(data_df, time):
    data_df = data_df.reset_index(drop=True)
    data = data_df.copy()
    data_true = data.copy()
    data["class"] = data["class"] - 1

    data_0 = data[data["class"] == 0]
    data_1 = data[data["class"] == 1]
    data_0_mask = data_0.sample(n=int(len(data_0) * 0.7), random_state=0)
    data_1_mask = data_1.sample(n=int(len(data_1) * 0.7), random_state=0)
    data_mask = pd.concat((data_0_mask, data_1_mask), axis=0)

    data_1_know = data_1.drop(data_1_mask.index)
    data_0_know = data_0.drop(data_0_mask.index)
    data_know = pd.concat((data_0_know, data_1_know), axis=0)

    X_train, X_test, y_train, y_test = data_know.iloc[:, 2:-1], data_mask.iloc[:, 2:-1], data_know.iloc[:,
                                                                                         -1], data_mask.iloc[:, -1]
    X_total, y_total = data.iloc[:, 2:-1], data.iloc[:, -1]
    dtotal = xgb.DMatrix(X_total, label=y_total)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    params = {
        'max_depth': 8,
        'eta': 0.3,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss'
    }
    num_boost_round = 300
    bst = xgb.train(params, dtrain, num_boost_round=num_boost_round)


    y_pred = bst.predict(dtotal)

    y_pred = (y_pred > 0.5).astype(int)

    df = pd.DataFrame({'txId': data['txId'], 'predict': y_pred})
    df.to_csv(path_or_buf=f'.\semi_supervised_results\\xgboost\XGboost{time}.csv', sep=',', index=False)

    print(f'xgboost{time}.csv saved')

    return 0
