import numpy
import numpy as np
import torch
from sklearn import metrics
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score

from util import myplot


def eva(y_true: numpy, y_pred: numpy, str=''):
    ''''''
    if max(y_true) > 1:
        indices = np.where(y_true != 2)[0]
        y_true = y_true[indices]
        y_pred = y_pred[indices]

    # acc, f1 = cluster_acc(y_true, y_pred)
    acc = metrics.accuracy_score(y_true, y_pred)

    precision_macro = metrics.precision_score(y_true, y_pred, average=None)

    recall = metrics.recall_score(y_true, y_pred, average='micro')
    recall_micro_N = metrics.recall_score(y_true, y_pred, average='micro', labels=[0])
    recall_micro_p = metrics.recall_score(y_true, y_pred, average='micro', labels=[1])

    f1_macro_N = metrics.f1_score(y_true, y_pred, average='binary', pos_label=0)
    f1_macro_p = metrics.f1_score(y_true, y_pred, average='binary', pos_label=1)
    f1_macro = metrics.f1_score(y_true, y_pred, average='macro')

    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)

    print(str)

    print(f'Total transactions: {len(y_true)}')
    print(f'Total transactions: {len(y_pred)}')
    print(f'Anomalous transactions: {np.sum(y_true == 0)}')
    print(f'Accuracy: {acc:.4f}\n',
          'Precision: True negative/Predicted negative: {:.4f} True positive/Predicted positive: {:.4f}\n'.format(
              precision_macro[0], precision_macro[1]),
          f'Recall: {recall:.4f} Correct predictions in negative class: {recall_micro_N:.4f} Correct predictions in positive class: {recall_micro_p:.4f}\n'
          f'Overall F1 score: {f1_macro:.4f} Negative class F1: {f1_macro_N:.4f} Positive class F1: {f1_macro_p:.4f}')
    print('{:.4f}'.format(acc), '{:.4f}'.format(precision_macro[0]), '{:.4f}'.format(precision_macro[1]),
          '{:.4f}'.format(recall_micro_N), '{:.4f}'.format(recall_micro_p),
          f'Negative class F1: {f1_macro_N:.4f} Positive class F1: {f1_macro_p:.4f}')
    print('---------------------------------------------')
    return acc, f1_macro_N, f1_macro_p, f1_macro, precision_macro[0], precision_macro[1], recall_micro_N, recall_micro_p



def CoSemiGNNeval(model, CMOS, data, timeList, training_params):

    predict_timeList = timeList
    preditcList_acc = []
    preditcList_f1_N = []
    preditcList_f1_p = []
    preditcList_f1 = []
    preditcList_precision_N = []
    preditcList_precision_P = []
    preditcList_recall_N = []
    preditcList_recall_P = []

    feature_tensorlist = data[0]
    adj_matrix_tensorlist = data[1]
    labels_tensorlist = data[2]
    ca_matrixList = data[3]
    ca_weightsList = data[4]
    semi_resultList = data[5]
    fakelabelList = data[6]
    ca_featureList = data[7]

    preditcList_acc = []
    preditcList_f1_N = []
    preditcList_f1_p = []
    preditcList_f1 = []
    preditcList_precision_N = []
    preditcList_precision_P = []
    preditcList_recall_N = []
    preditcList_recall_P = []

    for time in predict_timeList:
        print(f"time: {time}:")

        model.eval()

        with torch.no_grad():
            out, h = model(feature_tensorlist[time], adj_matrix_tensorlist[time], ca_weightsList[time])

            predicted = (torch.sigmoid(out) > 0.5).long()

            acc, f1_macro_N, f1_macro_p, f1_macro, precision_N, precision_P, recall_N, recall_P = \
                eva(labels_tensorlist[time].cpu().numpy(), predicted.cpu().numpy())

        preditcList_acc.append(acc)
        preditcList_f1_N.append(f1_macro_N)
        preditcList_f1_p.append(f1_macro_p)
        preditcList_f1.append(f1_macro)
        preditcList_precision_N.append(precision_N)
        preditcList_precision_P.append(precision_P)
        preditcList_recall_N.append(recall_N)
        preditcList_recall_P.append(recall_P)

    print("Accuracy", preditcList_acc)
    print("F1 (Negative)", preditcList_f1_N)
    print("F1 (Positive)", preditcList_f1_p)
    print("F1 (Overall)", preditcList_f1)
    print("Precision (Negative)", preditcList_precision_N)
    print("Precision (Positive)", preditcList_precision_P)
    print("Recall (Negative)", preditcList_recall_N)
    print("Recall (Positive)", preditcList_recall_P)
    print(f'-----f1-avg{sum(preditcList_f1_N) / len(preditcList_f1_N)}')
    myplot(x=predict_timeList, y1=preditcList_f1_N, title="Illegal f1 score at each time")
