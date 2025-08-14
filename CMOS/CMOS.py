import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch_geometric.nn import GCNConv

from evaluation import eva
from util import myplot


class CMOS(nn.Module):
    def __init__(self, input_size, hidden_size0=128, hidden_size1=256, output_size=1, num_layers=1,bidirectional=False, dropout=0):
        super(CMOS, self).__init__()

        self.fc0 = nn.Linear(input_size, hidden_size0)
        self.GNN1 = GCNConv(input_size, hidden_size0)


        self.GNN2 = GCNConv(hidden_size0, int(hidden_size1))

        self.dropout = nn.Dropout(0.2)

        self.lstm = nn.LSTM(input_size=hidden_size1,
                            hidden_size=hidden_size1,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional,
                            dropout=dropout)

        self.h0 = torch.zeros(num_layers, 3000, hidden_size1).cuda()
        self.c0 = torch.zeros(num_layers, 3000, hidden_size1).cuda()


        self.classifier = nn.Linear(hidden_size1, output_size)

    def expanded(self, x):

        original_rows = x.size(0)

        num_new_rows = 3000 - original_rows

        new_rows = torch.zeros(num_new_rows, x.size(1)).cuda()

        expanded_tensor = torch.cat((x, new_rows), dim=0)
        return expanded_tensor
    def forward(self, x, adj, weight):

        x0 = F.leaky_relu(self.fc0(x))
        x1 = F.leaky_relu(self.GNN1(x, adj, edge_weight=weight))
        x1 = self.dropout(x1)
        x = torch.cat([x0, x1], dim=1)

        x_len = x.size(0)
        x = self.expanded(x)
        x = x.unsqueeze(1)
        x0 = x
        x, (self.h0, self.c0) = self.lstm(x, (self.h0, self.c0))
        self.h0 = self.h0.detach()
        self.c0 = self.h0.detach()
        x = x0 + x
        x = x.squeeze(1)[:x_len,:]


        out = self.classifier(x)
        out = out.squeeze()
        return out


def CreateCMOS(data=None, load_pretrain=True):

    featureList = data[0]
    adjList = data[1]
    lableLsit = data[2]
    ca_matrixList = data[3]
    ca_weightsList = data[4]
    semi_resultList = data[5]
    fakelabelList = data[6]
    ca_featureList = data[7]

    k = 35
    train_timeList = [i for i in range(1, k)]
    predict_timeList = [i for i in range(k, 50)]

    model = CMOS(input_size=ca_featureList[1].size(1)).cuda()
    print(model)

    if load_pretrain is True:
        model.load_state_dict(torch.load('./weight/CMOS_params.pth'))
        return model

    optimizer = optim.Adam(model.parameters(), lr=0.0001)  
    epochs = 200
    losses = []

    for epoch in range(epochs):
        loss_value = 0
        loss_tatol = torch.tensor([0], dtype=torch.float32, device='cuda', requires_grad=True)

        for time in train_timeList:  
            optimizer.zero_grad()  
            outputs = model(ca_featureList[time], adjList[time], ca_weightsList[time])  

            labels = lableLsit[time].float()

            pos_counts = labels.sum(dim=0)  
            neg_counts = (1 - labels).sum(dim=0)  
            epsilon = 1e-6
            pos_weight = neg_counts / (pos_counts + epsilon)
            pos_weight = pos_weight.cuda() * 10
            loss = F.binary_cross_entropy_with_logits(outputs, labels, pos_weight=pos_weight)
            loss.backward() 
            optimizer.step() 

            loss_tatol = loss_tatol + loss


        loss_value += loss_tatol.item()

        print(f'{time}Epoch [{epoch + 1}/{epochs}]Loss: {loss_tatol.item():.8f}')

        avg_loss = loss_value / len(train_timeList)
        losses.append(avg_loss)
    myplot(x=[i for i in range(len(losses))], y1=losses, title="teacher")


    predicted = (torch.sigmoid(outputs) > 0.5).long()

    acc, f1_macro_N, f1_macro_p, f1_macro, precision_N, precision_P, recall_N, recall_P = eva(
        lableLsit[time].cpu().numpy(), predicted.cpu().numpy(), str=f"time{time}")


    preditcList_acc = []
    preditcList_f1_N = []
    preditcList_f1_p = []
    preditcList_f1 = []
    preditcList_precision_N = []
    preditcList_precision_P = []
    preditcList_recall_N = []
    preditcList_recall_P = []

    print(model)
    for time in predict_timeList:

        with torch.no_grad():

            outputs = model(ca_featureList[time], adjList[time], ca_weightsList[time])

            predicted = (torch.sigmoid(outputs) > 0.5).long()


            acc, f1_macro_N, f1_macro_p, f1_macro, precision_N, precision_P, recall_N, recall_P = eva(
                lableLsit[time].cpu().numpy(), predicted.cpu().numpy(), str=f"time{time}")
            preditcList_acc.append(acc)
            preditcList_f1_N.append(f1_macro_N)
            preditcList_f1_p.append(f1_macro_p)
            preditcList_f1.append(f1_macro)
            preditcList_precision_N.append(precision_N)
            preditcList_precision_P.append(precision_P)
            preditcList_recall_N.append(recall_N)
            preditcList_recall_P.append(recall_P)
            print()

    print("Accuracy", preditcList_acc)
    print("F1 (Negative)", preditcList_f1_N)
    print("F1 (Positive)", preditcList_f1_p)
    print("F1 (Overall)", preditcList_f1)
    print("Precision (Negative)", preditcList_precision_N)
    print("Precision (Positive)", preditcList_precision_P)
    print("Recall (Negative)", preditcList_recall_N)
    print("Recall (Positive)", preditcList_recall_P)
    myplot(x=predict_timeList, y1=preditcList_f1_N, title="teacher")

    torch.save(model.state_dict(), './weight/CMOS_params.pth')

    return model