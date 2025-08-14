import torch
import torch.nn.functional as F

from evaluation import eva
from util import myplot


def CoSemiGNNtrain(model, CMOS, data, timeList, training_params):
    feature_tensorlist = data[0]
    adj_matrix_tensorlist = data[1]
    labels_tensorlist = data[2]
    ca_matrixList = data[3]
    ca_weightsList = data[4]
    semi_resultList = data[5]
    fakelabelListn = data[6]
    ca_featureList = data[7]

    model = model.to(training_params['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=training_params['learning_rate'])
    model.train()
    CMOS.train()
    total_lossList = []
    for epoch in range(training_params["epochs"]):
        loss = 0
        for time in timeList:
            out, h = model(feature_tensorlist[time], adj_matrix_tensorlist[time], ca_weightsList[time])
            CMOSout = CMOS(ca_featureList[time], adj_matrix_tensorlist[time], ca_weightsList[time])

            input = torch.sigmoid(out)
            target = torch.sigmoid(CMOSout)

            input_probs = torch.stack([torch.log(1 - input), torch.log(input)], dim=1)
            target_probs = torch.stack([1 - target, target], dim=1)  # [q, 1-q]

            kl_loss = F.kl_div(input=input_probs, target=target_probs, reduction='batchmean')

            labels = labels_tensorlist[time].float()

            pos_counts = labels.sum(dim=0)
            neg_counts = (1 - labels).sum(dim=0)
            epsilon = 1e-6
            pos_weight = neg_counts / (pos_counts + epsilon)
            pos_weight = pos_weight.cuda() / 2

            #  logits + pos_weight
            ce_loss = F.binary_cross_entropy_with_logits(out, labels, pos_weight=pos_weight)
            alpha = 0.6


            total_loss = ce_loss + (1 - alpha) * kl_loss
            loss += total_loss.item()
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        total_lossList.append(loss)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')


    predicted = (torch.sigmoid(out) > 0.5).long()
    eva(labels_tensorlist[timeList[-1]].cpu().numpy(), predicted.cpu().numpy())
    myplot(x=[i for i in range(len(total_lossList))], y1=total_lossList, title="teacher")

    torch.save(model.state_dict(), 'CoSemiGNN_params.pth')
    return model
