import argparse
import torch
import time as t
from CMOS.CMOS import CreateCMOS
from CMOS.semi.dataloader import dataload_preprocess
from evaluation import CoSemiGNNeval
from model.CoSemiGNN import CoSemiGNN
from train import CoSemiGNNtrain
from util import print_layer_parameters

def main(args):

    k = 35
    train_timeList = [i for i in range(1, k)]
    predict_timeList = [i for i in range(k, 50)]

    load_pretrain = args.load_pretrain

    training_params = {
        'learning_rate': 0.0001,
        'batch_size': 32,
        'epochs': 500,
        'optimizer': 'adam',
        'device': "cuda"
    }

    data = dataload_preprocess(tr_path=args.tr_feature,edge_path=args.tr_edge)


    start_time = t.time()

    print('loading CMOS............')
    CMOS = CreateCMOS(data=data, load_pretrain=load_pretrain)

    print('loading SDGM............')
    model = CoSemiGNN(feature_in=data[0][1].size(1)).cuda()


    if load_pretrain is True:
        model.load_state_dict(torch.load('CoSemiGNN_params.pth'))
    else:
        start_time = t.time()
        model = CoSemiGNNtrain(model=model,
                               CMOS=CMOS,
                               data=data,
                               timeList=train_timeList,
                               training_params=training_params)
        end_time = t.time()
        elapsed = end_time - start_time
        print(f"time: {elapsed:.4f} s")


    CoSemiGNNeval(model=model,
                          CMOS=CMOS,
                          data=data,
                          timeList=predict_timeList,
                          training_params=training_params)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="please type tr_path and edge_path")
    parser.add_argument('--tr_feature', required=False,default="data/txs_label.csv",help='feature')
    parser.add_argument('--tr_edge', required=False,default="data/txs_edgelist.csv", help='edge')
    parser.add_argument('--load_pretrain', action='store_true', help='pretrain')
    args = parser.parse_args()
    print(args.load_pretrain)
    main(args)