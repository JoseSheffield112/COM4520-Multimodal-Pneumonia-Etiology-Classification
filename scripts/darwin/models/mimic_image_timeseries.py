import sys
import os
import torch
from torch import nn
from multiprocessing import freeze_support
import pickle
import pandas as pd

sys.path.append(os.getcwd())

from MultiBench.unimodals.common_models import MLP, GRU 
from get_data import get_dataloader 
from MultiBench.fusions.common_fusions import Concat 
from training_structures.Supervised_Learning import train, test 

from scripts.darwin.models.xrv_model import DenseNetXRVFeature

import scripts.darwin.const as const
import config.darwin.config as config



def main():

    image_model = DenseNetXRVFeature(pretrain_weights="densenet121-res224-all")
    image_model.load_state_dict(torch.load(config.pretrained_root + '/densenet_P_etiology.pth'))

    traindata, validdata, testdata = get_dataloader(
        1, imputed_path=config.dataPath, model = const.Models.timeseries_image)


    encoders = [GRU(const.nr_timeseries_features, 30, dropout=False, batch_first=True).cuda(), image_model.cuda()]
    head = MLP(const.image_encoder_output_size + 720, 40, 2, dropout=False).cuda()
    fusion = Concat().cuda()

    # train
    stats = train(encoders, fusion, head, traindata, validdata, 20, auprc=True)

    # test
    print("Testing: ")
    model = torch.load('best.pt').cuda()

    # dataset = 'mimic mortality', 'mimic 1', 'mimic 7'
    test(model, testdata, dataset='mimic 7', auprc=True)

    outputStats(stats)

def outputStats(stats):
    pd.DataFrame(stats['valid'],columns = ['epoch','acc','valloss']).to_csv(config.stats_root +'/image_timeseries_val_perform_while_training.csv')


if __name__ == '__main__':
    freeze_support()
    main()
