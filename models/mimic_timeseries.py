import sys
import os
sys.path.append(os.getcwd())


import torch
from torch import nn
from multiprocessing import freeze_support
import pandas as pd
import scripts.config as config


from unimodals.common_models import MLP, GRU # noqa
from get_data import get_dataloader # noqa
from fusions.common_fusions import Concat # noqa
from training_structures.Supervised_Learning import train, test # noqa

import scripts.const as const


def main():

    traindata, validdata, testdata = get_dataloader(
        7, imputed_path=config.impkPath, model = const.Models.timeseries)


    encoders = [GRU(const.nr_timeseries_features, 30, dropout=False, batch_first=True).cuda()]
    head = MLP(720, 40, 2, dropout=False).cuda()
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
    # Outputs statistics to stats folder
    pd.DataFrame(stats['valid'],columns = ['epoch','acc','valloss']).to_csv(config.stats_root +'/timeseries_val_perform_while_training.csv')

if __name__ == '__main__':
    freeze_support()
    main()
