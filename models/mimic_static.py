import sys
import os
import torch
from torch import nn
from multiprocessing import freeze_support

sys.path.append(os.getcwd())

from unimodals.common_models import MLP, GRU # noqa
from get_data import get_dataloader # noqa
from fusions.common_fusions import Concat # noqa
from training_structures.Supervised_Learning import train, test # noqa

import scripts.const as const
import pandas as pd
import scripts.config as config

# Point this to the resulting file of our preprocessing code (/output/im.pk)
PATH_TO_DATA = 'C:\dev\darwin\datasetExploration\data\ourim.pk'

def main():

    traindata, validdata, testdata = get_dataloader(
        7, imputed_path=PATH_TO_DATA, model = const.Models.static)
 

    encoders = [MLP(const.nr_static_features, 10, 10, dropout=False).cuda()]
    head = MLP(10, 40, 2, dropout=False).cuda()
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
    pd.DataFrame(stats['valid'],columns = ['epoch','acc','valloss']).to_csv(config.stats_root +'/static_val_perform_while_training.csv')


if __name__ == '__main__':
    freeze_support()
    main()
