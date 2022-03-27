import sys
import os
import torch
from torch import nn
from multiprocessing import freeze_support
import pickle
import pandas as pd

sys.path.append(os.getcwd())

from unimodals.common_models import MLP, GRU # noqa
from get_data import get_dataloader # noqa
from fusions.common_fusions import Concat # noqa
from training_structures.Supervised_Learning import train, test # noqa

from mimic_cxr.models.xrv_model import DenseNetXRVFeature

import scripts.const as const
import scripts.config as config

# Point this to the resulting file of our preprocessing code (/output/im.pk)
PATH_TO_DATA = 'C:\dev\darwin\datasetExploration\data\ourim.pk'

def main():

    image_model = DenseNetXRVFeature(pretrain_weights="densenet121-res224-all")
    image_model.load_state_dict(torch.load(config.pretrained_root + '/densenet_P_etiology.pth'))

    traindata, validdata, testdata = get_dataloader(
        1, imputed_path=PATH_TO_DATA, model = const.Models.static_timeseries_image)
 
    encoders = [MLP(const.nr_static_features, 10, 10, dropout=False).cuda(),
              GRU(const.nr_timeseries_features, 30, dropout=False, batch_first=True).cuda(),
              image_model.cuda()]
    head = MLP(const.image_encoder_output_size + 10 + 720, 40, 2, dropout=False).cuda()
    fusion = Concat().cuda()

    # I modified the train function from multibench to return some statistics (epoch,accuracy,validloss) to make statistics later on. We can either make a fork of multibench with this change and
    # have our repo depend on the fork. Or we could steal the code.
    stats = train(encoders, fusion, head, traindata, validdata, 20, auprc=True)

    # test
    print("Testing: ")
    model = torch.load('best.pt').cuda()

    # dataset = 'mimic mortality', 'mimic 1', 'mimic 7'
    test(model, testdata, dataset='mimic 7', auprc=True)

    outputStats(stats)

def outputStats(stats):
    pd.DataFrame(stats['valid'],columns = ['epoch','acc','valloss']).to_csv(config.stats_root +'/image_static_timeseries_val_perform_while_training.csv')

if __name__ == '__main__':
    freeze_support()
    main()
