import sys
import os
import torch
from torch import nn
from multiprocessing import freeze_support
import pickle
import pandas as pd

sys.path.append(os.getcwd())

from unimodals.common_models import MLP, GRU # noqa
from models.get_data import get_dataloader # noqa
from fusions.common_fusions import Concat # noqa
from training_structures.Supervised_Learning import train, test # noqa

from mimic_cxr.models.xrv_model import DenseNetXRVFeature

import scripts.const as const
import scripts.config as config

def runModel(nrRuns,outputRoot,nrEpochs,shuffle_split = True, lr = 0.001):
    # Point this to the resulting file of our preprocessing code (/output/im.pk)
    MODEL_NAME = "image_static"

    static_output_size = 100
    test_accuracies = []
    for i in range(nrRuns):
        #TOFIX: Currently this model only works with batchsize = 1. This is a temporary fix to a bug.
        traindata, validdata, testdata = get_dataloader(
            1, imputed_path=config.impkPath, model = const.Models.static_image,shuffle_split = shuffle_split)

        image_model = DenseNetXRVFeature(pretrain_weights="densenet121-res224-all")
        encoders = [MLP(const.nr_static_features, 50, static_output_size, dropout=False).cuda(),image_model.cuda()]
        head = MLP(const.image_encoder_output_size + static_output_size, 40, 2, dropout=False).cuda()
        fusion = Concat().cuda()

        # train
        stats = train(encoders, fusion, head, traindata, validdata, nrEpochs, auprc=True,lr = lr)

        # test
        print("Testing: ")
        model = torch.load('best.pt').cuda()

        rob_curve = test(model, testdata, dataset='mimic 7', auprc=True)
        test_acc = rob_curve['Accuracy'][0] 
        test_accuracies.append(test_acc)

        outputStats(stats,outputRoot, "/run-{}-{}-validation.csv".format(str(i), MODEL_NAME))
    
    pd.DataFrame(test_accuracies,columns=['acc']).to_csv(outputRoot + "/{}-test.csv".format(MODEL_NAME))


    #Write the arhitecture of the model to a file
    #TODO: Be more specific about the arhitecture. Write down the size of every layer, not just the sizes of the output layers of the encoders
    with open(outputRoot + "/model_arhitecture.txt", 'w') as f:
        f.write('Static output layer: ' + str(static_output_size) + '\n')
        f.write('Image output layer: ' + str(const.image_encoder_output_size) + '\n')

def outputStats(stats,root,csvName):
    # Outputs statistics to csvPath
    pd.DataFrame(stats['valid'],columns = ['epoch','acc','valloss']).to_csv(root + csvName)

if __name__ == '__main__':
    freeze_support()
    runModel(nrRuns = 1,outputRoot = config.stats_root)
