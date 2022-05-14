import sys
import os
import torch
from torch import nn
from multiprocessing import freeze_support
import pickle

sys.path.append(os.getcwd())

from MultiBench.unimodals.common_models import MLP, GRU # noqa
from get_data import get_dataloader # noqa
from MultiBench.fusions.common_fusions import Concat # noqa
from MultiBench.training_structures.Supervised_Learning import train, test # noqa

from darwin.models.xrv_model import DenseNetXRVFeature

import darwin.const as const
import darwin.config as config

# Point this to the resulting file of our preprocessing code (/output/im.pk)
PATH_TO_DATA = 'C:\dev\darwin\datasetExploration\data\ourim.pk'

def main():

    image_model = DenseNetXRVFeature(pretrain_weights="densenet121-res224-all")
    image_model.load_state_dict(torch.load(config.pretrained_root + '/densenet_P_etiology.pth'))
    
    f = open(config.image_data_pickled_root + '/test.pk', 'rb')
    test_images = pickle.load(f)
    f.close()

    #print("Input from the original dict:",test_images[0]['img'])
    #print("Shape of original dict:",test_images[0]['img'])
    inp = test_images[0]['img']
    output = image_model(inp)

    inpArray = inp.detach().cpu().numpy()
    print(inpArray)
    print(inpArray.shape)

    traindata, validdata, testdata = get_dataloader(
        7, imputed_path=PATH_TO_DATA, model = const.Models.image)

    traindata_static, validdata_static, testdata_static = get_dataloader(
        7, imputed_path=PATH_TO_DATA, model = const.Models.static)

    #for element in traindata_static:
    #    print(element)
    #for element in traindata:
    #    print(element)
 

    # build encoders, head and fusion layer. Only changed the first argument of MLP and GRU (input dimensions) to make them match the shape of our data
    encoders = [image_model.cuda()]
    head = MLP(1024, 40, 2, dropout=False).cuda()
    fusion = Concat().cuda()

    # train
    train(encoders, fusion, head, traindata, validdata, 20, auprc=True)

    # test
    print("Testing: ")
    model = torch.load('best.pt').cuda()

    # dataset = 'mimic mortality', 'mimic 1', 'mimic 7'
    test(model, testdata, dataset='mimic 7', auprc=True)


if __name__ == '__main__':
    freeze_support()
    main()
