from unimodals.common_models import LeNet, MLP, Constant
from private_test_scripts.all_in_one import all_in_one_train, all_in_one_test
import torch
from torch import nn
from datasets.avmnist.get_data import get_dataloader
from fusions.common_fusions import Concat, MultiplicativeInteractions2Modal
from training_structures.Simple_Late_Fusion import train, test
import sys
import os
sys.path.append(os.getcwd())


filename = 'bestmi.pt'
traindata, validdata, testdata = get_dataloader(
    '/data/yiwei/avmnist/_MFAS/avmnist')
channels = 6
encoders = [LeNet(1, channels, 3).cuda(), LeNet(1, channels, 5).cuda()]
head = MLP(channels*40, 100, 10).cuda()

# fusion=Concat().cuda()
fusion = MultiplicativeInteractions2Modal(
    [channels*8, channels*32], channels*40, 'matrix')


def trpr():
    train(encoders, fusion, head, traindata, validdata, 25,
          optimtype=torch.optim.SGD, lr=0.05, weight_decay=0.0001, save=filename)


all_in_one_train(trpr, [encoders[0], encoders[1], fusion, head])
print("Testing:")
model = torch.load(filename).cuda()


def tepr():
    test(model, testdata)


all_in_one_test(tepr, [model])
