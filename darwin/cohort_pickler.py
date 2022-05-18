# Import libraries
from datetime import timedelta
import os
import sys
import numpy as np
import pandas as pd
from darwin.config import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torchxrayvision as xrv
import torch
import torch.nn as nn
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, utils
from models.xrv_model import DenseNetXRVFeature
import pneumonia_dataset

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import pickle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                xrv.datasets.XRayResizer(224),
                                ])

train_dataset = pneumonia_dataset.MIMIC_Pneumonia_Dataset(imgpath=cxr_jpg_root,
                                     csvpath=cxr_negbio_csv,
                                     metacsvpath=cxr_metadata_csv,
                                     option='train',
                                     pre_processed=False,
                                     transform=transform,
                                     remove_duplicate_hadm=True,
                                     )
test_dataset = pneumonia_dataset.MIMIC_Pneumonia_Dataset(imgpath=cxr_jpg_root,
                                     csvpath=cxr_negbio_csv,
                                     metacsvpath=cxr_metadata_csv,
                                     option='test',
                                     pre_processed=False,
                                     transform=transform,
                                     remove_duplicate_hadm=True,
                                     )
valid_dataset = pneumonia_dataset.MIMIC_Pneumonia_Dataset(imgpath=cxr_jpg_root,
                                     csvpath=cxr_negbio_csv,
                                     metacsvpath=cxr_metadata_csv,
                                     option='valid',
                                     pre_processed=False,
                                     transform=transform,
                                     remove_duplicate_hadm=True,
                                     )

batch_size = 1

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

train_samples = []
for i, data in enumerate(train_loader):
    train_samples.append(data)
    print(data['img'].squeeze(1).shape)
print(len(train_samples))
print(train_samples[0]['img'])
pickle.dump( train_samples, open( image_data_pickled_root + "/train.pk", "wb" ) )

test_samples = []
for i, data in enumerate(test_loader):
    test_samples.append(data)
print(len(test_samples))
pickle.dump( test_samples, open( image_data_pickled_root + "/test.pk", "wb" ) )

valid_samples = []
for i, data in enumerate(valid_loader):
    valid_samples.append(data)
print(len(valid_samples))
pickle.dump( valid_samples, open( image_data_pickled_root + "/valid.pk", "wb" ) )