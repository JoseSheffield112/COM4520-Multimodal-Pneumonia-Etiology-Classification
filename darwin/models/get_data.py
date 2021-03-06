from sklearn.utils import shuffle
from MultiBench.robustness.tabular_robust import add_tabular_noise
from MultiBench.robustness.timeseries_robust import add_timeseries_noise
import numpy as np
import torch
from torch.utils.data import DataLoader
import darwin.const as const
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from darwin.darwin_datasets import darwin_multimodal_dataset
from torchvision import transforms
import torchxrayvision as xrv
#sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))



def get_dataloader(batch_size=40, num_workers=1, train_shuffle=True, imputed_path='im.pk', model = const.Models.static_timeseries,shuffle_split = False, kfold = 0,augment_images = True):
    '''
    * kfold : If >= 2 then k-fold cross validation is performed. This function will then instead return a list of tuples containing kfold splits of training and validation. 
    And a test dataloader as normal. (Technically a test dict containing a dataloader under the 'timeseries' key. Just for the Multibench test function compatibility)
    * augment_images: If true, then every time a sample is pulled from the train dataloader, the images will be augmented (rotated, scaled) randomly. This is supposed to help with the model's ability to generalize.

    Gets the training,validation and testing dataloaders when pointed to our processed data.
    '''
    # Set the transform/data augmentation to apply to images
    transform = None 
    data_aug = None
    if (augment_images == True):
        transform = transforms.Compose([xrv.datasets.XRayCenterCrop(),
                    xrv.datasets.XRayResizer(224),])
        data_aug = transforms.Compose([xrv.datasets.ToPILImage(),
                               transforms.RandomAffine(15, translate=(0.1, 0.1), scale=(0.5, 1.5)),
                               #transforms.GaussianBlur(kernel_size=55,sigma=(0.01,2)),
                               transforms.ToTensor()
                               ])

    f = open(imputed_path, 'rb')
    datafile = pickle.load(f)
    f.close()

    #Converting labels from 1 - 2 to binary
    datafile['test']['labels']  = datafile['test']['labels'] - 1 
    datafile['train']['labels'] = datafile['train']['labels'] - 1 
    datafile['valid']['labels'] = datafile['valid']['labels'] - 1
    datafile['cohort']['labels'] = datafile['cohort']['labels'] - 1

    if (shuffle_split == False and kfold < 2):
        train_data,valids_data,test_data,imgidx = order_data_static()
    elif (shuffle_split == True) and (kfold < 2):

        datasets,imgidx = order_data_random(datafile,model)

        #Create the splits by doing a random shuffle. We can ignore the y values for the most part since the tuple has the labels inside it either way
        X_train, X_test_val, y_train, y_test_val = train_test_split(datasets, datafile['cohort']['labels'], test_size=0.40, random_state=None,stratify = datafile['cohort']['labels'])

        X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=None,stratify = y_test_val)
        
        train_data = X_train
        valids_data = X_val
        test_data = X_test
    elif (kfold >=2):
        # You chose k-fold cross validation
        
        train_data,valids_data,X_test,imgidx = order_data_static(datafile,model)
        
        X_all = train_data + valids_data + X_test
        y_all = [tpl[-1] for tpl in X_all]
        #X_train_val, X_test, y_train_val, y_test = train_test_split(datasets, datafile['cohort']['labels'], test_size=0.40, random_state=None,stratify = datafile['cohort']['labels'])
        
        skf = StratifiedKFold(n_splits=kfold)
        kSplits = []
        
        #Just so we can use smart indexing
        X_all = np.asarray(X_all)
        for train_index, val_index in skf.split(X_all, y_all):

            X_train, X_val = X_all[train_index], X_all[val_index]
            kSplits.append((X_train, X_val))

        #Transform to dataloaders
        #Train and val
        train_val_dataloaders = [(DataLoader(darwin_multimodal_dataset(split[0].tolist(),imgidx,transform,data_aug), shuffle=train_shuffle,num_workers=num_workers, batch_size=batch_size),
                                  DataLoader(split[1].tolist(), shuffle=False,num_workers=num_workers, batch_size=batch_size))
                                  for split in kSplits]

        #Note there's a different return statement in the case of kfold<=2 (This is bad programming practice, as it makes the function much harder to read. Maybe to refactor in the future?)
        return train_val_dataloaders

    if (kfold < 2): #This if is just for code clarity since if kfold>=2 the function will have returned by this point
        valids = DataLoader(valids_data, shuffle=False,
                            num_workers=num_workers, batch_size=batch_size)
        trains = DataLoader(darwin_multimodal_dataset(train_data,imgidx,transform,data_aug), shuffle=train_shuffle,
                            num_workers=num_workers, batch_size=batch_size)

                           
        tests = dict()
        tests['timeseries'] = []
        tests['timeseries'].append(DataLoader(test_data, shuffle=False,
                            num_workers=num_workers, batch_size=batch_size))

    #Note that there's a different return statement in the case of kfold>2 (This is bad programming practice, as it makes the function much harder to read. Maybe refactor in the future)
    return trains, valids, tests

def order_data_static(datafile,model):
    if (model == const.Models.static_timeseries_image):
        le = len(datafile['valid']['labels'])
        valids_data = [(datafile['valid']['static'][i],datafile['valid']['timeseries'][i],datafile['valid']['image'][i],datafile['valid']['labels'][i]) for i in range(le)]

        le = len(datafile['train']['labels'])
        train_data = [(datafile['train']['static'][i],datafile['train']['timeseries'][i],datafile['train']['image'][i],datafile['train']['labels'][i]) for i in range(le)]

        le = len(datafile['test']['labels'])
        test_data = [(datafile['test']['static'][i],datafile['test']['timeseries'][i],datafile['test']['image'][i],datafile['test']['labels'][i]) for i in range(le)]
        imgidx = 2
    elif (model == const.Models.static_timeseries):
        le = len(datafile['valid']['labels'])
        valids_data = [(datafile['valid']['static'][i],datafile['valid']['timeseries'][i],datafile['valid']['labels'][i]) for i in range(le)]

        le = len(datafile['train']['labels'])
        train_data = [(datafile['train']['static'][i],datafile['train']['timeseries'][i],datafile['train']['labels'][i]) for i in range(le)]

        le = len(datafile['test']['labels'])
        test_data = [(datafile['test']['static'][i],datafile['test']['timeseries'][i],datafile['test']['labels'][i]) for i in range(le)]
        imgidx = -1
    elif (model == const.Models.static_image):
        le = len(datafile['valid']['labels'])
        valids_data = [(datafile['valid']['static'][i],datafile['valid']['image'][i],datafile['valid']['labels'][i]) for i in range(le)]

        le = len(datafile['train']['labels'])
        train_data = [(datafile['train']['static'][i],datafile['train']['image'][i],datafile['train']['labels'][i]) for i in range(le)]

        le = len(datafile['test']['labels'])
        test_data = [(datafile['test']['static'][i],datafile['test']['image'][i],datafile['test']['labels'][i]) for i in range(le)]
        imgidx = 1
    elif (model == const.Models.timeseries_image):
        le = len(datafile['valid']['labels'])
        valids_data = [(datafile['valid']['timeseries'][i],datafile['valid']['image'][i],datafile['valid']['labels'][i]) for i in range(le)]

        le = len(datafile['train']['labels'])
        train_data = [(datafile['train']['timeseries'][i],datafile['train']['image'][i],datafile['train']['labels'][i]) for i in range(le)]

        le = len(datafile['test']['labels'])
        test_data = [(datafile['test']['timeseries'][i],datafile['test']['image'][i],datafile['test']['labels'][i]) for i in range(le)]
        imgidx = 1
    elif (model == const.Models.static):
        le = len(datafile['valid']['labels'])
        valids_data = [(datafile['valid']['static'][i],datafile['valid']['labels'][i]) for i in range(le)]

        le = len(datafile['train']['labels'])
        train_data = [(datafile['train']['static'][i],datafile['train']['labels'][i]) for i in range(le)]

        le = len(datafile['test']['labels'])
        test_data = [(datafile['test']['static'][i],datafile['test']['labels'][i]) for i in range(le)]
        imgidx = -1
    elif (model == const.Models.timeseries):
        le = len(datafile['valid']['labels'])
        valids_data = [(datafile['valid']['timeseries'][i],datafile['valid']['labels'][i]) for i in range(le)]

        le = len(datafile['train']['labels'])
        train_data = [(datafile['train']['timeseries'][i],datafile['train']['labels'][i]) for i in range(le)]

        le = len(datafile['test']['labels'])
        test_data = [(datafile['test']['timeseries'][i],datafile['test']['labels'][i]) for i in range(le)]
        imgidx = -1
    elif (model == const.Models.image):
        le = len(datafile['valid']['labels'])
        valids_data = [(datafile['valid']['image'][i],datafile['valid']['labels'][i]) for i in range(le)]

        le = len(datafile['train']['labels'])
        train_data = [(datafile['train']['image'][i],datafile['train']['labels'][i]) for i in range(le)]

        le = len(datafile['test']['labels'])
        test_data = [(datafile['test']['image'][i],datafile['test']['labels'][i]) for i in range(le)]
        imgidx = 0

    return train_data,valids_data,test_data,imgidx


def order_data_random(datafile,model):
    '''
    Returns a list of tuples ordered appropriately based on the model you want to run.
    '''
    le = len(datafile['cohort']['labels'])
    #Order the tuple appropriately for the model you're running. The data in the tuple needs to be in the same order as the models are in the encoders list of the MMDL model.
    if (model == const.Models.static_timeseries_image):
        datasets = [(datafile['cohort']['static'][i], datafile['cohort']['timeseries'][i], datafile['cohort']['image'][i],datafile['cohort']['labels'][i]) for i in range(le)]
        imgidx = 2
    elif(model == const.Models.static_timeseries):
        datasets = [(datafile['cohort']['static'][i], datafile['cohort']['timeseries'][i],datafile['cohort']['labels'][i]) for i in range(le)]
        imgidx = -1
    elif(model == const.Models.static_image):
        datasets = [(datafile['cohort']['static'][i], datafile['cohort']['image'][i],datafile['cohort']['labels'][i]) for i in range(le)]
        imgidx = 1
    elif(model == const.Models.timeseries_image):
        datasets = [(datafile['cohort']['timeseries'][i], datafile['cohort']['image'][i],datafile['cohort']['labels'][i]) for i in range(le)]
        imgidx = 1
    elif(model == const.Models.static):
        datasets = [(datafile['cohort']['static'][i],datafile['cohort']['labels'][i]) for i in range(le)]
        imgidx = -1
    elif(model == const.Models.timeseries):
        datasets = [(datafile['cohort']['timeseries'][i],datafile['cohort']['labels'][i]) for i in range(le)]
        imgidx = -1
    elif(model == const.Models.image):
        datasets = [(datafile['cohort']['image'][i],datafile['cohort']['labels'][i]) for i in range(le)]
        imgidx = 0

    return datasets, imgidx