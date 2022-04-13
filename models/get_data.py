from sklearn.utils import shuffle
from robustness.tabular_robust import add_tabular_noise
from robustness.timeseries_robust import add_timeseries_noise
import numpy as np
import torch
from torch.utils.data import DataLoader
import scripts.const as const
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from scripts.darwin_datasets import darwin_multimodal_dataset
from torchvision import transforms
import torchxrayvision as xrv
#sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))



def get_dataloader(batch_size=40, num_workers=1, train_shuffle=True, imputed_path='im.pk', model = const.Models.static_timeseries,shuffle_split = False, kfold = 0,augment_images = True):
    '''
    * kfold : If >= 2 then k-fold cross validation is performed. This function will then instead return a list of tuples containing kfold splits of training and validation. 
    And a test dataloader as normal. (Technically a test dict containing a dataloader under the 'timeseries' key. Just for the Multibench test function compatibility)

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


    #some_image = datafile['cohort']['image'][0]

    #print("Before squeeze: ",some_image)
    #some_image = np.squeeze(some_image, axis=0)
    #print("After squeeze: ",some_image)
    #transformed_image = transform(some_image)
    #rint("After transform: ",some_image)
    #Re-adding the dimension
    #some_image = np.array(some_image)
    #print("After adding to list: ",some_image)



    #Converting labels from 1 - 2 to binary
    datafile['test']['labels']  = datafile['test']['labels'] - 1 
    datafile['train']['labels'] = datafile['train']['labels'] - 1 
    datafile['valid']['labels'] = datafile['valid']['labels'] - 1
    datafile['cohort']['labels'] = datafile['cohort']['labels'] - 1
    #le = len(y)
    #Make a tuple
    if (shuffle_split == False and kfold < 2):
        train_data,valids_data,test_data,imgidx = order_data_static()
    elif (shuffle_split == True) and (kfold < 2):

        datasets,imgidx = order_data_random(datafile,model)

        #Create the splits by doing a random shuffle. We can ignore the y values for the most part since the tuple has the labels inside it either way
        X_train, X_test_val, y_train, y_test_val = train_test_split(datasets, datafile['cohort']['labels'], test_size=0.40, random_state=None,stratify = datafile['cohort']['labels'])

        X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size=0.66, random_state=None,stratify = y_test_val)
        
        train_data = X_train
        valids_data = X_val
        test_data = X_test
    elif (kfold >=2):
        # You chose k-fold cross validation
        
        train_data,valids_data,X_test,imgidx = order_data_static(datafile,model)
        
        X_train_val = train_data + valids_data
        y_train_val = [tpl[-1] for tpl in X_train_val]
        #X_train_val, X_test, y_train_val, y_test = train_test_split(datasets, datafile['cohort']['labels'], test_size=0.40, random_state=None,stratify = datafile['cohort']['labels'])
        
        skf = StratifiedKFold(n_splits=kfold)
        kSplits = []
        
        #Just so we can use smart indexing
        X_train_val = np.asarray(X_train_val)
        for train_index, val_index in skf.split(X_train_val, y_train_val):

            X_train, X_val = X_train_val[train_index], X_train_val[val_index]
            kSplits.append((X_train, X_val))
            #y_train, y_test = y[train_index], y[test_index]

        #Transform to dataloaders
        #Test. Not that this will be used anyway on k-fold cross validation
        test_data = X_test
        tests = dict()
        tests['timeseries'] = []
        tests['timeseries'].append(DataLoader(test_data, shuffle=False,
                            num_workers=num_workers, batch_size=batch_size))
        #Train and val
        train_val_dataloaders = [(DataLoader(darwin_multimodal_dataset(split[0].tolist(),imgidx,transform,data_aug), shuffle=train_shuffle,num_workers=num_workers, batch_size=batch_size),
                                  DataLoader(split[1].tolist(), shuffle=False,num_workers=num_workers, batch_size=batch_size))
                                  for split in kSplits]


        return train_val_dataloaders,tests

    if (kfold < 2): #This if is just for show since if kfold>=2 the function will have returned by this point
        valids = DataLoader(valids_data, shuffle=False,
                            num_workers=num_workers, batch_size=batch_size)
        trains = DataLoader(darwin_multimodal_dataset(train_data,imgidx,transform,data_aug), shuffle=train_shuffle,
                            num_workers=num_workers, batch_size=batch_size)


        #testing code:
        for sample in trains:
            print(sample)                            
        tests = dict()
        tests['timeseries'] = []
        tests['timeseries'].append(DataLoader(test_data, shuffle=False,
                            num_workers=num_workers, batch_size=batch_size))
    
    '''
    for noise_level in tqdm(range(11)):
        dataset_robust = copy.deepcopy(datasets[le//10:le//5])
        if tabular_robust:
            X_s_robust = add_tabular_noise([dataset_robust[i][0] for i in range(
                len(dataset_robust))], noise_level=noise_level/10)
        else:
            X_s_robust = [dataset_robust[i][0]
                          for i in range(len(dataset_robust))]
        if timeseries_robust:
            X_t_robust = add_timeseries_noise([[dataset_robust[i][1] for i in range(
                len(dataset_robust))]], noise_level=noise_level/10)[0]
        else:
            X_t_robust = [dataset_robust[i][1]
                          for i in range(len(dataset_robust))]
        y_robust = [dataset_robust[i][2] for i in range(len(dataset_robust))]
        if flatten_time_series:
            tests['timeseries'].append(DataLoader([(X_s_robust[i], X_t_robust[i].reshape(timestep*series_dim), y_robust[i])
                                       for i in range(len(y_robust))], shuffle=False, num_workers=num_workers, batch_size=batch_size))
        else:
            tests['timeseries'].append(DataLoader([(X_s_robust[i], X_t_robust[i], y_robust[i]) for i in range(
                len(y_robust))], shuffle=False, num_workers=num_workers, batch_size=batch_size))
    '''

    #Note there's a different return statement in the case of kfold>2 (This is bad programming practice, as it makes the function harder to read. Maybe to refactor in the future?)
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