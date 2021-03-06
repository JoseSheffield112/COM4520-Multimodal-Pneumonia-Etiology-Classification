import sys
import os
import torch
from torch import nn
from multiprocessing import freeze_support
import pandas as pd

sys.path.append(os.getcwd())

from MultiBench.unimodals.common_models import MLP, GRU 
from models.get_data import get_dataloader 
from MultiBench.fusions.common_fusions import Concat 
from MultiBench.training_structures.Supervised_Learning import train, test 

from darwin.models.xrv_model import DenseNetXRVFeature

import darwin.const as const
import darwin.config as config
import pickle

def get_encoders_head_fusion(static_output_size,dropout,dropoutP):
    image_model = DenseNetXRVFeature(pretrain_weights="densenet121-res224-all")
    encoders = [MLP(const.nr_static_features, 50, static_output_size, dropout=dropout,dropoutp=dropoutP).cuda(),image_model.cuda()]
    head = MLP(const.image_encoder_output_size + static_output_size, 40, 2, dropout=False).cuda()
    fusion = Concat().cuda()
    return encoders,head,fusion

def save_model_and_test_data(model,testDataLoader,outputRoot,run,MODEL_NAME):
    # Create the models folder and the test data folder
    if not os.path.exists(outputRoot + "/models"):
        os.makedirs(outputRoot + "/models")
    if not os.path.exists(outputRoot + "/data"):
        os.makedirs(outputRoot + "/data")
    #Save model
    torch.save(model,outputRoot + "/models/run-{}-{}.pt".format(str(run),MODEL_NAME))
    #Save the test DataLoader (serialize it)
    f = open(outputRoot + "/data/run-{}-{}.pk".format(str(run),MODEL_NAME), mode='wb')
    pickle.dump(testDataLoader, file=f)
    f.close()

def runModel(nrRuns,outputRoot,nrEpochs,shuffle_split = True,lr =0.001,dropout=False,dropoutP=0.1,
            optimizer=torch.optim.RMSprop,earlyStop = True,kfold = 0,augmentImages = True,batch_size = 3,save_models=False,patience=7,
            early_stop_metric = 'acc'):

    MODEL_NAME = "image_static"
    static_output_size = 100

    if (kfold < 2):
        test_statistics = []
        for i in range(nrRuns):
            traindata, validdata, testdata = get_dataloader(
                batch_size, imputed_path=config.dataPath, model = const.Models.static_image,shuffle_split = shuffle_split,augment_images=augmentImages)
        
            encoders, head, fusion = get_encoders_head_fusion(static_output_size,dropout,dropoutP)

            # train
            stats,model,bestacc = train(encoders, fusion, head, traindata, validdata, nrEpochs, auprc=True,lr = lr,early_stop=earlyStop,
            optimtype=optimizer,max_patience=patience,early_stop_metric=early_stop_metric)

            if save_models:
                save_model_and_test_data(model,testdata,outputRoot,i,MODEL_NAME)
            # test
            print("Testing: ")
            model = torch.load('best.pt').cuda()

            rob_curve = test(model, testdata, dataset='mimic 7', auprc=True)
            test_stats = (rob_curve['Accuracy'][0],rob_curve['f1_score_1'][0],rob_curve['f1_score_2'][0],rob_curve['precision_1'][0],
                          rob_curve['precision_2'][0],rob_curve['recall_1'][0],rob_curve['recall_2'][0],rob_curve['true'][0],rob_curve['predicted'][0])

            test_statistics.append(test_stats)

            outputStats(stats,outputRoot, "/run-{}-{}-validation.csv".format(str(i), MODEL_NAME))
        #Output test statistics to csv file
        pd.DataFrame(test_statistics,columns=['acc','f1_score_1','f1_score_2','precision_1','precision_2','recall_1','recall_2','true','predicted']).to_csv(outputRoot + "/{}-test-stats.csv".format(MODEL_NAME))    
    else:# K-fold cross validation
        avg_val_accuracies = []
        for i in range(nrRuns):
            train_val_splits = get_dataloader(
                batch_size, imputed_path=config.dataPath, model = const.Models.static_image,shuffle_split = shuffle_split,kfold=kfold,augment_images=augmentImages)
            allAcc = []
            for s_i,(traindata,testdata) in enumerate(train_val_splits):

                encoders, head, fusion = get_encoders_head_fusion(static_output_size,dropout,dropoutP)
                # train
                stats,model,bestacc = train(encoders, fusion, head, traindata, testdata, nrEpochs,
                 auprc=True,lr = lr,early_stop=earlyStop,optimtype=optimizer,max_patience=patience,early_stop_metric=early_stop_metric)
                allAcc.append(bestacc)

                print("Best accuracy on validation for this split: {}\n".format(bestacc))

                # Output results of this split
                if not os.path.exists(outputRoot + "/run-{}".format(i)):
                    os.makedirs(outputRoot + "/run-{}".format(i))
                outputStats(stats,outputRoot, "/run-{}/split-{}-{}-validation.csv".format(str(i),s_i, MODEL_NAME))
                #Output all the best accuracies of each split to a csv file 
                pd.DataFrame(allAcc,columns=['acc']).to_csv(outputRoot + "/run-{}/{}-bestacc-all-splits.csv".format(str(i), MODEL_NAME))
            
            avgAcc = sum(allAcc) / len(train_val_splits)
            avg_val_accuracies.append(avgAcc)
            print("Average of best accuracies of all splits: {}\n".format(avgAcc))
        
        #Output average val accuracies
        pd.DataFrame(avg_val_accuracies,columns=['acc']).to_csv(outputRoot + "/{}-kfold-avgvalidacc.csv".format(MODEL_NAME))


    #Write the arhitecture of the model to a file
    with open(outputRoot + "/model_arhitecture.txt", 'w') as f:
        f.write('Static output layer: ' + str(static_output_size) + '\n')
        f.write('Image output layer: ' + str(const.image_encoder_output_size) + '\n')

def outputStats(stats,root,csvName):
    # Outputs statistics to csvPath
    pd.DataFrame(stats['valid'],columns = ['epoch','acc','valloss']).to_csv(root + csvName)

if __name__ == '__main__':
    freeze_support()
    runModel(nrRuns = 1,outputRoot = config.stats_root,nrEpochs = 20,kfold=2)
