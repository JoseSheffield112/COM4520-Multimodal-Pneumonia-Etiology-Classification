import sys
import os
import torch
from multiprocessing import freeze_support
import pandas as pd

sys.path.append(os.getcwd())

from unimodals.common_models import MLP, GRU # noqa
from models.get_data import get_dataloader # noqa
from fusions.common_fusions import Concat # noqa
from training_structures.Supervised_Learning import train, test # noqa

from mimic_cxr.models.xrv_model import DenseNetXRVFeature

import scripts.const as const
import scripts.config as config


#############

def get_encoders_head_fusion(static_output_size,dropout,dropoutP):
    image_model = DenseNetXRVFeature(pretrain_weights="densenet121-res224-all")
    #image_model.load_state_dict(torch.load(config.pretrained_root + '/densenet_P_etiology.pth'))
    encoders = [image_model.cuda()]
    head = MLP(const.image_encoder_output_size, 40, 2, dropout=False).cuda()
    fusion = Concat().cuda()
    return encoders,head,fusion


def runModel(nrRuns,outputRoot,nrEpochs,shuffle_split = True,lr =0.001,dropout=False,dropoutP=0.1,optimizer=torch.optim.RMSprop,earlyStop = False,kfold = 0,augmentImages = True,batch_size = 3):

    MODEL_NAME = "image"
    static_output_size = 100

    if (kfold < 2):
        test_accuracies = []
        for i in range(nrRuns):
            traindata, validdata, testdata = get_dataloader(
                batch_size, imputed_path=config.impkPath, model = const.Models.image,shuffle_split = shuffle_split,augment_images=augmentImages)


            encoders, head, fusion = get_encoders_head_fusion(static_output_size,dropout,dropoutP)

            # train
            stats = train(encoders, fusion, head, traindata, validdata, nrEpochs, auprc=True,lr = lr,early_stop=earlyStop,optimtype=optimizer)

            # test
            print("Testing: ")
            model = torch.load('best.pt').cuda()

            rob_curve = test(model, testdata, dataset='mimic 7', auprc=True)
            test_acc = rob_curve['Accuracy'][0] 
            test_accuracies.append(test_acc)

            outputStats(stats,outputRoot, "/run-{}-{}-validation.csv".format(str(i), MODEL_NAME))
        #Output test accuracy
        pd.DataFrame(test_accuracies,columns=['acc']).to_csv(outputRoot + "/{}-test.csv".format(MODEL_NAME))
    
    else:# Perform k cross validation
        avg_val_accuracies = []
        for i in range(nrRuns):
            train_val_splits, testdata = get_dataloader(
                batch_size, imputed_path=config.impkPath, model = const.Models.image,shuffle_split = shuffle_split,kfold=kfold,augment_images=augmentImages)
            bestModel = None
            bestbestAcc = 0
            sumAcc = 0
            allAcc = []
            for s_i,(traindata,validdata) in enumerate(train_val_splits):

                encoders, head, fusion = get_encoders_head_fusion(static_output_size,dropout,dropoutP)
                # train
                stats,model,bestacc = train(encoders, fusion, head, traindata, validdata, nrEpochs, auprc=True,lr = lr,early_stop=earlyStop,optimtype=optimizer)
                allAcc.append(bestacc)
                if (bestbestAcc < bestacc):
                    bestModel = model
                    bestbestAcc = bestacc
                sumAcc += bestacc
                print("Best accuracy on validation for this split: {}\n".format(bestacc))

                # Output results of this split
                if not os.path.exists(outputRoot + "/run-{}".format(i)):
                    os.makedirs(outputRoot + "/run-{}".format(i))
                outputStats(stats,outputRoot, "/run-{}/split-{}-{}-validation.csv".format(str(i),s_i, MODEL_NAME))
            
            avgAcc = sumAcc / len(train_val_splits)
            avg_val_accuracies.append(avgAcc)
            #Output all the best accuracies of each split to a csv file for inspection. I assume mostly to see variance 
            pd.DataFrame(allAcc,columns=['acc']).to_csv(outputRoot + "/run-{}/{}-bestacc-all-splits.csv".format(str(i), MODEL_NAME))
            #Don't test the model on the testing set anymore. Just print the average of the peak accuracies of each split:
            print("Average of best accuracy on validation of each split: {}\n".format(avgAcc))
        
        #Output average val accuracies
        pd.DataFrame(avg_val_accuracies,columns=['acc']).to_csv(outputRoot + "/{}-kfold-avgvalidacc.csv".format(MODEL_NAME))


    #Write the arhitecture of the model to a file
    with open(outputRoot + "/model_arhitecture.txt", 'w') as f:
        f.write('Static output layer: ' + str(static_output_size) + '\n')

def outputStats(stats,root,csvName):
    # Outputs statistics to csvPath
    pd.DataFrame(stats['valid'],columns = ['epoch','acc','valloss']).to_csv(root + csvName)

if __name__ == '__main__':
    freeze_support()
    runModel(nrRuns = 1,outputRoot = config.stats_root,nrEpochs = 20,kfold=2)