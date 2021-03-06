import os
import sys
sys.path.append(os.getcwd())
import shutil
from argparse import ArgumentParser
import darwin.models.mimic_image as mimic_image
import darwin.models.mimic_static as mimic_static
import darwin.models.mimic_image_static as mimic_image_static
import darwin.models.mimic_timeseries as mimic_timeseries
import torch
import darwin.config as config

# Script used to perform an experiment



#Expected format of the data output to the experiments directory:
#Root directory - > experiment directory - > ModelName -> 
#- A csv file for each run of the model containing its accuracy on validation data while training
#- A model_name-test.csv file containing accuracy on the test data for each run.

def t_or_f(arg):
    ua = str(arg).upper()
    if 'TRUE'.startswith(ua):
       return True
    elif 'FALSE'.startswith(ua):
       return False
    else:
       print("ERROR!!")

def main():
    parser = ArgumentParser(prog='experiment.py')
    parser.add_argument('-nr', '--numRuns', default='5',type=int)
    parser.add_argument('-ne', '--numEpochs', default='20',type=int)
    parser.add_argument('-bs', '--batchSize', default='3',type=int,help = "Batch size while training")
    parser.add_argument('-pat', '--patience', default='7',type=int,help = "Used with --earlyStop. Number of epochs to wait for an improvement for.")
    parser.add_argument('-estm', '--earlyStopMetric',default = 'acc', choices = ['valid', 'acc'],help = "What metric to use for early stopping.")
    parser.add_argument('-sf', '--shuffle',default = 'True', choices = [True, False],type=t_or_f,help = "Whether to use a pre-determined split or to randomly shuffle the split between each run.")
    parser.add_argument('-care', '--careful',default = 'True', choices = [True, False],type=t_or_f,help = "Whether to have the script ask you if you're sure if you want to \
overwrite the experiments folder. Turning this off helps if you want to run the tests via a bash or batch script")
    parser.add_argument('-m', '--model',required= True, choices = ['static', 'image', 'image_static', 'timeseries','static_timeseries', 'timeseries_image','static_timeseries_image'])
    parser.add_argument('-o', '--rootDir',required = True, help = "Directory where the experiments are saved to. Note that this path should be written with forward slashes.")
    parser.add_argument('-en', '--experimentName',required = True, help = "Name of the experiment. A folder with this name is created inside the root directory where all results will be output to.")
    parser.add_argument('-lr', '--learningRate',default = '0.001', help = "Learning rate of the model while training.",type=float)
    parser.add_argument('-dp', '--dropOutP',default = '0', help = "DOES NOT WORK. Weird bug, if anyone wants to help with that, that'd be great. Dropout percentage of the static model while training.",type=float)
    parser.add_argument('-kf', '--kFold',default = '0', help = "Whether to perform kfold cross validation. Number of folds to do. If < 2, then kfold cross validation won't be performed",type=int)
    parser.add_argument('-opt', '--optimizer',default = 'RMSprop', choices = ['adam', 'RMSprop'], help = "Optimizer to use while training")
    parser.add_argument('-estp', '--earlyStop',default = 'True', choices = [True, False],type=t_or_f,help = "Whether to stop after 7 epochs where validation accuracy did not improve.")
    parser.add_argument('-aug', '--augmentImage',default = 'False', choices = [True, False],type=t_or_f,help = "Whether to augment the images whenver they are retreived from the dataloader.\
This should increase the image model's ability to generalize")
    parser.add_argument('-save', '--saveModels',default = 'False', choices = [True, False],type=t_or_f,help = "Whether to save the model with best accuracy on validation for every run. This will also save the test dataloader used\
by the model. Note this will save as many models as runs.")

    args = parser.parse_args()
    
    rootExperimentDir = args.rootDir + "/{}".format(args.experimentName)
    modelResultsDir = rootExperimentDir + "/{}".format(args.model)

    

    if args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam

    print("\n\nPerforming experiment {} on the {} model for {} epochs and {} runs. \nThe results will be stored in the: '{}' directory \n\n".format(args.experimentName,args.model,str(args.numEpochs),str(args.numRuns),modelResultsDir))
    
    print("* Early stop = {}\n".format(str(args.earlyStop)))

    print("* Patience : {}\n".format(args.patience))

    print("* Early stop metric : {}\n".format(args.earlyStopMetric))

    if args.shuffle and args.kFold < 2:
        print("* The valiation/training/testing split will be randomly shuffled for each run. The shuffle will be stratified.\n")
    elif args.kFold < 2:
        print("* The valiation/training/testing split will be the same for every run.\n")
    elif args.kFold >=2:
        print("* Every run will be a {}-fold cross validation\n".format(args.kFold))


    print("* Learning rate: {}\n".format(str(args.learningRate)))

    if (args.model == 'static' or args.model == 'image_static'):
        print("* Dropout percentage of the static model is: {}\n".format(args.dropOutP))
    
    print("* Optimizer used: {}\n".format(args.optimizer))

    print("* Batch size: {}\n".format(args.batchSize))

    print("* Augment images: {}\n".format(args.augmentImage))

    print("* Save models: {}\n \n".format(args.saveModels))


    if not os.path.exists(args.rootDir):
        os.makedirs(args.rootDir)
    if not os.path.exists(rootExperimentDir):
        os.makedirs(rootExperimentDir)
    if not os.path.exists(modelResultsDir):
        os.makedirs(modelResultsDir)
    else:
        #If the directory containing results exists, delete it.
        if (args.careful == False):
            shutil.rmtree(modelResultsDir)
            os.makedirs(modelResultsDir)
        else:
            #A little safeguard in place just in case the user accidentally somehow inputted the wrong outputdir. And the results directory ends up being a valuable directory in their system. 
            answer = 'asdgsd'
            while (not (answer == 'y' or answer == 'n')):
                answer = input("The script is about to delete the {} folder and recursively delete everything under it in order to replace it with the experiment results.\n\
Are you sure you want to continue?: (y/n)".format(modelResultsDir))
            if (answer == 'y'):
                shutil.rmtree(modelResultsDir)
                os.makedirs(modelResultsDir)
            else:
                print("You've terminated the script")
                return -1

    
    # Print out the arguments used to the script in the experiment folder
    with open(modelResultsDir + '/arguments.txt', 'w') as f:
        for arg in sys.argv:
            f.write(arg + ' ')
        f.write("\n Impk path : {}\n".format(config.dataPath))


    # Run the model

    if (args.model == "static"):
        mimic_static.runModel(args.numRuns,modelResultsDir,args.numEpochs,args.shuffle,args.learningRate,args.dropOutP != 0.0,args.dropOutP,optimizer=optimizer,
        earlyStop=args.earlyStop,kfold=args.kFold,batch_size=args.batchSize,save_models=args.saveModels,patience=args.patience,early_stop_metric=args.earlyStopMetric)
    elif (args.model == "image"):
        mimic_image.runModel(args.numRuns,modelResultsDir,args.numEpochs,args.shuffle,args.learningRate,optimizer=optimizer,
        earlyStop=args.earlyStop,kfold=args.kFold,batch_size=args.batchSize,augmentImages=args.augmentImage,save_models=args.saveModels,patience=args.patience,early_stop_metric=args.earlyStopMetric)
    elif (args.model == "image_static"):
        mimic_image_static.runModel(args.numRuns,modelResultsDir,args.numEpochs,args.shuffle,args.learningRate,args.dropOutP != 0.0,args.dropOutP,optimizer=optimizer,
        earlyStop=args.earlyStop,kfold=args.kFold,batch_size=args.batchSize,augmentImages=args.augmentImage,save_models=args.saveModels,patience=args.patience,early_stop_metric=args.earlyStopMetric)
    elif (args.model == "timeseries"):
        mimic_timeseries.runModel(args.numRuns,modelResultsDir,args.numEpochs,args.shuffle,args.learningRate,optimizer=optimizer,
        earlyStop=args.earlyStop,kfold=args.kFold,batch_size=args.batchSize,save_models=args.saveModels,patience=args.patience,early_stop_metric=args.earlyStopMetric)
    elif (args.model == "static_timeseries"):
        mimic_static.runModel(args.numRuns,modelResultsDir,args.numEpochs,args.shuffle,args.learningRate,args.dropOutP != 0.0,args.dropOutP,optimizer=optimizer,
        earlyStop=args.earlyStop,kfold=args.kFold,batch_size=args.batchSize,save_models=args.saveModels,patience=args.patience,early_stop_metric=args.earlyStopMetric)
    elif (args.model == "timeseries_image"):
        mimic_image.runModel(args.numRuns,modelResultsDir,args.numEpochs,args.shuffle,args.learningRate,optimizer=optimizer,
        earlyStop=args.earlyStop,kfold=args.kFold,batch_size=args.batchSize,augmentImages=args.augmentImage,save_models=args.saveModels,patience=args.patience,early_stop_metric=args.earlyStopMetric)
    elif (args.model == "static_timeseries_image"):
        mimic_image_static.runModel(args.numRuns,modelResultsDir,args.numEpochs,args.shuffle,args.learningRate,args.dropOutP != 0.0,args.dropOutP,optimizer=optimizer,
        earlyStop=args.earlyStop,kfold=args.kFold,batch_size=args.batchSize,augmentImages=args.augmentImage,save_models=args.saveModels,patience=args.patience,early_stop_metric=args.earlyStopMetric)



if __name__ == '__main__':
    main()