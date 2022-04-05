import os
import sys
sys.path.append(os.getcwd())
import shutil
from argparse import ArgumentParser
import models.mimic_image
import models.mimic_static
import models.mimic_image_static

# Script used to perform an experiment


'''
Expected format of the data output to the experiments directory:
Root directory - > experiment directory - > ModelName -> 
- A csv file for each run of the model containing its accuracy on validation data while training
- A model_name-test.csv file containing accuracy on the test 
data for each run.
'''


def main():
    parser = ArgumentParser(prog='experiment.py')
    parser.add_argument('-nr', '--numRuns', default='5',type=int)
    parser.add_argument('-m', '--model',required= True, choices = ['static', 'image', 'image_static'])
    parser.add_argument('-o', '--rootDir',required = True, help = "Directory where the experiments are saved to.")
    parser.add_argument('-en', '--experimentName',required = True, help = "Name of the experiment. A folder with this name is created inside the root directory where all results will be output to.")
    
    args = parser.parse_args()
    
    print("Performing experiment {} on the {} model for {} runs. \nThe results will be stored in the {} directory".format(args.experimentName,args.model,str(args.numRuns),args.rootDir))


    rootExperimentDir = args.rootDir + "/{}".format(args.experimentName)
    modelResultsDir = rootExperimentDir + "/{}".format(args.model)
    if not os.path.exists(args.rootDir):
        os.makedirs(args.rootDir)
    if not os.path.exists(rootExperimentDir):
        os.makedirs(rootExperimentDir)
    if not os.path.exists(modelResultsDir):
        os.makedirs(modelResultsDir)
    else:
        #If the directory containing results exists, delete it.
        #A little safeguard in place just in case the user accidentally somehow inputted the wrong outputdir. And the results directory ends up being a valuable directory in their system. 
        answer = 'asdgsd'
        while (not (answer == 'y' or answer == 'n')):
            answer = input("The script is about to delete the {} folder and everything inside it in order to replace it with the experiment results.\
                 Are you sure you want to continue?: (y/n)".format(modelResultsDir))
        if (answer == 'y'):
            shutil.rmtree(modelResultsDir)
            os.makedirs(modelResultsDir)
        else:
            print("You've terminated the script")
            return -1
    
    if (args.model == "static"):
        models.mimic_static.runModel(args.numRuns,modelResultsDir)
    elif (args.model == "image"):
        models.mimic_image.runModel(args.numRuns,modelResultsDir)
    elif (args.model == "image_static"):
        models.mimic_image_static.runModel(args.numRuns,modelResultsDir)
    
    #Then I could make the code that calculates maximum, average and does the plots based on what model folder we have.
    #I would first go for each individual model folder, then for that



if __name__ == '__main__':
    main()