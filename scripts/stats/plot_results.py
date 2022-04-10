from cgi import test
from pyexpat import model
import sys
import os
from os.path import exists
sys.path.append(os.getcwd()) # To make this script easy to run from the terminal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scripts.config as config
import re
from argparse import ArgumentParser


def plotXYPoints(points,labels,plotTitle,outputPath):
    '''
    PARAMETERS:
    * points - A list of tuples. Each tuple contains 2 lists. One contains values on the x axis and one contains valeus on the y axis
    * labels - A list of equal size to points that contains the labels for each line to be plot
    * plotTitle - A string to name the plot
    * outputPath - An output path to save the figure to
    '''
    # Assume that for each csv path a label is provided
    lineColours = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white']
    x = None # Declare variable in this scope for use later to set xticks

    for i in range(len(points)):
        x = points[i][0]
        y = points[i][1]
        if len(labels) > 0:
            plt.plot(x, y, color=lineColours[i % len(lineColours)], linestyle='dashed', linewidth = 3, label = labels[i],
            marker='o', markerfacecolor='blue', markersize=6)
        else:
            plt.plot(x, y, color=lineColours[i % len(lineColours)], linestyle='dashed', linewidth = 3,
            marker='o', markerfacecolor='blue', markersize=6)

    # naming the x axis
    plt.xlabel('epochs')
    # naming the y axis
    plt.ylabel('accuracy')
    # giving a title to my graph
    plt.title(plotTitle)
    
    # show a legend on the plot
    if len(labels) > 0:
        plt.legend()
    # Limit the axis range
    plt.ylim(0,1)
    plt.xticks(x)
    # I would've used np.round_(np.linspace(0,1,20), decimals = 2), but for some reason it doesn't output multiples of 0.05
    plt.yticks([0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1])
    plt.savefig(outputPath)
    plt.clf()
    


def valcsvListToTupleList(csvPaths):
    '''
    Turns a list of paths to csv files containing results of accuracy on validation data while training into a list of tuples.
    '''
    tupleList = []
    for csv in csvPaths:
        df = pd.read_csv(csv)
        x = df['epoch'].values
        y = df['acc'].values
        tupleList.append((x,y))
    return tupleList

def plotValidationAccWhileTraining(csvPaths, labels, plotTitle, outputPath):
    """
    Generalised function to plot validation accuracy whilst training

    PARAMETERS:
    * csvPaths - A list of input paths to csv files to plot
    * labels - A list of equal size to csvPaths that contains the labels each line to be plot
    * plotTitle - A string to name the plot
    * outputPath - An output path to save the figure to
    """
    #Convert the csvPaths into tuples
    tupleList = valcsvListToTupleList(csvPaths)
    plotXYPoints(tupleList,labels,plotTitle,outputPath)



#Expected format of the data output to the experiments directory:
#Root directory - > experiment directory - > ModelName -> 
#- A csv file for each run of the model containing its accuracy on validation data while training
#- A model_name-test.csv file containing accuracy on the test 
#data for each run.

def calculateAverageData(experimentDir,modelName):
    '''
    PARAMETERS:
    * modelDir - Path to directory holding results of an experiment relating to a specific model

    Returns the data of the average performance of a model over all of its runs in the experiment. 
    The first element of the tuple is a tuple of lists containing the x and y points for the model's accuracy on validation data while training.     
    The second element of the tuple contains the test accuracy.
    '''
    resultsDir = experimentDir + "/{}".format(modelName)
    validationCsvFiles = [resultsDir + '/' + x for x in os.listdir(resultsDir) if x.startswith("run")]
    results_all_runs = valcsvListToTupleList(validationCsvFiles)
    
    avgpoints = [[],[]]
    nrEpochs = len(results_all_runs[0][0])
    nrRuns = len(results_all_runs)
    for epoch in range(nrEpochs):
        sum = 0
        for run in range(nrRuns):
            sum += results_all_runs[run][1][epoch]
        avgpoints[0].append(epoch)
        avgpoints[1].append(sum/nrRuns)
        
    #convert to numpy array 
    avgpoints[0] = np.array(avgpoints[0])
    avgpoints[1] = np.array(avgpoints[1])
    
    testcsvPath = [resultsDir + '/' + x for x in os.listdir(resultsDir) if x.endswith("test.csv")]
    testavg = pd.read_csv(testcsvPath[0])['acc'].mean()
    
    return (avgpoints,testavg) 
    

def calculateMaxData(experimentDir,modelName):
    '''
    PARAMETERS:
    * modelDir - Path to directory holding results of an experiment relating to a specific model

    Returns the data of the model that had the best test accuracy as a tuple. 
    The first element of the tuple is a tuple of lists containing the x and y points for the model's accuracy on validation data while training.     
    The second element of the tuple contains the test accuracy.
    '''
    resultsDir = experimentDir + "/{}".format(modelName)

    testcsvPath = [resultsDir + '/' + x for x in os.listdir(resultsDir) if x.endswith("test.csv")]
    testDataFrame = pd.read_csv(testcsvPath[0])['acc']
    testMaxIdx = testDataFrame.idxmax()
    csvMax = [resultsDir + '/' + x for x in os.listdir(resultsDir) if x.startswith("run-{}".format(str(testMaxIdx)))][0]
    
    maxPoints = valcsvListToTupleList([csvMax])[0]
    return (maxPoints,testDataFrame.iloc[testMaxIdx])

def barplot(bottomNames, values,title,outputPath):
    def add_value_label(x_list,y_list):
        for i in range(0, len(x_list)):
            plt.text(i,y_list[i],y_list[i], ha="center")

    plt.yticks([0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1])
    #plt.ylim(0,1)
    plt.bar(bottomNames, values)
    plt.title(title)
    plt.ylabel("Accuracy")
    add_value_label(list(range(0,len(bottomNames))),values)
    plt.savefig(outputPath)
    plt.clf()

def barPlotModelTestAccuracies(experimentDir,modelName,outputDir):
    '''
    Creates and saves a bar plot with all of the model's testing accuracies
    '''
    resultsDir = experimentDir + "/{}".format(modelName)

    testcsvPath = [resultsDir + '/' + x for x in os.listdir(resultsDir) if x.endswith("test.csv")]
    testAccuracies = pd.read_csv(testcsvPath[0])['acc']
    testAccuracies = testAccuracies.tolist()

    barplot(list(range(len(testAccuracies))), testAccuracies,"All accuracies on testing data for all runs on the {} model".format(modelName),outputDir)

def main():
    parser = ArgumentParser(prog='plot_results.py')
    parser.add_argument('-i', '--experimentsRoot',required = True, help = "Root directory of an experiment.")
    parser.add_argument('-n', '--experimentName',required = True, help = "Experiment name")
    parser.add_argument('-o','--outputDir',required = True, help = "Root output directory.")
    args = parser.parse_args()

    experimentDir = args.experimentsRoot + '/' + args.experimentName
    outputDirRoot = args.outputDir

    #Only static model
    (avg_points_static,average_testacc_static) = calculateAverageData(experimentDir,'static')
    (max_points_static,max_testacc_static) = calculateMaxData(experimentDir,'static')
    
    plotXYPoints([avg_points_static],[],"Static model: Average accuracy on validation while training",outputDirRoot + '/static-avg-validation.png')
    plotXYPoints([max_points_static],[],"Static model: Best run accuracy on validation while training",outputDirRoot + '/static-max-validation.png')
    
    #Image and static model
    (avg_points_imagestatic,average_testacc_imagestatic) = calculateAverageData(experimentDir,'image_static')
    (max_points_imagestatic,max_testacc_imagestatic) = calculateMaxData(experimentDir,'image_static')

    plotXYPoints([avg_points_imagestatic],[],"Image_Static model: Average accuracy on validation while training",outputDirRoot + '/image_static-avg-validation.png')
    plotXYPoints([max_points_imagestatic],[],"Image_Static model: Best run accuracy on validation while training",outputDirRoot + '/image_static-max-validation.png')

    barplot(['static','image_static'],[average_testacc_static,average_testacc_imagestatic],'Average test accuracies of: image_static and static',outputDirRoot + '/compare-average-testacc-static-vs-image_static.png')
    barplot(['static','image_static'],[max_testacc_static,max_testacc_imagestatic],'Test accuracies of the best runs of: image_static and static',outputDirRoot + '/compare-max-testacc-static-vs-image_static.png')

    #Comparison of averages / max performance
    plotXYPoints([avg_points_static,avg_points_imagestatic],['static','image_static'],"Comparison of average accuracy while training: static vs static_image",outputDirRoot + '/comparison-avg-validation-staticAndImageStatic.png')
    plotXYPoints([max_points_static,max_points_imagestatic],['static','image_static'],"Comparison of best models accuracy while training: static vs static_image",outputDirRoot + '/comparison-max-validation-staticAndImageStatic.png')

    barPlotModelTestAccuracies(experimentDir,'static',outputDirRoot + '/all-static-test_acc.png')
    barPlotModelTestAccuracies(experimentDir,'image_static',outputDirRoot + '/all-image_static-test_acc.png')

    # Images model
    #TODO: Finish the only image modality


def legacyGraphs():
    #TODELTE. Keeping it for now for inspiration
    #Accuracy on validation data while training for each individual model
    inputCsvPath = config.stats_root +'/image_static_timeseries_val_perform_while_training.csv'
    if(exists(inputCsvPath)):
        plotValidationAccWhileTraining([inputCsvPath],
                                       ['3-modality validation accuracy while training'],
                                       '3-modality validation accuracy while training',
                                       config.graphs_root +'/image_static_timeseries_validationacc_while_training.png')

    inputCsvPath = config.stats_root +'/image_static_val_perform_while_training.csv'
    if(exists(inputCsvPath)):
        plotValidationAccWhileTraining([inputCsvPath],
                                       ['image and static validation accuracy while training'],
                                       'image and static validation accuracy while training',
                                       config.graphs_root +'/image_static_validationacc_while_training.png')

    inputCsvPath = config.stats_root +'/image_timeseries_val_perform_while_training.csv'
    if(exists(inputCsvPath)):
        plotValidationAccWhileTraining([inputCsvPath],
                                       ['image and timeseries validation accuracy while training'],
                                       'image and timeseries validation accuracy while training',
                                        config.graphs_root +'/image_timeseries_validationacc_while_training.png')

    inputCsvPath = config.stats_root +'/static_timeseries_val_perform_while_training.csv'
    if(exists(inputCsvPath)):
        plotValidationAccWhileTraining([inputCsvPath],
                                       ['static and timeseries model validation accuracy while training'],
                                       'static and timeseries model validation accuracy while training',
                                        config.graphs_root +'/static_timeseries_validationacc_while_training.png')

    inputCsvPath = config.stats_root +'/image_val_perform_while_training.csv'
    if(exists(inputCsvPath)):
        plotValidationAccWhileTraining([inputCsvPath],
                                       ['image model validation accuracy while training'],
                                       'image model validation accuracy while training',
                                       config.graphs_root +'/image_validationacc_while_training.png')

    inputCsvPath = config.stats_root +'/static_val_perform_while_training.csv'
    if(exists(inputCsvPath)):
        plotValidationAccWhileTraining([inputCsvPath],
                                       ['static model validation accuracy while training'],
                                       'static model validation accuracy while training',
                                       config.graphs_root +'/static_validationacc_while_training.png')


    inputCsvPath = config.stats_root +'/timeseries_val_perform_while_training.csv'
    if(exists(inputCsvPath)):
        plotValidationAccWhileTraining([inputCsvPath],
                                       ['timeseries model validation accuracy while training'],
                                       'timeseries model validation accuracy while training',
                                       config.graphs_root +'/timeseries_validationacc_while_training.png')

    #Accuracy on validation data while training for multiple models in the same plot

    # Static by itself + image by itself + image and static together

    inputCsvPath1 = config.stats_root +'/image_static_val_perform_while_training.csv'
    inputCsvPath2 = config.stats_root +'/image_val_perform_while_training.csv'
    inputCsvPath3 = config.stats_root +'/static_val_perform_while_training.csv'
    if(exists(inputCsvPath1) and exists(inputCsvPath2) and exists(inputCsvPath3)):
        plotValidationAccWhileTraining([inputCsvPath1,inputCsvPath2,inputCsvPath3],
                                       ['image + static','image','static'],
                                       'Image, static and image + static',
                                       config.graphs_root +'/static_and_image_alongside.png')

if __name__ == '__main__':
    main()