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

def calculateAverageData(experimentDir,modelName,args):
    '''
    PARAMETERS:
    * modelDir - Path to directory holding results of an experiment relating to a specific model

    Returns the data of the average performance of a model over all of its runs in the experiment. 
    The first element of the tuple is a tuple of lists containing the x and y points for the model's accuracy on validation data while training.     
    The second element of the tuple contains the test accuracy.
    '''
    resultsDir = experimentDir + "/{}".format(modelName)
    avgpoints = None
    if (args.avgEpochPlots):
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
    
    testcsvPath = [resultsDir + '/' + x for x in os.listdir(resultsDir) if x.endswith("test-stats.csv")]
    acc_avg = pd.read_csv(testcsvPath[0])['acc'].mean()
    f1_1_avg = pd.read_csv(testcsvPath[0])['f1_score_1'].mean()
    f1_2_avg = pd.read_csv(testcsvPath[0])['f1_score_2'].mean()

    precision_1_avg = pd.read_csv(testcsvPath[0])['precision_1'].mean()
    precision_2_avg = pd.read_csv(testcsvPath[0])['precision_2'].mean()

    recall_1_avg = pd.read_csv(testcsvPath[0])['recall_1'].mean()
    recall_2_avg = pd.read_csv(testcsvPath[0])['recall_2'].mean()
    return (avgpoints,acc_avg,f1_1_avg,f1_2_avg,precision_1_avg,precision_2_avg,recall_1_avg,recall_2_avg) 
    

def calculateMaxData(experimentDir,modelName):
    '''
    PARAMETERS:
    * modelDir - Path to directory holding results of an experiment relating to a specific model

    Returns the data of the model that had the best test accuracy as a tuple. 
    The first element of the tuple is a tuple of lists containing the x and y points for the model's accuracy on validation data while training.     
    The second element of the tuple contains the test accuracy.
    '''
    resultsDir = experimentDir + "/{}".format(modelName)

    testcsvPath = [resultsDir + '/' + x for x in os.listdir(resultsDir) if x.endswith("test-stats.csv")]
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
    plt.savefig(outputPath, format='png')
    plt.clf()

def barPlotModelTestAccuracies(experimentDir,modelName,outputDir):
    '''
    Creates and saves a bar plot with all of the model's testing accuracies
    '''
    resultsDir = experimentDir + "/{}".format(modelName)

    testcsvPath = [resultsDir + '/' + x for x in os.listdir(resultsDir) if x.endswith("test-stats.csv")]
    testAccuracies = pd.read_csv(testcsvPath[0])['acc']
    testAccuracies = testAccuracies.tolist()

    barplot(list(range(len(testAccuracies))), testAccuracies,"All accuracies on testing data for all runs on the {} model".format(modelName),outputDir)

def t_or_f(arg):
    ua = str(arg).upper()
    if 'TRUE'.startswith(ua):
       return True
    elif 'FALSE'.startswith(ua):
       return False
    else:
       print("This part of the code should never run, idk how you got here.")
def main():
    parser = ArgumentParser(prog='plot_results.py')
    parser.add_argument('-i', '--experimentsRoot',required = True, help = "Root directory of an experiment.")
    parser.add_argument('-n', '--experimentName',required = True, help = "Experiment name")
    parser.add_argument('-o','--outputDir',required = True, help = "Root output directory.")
    parser.add_argument('-avgepc', '--avgEpochPlots',default = 'True', choices = [True, False],type=t_or_f,help = "Whether to make the plots for average accuracy on validation per epoch over time.")
    args = parser.parse_args()

    
    experimentDir = args.experimentsRoot + '/' + args.experimentName
    outputDirRoot = args.outputDir

    #Only static model
    staticDir = experimentDir + "/{}".format('static')
    if(exists(staticDir)):
        (static_avg_points,static_average_testacc,static_f1_1_avg,static_f1_2_avg,static_precision_1_avg,
        static_precision_2_avg,static_recall_1_avg,static_recall_2_avg) = calculateAverageData(experimentDir,'static',args)
        (static_max_points,static_max_testacc) = calculateMaxData(experimentDir,'static')
        
        #plotXYPoints([avg_points_static],[],"Static model: Average accuracy on validation while training",outputDirRoot + '/static-avg-validation.png')
        #plotXYPoints([max_points_static],[],"Static model: Best run accuracy on validation while training",outputDirRoot + '/static-max-validation.png')
        
        barPlotModelTestAccuracies(experimentDir,'static',outputDirRoot + '/all-static-test_acc.png')    
    
    #Only the image model
    imageDir = experimentDir + "/{}".format('image')
    if(exists(imageDir)):
        (image_avg_points,image_average_testacc,image_f1_1_avg,image_f1_2_avg,image_precision_1_avg,
        image_precision_2_avg,image_recall_1_avg,image_recall_2_avg) = calculateAverageData(experimentDir,'image',args)
        (image_max_points,image_max_testacc) = calculateMaxData(experimentDir,'image')
        #plotXYPoints([avg_points_image],[],"Image model: Average accuracy on validation while training",outputDirRoot + '/image-avg-validation.png')
        #plotXYPoints([max_points_image],[],"Image model: Best run accuracy on validation while training",outputDirRoot + '/image-max-validation.png')

        barplot(['image'],[image_average_testacc],'Average test accuracy of the image model',outputDirRoot + '/image_avg_test_accuracy.png')
        barplot(['image'],[image_max_testacc],'Test accuracy of the best run of the image model',outputDirRoot + '/image_max_test_accuracy.png')        

    #Image and static model
    image_staticDir = experimentDir + "/{}".format('image_static')
    if(exists(image_staticDir)):
        (imagestatic_avg_points,imagestatic_average_testacc,imagestatic_f1_1_avg,imagestatic_f1_2_avg,imagestatic_precision_1_avg,
        imagestatic_precision_2_avg,imagestatic_recall_1_avg,imagestatic_recall_2_avg) = calculateAverageData(experimentDir,'image_static',args)
        (imagestatic_max_points,imagestatic_max_testacc) = calculateMaxData(experimentDir,'image_static')

        #plotXYPoints([avg_points_imagestatic],[],"Image_Static model: Average accuracy on validation while training",outputDirRoot + '/image_static-avg-validation.png')
        #plotXYPoints([max_points_imagestatic],[],"Image_Static model: Best run accuracy on validation while training",outputDirRoot + '/image_static-max-validation.png')

        barplot(['image_static'],[imagestatic_average_testacc],'Average test accuracy of: image_static',outputDirRoot + '/average-testacc-image_static.png')
        
        barPlotModelTestAccuracies(experimentDir,'image_static',outputDirRoot + '/all-image_static-test_acc.png')
    

    #Comparison of static and imageStatic: averages / max performance
    if(exists(staticDir) and exists(image_staticDir)):
        barplot(['static','image_static'],[static_average_testacc,imagestatic_average_testacc],'Average test accuracies of: image_static and static',outputDirRoot + '/compare-average-testacc-static-vs-image_static.png')
        barplot(['static','image_static'],[static_max_testacc,imagestatic_max_testacc],'Test accuracies of the best runs of: image_static and static',outputDirRoot + '/compare-max-testacc-static-vs-image_static.png')
        
        #plotXYPoints([avg_points_static,avg_points_imagestatic],['static','image_static'],"Comparison of average accuracy while training: static vs static_image",outputDirRoot + '/comparison-avg-validation-staticAndImageStatic.png')
        #plotXYPoints([max_points_static,max_points_imagestatic],['static','image_static'],"Comparison of best models accuracy while training: static vs static_image",outputDirRoot + '/comparison-max-validation-staticAndImageStatic.png')

    #Comparison of static and imageStatic and image:
    if(exists(staticDir) and exists(image_staticDir) and exists(imageDir)):
        #Average stats:
        barplot(['image','static','image_static'],[image_average_testacc,static_average_testacc,imagestatic_average_testacc],'Average test accuracies of: image_static, static and image',outputDirRoot + '/compare-average-testacc-all.png')
        
        barplot(['image','static','image_static'],[image_f1_1_avg,static_f1_1_avg,imagestatic_f1_1_avg],'Average f1 score on the viral class of: image_static, static and image',outputDirRoot + '/compare-average-f1_1-all.png')
        barplot(['image','static','image_static'],[image_f1_2_avg,static_f1_2_avg,imagestatic_f1_2_avg],'Average f1 score on the bacterial class of: image_static, static and image',outputDirRoot + '/compare-average-f1_2-all.png')

        barplot(['image','static','image_static'],[image_precision_1_avg,static_precision_1_avg,imagestatic_precision_1_avg],'Average precision on the viral class of: image_static, static and image',outputDirRoot + '/compare-average-precision_1-all.png')
        barplot(['image','static','image_static'],[image_precision_2_avg,static_precision_2_avg,imagestatic_precision_2_avg],'Average precision on the bacterial class of: image_static, static and image',outputDirRoot + '/compare-average-precision_2-all.png')

        barplot(['image','static','image_static'],[image_recall_1_avg,static_recall_1_avg,imagestatic_recall_1_avg],'Average recall on the viral class of: image_static, static and image',outputDirRoot + '/compare-average-recall_1-all.png')
        barplot(['image','static','image_static'],[image_recall_2_avg,static_recall_2_avg,imagestatic_recall_2_avg],'Average recall on the bacterial class of: image_static, static and image',outputDirRoot + '/compare-average-recall_2-all.png')
        
        barplot(['image','static','image_static'],[image_max_testacc,static_max_testacc,imagestatic_max_testacc],'Test accuracies of the best runs of: image_static, static and image',outputDirRoot + '/compare-max-testacc-all.png')
        #Upload these to a csv so that they can be put into a table:
        [(image_average_testacc,image_f1_1_avg,image_f1_2_avg,image_precision_1_avg,image_precision_2_avg,image_recall_1_avg,image_recall_2_avg)]

        #plotXYPoints([avg_points_image,avg_points_static,avg_points_imagestatic],['image','static','image_static'],"Comparison of average accuracy while training: all",outputDirRoot + '/comparison-avg-validation-all.png')
        #plotXYPoints([max_points_image,max_points_static,max_points_imagestatic],['image','static','image_static'],"Comparison of best models accuracy while training: all",outputDirRoot + '/comparison-max-validation-all.png')

    #Comparison of static and image: averages / max performance
    if(exists(staticDir) and exists(imageDir)):
        barplot(['static','image'],[static_average_testacc,image_average_testacc],'Average test accuracies of: static and image',outputDirRoot + '/compare-average-testacc-static-vs-image.png')
        barplot(['static','image'],[static_max_testacc,image_max_testacc],'Test accuracies of the best runs of: static and image',outputDirRoot + '/compare-max-testacc-static-vs-image.png')
        
        #plotXYPoints([avg_points_static,avg_points_image],['static','image'],"Comparison of average accuracy while training: static vs image",outputDirRoot + '/comparison-avg-validation-staticAndImage.png')
        #plotXYPoints([max_points_static,max_points_image],['static','image'],"Comparison of best models accuracy while training: static vs image",outputDirRoot + '/comparison-max-validation-staticAndImage.png')

    

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