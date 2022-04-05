import sys
import os
from os.path import exists
sys.path.append(os.getcwd()) # To make this script easy to run from the terminal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scripts.config as config

# assume that we get the csv

def plotValidationAccWhileTraining(csvPaths, labels, plotTitle, outputPath):
    """
    Generalised function to plot validation accuracy whilst training

    PARAMETERS:
    * csvPaths - A list of input paths to csv files to plot
    * labels - A list of equal size to csvPaths that contains the labels each line to be plot
    * plotTitle - A string to name the plot
    * outputPath - An output path to save the figure to

    """
    # Assume that for each csv path a label is provided
    lineColours = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white']
    x = None # Declare variable in this scope for use later to set xticks

    for i in range(len(csvPaths)):
        df = pd.read_csv(csvPaths[i])
        x = df['epoch'].values
        y = df['acc'].values

        plt.plot(x, y, color=lineColours[i % len(lineColours)], linestyle='dashed', linewidth = 3, label = labels[i],
        marker='o', markerfacecolor='blue', markersize=6)

    # naming the x axis
    plt.xlabel('epochs')
    # naming the y axis
    plt.ylabel('accuracy')
    # giving a title to my graph
    plt.title(plotTitle)
    
    # show a legend on the plot
    plt.legend()
    # Limit the axis range
    plt.ylim(0,1)
    plt.xticks(x)
    # I would've used np.round_(np.linspace(0,1,20), decimals = 2), but for some reason it's not straight on 0.05
    plt.yticks([0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1])
    plt.savefig(outputPath)
    plt.clf()

def main():

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