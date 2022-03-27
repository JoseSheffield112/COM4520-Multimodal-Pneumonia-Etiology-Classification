import sys
import os
from os.path import exists
sys.path.append(os.getcwd()) # To make this script easy to run from the terminal

import matplotlib.pyplot as plt
import pandas as pd
import scripts.config as config

# assume that we get the csv

def plotValidationAccWhileTraining(csvPath,plotTitle,outputPath):
    dataFrame = pd.read_csv(csvPath)
    x1 = dataFrame['epoch'].values
    y1 = dataFrame['acc'].values
    # plotting the line 1 points
    plt.plot(x1, y1, color='green', linestyle='dashed', linewidth = 3,
         marker='o', markerfacecolor='blue', markersize=6)

    # naming the x axis
    plt.xlabel('epochs')
    # naming the y axis
    plt.ylabel('accuracy')
    # giving a title to my graph
    plt.title(plotTitle)

    # show a legend on the plot
    #plt.legend()
    # Limit the axis range
    plt.ylim(0,1)
    plt.xticks(x1)
    plt.savefig(outputPath)

def main():
    
    inputCsvPath = config.stats_root +'/image_static_timeseries_val_perform_while_training.csv'
    if(exists(inputCsvPath)):
        plotValidationAccWhileTraining(inputCsvPath,
                                '3-modality validation accuracy while training',
                                config.graphs_root +'/image_static_timeseries_validationacc_while_training.png')

    inputCsvPath = config.stats_root +'/image_static_val_perform_while_training.csv'
    if(exists(inputCsvPath)):
        plotValidationAccWhileTraining(inputCsvPath,
                                'image and static validation accuracy while training',
                                config.graphs_root +'/image_static_validationacc_while_training.png')

    inputCsvPath = config.stats_root +'/image_timeseries_val_perform_while_training.csv'
    if(exists(inputCsvPath)):
        plotValidationAccWhileTraining(inputCsvPath,
                                'image and timeseries validation accuracy while training',
                                config.graphs_root +'/image_timeseries_validationacc_while_training.png')

    inputCsvPath = config.stats_root +'/static_timeseries_val_perform_while_training.csv'
    if(exists(inputCsvPath)):
        plotValidationAccWhileTraining(inputCsvPath,
                                'static and timeseries model validation accuracy while training',
                                config.graphs_root +'/static_timeseries_validationacc_while_training.png')

    inputCsvPath = config.stats_root +'/image_val_perform_while_training.csv'
    if(exists(inputCsvPath)):
        plotValidationAccWhileTraining(inputCsvPath,
                                'image model validation accuracy while training',
                                config.graphs_root +'/image_validationacc_while_training.png')

    inputCsvPath = config.stats_root +'/static_val_perform_while_training.csv'
    if(exists(inputCsvPath)):
        plotValidationAccWhileTraining(inputCsvPath,
                                'static model validation accuracy while training',
                                config.graphs_root +'/static_validationacc_while_training.png')


    inputCsvPath = config.stats_root +'/timeseries_val_perform_while_training.csv'
    if(exists(inputCsvPath)):
        plotValidationAccWhileTraining(inputCsvPath,
                                'timeseries model validation accuracy while training',
                                config.graphs_root +'/timeseries_validationacc_while_training.png')


if __name__ == '__main__':
    main()