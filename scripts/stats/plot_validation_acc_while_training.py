import sys
import os
from os.path import exists
sys.path.append(os.getcwd()) # To make this script easy to run from the terminal
import numpy as np
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
    # I would've used np.round_(np.linspace(0,1,20), decimals = 2), but for some reason it's not straight on 0.05
    plt.yticks([0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1])
    plt.savefig(outputPath)
    plt.clf()
    

# Could just pass in a list of csvPaths and make this a more generic function accepting any number of lines
def plotValidationAccWhileTraining2atonce(csvPath1,csvPath2,label1,label2,plotTitle,outputPath):
    dataFrame1 = pd.read_csv(csvPath1)
    dataFrame2 = pd.read_csv(csvPath2)
    x1 = dataFrame1['epoch'].values
    y1 = dataFrame1['acc'].values

    x2 = dataFrame2['epoch'].values
    y2 = dataFrame2['acc'].values
    # plotting the line 1 points
    plt.plot(x1, y1, color='green', linestyle='dashed', linewidth = 3, label = label1,
         marker='o', markerfacecolor='blue', markersize=6)

    # plotting the line 2 points
    plt.plot(x2, y2, color='blue', linestyle='dashed', linewidth = 3, label = label2,
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
    plt.xticks(x1)
    # I would've used np.round_(np.linspace(0,1,20), decimals = 2), but for some reason it's not straight on 0.05
    plt.yticks([0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1])
    plt.savefig(outputPath)
    plt.clf()

def plotValidationAccWhileTraining3atonce(csvPath1,csvPath2,csvPath3,label1,label2,label3,plotTitle,outputPath):
    dataFrame1 = pd.read_csv(csvPath1)
    dataFrame2 = pd.read_csv(csvPath2)
    dataFrame3 = pd.read_csv(csvPath3)
    x1 = dataFrame1['epoch'].values
    y1 = dataFrame1['acc'].values

    x2 = dataFrame2['epoch'].values
    y2 = dataFrame2['acc'].values

    x3 = dataFrame3['epoch'].values
    y3 = dataFrame3['acc'].values
 
    # plotting the line 1 points
    plt.plot(x1, y1, color='green', linestyle='dashed', linewidth = 3, label = label1,
         marker='o', markerfacecolor='blue', markersize=6)

    # plotting the line 2 points
    plt.plot(x2, y2, color='blue', linestyle='dashed', linewidth = 3, label = label2,
         marker='o', markerfacecolor='blue', markersize=6)

    # plotting the line 3 points
    plt.plot(x3, y3, color='red', linestyle='dashed', linewidth = 3, label = label3,
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
    plt.xticks(x1)
    # I would've used np.round_(np.linspace(0,1,20), decimals = 2), but for some reason it's not straight on 0.05
    plt.yticks([0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1])
    plt.savefig(outputPath)
    plt.clf()    

def main():

    #Accuracy on validation data while training for each individual model
    
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

    #Accuracy on validation data while training for multiple models in the same plot

    # Static by itself + image by itself + image and static together

    inputCsvPath1 = config.stats_root +'/image_static_val_perform_while_training.csv'
    inputCsvPath2 = config.stats_root +'/image_val_perform_while_training.csv'
    inputCsvPath3 = config.stats_root +'/static_val_perform_while_training.csv'
    if(exists(inputCsvPath1) and exists(inputCsvPath2)):
        plotValidationAccWhileTraining3atonce(inputCsvPath1,inputCsvPath2,inputCsvPath3,'image + static','image','static',
                                'Image, static and image + static',
                                config.graphs_root +'/static_and_image_alongside.png')

if __name__ == '__main__':
    main()