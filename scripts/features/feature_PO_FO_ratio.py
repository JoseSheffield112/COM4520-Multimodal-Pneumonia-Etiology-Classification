import os
import sys
sys.path.append(os.getcwd()) # Append current directory to sys.path. Makes it easier to run this script individually from the terminal.

import json
import psycopg2
import pandas as pd
import numpy as np
import pickle
from itertools import compress
from pathlib import Path
import scripts.config as const


#Paths to sql scripts
bg_sqlfile = 'mimic-iv/concepts/postgres/measurement/bg.sql'
postgresfunctions_sqlfile = 'mimic-iv/concepts/postgres/postgres-functions.sql'

# Set this to True if you've already created the bg table (Blood and gasses table).
BG_EXISTS = True

def main():
    conn = getConnection()
    cur = conn.cursor()
    
    create_bg_table(cur,conn)
    pao2fio2Dict = extract_pao2fio2ratio(cur,conn)
    return pao2fio2Dict



def create_bg_table(cur,conn):
    '''
    Creates the bg table (Blood and gases) containing the pao2fio2ratio data. This uses an sql script (bg.sql) from the mimic-codes repository:
    https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv/concepts/postgres/measurement
    '''
    # Define sql functions the sql scripts depend on
    print("Mimic code root: ",const.mimic_code_root)
    abs_path_sqlfunctions = os.path.join(const.mimic_code_root, postgresfunctions_sqlfile)
    if not os.path.exists(abs_path_sqlfunctions):
        print(abs_path_sqlfunctions + " was not found.")
        raise
    with open(abs_path_sqlfunctions, 'r') as f:
        sql = f.read()
        cur.execute(sql)
        conn.commit()
    
    # Create the bg table (Blood and gasses table) only if it does not already exist. I could query the database and determine this dynamically, but there's no reason to spent time on that
    if (not BG_EXISTS):
        absolute_sqlfilepath = os.path.join(const.mimic_code_root,bg_sqlfile)
        if not os.path.exists(absolute_sqlfilepath):
            print(absolute_sqlfilepath + " was not found.")
            raise
        
        print('executing {0}...'.format(absolute_sqlfilepath))
        with open(absolute_sqlfilepath, 'r') as f:
            sql = f.read()
            cur.execute(sql)
            conn.commit()
        print('finish executing {0}!'.format(absolute_sqlfilepath))


def extract_pao2fio2ratio(cur,conn):
    '''
    Returns a dictionary whose keys are hadm_id's and values are (24,1) numpy arrays containing pao2fio2ratio values for the first 24 hours after the
    first pao2fao2ratio reading for that admission.

    Also serializes the dictionary as a python pickled file and stores it in the repo-root/features directory
    '''
    conn = getConnection()
    cur = conn.cursor()

    sql = "SELECT subject_id, hadm_id, charttime, pao2fio2ratio FROM bg;"
    cur.execute(sql)
    conn.commit()

    resultsRatio = cur.fetchall()

    #Convert results to DataFrame
    resultsRatio = pd.DataFrame(resultsRatio, columns =['subject_id', 'hadm_id', 'charttime','pao2fio2ratio'])

    # Throw away any readings that don't have pao2fio2ratio
    resultsRatio = resultsRatio[~resultsRatio['pao2fio2ratio'].isnull()]

    print("Get a list containing each admission represented as a dataframe")
    data = [resultsRatio[resultsRatio.hadm_id == hadmID] for hadmID in resultsRatio.hadm_id.unique()]

    # Filter out admissions that have too few readings
    maskReadingsMoreThan5 = [admissionDataFrame.shape[0] > 5 for admissionDataFrame in data]
    filteredDataFrames = list(compress(data,maskReadingsMoreThan5))


    filteredDataFrames = processAdmissions(filteredDataFrames)
    #Fix types
    for idx,admissionData in enumerate(filteredDataFrames):
        filteredDataFrames[idx] = admissionData.astype({'hadm_id': 'int32'})


    # Shape the output to be a dataframe with hadm_id as the index and the (24,1) numpy array as its single value.
    listOfAdmIdArrayTuples = [(dataFrame.iloc[0].hadm_id,convertAdmisionDataframeToNumpyArray(dataFrame)) for dataFrame in filteredDataFrames]
    outputDataframe = pd.DataFrame(listOfAdmIdArrayTuples, columns=['hadm_id', 'value']).set_index('hadm_id')

    #Serialize the dictionary
    f = open(Path(const.feature_root + '/pao2fio2ratio.pyc'), mode='wb')
    pickle.dump(outputDataframe, file=f)
    f.close()

    return outputDataframe

def getConnection(db='mimiciv'):
    '''
    Returns a connection to a specified database by using a configuration file whose path is specified in the scripts.config python module.
    '''
    with open(const.connection_json_root, 'r') as f:
        conn_params = json.load(f)[db]
        conn = psycopg2.connect(
            "dbname='{dbname}' user='{user}' host='{host}' password='{password}' port='{port}'".format(**conn_params)
        )
        return conn


def convertAdmisionDataframeToNumpyArray(admissionData):
    return np.asarray([reading.pao2fio2ratio for (_,reading) in admissionData.iterrows()])

def interpolateGap(deltaHours,gapSize,leftGap,rightGapRatio,admissionData):
    '''
    deltaHours    = Nr of hours between the minimum of the gap and maximum of the gap
    gapSize       = Size of gap to fill. In most situations = deltaHours - 1 , but sometimes not. This is for the caveat where 
                    the linear interpolation towards the final 24 hour is done with a value from > 24 hours away from the first reading.
    leftGap       = Pandas Series representing the reading of the minimum extreme of the gap
    rightGapRatio = pao2fio2ratio of the Pandas Series representing the reading of the maximum extreme of the gap
    admissionData = pandas DataFrame containing readings for an admission
    
    Returns a pandas DataFrame identical with admissionData, except for it having filled a described hour gap between readings 
    using linear interpolation of the pao2fio2ratio values of the extremes of the gap.
    '''
    increment = (rightGapRatio - leftGap.pao2fio2ratio) / deltaHours
    while (gapSize > 0):
        artificialReading = leftGap.copy(deep=True)
        artificialReading.name = leftGap.name + 1 #.name = hour since the hour column is set as the index of the series
        artificialReading.pao2fio2ratio = leftGap.pao2fio2ratio + increment
        artificialReading.charttime = pd.Timestamp('2069-01-01T12') #Meme charttime for no real reason. Maybe to later identify this as an artificially created reading.
        admissionData = admissionData.append(artificialReading)
        
        leftGap = artificialReading
        gapSize -= 1
    return admissionData

def interpolateAdmission(admissionData):    
    '''
    admissionData = pandas dataframe containing readings for an admission
    
    Given an admissionData dataframe containing readings for a particular admission, create artificial readings for missing
    hours by doing a linear interpolation between the beginning and the end of a gap in hours.
    It is assumed that the admissionData dataframe has been set up with an hours column for each reading.  
    '''
    
    #Pandas Series from admissionData that represent readings that are either the minimum extreme of the gap or maximum extreme of the gap
    leftGap = None
    rightGap = None
    for hourIndx in range(0,24):
        if (hourIndx in admissionData.index):
            leftGap = rightGap
            rightGap = admissionData.loc[hourIndx]
            delta = (rightGap.name - leftGap.name) if not (leftGap is None) else 0 #.name is the hour column. Since the hour column is set as the index of the dataframe
            if (delta > 1): # Interpolate only if there is a gap between the hours
                admissionData = interpolateGap(deltaHours=delta,gapSize=(delta-1),leftGap=leftGap\
                                               ,rightGapRatio=rightGap.pao2fio2ratio,\
                                               admissionData=admissionData)
            continue
        elif (hourIndx == 23) and (admissionData.iloc[-1].name > 23):
            #If there is no reading for the 24th hour, then interpolate between last value and the next closest value (that is going to be > 24 hours away)
            firstReadingAfter24hrs = admissionData.iloc[-1]
            delta = firstReadingAfter24hrs.name - rightGap.name #Again .name is the hour, since the hour column is set as the index of the dataframe.
            admissionData = interpolateGap(deltaHours=delta,gapSize=(24 - rightGap.name - 1)\
                                           ,leftGap=rightGap,rightGapRatio=firstReadingAfter24hrs.pao2fio2ratio,\
                                           admissionData=admissionData)
            continue
        elif (hourIndx == 23):
            #24th hour doesn't exist, and there is no next value to interpolate towards. Interpolate towards the average pao2fio2ratio of this admission
            avgRatio = admissionData['pao2fio2ratio'].mean()
            delta = 24 - rightGap.name
            admissionData = interpolateGap(deltaHours=delta,gapSize=(24 - rightGap.name - 1)\
                                           ,leftGap=rightGap,rightGapRatio=avgRatio,\
                                           admissionData=admissionData)              
    return admissionData

# Merge these into the same for eventually
# Go through each admission

def processAdmissions(admissions):
    '''
    admissions = A list of DataFrames, each DataFrame containing readings for an admission
    
    This function associates an hour value to each reading, filters out readings > 24 hours away from the first reading. (while keeping a single reading > 24 hours,
    this is for preprocessing admissions that have a small amount of readings) and filters out admissions that have too few readings.
    
    Finally, each admission dataframe is filled to have 24 values, by linearly interpolating between gaps in readings. 
    '''
    MIN_READINGS_PER_ADMISSION = 3
    for idx,admissionData in enumerate(admissions):
        #Associate an hour to each reading
        admissionData.sort_values(by=['charttime'], ascending = True, inplace=True)
        firstReading = admissionData.iloc[0]
        admissionData['hour'] = ((admissionData.charttime - firstReading.charttime) / np.timedelta64(1, 's')/(1*60*60)).round(0).astype(int)

        #Keep only readings that are <= 24 hours
        first24 = admissionData[admissionData.hour <= 23]
        firstValueOver24 = admissionData[admissionData.hour > 23].iloc[0] \
                            if (admissionData[admissionData.hour > 23].size > 0) else None
        admissions[idx] = first24.append(firstValueOver24).drop_duplicates(subset = ['hour'])

    #Filter out admissions where there are not enough readings in the first 24 hrs for the feature to be meaningful
    admissions = list(compress(admissions,[adm.shape[0] > MIN_READINGS_PER_ADMISSION for adm in admissions]))
    
    print("Interpolating ... This will take the longest")
    for idx,admissionData in enumerate(admissions):
        admissionData.set_index(keys='hour', inplace = True)
        admissionData = interpolateAdmission(admissionData).sort_index()
        #If this dataframe has 25 values, drop the last one, it's no longer needed
        if(admissionData.shape[0] == 25):
            admissionData = admissionData.head(admissionData.shape[0] - 1)
        admissions[idx] = admissionData
    return admissions



if __name__ == '__main__':
    main()