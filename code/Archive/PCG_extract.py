"""
PCG_extract (Parity, Chart, and Graph Features extract) seeks to extract features from .csv files generated
from ImageJ analysis

Make sure to do the ImageJ analyses first, and then export the data to this script

Coded in Python 3.12.4 using Anaconda Interpreter
"""

import pandas as pd
import numpy as np
import os

############################################
# DEFINITIONS FOR EXTRACTION
############################################

def printHeader():
    print('#########################################################')

def appendItems(inputFDF, inputGT, inputML, inputClass, inputName):
    inputFDF['GT'].append(inputGT)
    inputFDF['ML'].append(inputML)
    inputFDF['Class'].append(inputClass)
    inputFDF['ImageName'].append(inputName)

#takes in a list of .csv file measurements
def extractMLFibCount(mList):
    #takes the length of dataset as # of fibrils
    features = {
        'GT': [],
        'ML': [],
        'Class': [],
        'ImageName': []
    }
    for i in range(0,len(mList)):
        fC = len(mList[i])
        appendItems(features, 'N/A', fC, 'FIRM', 'Image ' + str(i+1))

    return features

#takes in a list of .csv file measurements
def extractMLAvgDia(mList):
    #takes the mean of the 'Minor" axis column and rounds to the 3rd decimal place
    features = {
        'GT': [],
        'ML': [],
        'Class': [],
        'ImageName': []
    }
    for i in range(0,len(mList)):
        aD = (np.mean(mList[i]['Minor'])).round(3)
        appendItems(features, 'N/A', aD, 'FIRM', 'Image ' + str(i+1))
    return features

#takes in individual .csv file distributions
def extractMLDist(df, imageNum):
    bC = df['count']
    features = {
        'GT': [],
        'ML': [],
        'Class': [],
        'ImageName': []
    }
    #iterates through the 'Count' column and appends its contents to the ML column
    for i in bC:
        appendItems(features, 'N/A', i, 'FIRM', 'Image ' + str(imageNum))

    return features

############################################
# EXECUTE IMAGEJ COMMANDS TO GENERATE .CSVs
############################################

pass

############################################
# EXTRACT FEATURES FROM .CSV
# -iterate through folder solution provided by https://www.geeksforgeeks.org/how-to-iterate-over-files-in-directory-using-python/
############################################

mList = []
dList = []

print('Enter the path for ImageJ measurements')
mFolderPath = input('Format: C:\\Users\\Name\\FIRM-image-analysis\\Measurements: ')
for file in os.scandir(mFolderPath):
    if file.is_file():
        temp = pd.read_csv(file.path)
        mList.append(temp)
printHeader()
print('Enter the path for ImageJ distributions')
dFolderPath = input('Format: C:\\Users\\Name\\FIRM-image-analysis\\Distributions: ')
for file in os.scandir(dFolderPath):
    if file.is_file():
        temp = pd.read_csv(file.path)
        dList.append(temp)

fibCount = pd.DataFrame(extractMLFibCount(mList))
avgDia = pd.DataFrame(extractMLAvgDia(mList))
#reference for solution for creating new variables: https://stackoverflow.com/questions/6181935/how-do-you-create-different-variable-names-while-in-a-loop
d = {}
for i in range(1,len(dList)+1):
    d['dfd'+str(i)] = pd.DataFrame(extractMLDist(dList[i-1], i))

############################################
# CREATE FORMATTED .CSV FILES IN PATH
############################################

printHeader()
print('Enter path you would like PPEcsvs to be created: ')
path = input('Format: C:\\Users\\Name\\FIRM-image-analysis\\: ')

os.makedirs(path + 'PPEcsvs', exist_ok=True)
fibCount.to_csv(path + '\\PPEcsvs\\fibCount.csv', index=False)
avgDia.to_csv(path + '\\PPEcsvs\\avgDia.csv', index=False)
for i in range(1,len(dList)+1):
    d['dfd'+str(i)].to_csv(path + '\\PPEcsvs\\dist' + str(i) + '.csv', index=False)

printHeader()
print('Features extracted! Check PPEcsvs in the designated folder.')

