"""
PCG_gen (Parity, Chart, and Graph generator) seeks to generate customizable parity plots from importing a
.csv file

Required files:
    -.csv file for analysis

Coded in Python 3.12.4 using Anaconda Interpreter
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

pd.options.mode.copy_on_write = True

#may generate warning that there are more markers passed than used
#this is for the purpose of accomodating multiple potential graphs
m = ['o', '^', 's', 'v', 'd']

#dictionary to manage graph charateristics variables (gcv)
gcv = {
    'csv': None,
    'inputGTValues': None,
    'pcgMode': None,
    'max': None,
    'min': None,
    'error': None,
    'xLabel': None,
    'yLabel': None,
    'title': None
}

############################################
# DEFINITIONS FOR PCGs
############################################

# df - the list
# GT - the Ground Truth measurements
# M - the Measured values (FIRM) measurements
def calculateRValues(df, G, M):
    #generates frames of columns and calculates R-values from each one
    #declare r-values equal to
    sub1 = df[df['Class'] == 'FIRM']
    rFIRM = np.corrcoef(sub1[G], sub1[M])
    ''' # HUMAN CODE - RESERVED FOR NOW #
    sub2 = df[df['Class'] == 'Human 1']
    rH1 = np.corrcoef(sub2[G], sub2[M])
    sub3 = df[df['Class'] == 'Human 2']
    rH2 = np.corrcoef(sub3[G], sub3[M])
    sub4 = df[df['Class'] == 'Human 3']
    rH3 = np.corrcoef(sub4[G], sub4[M])
    '''

    #iterates through the list 'class' column and adds the r^2 value rounded to 3 decimal places
    for i in df['Class']:
        if i == 'FIRM':
            df = df.replace(i, i + ' R^2-value: ' + str(((rFIRM[0,1])**2).round(3)))
    ''' # HUMAN CODE - RESERVED FOR NOW #
    for i in df['Class']:
        if i == 'Human 1':
            df = df.replace(i, i + ' R^2-value: ' + str((rH1[0,1])**2).round(3))
    for i in df['Class']:
        if i == 'Human 2':
            df = df.replace(i, i + ' R^2-value: ' + str((rH2[0,1])**2).round(3))
    for i in df['Class']:
        if i == 'Human 3':
            df = df.replace(i, i + ' R^2-value: ' + str((rH3[0,1])**2).round(3))
    '''
    return df

def strToFloatList(l):
    nL = []
    l = l.split(', ')
    for i in l:
        nL.append(float(i))
    return nL

def appendItems(df, newDf, c1, c2, nc1, nc2, nc3):
    for i in df[c1]:
        newDf[nc1].append(i)
    for i in df[c2]:
        newDf[nc2].append(i)
        newDf[nc3].append(c1)

def printHeader():
    print('#########################################################')
    print('---------------------------------------------------------')
    print('#########################################################')

def printGraphNum(j):
    print('#########################################################')
    print('Setting up Graph '+str(j)+'---------------------------------------')
    print('#########################################################')

def setLabels():
    #sets up the text features
    plt.xlabel(gcv['xLabel']) #to change font size, use fontsize = 20
    plt.ylabel(gcv['yLabel']) #to change font size, use fontsize = 20
    plt.title(gcv['title']) #to change font size, use fontsize = 20
    #plt.rcParams['font.size']=20

def distCumulative(df):
    temp = None
    for i in range(0,len(df)-1):
        temp = df[i]
        df[i+1] += temp
    return df

def askInputs():
    #input for PCG mode
    print('Enter a pcg mode: ')
    gcv['pcgMode'] = str(input("'p' for parity plots, 'c' for bar charts, or 'g' for size distribution graphs: "))

    #input for .csv file
    print('Enter path for .csv file: ')
    gcv['csv'] = str(input("Format: C:\\Users\\name\\FIRM-image-analysis\\example.csv: "))

    #input for GT values
    print('Enter GT values: ')
    gcv['inputGTValues'] = {'GT': strToFloatList(input(
        'Type your values in a list in the format of 1.2, 3.6, 9.3: '))}
    gcv['inputGTValues'] = pd.DataFrame(gcv['inputGTValues'])
    print(gcv['inputGTValues'])

    #input for min/max
    if gcv['pcgMode'] == 'p':
        gcv['max'] = int(input('Enter a max x and y boundary for the graph: '))
        gcv['min'] = int(input('Enter a min x and y boundary for the graph: '))
    elif gcv['pcgMode'] == 'g' or 'c':
        pass

    #error bars
    if gcv['pcgMode'] != 'g':
        response1 = str(input("Enter error bars? Type 'y' or 'n': "))
        gcv['error'] = None
        if response1 == 'y':
            errorType = int(input("Press '1' to enter a single error value or '2' to enter multiple error values as a list: "))
            if errorType == 1:
                gcv['error'] = float(input('Type your error as a single number: '))
            elif errorType == 2:
                gcv['error'] = strToFloatList(input('Type your errors in a list in the format of 1.2, 3.6, 9.3: '))
        elif response1 == 'n':
            pass

    #input for labels
    gcv['xLabel'] = str(input('Enter a label for the x-axis: '))
    gcv['yLabel'] = str(input('Enter a label for the y-axis: '))
    response2 = str(input("Enter a title? Type 'y' or 'n': "))
    if response2 == 'y':
        gcv['title'] = str(input('Enter a title:'))
    elif response2 == 'n':
        gcv['title'] = None

def clearDict(dict):
    for key in dict.keys():
        dict[key] = None

############################################
# CODE THAT GENERATES PCGs
############################################

print('PCG_gen - Generate parity plots, bar charts, and size distribution graphs')
print('PCG_extract implemented soon')
#inputs for number of graphs
printHeader()
iterations = int(input('How many graphs would you like to generate? '))

#generates graphs j times
for j in range(1, iterations+1):
    clearDict(gcv)
    printGraphNum(j)
    askInputs()

    # PARITY PLOT CODE #########################
    if gcv['pcgMode'] == 'p':
        #read .csv file and run the code
        df = pd.read_csv(gcv['csv'])

        #inputs for GT values
        for i in range(0, len(df['GT'])):
            df.loc[i, 'GT'] = gcv['inputGTValues'].loc[i, 'GT']
            print(df['GT'][i])
        print(df)
        df = calculateRValues(df, 'GT', 'ML')
        print(df)

        #sets up the x=y line
        line = pd.DataFrame({'x': [0,1],'y': [0,1]})
        newLine = line.replace(1, gcv['max'])

        #plots the graph
        sb.pairplot(x_vars='GT', y_vars='ML', kind='scatter', hue='Class', markers=m, data=df)
        sb.lineplot(x='x', y='y', linestyle='dashed', color='grey', data=newLine) 
        plt.axis([gcv['min'], gcv['max'], gcv['min'], gcv['max']]) 
        plt.errorbar(x=df['GT'], y=df['ML'], yerr=gcv['error'], fmt='none', capsize=7, elinewidth=2)
        setLabels()
        #for some reason, if this line of code is at the end, it removes the extra blank graph window. Watch
        #out for this line of code if a blank window is generated
        plt.figure(j)

    # BAR CHART CODE #########################
    if gcv['pcgMode'] == 'c':
        #read .csv file and run the code
        df = pd.read_csv(gcv['csv'])

        #pre-formatting (does not belong in PCG_F_extract)
        features = {
            'Quantity': [],
            'GTorML': [],
            'ImageName': []
        } 

        #inputs for GT values
        for i in range(0, len(df['GT'])):
            df.loc[i, 'GT'] = gcv['inputGTValues'].loc[i, 'GT']
            print(df['GT'][i])  

        #append df values to newly formatted dataset
        GTvalues = df[['GT', 'ImageName']]
        appendItems(GTvalues, features, 'GT', 'ImageName', 'Quantity', 'ImageName', 'GTorML')
        MLvalues = df[['ML', 'ImageName']]
        appendItems(MLvalues, features, 'ML', 'ImageName', 'Quantity', 'ImageName', 'GTorML')
        df = pd.DataFrame(features)
        print(df)

        #plots the graph
        plt.figure(j)
        sb.barplot(x='ImageName', y='Quantity', hue='GTorML', edgecolor='0', data=df)
        setLabels()

    # SIZE DISTRIBUTION GRAPH CODE #########################
    if gcv['pcgMode'] == 'g':
        #read .csv file and run the code
        df = pd.read_csv(gcv['csv'])

        #pre-formatting (does not belong in PCG_F_extract)
        features = {
            'Bins': [],
            'Fibril%': [],
            'GTorML': [],
            'Class': []
        }

        #create bins based on length of dataset
        Bins = pd.DataFrame({'Bins': range(0,len(df)*10,10)})
        print(Bins)

        #inputs for GT values
        for i in range(0, len(df['GT'])-1):
            df.loc[i, 'GT'] = gcv['inputGTValues'].loc[i, 'GT']
            print(df['GT'][i])

        #append df values to newly formatted dataset, also makes values cumulative
        df['GT'] = distCumulative(df['GT'])
        GTvalues = df[['GT', 'Class']]
        appendItems(GTvalues, features, 'GT', 'Class', 'Fibril%', 'Class', 'GTorML')
        for i in Bins['Bins']:
            features['Bins'].append(i)
        df['ML'] = distCumulative(df['ML'])
        MLvalues = df[['ML', 'Class']]
        appendItems(MLvalues, features, 'ML', 'Class', 'Fibril%', 'Class', 'GTorML')
        for i in Bins['Bins']:
            features['Bins'].append(i)
        df = pd.DataFrame(features)
        print(df)

        #creates a lineplot of the data, sets the hue for the different classes, and provides the values for the size ranges
        plt.figure(j)
        sb.lineplot(x='Bins', y='Fibril%', hue='GTorML', data=df)

        #creates a scatterplot of the data, maps the size ranges to the index, provides different markers
        sb.scatterplot(x='Bins', y='Fibril%', data=df, hue='GTorML', style='GTorML', s=50, markers=m)
        setLabels()

print('Remember to open application in full screen!')
plt.show()

############################################
# CREDITS
############################################

#multiple windows solution by: https://stackoverflow.com/questions/5993206/is-it-possible-to-have-multiple-pyplot-windows-or-am-i-limited-to-subplots
#clear dictionary solution by: https://stackoverflow.com/questions/22991888/how-to-reset-all-values-in-a-dictionary
#solution concerning 'SettingWithCopyWarning' by: https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas