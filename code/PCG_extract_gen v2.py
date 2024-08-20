"""
PCG_extract_gen v2 (Parity, Chart, and Size Distribution Graph) seeks to extract features from ImageJ analysis files 
and automatically generate graphs for the purpose of analyzing correlation, total/average parameters, and size
distribution

Make sure to do the ImageJ analyses first, and then export the data to here

Coded in Python 3.12.4 using Anaconda Interpreter
"""

print('Loading Packages...')

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sb
import math

sb.set_theme(font='Times New Roman', style='white', palette=None)
pd.options.mode.copy_on_write = True
rng = np.random.default_rng()

#may generate warning that there are more markers passed than used
#this is for the purpose of accomodating multiple potential graphs
m = ['o', '^'] #'s', 'v', 'd'

############################################
# DEFINITIONS FOR EXTRACTION
############################################

def printHeader():
    print('#########################################################')
    print('---------------------------------------------------------')
    print('#########################################################')

def appendItems(inputFDF, inputGT, inputML, inputClass, inputName):
    inputFDF['GT'].append(inputGT)
    inputFDF['ML'].append(inputML)
    inputFDF['Class'].append(inputClass)
    inputFDF['ImageName'].append(inputName)

#takes in a list of .csv file measurements
def extractFibCount(mList, Mode):
    #takes the length of dataset as # of fibrils
    features = {
        'GT': [],
        'ML': [],
        'Class': [],
        'ImageName': []
    }
    for i in range(0,len(mList)):
        fC = len(mList[i])
        if Mode == 'ML':
            appendItems(features, 'N/A', fC, 'FIRM', 'Image ' + str(i+1))
        elif Mode == 'GT':
            appendItems(features, fC, 'N/A', 'FIRM', 'Image ' + str(i+1))
        
    return features

#takes in a list of .csv file measurements
def extractMinorAxis(mList, Mode):
    features = {
        'GT': [],
        'ML': [],
        'Class': [],
        'ImageName': []
    }
    for i in range(0,len(mList)):
        temp = mList[i]
        for j in temp['Minor']:
            if Mode == 'ML':
                appendItems(features, 'N/A', j, 'FIRM', 'Image ' + str(i+1))
            elif Mode == 'GT':
                appendItems(features, j, 'N/A', 'FIRM', 'Image ' + str(i+1))

    return features

def extractMinorAxisMeans(mList, Mode):
    #takes the mean of the 'Minor" axis column and rounds to the 3rd decimal place
    features = {
        'GT': [],
        'ML': [],
        'Class': [],
        'ImageName': []
    }
    for i in range(0,len(mList)):
        aD = (np.mean(mList[i]['Minor'])).round(3)
        if Mode == 'ML':
            appendItems(features, 'N/A', aD, 'FIRM', 'Image ' + str(i+1))
        elif Mode == 'GT':
            appendItems(features, aD, 'N/A', 'FIRM', 'Image ' + str(i+1))

    return features

#takes in individual .csv file distributions
def extractDist(dList, imageNum, Mode):
    bC = dList['Counts']
    features = {
        'GT': [],
        'ML': [],
        'Class': [],
        'ImageName': []
    }
    #iterates through the 'Count' column and appends its contents to the ML column
    for i in bC:
        if Mode == 'ML':
            appendItems(features, 'N/A', i, 'FIRM', 'Image ' + str(imageNum))
        elif Mode == 'GT':
            appendItems(features, i, 'N/A', 'FIRM', 'Image ' + str(imageNum))
    return features

def strToFloatList(l):
    nL = []
    l = l.split(', ')
    for i in l:
        nL.append(float(i))
    return nL

############################################
# EXECUTE IMAGEJ COMMANDS TO GENERATE .CSVs
# Currently Unavailable
############################################

import imagej 
import scyjava
from scyjava import jimport #use Java object/resources in Python
scyjava.config.add_option('-Xmx6g')

printHeader()
print('PCG_extract_gen v1.0: Extract features and automatically generate graphs')
printHeader()

#sets imagej to a reproducible version
print("Loading ImageJ...")
ij = imagej.init('net.imagej:imagej:2.14.0', mode='interactive')
print(f"ImageJ2 version: {ij.getVersion()}")

print('Enter the folder path for ML images')
imgsMLFolder = input(r'Format: C:\Users\Name\FIRM-image-analysis\ImageSet\ML images\: ')
imgsMLPaths = []
for file in os.scandir(imgsMLFolder):
    if file.is_file():
        temp = str(file.path)
        imgsMLPaths.append(temp)

print('Enter the folder path for GT images')
imgsGTFolder = input(r'Format: C:\Users\Name\FIRM-image-analysis\ImageSet\GT images\: ')
imgsGTPaths = []
for file in os.scandir(imgsGTFolder):
    if file.is_file():
        temp = str(file.path)
        imgsGTPaths.append(temp)

print('Enter the folder path for where you want to store the IJ csvs')
pathIJ = input(r'Format: C:\Users\Name\FIRM-image-analysis\ImageSet\: ')
os.makedirs(pathIJ + 'MLcsvs', exist_ok=True); os.makedirs(pathIJ + 'MLcsvs' + r'\Measurements', exist_ok=True); os.makedirs(pathIJ + 'MLcsvs' + r'\Distributions', exist_ok=True)
os.makedirs(pathIJ + 'GTcsvs', exist_ok=True); os.makedirs(pathIJ + 'GTcsvs' + r'\Measurements', exist_ok=True); os.makedirs(pathIJ + 'GTcsvs' + r'\Distributions', exist_ok=True)
measurementsMLPath = (pathIJ.replace("\\", "/")) + "MLcsvs/" + "Measurements"
distributionsMLPath = (pathIJ.replace("\\", "/")) + "MLcsvs/" + "Distributions"
measurementsGTPath = (pathIJ.replace("\\", "/")) + "GTcsvs/" + "Measurements"
distributionsGTPath = (pathIJ.replace("\\", "/")) + "GTcsvs/" + "Distributions"

print('Enter the folder path for where you want to store the PCG csvs')
pathPCG = input(r'Format: C:\Users\Name\FIRM-image-analysis\: ')
pathPCG = pathPCG.replace("\\", "\\\\")

print('Enter the path where you would like to save your figures: ')
figPath = input(r"Format: C:\Users\Name\FIRM-image-analysis\figures\: ")
figPath = figPath.replace("\\", "\\\\")

print('Enter a range for size filtering (unit)')
sizeFilter = str(input('Format: 1000-8000: '))

print('Enter a list for ML image scales in the order of images top->down (nm/1 pix)')
scaleML = strToFloatList(input('Format: 10, 20, 30: '))

print('Enter a list for GT image scales in the order of images top->down (nm/1 pix)')
scaleGT = strToFloatList(input('Format: 10, 20, 30: '))


def runMLMacro(imgPath, mPath, dPath,scalePer1Pix, unit):
    macro = """
    open("""+imgPath+""");
    run("8-bit");
    setAutoThreshold("Li dark no-reset");
    setOption("BlackBackground", true);
    run("Convert to Mask");
    run("Erode");
    run("Dilate");
    run("Dilate");
    run("Watershed");
    run("Set Scale...", "distance=1 known="""+str(scalePer1Pix)+""" unit="""+unit+"""");
    run("Analyze Particles...", "size="""+sizeFilter+""" circularity=0.85-1.00 show=Outlines display clear summarize overlay");
    saveAs("Results", """+mPath+""");
    close();
    run("Distribution...", "parameter=Minor or=21 and=0-210");
    selectWindow("Minor Distribution");
    Plot.getValues(bins, counts);
    Table.create("Table 1")
    Table.setColumn("Bins", bins);
    Table.setColumn("Counts", counts);
    selectWindow("Table 1");
    saveAs("Text", """+dPath+""")
    close();
    """
    print(macro)
    ij.py.run_macro(macro)

def runGTMacro(imgPath, mPath, dPath, scalePer1Pix, unit):
    macro = """
    open("""+imgPath+""");
    setAutoThreshold("Li dark no-reset");
    setOption("BlackBackground", true);
    run("Convert to Mask");
    run("Set Scale...", "distance=1 known="""+str(scalePer1Pix)+""" unit="""+unit+"""");
    run("Analyze Particles...", "size=0-infinity circularity=0-1 show=Outlines display clear summarize overlay");
    saveAs("Results", """+mPath+""");
    close();
    run("Distribution...", "parameter=Minor or=21 and=0-210");
    selectWindow("Minor Distribution");
    Plot.getValues(bins, counts);
    Table.create("Table 1")
    Table.setColumn("Bins", bins);
    Table.setColumn("Counts", counts);
    selectWindow("Table 1");
    saveAs("Text", """+dPath+""")
    close();
    """
    print(macro)
    ij.py.run_macro(macro)

print("Performing Image Analysis...")
#ML Image Analysis
for i in range(0, len(imgsMLPaths)):
    imgPath = imgsMLPaths[i].replace('\\', '/')
    imgPath = '''"''' + imgPath + '''"'''
    mPath = '''"''' + measurementsMLPath + "/Results(" + str(i+1) + ").csv" + '''"'''
    dPath = '''"''' + distributionsMLPath + "/Minor Distribution(" + str(i+1) + ").csv" + '''"'''

    print('Using scale: ' + str(scaleML[i]) + 'nm/pix')
    runMLMacro(imgPath, mPath, dPath, scaleML[i], """nm""")

#GT Image Analysis
for i in range(0, len(imgsGTPaths)):
    imgPath = imgsGTPaths[i].replace('\\', '/')
    imgPath = '''"''' + imgPath + '''"'''
    mPath = '''"''' + measurementsGTPath + "/Results(" + str(i+1) + ").csv" + '''"'''
    dPath = '''"''' + distributionsGTPath + "/Minor Distribution(" + str(i+1) + ").csv" + '''"'''

    print('Using scale: ' + str(scaleGT[i]) + 'nm/pix')
    runGTMacro(imgPath, mPath, dPath, scaleGT[i], """nm""")

print('ImageJ Analysis Done!')


############################################
# EXTRACT FEATURES FROM .CSV
# -iterate through folder solution provided by https://www.geeksforgeeks.org/how-to-iterate-over-files-in-directory-using-python/
############################################

mListML = []
dListML = []
mListGT = []
dListGT = []

#Enters ML Measurements
mFolderPathML = measurementsMLPath.replace("/", "\\")
for file in os.scandir(mFolderPathML):
    if file.is_file():
        temp = pd.DataFrame(pd.read_csv(file.path))
        mListML.append(temp)
#Enters ML Distributions
dFolderPathML = distributionsMLPath.replace("/", "\\")
for file in os.scandir(dFolderPathML):
    if file.is_file():
        temp = pd.DataFrame(pd.read_csv(file.path))
        dListML.append(temp)

#Enters GT Measurements
mFolderPathGT = measurementsGTPath.replace("/", "\\")
for file in os.scandir(mFolderPathGT):
    if file.is_file():
        temp = pd.read_csv(file.path)
        mListGT.append(temp)

#Enters GT Distributions
dFolderPathGT = distributionsGTPath.replace("/", "\\")
for file in os.scandir(dFolderPathGT):
    if file.is_file():
        temp = pd.read_csv(file.path)
        dListGT.append(temp)

printHeader()

minorAxisML = pd.DataFrame(extractMinorAxis(mListML, 'ML'))
minorAxisMeansML = pd.DataFrame(extractMinorAxisMeans(mListML, 'ML'))
fibCountML = pd.DataFrame(extractFibCount(mListML, 'ML'))
dML = {}
for i in range(1,len(dListML)+1):
    dML['dfd'+str(i)] = pd.DataFrame(extractDist(dListML[i-1], i, 'ML'))

minorAxisGT = pd.DataFrame(extractMinorAxis(mListGT, 'GT'))
minorAxisMeansGT = pd.DataFrame(extractMinorAxisMeans(mListGT, 'GT'))
fibCountGT = pd.DataFrame(extractFibCount(mListGT, 'GT'))
dGT = {}
for i in range(1,len(dListGT)+1):
    dGT['dfd'+str(i)] = pd.DataFrame(extractDist(dListGT[i-1], i, 'GT'))

############################################
# CREATE FORMATTED .CSV FILES IN PATH
############################################

os.makedirs(pathPCG + 'MLcsvs', exist_ok=True)
fibCountML.to_csv(pathPCG + 'MLcsvs\\fibCountML.csv', index=False)
minorAxisML.to_csv(pathPCG + 'MLcsvs\\minorAxisML.csv', index=False)
minorAxisMeansML.to_csv(pathPCG + 'MLcsvs\\minorAxisMeansML.csv', index=False)
for i in range(1,len(dListML)+1):
    dML['dfd'+str(i)].to_csv(pathPCG + 'MLcsvs\\dist' + str(i) + '.csv', index=False)

os.makedirs(pathPCG + 'GTcsvs', exist_ok=True)
fibCountGT.to_csv(pathPCG + 'GTcsvs\\fibCountGT.csv', index=False)
minorAxisGT.to_csv(pathPCG + 'GTcsvs\\minorAxisGT.csv', index=False)
minorAxisMeansGT.to_csv(pathPCG + 'GTcsvs\\minorAxisMeansGT.csv', index=False)
for i in range(1,len(dListGT)+1):
    dGT['dfd'+str(i)].to_csv(pathPCG + 'GTcsvs\\dist' + str(i) + '.csv', index=False)

printHeader()
print('Features extracted! Check PPEcsvs in the designated folder.')
#dictionary to manage graph charateristics variables (gcv)
gcv = {
    'pcgMode': None,
    'max': None,
    'min': None,
    'error': None,
    'xLabel': None,
    'yLabel': None,
    'title': None,
    'csvCountML': None,
    'csvAvgML': None
}

############################################
# DEFINITIONS FOR GENERATION
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

def calculateSE(data):
    StdE = np.std(data, ddof=1) / np.sqrt(np.size(data))
    return StdE

def appendItemsReformat(df, newDf, c1, c2, nc1, nc2, nc3):
    for i in df[c1]:
        newDf[nc1].append(i)
    for i in df[c2]:
        newDf[nc2].append(i)
        newDf[nc3].append(c1)

def printGraphNum(j):
    print('#########################################################')
    print('Setting up Graph '+str(j)+'---------------------------------------')
    print('#########################################################')

def setLabels(xLabel, yLabel, title):
    #sets up the text features
    plt.xlabel(xLabel) #to change font size, use fontsize = 20
    plt.ylabel(yLabel) #to change font size, use fontsize = 20
    plt.title(title) #to change font size, use fontsize = 20
    #plt.rcParams['font.size']=20

def distCumulative(df):
    temp = None
    for i in range(0,len(df)-1):
        temp = df[i]
        df[i+1] += temp
    return df

def clearDict(dict):
    for key in dict.keys():
        dict[key] = None

def parityPlot(dataML, gcvGT, mode, xLabel, yLabel, title, figureNum):
    printGraphNum(figureNum)

    errList = None

    #read .csv file and run the code
    df = pd.read_csv(dataML)

    #inputs for GT values
    for i in range(0, len(df)):
        df.loc[i, 'GT'] = gcvGT.loc[i, 'GT']

    if mode == 'Dist':
        #pre-formatting (does not belong in PCG_F_extract)
        features = pd.DataFrame({
            'GT': [], #In %/bin
            'ML': [], #In %/bin
            'Class': [],
            'Bins': []
        })

        #create bins based on length of dataset
        Bins = pd.DataFrame({'Bins': range(0,len(df)*10,10)})
        for i in range(0,len(df)):
            features.loc[i, 'Bins'] = Bins.loc[i, 'Bins']
            features.loc[i, 'Class'] = 'FIRM'

        #append df values to newly formatted dataset, also makes values cumulative and percentage
        totalGT = float(sum(df['GT']))
        for i in range(0, len(df['GT'])):
            df.loc[i, 'GT'] = (df.loc[i, 'GT']/totalGT)*100
            features.loc[i, 'GT'] = df.loc[i, 'GT']
            
        #append df values to newly formatted dataset, also makes values cumulative and percentage
        totalML = float(sum(df['ML']))
        for i in range(0, len(df['ML'])):
            df.loc[i, 'ML'] = (df.loc[i, 'ML']/totalML)*100
            features.loc[i, 'ML'] = df.loc[i, 'ML']
        
        df = features
    elif mode == 'Avg':
        features = {
            'GT': [],
            'ML': [],
            'Class': []
        }
        errList = []
    
        for i in mListML:
            temp = i['Minor']
            mean = np.mean(temp)
            features['ML'].append(mean)
            err = calculateSE(temp); errList.append(err)
            print('Standard Error: ' + str(err))
        for i in mListGT:
            temp = i['Minor']
            mean = np.mean(temp)
            features['GT'].append(mean)
            features['Class'].append('FIRM')
        
        df = pd.DataFrame(features)

    elif mode == 'Count':
        pass

    #calculate R^2 values
    df = calculateRValues(df, 'GT', 'ML')
    print(df)
    
    #auto boundaries
    Gmax = None; Gmin = None
    maxML = max(df['ML']); maxGT = max(df['GT']); minML = min(df['ML']); minGT = min(df['GT'])
    if maxML > maxGT: Gmax = maxML 
    else: Gmax = maxGT
    if minML < minGT: Gmin = minML
    else: Gmin = minGT
    Gmax = autoBoundary(Gmax, 'Max'); Gmin = autoBoundary(Gmin, 'Min')

    #sets up the x=y line
    line = pd.DataFrame({'x': [Gmin,Gmax],'y': [Gmin,Gmax]})

    #plots the graph
    sb.pairplot(x_vars='GT', y_vars='ML', kind='scatter', hue='Class', markers=m, data=df)
    sb.lineplot(x='x', y='y', linestyle='dashed', color='grey', data=line) 
    plt.axis([Gmin, Gmax, Gmin, Gmax]) 
    plt.errorbar(x=df['GT'], y=df['ML'], yerr=errList, fmt='none', capsize=7, elinewidth=2)
    print(errList)
    setLabels(xLabel, yLabel, title)
    #for some reason, if this line of code is at the end, it removes the extra blank graph window. Watch
    #out for this line of code if a blank window is generated
    plt.figure(figureNum)

def barChart(dataML, gcvGT, mode, xLabel, yLabel, title, figureNum):
    printGraphNum(figureNum)
    #read .csv file and run the code
    df = pd.read_csv(dataML)
    gcvGT = pd.DataFrame(gcvGT)

    #pre-formatting (does not belong in PCG_F_extract)
    features = {
        'Quantity': [],
        'GTorML': [],
        'ImageName': []
    } 

    if mode == 'Count':
        #inputs for GT values
        for i in range(0, len(df['GT'])):
            df.loc[i, 'GT'] = gcvGT.loc[i, 'GT']

        #append df values to newly formatted dataset
        GTvalues = df[['GT', 'ImageName']]
        appendItemsReformat(GTvalues, features, 'GT', 'ImageName', 'Quantity', 'ImageName', 'GTorML')
        MLvalues = df[['ML', 'ImageName']]
        appendItemsReformat(MLvalues, features, 'ML', 'ImageName', 'Quantity', 'ImageName', 'GTorML')
        df = pd.DataFrame(features)

    elif mode == 'Average':
        #append GT values
        appendItemsReformat(gcvGT, features, 'GT', 'ImageName', 'Quantity', 'ImageName', 'GTorML')
        #append ML values
        appendItemsReformat(df, features, 'ML', 'ImageName', 'Quantity', 'ImageName', 'GTorML')
        df = pd.DataFrame(features)
    
    print(df)
    #plots the graph
    plt.figure(figureNum)
    sb.barplot(x='ImageName', y='Quantity', hue='GTorML', edgecolor='0', errorbar=('se'), err_kws={"linewidth": 1}, capsize=.20 , data=df)
    setLabels(xLabel, yLabel, title)

def sizeDistGraph(dataML, gcvGT, xLabel, yLabel, title, figureNum):
    printGraphNum(figureNum)
    #read .csv file and run the code
    df = pd.read_csv(dataML)

    #pre-formatting (does not belong in PCG_F_extract)
    features = {
        'Bins': [],
        'Fibril%': [],
        'GTorML': [],
        'Class': []
    }

    #create bins based on length of dataset
    Bins = pd.DataFrame({'Bins': range(0,len(df)*10,10)})

    #inputs for GT values
    for i in range(0, len(df['GT'])):
        df.loc[i, 'GT'] = gcvGT.loc[i, 'GT']

    #append df values to newly formatted dataset, also makes values cumulative and percentage
    totalGT = float(sum(df['GT']))
    for i in range(0, len(df['GT'])):
        df.loc[i, 'GT'] = (df.loc[i, 'GT']/totalGT)*100
    df['GT'] = distCumulative(df['GT'])

    GTvalues = df[['GT', 'Class']]
    appendItemsReformat(GTvalues, features, 'GT', 'Class', 'Fibril%', 'Class', 'GTorML')
    for i in Bins['Bins']:
        features['Bins'].append(i)
    
    #append df values to newly formatted dataset, also makes values cumulative and percentage
    totalML = float(sum(df['ML']))
    for i in range(0, len(df['ML'])):
        df.loc[i, 'ML'] = (df.loc[i, 'ML']/totalML)*100
    df['ML'] = distCumulative(df['ML'])    

    MLvalues = df[['ML', 'Class']]
    appendItemsReformat(MLvalues, features, 'ML', 'Class', 'Fibril%', 'Class', 'GTorML')
    for i in Bins['Bins']:
        features['Bins'].append(i)
    
    df = pd.DataFrame(features)
    print(df)

    #creates a lineplot of the data, sets the hue for the different classes, and provides the values for the size ranges
    plt.figure(figureNum)
    sb.lineplot(x='Bins', y='Fibril%', hue='GTorML', data=df)

    #creates a scatterplot of the data, maps the size ranges to the index, provides different markers
    sb.scatterplot(x='Bins', y='Fibril%', data=df, hue='GTorML', style='GTorML', s=50, markers=m)
    setLabels(xLabel, yLabel, title)

def autoBoundary(input, mode):
    input = int(input)
    numLen = len(str(input))-1
    l1 = str(input)[numLen-1]
    l2 = str(input)[numLen]
    l = int(l1 + l2)
    if l >= 50:
        if mode == 'Max':
            return (int(math.ceil(input / 100.0)) * 100)
        else: 
            return (int(math.ceil(input / 100.0)) * 100) - 100
    else: 
        if mode == 'Max':
            return (int(math.floor(input / 100.0)) * 100) + 100
        else: 
            return (int(math.floor(input / 100.0)) * 100)

def maxWindow():
    figure = plt.gcf()
    figure.set_size_inches(8, 4.5)

############################################
# CODE THAT GENERATES PCGs
############################################

#set the ML values
gcv['csvCountML'] = pathPCG + 'MLcsvs\\fibCountML.csv'
gcv['csvAvgFullML'] = pathPCG + 'MLcsvs\\minorAxisML.csv'
gcv['csvAvgMeansML'] = pathPCG + 'MLcsvs\\minorAxisMeansML.csv'
for i in range(1, len(dListML)+1):
    gcv['csvDist'+str(i)+'ML'] = pathPCG + 'MLcsvs\\dist' + str(i) + '.csv'

#set the GT values 
gcv['csvCountGT'] = fibCountGT
gcv['csvAvgFullGT'] = minorAxisGT
gcv['csvAvgMeansGT'] = minorAxisMeansGT
for i in range(1, len(dListGT)+1):
    gcv['csvDist'+str(i)+'GT'] = dGT['dfd'+str(i)]

#generate parity plots
parityPlot(gcv['csvCountML'], gcv['csvCountGT'], 'Count', 'True Fibril Count (#)', 'Measured Fibril Count (#)', None, 1)
maxWindow()
plt.savefig(figPath + "(1) GT Fibril Count vs ML Fibril Count Parity Plot.png", bbox_inches='tight')
parityPlot(gcv['csvAvgMeansML'], gcv['csvAvgMeansGT'], 'Avg', 'True Fibril Diameter (nm)', 'Measured Fibril Diameter (nm)', None, 2)
maxWindow()
plt.savefig(figPath + "(2) GT Fibril Diameter vs ML Fibril Diameter Parity Plot.png", bbox_inches='tight')

#generate bar charts
barChart(gcv['csvCountML'], gcv['csvCountGT'], 'Count', 'Image Name', 'Number of Fibrils', None, 3)
maxWindow()
plt.savefig(figPath + "(3) GT Fibril Count vs ML Fibril Count Bar Chart.png", bbox_inches='tight')
barChart(gcv['csvAvgFullML'], gcv['csvAvgFullGT'], 'Average', 'Image Name', 'Average Fibril Diameter (nm)', None, 4)
maxWindow()
plt.savefig(figPath + "(4) GT Fibril Diameter vs ML Fibril Diameter Bar Chart.png", bbox_inches='tight')

#generate size distribution graphs
for i in range(1, len(dListML)+1):
    sizeDistGraph(gcv['csvDist'+str(i)+'ML'], gcv['csvDist'+str(i)+'GT'], 'Fibril Diameters (nm)', '% Identified Fibrils (%)', "Image "+str(i), 4+i)
    maxWindow()
    plt.savefig(figPath + "("+str(4+i)+") GT Size Dist vs ML Size Dist for Image "+str(i)+" Cumulative Dist.png", bbox_inches='tight')
for i in range(1, len(dListML)+1):
    parityPlot(gcv['csvDist'+str(i)+'ML'], gcv['csvDist'+str(i)+'GT'], 'Dist', 'True Size Distributions (%/bin)', 'Measured Size Distributions (%/bin)', "Image "+str(i), len(dListML)+4+i)
    maxWindow()
    plt.savefig(figPath + "("+str(len(dListML)+4+i)+") GT Size Dist vs ML Size Dist for Image "+str(i)+" Parity Plot.png", bbox_inches='tight')

print('Remember to open applications in full screen!')
print('Figures will appear in a moment')
plt.show()

############################################
# CREDITS
############################################

#multiple windows solution by: https://stackoverflow.com/questions/5993206/is-it-possible-to-have-multiple-pyplot-windows-or-am-i-limited-to-subplots
#clear dictionary solution by: https://stackoverflow.com/questions/22991888/how-to-reset-all-values-in-a-dictionary
#solution concerning 'SettingWithCopyWarning' by: https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
#new key for dictionary solution by: https://stackoverflow.com/questions/1024847/how-can-i-add-new-keys-to-a-dictionary
#reference for solution for creating new variables: https://stackoverflow.com/questions/6181935/how-do-you-create-different-variable-names-while-in-a-loop
#rounding to nearest 100 solution by: https://stackoverflow.com/questions/8866046/python-round-up-integer-to-next-hundred#:~:text=ceil%20(always%20rounds%20up).&text=Dividing%20by%20100%20first%20and,n%20%3D%203%20)%2C%20etc.
#exporting distribution to .csv solution by: https://forum.image.sc/t/macro-for-saving-bin-and-count-to-csv/22196/3
#creating ImageJ tables solution by: https://forum.image.sc/t/creating-custom-tables-in-imagej/33100/2
#calculating standard error solution by: https://www.statology.org/standard-error-of-mean-python/
#save figure to file solution by: https://stackoverflow.com/questions/9622163/save-plot-to-image-file-instead-of-displaying-it
#maximize window solution by: https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen