'''
Please see https://py.imagej.net/en/latest/Install.html for a guide on how to install
PyImageJ to your computer

Refer to https://github.com/imagej/pyimagej/tree/main/doc 
for using Jupyter Notebook Tutorials in PyImageJ

Video tutorial used for macro writing: https://www.youtube.com/watch?v=VEBQqoDsljM
'''

import imagej
import scyjava
from scyjava import jimport #use Java object/resources in Python
scyjava.config.add_option('-Xmx6g')
#sets imagej to a reproducible version
ij = imagej.init('net.imagej:imagej:2.14.0', mode='interactive')

dataset = ij.io().open('C:\\Users\\Dwoo413\\FIRM-image-analysis\\Testing Dataset for FIRM\\E15\\Export\\S2_B1_21Feb24_010_16.jpg\\')
imp = ij.py.to_imageplus(dataset)

ij.IJ.run(imp, "8-bit", "")
ij.IJ.setAutoThreshold(imp, "Li white")
ij.prefs.blackBackground = True
ij.IJ.run(imp, "Dilate", "")
ij.IJ.run(imp, "Watershed", "")
ij.IJ.run(imp, "Set Scale...", "distance=1 known=2.081 unit=nm")
ij.IJ.run(imp, "Analyze Particles...", "size=50-Infinity circularity=0.85-1.00 show=Outlines display clear summarize overlay")
ij.IJ.saveAs("Results", "C:/Users/Dwoo413/FIRM-image-analysis/Testing Dataset for FIRM/E15/ML csvs/Measurements/Results(1).csv")
ij.IJ.run("Distribution...", "parameter=Minor or=10 and=0-100")
ij.IJ.saveAs("Results", "C:/Users/Dwoo413/FIRM-image-analysis/Testing Dataset for FIRM/E15/ML csvs/Distributions/Minor Distribution(1).csv")



