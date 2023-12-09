import numpy as np
import math
%matplotlib widget
import matplotlib.pyplot as plt
import csv
import matplotlib.cm as cm
from IPython.display import clear_output
from PIL import Image

import os
wdir = "C:\\...\\wdir"
os.chdir(wdir)
small, large, fs = (5,3.5), (9,4), 10
Countries = ["Cambodia",  "Indonesia", "Laos", "Malaysia", "Myanmar", "Philippines", "Singapore", "Thailand", "Vietnam", "KSWS", "CT"]
Islands = ["Karimunjawa", "Lasia", "Maratua", "Simeulue"]

'''Extracts RGB pixels from an image'''
for country in Countries:
    file_data = Image.open(f"C:\\...\\{country}HabMap.png")
    file_data = file_data.convert('RGB'); data = file_data.load() # conversion to RGB
    img = Image.open(f"C:\\Users\\ALKochListon\\Downloads\\LTM\\NonIntrusiveAbundanceEstimation\\HabitabilityV3\\HabMap.png")
    RGBPixels = np.zeros((np.size(img)[0],np.size(img)[1],3))
    for i in range(np.size(img)[0]):
        clear_output(), print(100*(i+1)/np.size(img)[0])
        for j in range(np.size(img)[1]):
            RGBPixels[i][j] = data [i,j]
    RGBPixels = np.array(RGBPixels)
    print(np.shape(RGBPixels))
    np.save(f"{country}RGBPixels.npy",RGBPixels)


for island in Islands:
    file_data = Image.open(f"C:\\Users\\ALKochListon\\Downloads\\LTM\\NonIntrusiveAbundanceEstimation\\BiancaSplitV3\\{island}HabMap.png")
    file_data = file_data.convert('RGB'); data = file_data.load() # conversion to RGB
    img = Image.open(f"C:\\Users\\ALKochListon\\Downloads\\LTM\\NonIntrusiveAbundanceEstimation\\HabitabilityV3\\HabMap.png")
    RGBPixels = np.zeros((np.size(img)[0],np.size(img)[1],3))
    for i in range(np.size(img)[0]):
        clear_output(), print(100*(i+1)/np.size(img)[0])
        for j in range(np.size(img)[1]):
            RGBPixels[i][j] = data[i,j] 
    RGBPixels = np.array(RGBPixels)
    print(np.shape(RGBPixels))
    np.save(f"{island}RGBPixels.npy",RGBPixels)

# RGBPixels = np.load("RGBPixels.npy")
# KSWSRGBPixels = np.load("KSWSRGBPixels.npy")
# CTRGBPixels = np.load("CTRGBPixels.npy")
# np.shape(RGBPixels), np.shape(KSWSRGBPixels), np.shape(CTRGBPixels)
