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
    img = Image.open(f"C:\\...\\HabMap.png")
    RGBPixels = np.zeros((np.size(img)[0],np.size(img)[1],3))
    for i in range(np.size(img)[0]):
        clear_output(), print(100*(i+1)/np.size(img)[0])
        for j in range(np.size(img)[1]):
            RGBPixels[i][j] = data [i,j]
    RGBPixels = np.array(RGBPixels)
    print(np.shape(RGBPixels))
    np.save(f"{country}RGBPixels.npy",RGBPixels)


for island in Islands:
    file_data = Image.open(f"C:\\...\\{island}HabMap.png")
    file_data = file_data.convert('RGB'); data = file_data.load() # conversion to RGB
    img = Image.open(f"C:\\...\\HabMap.png")
    RGBPixels = np.zeros((np.size(img)[0],np.size(img)[1],3))
    for i in range(np.size(img)[0]):
        clear_output(), print(100*(i+1)/np.size(img)[0])
        for j in range(np.size(img)[1]):
            RGBPixels[i][j] = data[i,j] 
    RGBPixels = np.array(RGBPixels)
    print(np.shape(RGBPixels))
    np.save(f"{island}RGBPixels.npy",RGBPixels)

RGBPixels = np.load("RGBPixels.npy")
KSWSRGBPixels = np.load("KSWSRGBPixels.npy")
CTRGBPixels = np.load("CTRGBPixels.npy")
np.shape(RGBPixels), np.shape(KSWSRGBPixels), np.shape(CTRGBPixels)


'''Extracts Color Gradient from an image
    Requires a linear map of colors and its location, in pixels'''
img = Image.open("C:\\...\\ColorGrad.png")
ColGrad = np.zeros((np.size(img)[1],4))

for i in range(np.size(img)[1]):
    ColGrad[i][:3] = img.getpixel((20,i))[:3]
    ColGrad[i][3] = 1-(i/(np.size(img)[1]-1))

ColGrad = np.array(ColGrad)
np.save("ColGrad.npy",ColGrad)
ColGrad = np.load("ColGrad.npy")
lenColGrad = len(ColGrad)
np.shape(ColGrad)


'''Extracts Latitude and Longitude from QGIS .csv output file 
   and sorts it (+i => increasing longitutude; +j => decreasing latitude)'''
File = "C:\\...\\PixLonLat.csv"
with open(File, newline='') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader) #Skip label row
    LatLon = [[float(Row[3]),float(Row[2])] for Row in csvreader]
LatLon = np.array(LatLon)
sortI = np.lexsort((-LatLon[:,0], LatLon[:,1]))
LatLon = LatLon[sortI]
FormLatLon = LatLon.reshape(-1, np.shape(RGBPixels)[1], 2) #Formats LatLon into original map format
        
def ColDis(c1, c2):
    '''Computes the Euclidean "distance" between two RGB colors'''
    return np.linalg.norm(c1[0:3] - c2[0:3])

'''Associates pixels to Habitability and LatLon coordinates.'''
HabLatLon = np.zeros_like(RGBPixels, dtype=float)
lenRGB = len(RGBPixels)

for i, e in enumerate(RGBPixels):
    if (i + 1) % 10 == 0: clear_output(), print(100 * (i + 1) / lenRGB)
    for j, f in enumerate(e):
        if np.array_equal(f, [0, 0, 0]):  # Ignores black (0,0,0) pixels
            HabLatLon[i][j][0] = 0.0
        else:
            habitability = ColGrad[np.argmin([ColDis(c, f) for c in ColGrad])][3]
            HabLatLon[i][j][0] = habitability

HabLatLon[..., 1:] = FormLatLon
np.save("HabLatLon.npy", HabLatLon)

HabLatLon = np.load("HabLatLon.npy")

I,J = np.where(HabLatLon[:,:,0] != 0)
Map = np.array([HabLatLon[i][j] for i,j in zip(I,J)])

TempRGBPixels = KSWSRGBPixels
TempI,TempJ = np.where(np.sum(TempRGBPixels,axis=2) != 0)
KSWSMap = np.array([HabLatLon[i][j] for i,j in zip(TempI,TempJ) if HabLatLon[i,j][0] != 0])
KSWSI = np.array([i for i,j in zip(TempI,TempJ) if HabLatLon[i,j][0] != 0])
KSWSJ = np.array([j for i,j in zip(TempI,TempJ) if HabLatLon[i,j][0] != 0])

TempRGBPixels = CTRGBPixels
TempI,TempJ = np.where(np.sum(TempRGBPixels,axis=2) != 0)
CTMap = np.array([HabLatLon[i][j] for i,j in zip(TempI,TempJ) if HabLatLon[i,j][0] != 0])
CTI = np.array([i for i,j in zip(TempI,TempJ) if HabLatLon[i,j][0] != 0])
CTJ = np.array([j for i,j in zip(TempI,TempJ) if HabLatLon[i,j][0] != 0])


np.shape(HabLatLon), np.shape(Map), np.shape(KSWSMap), np.shape(CTMap), (np.union1d(KSWSMap,Map) == np.union1d(Map,Map)).all(), (np.union1d(CTMap,Map) == np.union1d(Map,Map)).all()



'''Plots historgrams for the Pixel Count according to habitability in entire Map'''
fig, ax = plt.subplots(figsize = large)
ax.tick_params(direction='in')
ax.set_xlabel("Habitat Preference",fontsize=1.25*fs)
ax.set_ylabel("Number of Pixel Spots",fontsize=1.25*fs)
ax.set_title("Histogram of Pixel Count\naccording to Habitability in the Entire Map",fontsize=1.25*fs)

'''Builds Histograms'''
X = ColGrad[:,-1][::-1][1:]
Y = np.histogram(Map[:,0], bins=lenColGrad-1)[0]
# plt.plot(X,Y,label="Histogram function")
# plt.hist(Map[:,0], bins=ColGrad[:,-1][::-1],label="Histogram")

'''Applies MeanFilter of given Thresholds to XY relation'''
ThresholdMaximum, ThresholdInterest = int(lenColGrad)-1, [0,1,4,7,69] # [0,1,3,6,29,57,113,282]
lenColors, iC = len(ThresholdInterest), 0
Colors = cm.coolwarm(np.linspace(0, 1, lenColors))
MegaMeanFilterPxCount = []
# MegaMeanFilterPxCount = np.load("MegaMeanFilterPxCount.npy")
for threshold in range(ThresholdMaximum+1):
    MeanFieldThreshold, MeanFilterPxCount = (threshold/(lenColGrad-1)), []
    for x,y in zip(X,Y):
        MeanFilterPxCount.append(np.mean(Y[(np.abs(X - x) <= MeanFieldThreshold)]))
    MegaMeanFilterPxCount.append(MeanFilterPxCount)
    # MeanFilterPxCount = MegaMeanFilterPxCount[threshold]
    if threshold in ThresholdInterest:
        plt.plot(X,MeanFilterPxCount, color=Colors[iC],linewidth = 0.08*fs,
                 label=f"Threshold = {np.round(100*threshold/(lenColGrad-1),2)}%")
        iC += 1
    # else:
    #     plt.plot(X,MeanFilterY,color=Colors[threshold],linewidth=0.05*fs)
np.save("MegaMeanFilterPxCount.npy",MegaMeanFilterPxCount)

plt.legend()
plt.savefig(MainFolder + 'SpotMekong.png', dpi=600, bbox_inches='tight')
plt.show()


fig, ax = plt.subplots(figsize= large )
ax.tick_params(direction='in')
ax.set_xlabel("Habitat Preference",fontsize=1.25*fs)
ax.set_ylabel("Number of Pixel Spots",fontsize=1.25*fs)

Y = Map[:,0]
Bins = ColGrad[1:][:,-1][::-1]
plt.hist(Y, bins=Bins,label="Entire Map")
Y = KSWSMap[:,0]
Bins = ColGrad[1:][:,-1][::-1]
plt.hist(Y, bins=Bins,label="KSWS")

ax.set_title("Histogram of Pixel Count\naccording to Habitability in the entire Map",fontsize=1.25*fs)
# plt.legend()
plt.show()


'''Plots histogram of areas of interest'''
fig, ax = plt.subplots(figsize= large )
ax.tick_params(direction='in')
ax.set_xlabel("Habitat Preference",fontsize=1.25*fs)
ax.set_ylabel("Number of Pixel Spots",fontsize=1.25*fs)

Y = KSWSMap[:,0]
Bins = ColGrad[1:][:,-1][::-1]
plt.hist(Y, bins=Bins,label="KSWS")

ax.set_title("Histogram of Pixel Count\naccording to Habitability in the KSWS Map",fontsize=1.25*fs)
plt.show()


fig, ax = plt.subplots(figsize= large )
ax.tick_params(direction='in')
ax.set_xlabel("Habitat Preference",fontsize=1.25*fs)
ax.set_ylabel("Number of Pixel Spots",fontsize=1.25*fs)

Y = CTMap[:,0]
Bins = ColGrad[1:][:,-1][::-1]
plt.hist(Y, bins=Bins,label="CT")

ax.set_title("Histogram of Pixel Count\naccording to Habitability in the CT Map",fontsize=1.25*fs)
plt.show()



def DisLatLon(LatLon1, LatLon2):
    '''Computes the surface distance (in km) between two coordinates on Earth'''
    R = 6371  # Radius of the Earth in kilometers
    #Convert latitudes and longitudes to radians
    Lat1, Lon1, Lat2, Lon2 = map(math.radians, [LatLon1[0], LatLon1[1], LatLon2[0], LatLon2[1]])
    #Calculate the differences between the latitudes and longitudes
    dLat = Lat2 - Lat1
    dLon = Lon2 - Lon1
    #Apply the haversine formula
    a = math.sin(dLat/2)**2 + math.cos(Lat1)*math.cos(Lat2)*math.sin(dLon/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R*c

'''Associates each signal to a pixel spot in the map'''
import pickle

CheckPointFile = "Check_HabMeanSig.pk1"
CheckPointValue = "Check_ilatlonsig.pk1"

LatLonSigs = np.load("Malaysia_LatLonSigsTime.npy",allow_pickle=True)
HabLatLon = np.load("HabLatLon.npy")
threshold = 6*DisLatLon(HabLatLon[0,0][1:],HabLatLon[1,1][1:])

HabMeanSig = []
Check_ilatlonsig = 0

CheckPointCheck = "n" # "Fresh Start? [y/n]"
if CheckPointCheck == "n":
    with open(CheckPointFile, 'rb') as chk:
        HabMeanSig = pickle.load(chk)
    with open(CheckPointValue, 'rb') as chk:
        Check_ilatlonsig = pickle.load(chk)

for ilatlonsig, latlonsig in enumerate(LatLonSigs[Check_ilatlonsig:]): #For every signal in data
    clear_output(), print(100*(ilatlonsig+1+Check_ilatlonsig)/len(LatLonSigs))
    # print(HabMeanSig)

    imin, jmin = min([(i,j) for i in range(np.shape(HabLatLon)[0]) #Find the corresponding indices 
                     for j in range(np.shape(HabLatLon)[1])],         #that minmize distance in the map
                        key=lambda x: DisLatLon(HabLatLon[x][1:],latlonsig[:2]) )

    if DisLatLon(HabLatLon[imin, jmin][1:],latlonsig[:2]) < threshold: #If distance between signal and assigned pixel
        HabMean = np.mean(HabLatLon[imin - 2 : imin + 3, jmin - 2 : jmin + 3], axis=(0, 1))[0] # is within five pixels
        
        if HabMean > 0: #If the mean habitability is non-zero (is within a subset of the map)
            HabMeanSig.append([HabMean,latlonsig[2]]) #Associates mean habitability with signal in data
    
    with open(CheckPointFile, 'wb') as chk:
        pickle.dump(HabMeanSig, chk)
    with open(CheckPointValue, 'wb') as chk:
        pickle.dump(ilatlonsig, chk)
 

HabMeanSig = np.array(HabMeanSig)
np.save("HabMeanSig.npy",HabMeanSig)
np.save("Malaysia_HabMeanSig.npy",HabMeanSig)

HabMeanSig = np.load("HabMeanSig.npy")
KSWSHabMeanSig = np.load("KSWSHabMeanSig.npy")
CTHabMeanSig = np.load("CTHabMeanSig.npy")
np.shape(HabMeanSig), np.shape(KSWSHabMeanSig), np.shape(CTHabMeanSig)

'''Maps HabMean values into ColGrad'''
HabSig = HabMeanSig
for ihabsig, habsig in enumerate(HabMeanSig): #For every signal in HabMeanSig
    HabSig[ihabsig][0] = min([h for h in ColGrad[:,-1]], key=lambda h: abs(habsig[0] - h)) #Collects habitability  
                                                                                    # from ColGrad closest to HabMean

HabSig = HabSig[np.argsort(HabSig[:,0])] #Sorts according to habitability
np.save("Malaysia_HabSig.npy",HabSig)

HabSig = np.load("Malaysia_HabSig.npy")
KSWSHabSig = np.load("KSWSHabSig.npy")
CTHabSig = np.load("CTHabSig.npy")
np.shape(HabSig), np.shape(KSWSHabSig), np.shape(CTHabSig) 
