import numpy as np
import math
%matplotlib widget
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
from IPython.display import clear_output
import pickle
from scipy.interpolate import UnivariateSpline
import os
wdir = "C:\\...\\wdir"
os.chdir(wdir)

small, large, fs = (5,3.5), (9,4), 10
Countries = ["Cambodia",  "Indonesia", "Laos", "Malaysia", "Myanmar", "Philippines", "Singapore", "Thailand", "Vietnam", "KSWS", "CT"]
Islands = ["Karimunjawa", "Lasia", "Maratua", "Simeulue"]
IslandsArea = [27, 15, 24, 1844]
IslandsReportedPopulations = [119, 43, 176, 259]

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
    distance = R*c
    return distance


'''Loads arrays from Habitabitat-Preference'''
RGBPixels = np.load("RGBPixels.npy")
KSWSRGBPixels = np.load("KSWSRGBPixels.npy")
CTRGBPixels = np.load("CTRGBPixels.npy")
CountriesRGBPixels = [np.load(f"{country}RGBPixels.npy") for country in Countries]
IslandsRGBPixels = [np.load(f"{island}RGBPixels.npy") for island in Islands]

ColGrad = np.load("ColGrad.npy")
lenColGrad = len(ColGrad)

HabLatLon = np.load("HabLatLon.npy")

I,J = np.where(HabLatLon[:,:,0] != 0)
Map = np.array([HabLatLon[i][j] for i,j in zip(I,J)])
np.save("Map.npy", Map)
Map = np.load("Map.npy")

TempRGBPixels = KSWSRGBPixels
TempI,TempJ = np.where(np.sum(TempRGBPixels,axis=2) != 0)
KSWSMap = np.array([HabLatLon[i][j] for i,j in zip(TempI,TempJ) if HabLatLon[i,j][0] != 0])
KSWSI = np.array([i for i,j in zip(TempI,TempJ) if HabLatLon[i,j][0] != 0])
KSWSJ = np.array([j for i,j in zip(TempI,TempJ) if HabLatLon[i,j][0] != 0])
np.save("KSWSMap.npy", KSWSMap)
KSWSMap = np.load("KSWSMap.npy")

TempRGBPixels = CTRGBPixels
TempI,TempJ = np.where(np.sum(TempRGBPixels,axis=2) != 0)
CTMap = np.array([HabLatLon[i][j] for i,j in zip(TempI,TempJ) if HabLatLon[i,j][0] != 0])
CTI = np.array([i for i,j in zip(TempI,TempJ) if HabLatLon[i,j][0] != 0])
CTJ = np.array([j for i,j in zip(TempI,TempJ) if HabLatLon[i,j][0] != 0])
np.save("CTMap.npy", CTMap)
CTMap = np.load("CTMap.npy")

CountriesMap = []
for icountry, country in enumerate(Countries):
    TempRGBPixels = CountriesRGBPixels[icountry]
    TempI,TempJ = np.where(np.sum(TempRGBPixels,axis=2) != 0)
    countryMap = np.array([HabLatLon[i][j] for i,j in zip(TempI,TempJ) if HabLatLon[i,j][0] != 0])
    countryI = np.array([i for i,j in zip(TempI,TempJ) if HabLatLon[i,j][0] != 0])
    countryJ = np.array([j for i,j in zip(TempI,TempJ) if HabLatLon[i,j][0] != 0])
    CountriesMap.append(countryMap)
with open('CountriesMap.pkl', 'wb') as f:
    pickle.dump(CountriesMap, f)
with open('CountriesMap.pkl', 'rb') as f:
    CountriesMap = pickle.load(f)

IslandsMap = []
for iisland, island in enumerate(Islands):
    TempRGBPixels = IslandsRGBPixels[iisland]
    TempI,TempJ = np.where(np.sum(TempRGBPixels,axis=2) != 0)
    islandMap = np.array([HabLatLon[i][j] for i,j in zip(TempI,TempJ) if HabLatLon[i,j][0] != 0])
    islandI = np.array([i for i,j in zip(TempI,TempJ) if HabLatLon[i,j][0] != 0])
    islandJ = np.array([j for i,j in zip(TempI,TempJ) if HabLatLon[i,j][0] != 0])
    IslandsMap.append(islandMap)
with open('IslandsMap.pkl', 'wb') as f:
    pickle.dump(IslandsMap, f)
with open('IslandsMap.pkl', 'rb') as f:
    IslandsMap = pickle.load(f)

MapiHab = np.argmin(np.abs(Map[:, 0, None] - ColGrad[:, -1][::-1][1:]), axis=1)
np.save("MapiHab.npy",MapiHab)
MapiHab = np.load("MapiHab.npy")
KSWSiHab = np.argmin(np.abs(KSWSMap[:, 0, None] - ColGrad[:, -1][::-1][1:]), axis=1)
np.save("KSWSiHab.npy", KSWSiHab)
KSWSiHab = np.load("KSWSiHab.npy")
CTiHab = np.argmin(np.abs(CTMap[:, 0, None] - ColGrad[:, -1][::-1][1:]), axis=1)
np.save("CTiHab.npy", CTiHab)
CTiHab = np.load("CTiHab.npy")

CountriesiHab = []
for icountrymap, countrymap in enumerate(CountriesMap):
    CountriesiHab.append(np.argmin(np.abs(countrymap[:, 0, None] - ColGrad[:, -1][::-1][1:]), axis=1))
with open('CountriesiHab.pkl', 'wb') as f:
    pickle.dump(CountriesiHab, f)
with open('CountriesiHab.pkl', 'rb') as f:
    CountriesiHab = pickle.load(f)

IslandsiHab = []
for iislandmap, islandmap in enumerate(IslandsMap):
    IslandsiHab.append(np.argmin(np.abs(islandmap[:, 0, None] - ColGrad[:, -1][::-1][1:]), axis=1))
with open('IslandsiHab.pkl', 'wb') as f:
    pickle.dump(IslandsiHab, f)
with open('IslandsiHab.pkl', 'rb') as f:
    IslandsiHab = pickle.load(f)

HabSig = np.load("Malaysia_HabSig.npy")
KSWSHabSig = np.load("KSWSHabSig.npy")
CTHabSig = np.load("CTHabSig.npy")
np.shape(HabSig), np.shape(KSWSHabSig), np.shape(CTHabSig), 
