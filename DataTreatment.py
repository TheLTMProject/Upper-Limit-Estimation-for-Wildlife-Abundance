'''This is an example script to convert raw data into a LatLonSigsTime vector, containing the latitude, longitude, number of signals and time of signals. 
It standardizes coordinates to EPSG:4326 and completes any missing data (if number of signals is not specified, it is set as "1").'''
import csv
import numpy as np
import math
import utm
from IPython.display import clear_output
import os

wdir = "C:\\..\\wdir" #Work directory
os.chdir(wdir)

'''Data Processing'''
def DataExtractor(filename):
    '''DEPRECATED AND POORLY CODED.
        Extracts uneven and raw data from .csv file into python
        If the data is provided as UTM coordinates, 
         converts it into Latitude and Longitude coordinates'''
    with open(filename,'r') as file:
        rows = []
        for row in csv.reader(file):
            if len(row[0]) >= 15: #Ignores all entries with less than 15 characters (missing data)
                rows.append(row)
            
    Data = []
    for row in rows:
        print(row[0].split(","))
        if row[0].split(",")[5] == "":
            #print(row[0])
            Data.append([  [float(row[0].split(",")[0][2:]), float(row[0].split(",")[1][1:-1]) ], 
                int(row[0].split(",")[2]) ,
                int(row[0].split(",")[3])   ])
                        
        else:
            #print(row[0])
            UTMRegion = row[0].split(",")[5]
            UTMCoordinates = list(utm.to_latlon( float(row[0].split(",")[0][2:]), 
                                                   float(row[0].split(",")[1][1:-1]),
                                                       int(UTMRegion[0:2]), UTMRegion[2] ))
            #print(UTMCoordinates)
            Data.append([   UTMCoordinates, 
                int(row[0].split(",")[2]) ,
                int(row[0].split(",")[3]),
                row[0].split(",")[5]   ])

    return Data

def DistanceCheck(LatLon1, LatLon2, threshold=0.797884561):
    '''Computes the surface distance (in km) between two coordinates on Earth,
    and checks if the distance exceeds a default threshold = sqrt(2/pi) = radius of a two kilometer^2 disk'''
    R = 6371  # Radius of the Earth in kilometers
    #Convert latitudes and longitudes to radians
    Lat1, Lon1, Lat2, Lon2 = map(math.radians, [LatLon1[0], LatLon1[1], LatLon2[0], LatLon2[1]])
    #Calculate the differences between the latitudes and longitudes
    dLat = Lat2 - Lat1
    dLon = Lon2 - Lon1
    #Apply the haversine formula
    a = math.sin(dLat/2)**2 + math.cos(Lat1)*math.cos(Lat2)*math.sin(dLon/2)**2
    # print(a)
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    #print(distance)
    if distance >= threshold:
        return "Far"
    else:
        return "Close"

def Data_of_Signals(LatLonData,IndivsData,TimeData):
    '''Loops over LatLonData and combines signals close to each other, to avoid double counting.
        Returns LatLonCoordinates, number of signals detected at that a given coordinate and days since detection.'''
    SignalData = [] #List of signals
    DoubleCount = [] #List to track double count
    for i in range(len(LatLonData)):
        clear_output(), print(100*(i+1)/len(LatLonData),"%")
        signal_count = IndivsData[i] #If a signal was registered, it is at least 1
        if i not in DoubleCount:
            for j in range(len(LatLonData)):
                if i < j and j not in DoubleCount:

                    if DistanceCheck(LatLonData[i],LatLonData[j]) == "Close":
                        signal_count += IndivsData[j] #Increase signal count for signals close
                        DoubleCount.append(j) #Ignore close signal for future iterations
                        LatLonData[i] = np.array([ (Data1 + Data2)/2 for Data1, Data2 
                                                 in zip(LatLonData[i],LatLonData[j]) ])
                                            #Averages coordinate locations of signals

                    elif DistanceCheck(LatLonData[i],LatLonData[j]) == "Far":
                        signal_count += 0

            SignalData.append([LatLonData[i][0],LatLonData[i][1],signal_count,TimeData[i]])
            
    Sorted_Signal_indices = np.argsort(np.array(SignalData,dtype=object)[:, 2])
    Sorted_SignalData = [SignalData[i] for i in Sorted_Signal_indices] 
    print("Number of distinct signal sources =",len(Sorted_SignalData))
    return np.array(Sorted_SignalData,dtype=object)

def UTMExtractor(File):
    '''Extracts UTM coordinates from .csv. IGNORES FIRST ROW.'''
    with open(File,'r') as File:
        Rows = []
        for row in csv.reader(File):
            Rows.append(row)
    UTMX = []
    UTMY = []
    for row in Rows[1:]:
        UTMX.append(float(row[0]))
        UTMY.append(float(row[1]))
    return UTMX,UTMY

def Lat_Lon(UTMX,UTMY,UTMRegion):
    '''Converts UTM coordinates to Latitude and Longitude coordinates'''
    LatLon_Data = []
    for x,y in zip(UTMX,UTMY):
        lat, lon = utm.to_latlon(x, y, int(UTMRegion[0:2]), UTMRegion[2])
        LatLon_Data.append([lat,lon])
    return np.array(LatLon_Data)



'''Mapping Signals onto Habitability Map'''
def CoordDistance(LatLon1, LatLon2):
    '''Computes the surface distance (in km) between two coordinates on Earth,'''
    R = 6371  # Radius of the Earth in kilometers
    #Convert latitudes and longitudes to radians
    Lat1, Lon1, Lat2, Lon2 = map(math.radians, [LatLon1[0], LatLon1[1], LatLon2[0], LatLon2[1]])

    #Calculate the differences between the latitudes and longitudes
    dLat = Lat2 - Lat1
    dLon = Lon2 - Lon1

    #Apply the haversine formula
    a = math.sin(dLat/2)**2 + math.cos(Lat1)*math.cos(Lat2)*math.sin(dLon/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    
    return distance

def Progress(i,lenA,Clear=True):
    '''Estimates percetage of a given loop'''
    if Clear:
        clear_output(wait=True); print(100*(i+1)/lenA,"%")
    else:
        print(100*(i+1)/lenA)

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


'''Preprocessing data'''
File = "C:\\...\\BaselineData.csv"

with open(File,'r', encoding="utf8") as File:
    Rows = []
    for row in csv.reader(File):
        Rows.append(row)
LatLon = []
Sigs = []
Time = []
for irow, row in enumerate(Rows[2:]):
    LatLon.append([float(row[0]),float(row[1])])
    Sigs.append(int(row[2]))
    Time.append(row[3])



'''Loops over LatLonData and combines signals close to each other, to avoid double counting.
    Returns LatLonCoordinates, number of signals detected at that a given coordinate and days since detection.'''
LatLonSigsTime = []
DoubleCount = [] #List to track double count
lenLatLon = len(LatLon)
treshold = 0.797884561

for i in range(lenLatLon):
    if i%100 == 0:
        clear_output(), print(100*(i+1)/lenLatLon,"%")
    sig = Sigs[i]
    time = Time[i]
    if i not in DoubleCount:
        for j in range(lenLatLon):
            if i < j and j not in DoubleCount:
                if DisLatLon(LatLon[i],LatLon[j]) <= 0.797884561:
                    sig += Sigs[j] #Increase signal count for nearby signal 
                    DoubleCount.append(j) #Ignore close signal for future iterations
                    LatLon[i] = np.array([ (latlon1 + latlon2)/2 for latlon1, latlon2 
                                             in zip(LatLon[i],LatLon[j]) ]) #Averages coordinate locations of signals

        LatLonSigsTime.append([LatLon[i][0],LatLon[i][1],sig,time])

LatLonSigsTime = np.array(LatLonSigsTime,dtype=object)

for ilatlonsigtime, latlonsigtime in enumerate(LatLonSigsTime): #Caps number of signals to 200
    LatLonSigsTime[ilatlonsigtime][0:2] = [float(entry) for entry in latlonsigtime[0:2]]
    LatLonSigsTime[ilatlonsigtime][2] = int(latlonsigtime[2])
    if LatLonSigsTime[ilatlonsigtime][2] > 200:
        LatLonSigsTime[ilatlonsigtime][2] = 200

LatLonSigsTime = LatLonSigsTime[np.argsort(LatLonSigsTime[:, 2])]

