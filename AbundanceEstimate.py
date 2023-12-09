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
np.shape(HabSig), np.shape(KSWSHabSig), np.shape(CTHabSig)



'''Associates Signals to Expected Population and Averages spots of equal habitability'''
Exp_NgivenS = np.load("Exp_NgivenS0.523813.npy")
KSWSPopValue = 1566
Exp_NgivenS = np.load("Exp_NgivenS0.0.npy")
KSWSPopValue = 792
Exp_NgivenS = np.load("Exp_NgivenS0.999999.npy")
KSWSPopValue = 3097

HabPop = np.zeros((len(HabSig),2))
for ihabsig, habsig in enumerate(HabSig):
    HabPop[ihabsig][:] = [habsig[0],Exp_NgivenS[int(habsig[1])]]

HabSet = np.array(list(set(HabPop[:,0])))
HabSet = HabSet[np.argsort(HabSet)]
HabPopSpots = np.zeros((len(HabSet),3))

for ihabset, habset in enumerate(HabSet):
    spots = np.where(HabPop[:,0] == habset)[0]
    pop = np.mean(HabPop[spots],axis=0)[1]
    HabPopSpots[ihabset] = [habset,pop,len(spots)]

np.save("HabPopSpots.npy",HabPopSpots)
HabPopSpots = np.load("HabPopSpots.npy")


'''Scans and saves all possible thresholds for mean-filter'''
matplotlib.use('Agg')

iDiffusion = np.argmin(np.abs(HabPopSpots[:, 0, None] - ColGrad[:, -1][::-1][1:]), axis=1)
ThresholdMaximum = lenColGrad-1
SmoothingFactor = 0.1
MegaMeanZipSpline = np.zeros((lenColGrad,lenColGrad))
for sanity in range(2,lenColGrad):
    PixelCount = np.load("MegaMeanFilterPxCount.npy")[sanity] #Using the pixel count for a sanity MeanFieldFilter
    PixelDiffusion = PixelCount[iDiffusion] #Obtains the diffusion for all habitabilities in HabPopSpots

    Diffused_HabPop = np.zeros((len(HabPopSpots),2))
    for ihabpopspot, habpopspot in enumerate(HabPopSpots): #For every signal in HabPopSpots
        Diffused_HabPop[ihabpopspot] = habpopspot[0], habpopspot[1]*(habpopspot[2]/PixelDiffusion[ihabpopspot])
            #Calculates the diffused populaiton as = (mean population_h)*(number of signal spots_h)
                                                                    # /(MeanFiltered number of spots_h)

    '''Applies MeanFieldFilter for the LTMs per Pixel Spot according to Habitability given relation above'''
    X,Y = Diffused_HabPop[:,0],Diffused_HabPop[:,1] 
    MegaMeanFilter_DiffusedHabPop = np.zeros((ThresholdMaximum+1, len(iDiffusion)))
    '''Applies MeanFilter of given Thresholds to XY relation'''
    for threshold in range(ThresholdMaximum+1):
        MeanFieldThreshold, MeanFilterY = (threshold/(lenColGrad-1)), []
        
        for x,y in zip(X,Y):
            MeanFilterY.append(np.mean(Y[(np.abs(X - x) <= MeanFieldThreshold)]))
        MegaMeanFilter_DiffusedHabPop[threshold] = MeanFilterY

    '''Using MeanFilters above, constructs associated splines for entire ColGrad'''
    MegaSplines = np.zeros((ThresholdMaximum+1,2,lenColGrad-1))
    for threshold in range(ThresholdMaximum+1):
        USpline = UnivariateSpline(Diffused_HabPop[:,0],MegaMeanFilter_DiffusedHabPop[threshold],
                                s = ThresholdMaximum/(SmoothingFactor*lenColGrad*threshold) if threshold != 0 else 5, 
                                k=2, ext=0)
        MegaSplines[threshold] = ColGrad[:,-1][::-1][1:], USpline(ColGrad[:,-1][::-1][1:])


    '''Using KSWSMap, identifies the IDFactos for each Spline and obtains its Mean'''
    IDFactors = KSWSPopValue / np.sum(MegaSplines[:,-1,KSWSiHab],axis=1)
    MeanZipSpline = np.mean([(IDfactor*Spline[-1,:]) for IDfactor, Spline in zip(IDFactors,MegaSplines)],axis=1)
    MegaMeanZipSpline[sanity] = MeanZipSpline
    
    fig, ax = plt.subplots(figsize=(9,4))
    ax.tick_params(direction='in')
    ax.set_xlabel("Habitat Preference",fontsize=1.25*fs)
    ax.set_ylabel("Expected Number of LTMs per Spot",fontsize=1.25*fs)
    plt.title(f"Threshold = {np.round(100*sanity/(lenColGrad-1),2)}%",fontsize=1.25*fs)

    plt.plot(ColGrad[:,-1][::-1], MeanZipSpline, 'g', linewidth = 0.1*fs)
    plt.show()

    plt.savefig(f"C:\\...\\wdir\\wdirPlots\\Threshold = {sanity}.png", dpi=300, bbox_inches='tight')
    plt.close()
    clear_output()
    
    '''Using Mean Spline, prints populations for Map and KSWS'''
    print("\nMap", np.sum(MeanZipSpline[MapiHab]) )
    print("KSWS", np.sum(MeanZipSpline[KSWSiHab]),)
    print("CT", np.sum(MeanZipSpline[CTiHab]),"\n")
    FinalSum = 0
    for icountry, country in enumerate(Countries):
            countryPopulation = np.sum(MeanZipSpline[CountriesiHab[icountry]])
            print(country, countryPopulation)
            FinalSum += countryPopulation 
    print("\nCountries Sum", FinalSum)
    print(100*sanity/lenColGrad)

np.save("MegaMeanZipSpline0.523813.npy", MegaMeanZipSpline)
np.save("MegaMeanZipSpline0.0.npy", MegaMeanZipSpline)
np.save("MegaMeanZipSpline0.999999.npy", MegaMeanZipSpline)


'''Uses average of scanning thresholds to obtain estimate wildlife abundance'''
fig, ax = plt.subplots(figsize=large)
ax.tick_params(direction='in')
ax.set_xlabel("Habitat Preference",fontsize=1.25*fs)
ax.set_ylabel("Expected LTMs per Spot (individuals)",fontsize=1.25*fs)
plt.title(f"LTM Population according to Habitat Preference",fontsize=1.25*fs)

MegaMeanZipSpline = np.load("MegaMeanZipSpline0.999999.npy")
MeanMeanZipSpline = np.mean(MegaMeanZipSpline[1:], axis=0)
plt.plot(ColGrad[1:,-1][::-1], MeanMeanZipSpline[1:], 'r', linewidth = 0.2*fs, label = "Upper Bound, $ρ_{upper}$")

MegaMeanZipSpline = np.load("MegaMeanZipSpline0.523813.npy")
MeanMeanZipSpline = np.mean(MegaMeanZipSpline[1:], axis=0)
plt.plot(ColGrad[1:,-1][::-1], MeanMeanZipSpline[1:], 'g', linewidth = 0.2*fs, label = "Best Estimation, $ρ_{sanctuary}$")

MegaMeanZipSpline = np.load("MegaMeanZipSpline0.0.npy")
MeanMeanZipSpline = np.mean(MegaMeanZipSpline[1:], axis=0)
plt.plot(ColGrad[1:,-1][::-1], MeanMeanZipSpline[1:], 'b', linewidth = 0.2*fs, label = "Lower Bound, $ρ_{lower}$")

plt.legend()

plt.savefig(f"C:\\...\\wdir\\Malaysia_MeanSplines.png", dpi=600, bbox_inches='tight')

plt.show()

MegaMeanZipSpline = np.load("MegaMeanZipSpline0.0.npy")
MeanMeanZipSpline = np.mean(MegaMeanZipSpline[1:], axis=0)
a = np.max(MeanMeanZipSpline) / 1.951 
print(a)
print(a*1.952)

MegaMeanZipSpline = np.load("MegaMeanZipSpline0.523813.npy")
MeanMeanZipSpline = np.mean(MegaMeanZipSpline[1:], axis=0)
b= np.max(MeanMeanZipSpline) / 1.951 
print(b)
print(b*1.952)

MegaMeanZipSpline = np.load("MegaMeanZipSpline0.999999.npy")
MeanMeanZipSpline = np.mean(MegaMeanZipSpline[1:], axis=0)

c = np.max(MeanMeanZipSpline) / 1.951 

print(c)
print(c*1.952)


'''Computes abudance per region'''
MegaMeanZipSpline = np.load("MegaMeanZipSpline0.523813.npy")
A = np.mean(MegaMeanZipSpline[1:], axis=0)

MegaMeanZipSpline = np.load("MegaMeanZipSpline0.999999.npy")
B = np.mean(MegaMeanZipSpline[1:], axis=0)

MegaMeanZipSpline = np.load("MegaMeanZipSpline0.0.npy")
C = np.mean(MegaMeanZipSpline[1:], axis=0)



Populations =[]


for j in range(0,3):
        Populations.append([])
        for i in [A,B,C][j:j+1]:
                MeanZipSpline = i
                print("\nMap", np.sum(MeanZipSpline[MapiHab]) )
                print("KSWS", np.sum(MeanZipSpline[KSWSiHab]),)
                print("CT", np.sum(MeanZipSpline[CTiHab]),"\n")
                
                FinalSum = 0
                
                for icountry, country in enumerate(Countries):
                        countryPopulation = np.sum(MeanZipSpline[CountriesiHab[icountry]])
                        print(country, int(np.round(countryPopulation)), int(np.round(countryPopulation**0.5)))
                        Populations[j].append(countryPopulation)
                        FinalSum += countryPopulation

                for iisland, island in enumerate(Islands):
                        islandPopulation = np.sum(MeanZipSpline[IslandsiHab[iisland]])
                        print(island, int(np.round(islandPopulation)), int(np.round(islandPopulation**0.5)))
                        Populations[j].append(islandPopulation)
                        FinalSum += islandPopulation

                print("\nCountries Sum", FinalSum,"\n\n\n")

# Create a vertical bar plot
LowerPopulation, OptimalPopulation, UpperPopulation = Populations[2], Populations[0], Populations[1]

LowerPlot = np.array([x/1e3 for x in LowerPopulation])
OptimalPlot = np.array([x/1e3 for x in OptimalPopulation])
UpperPlot = np.array([x/1e3 for x in UpperPopulation])
Countries = np.array(Countries)
iCountriesFocus = [0,2,3,-3]

plt.figure(figsize=large)
plt.bar(Countries[iCountriesFocus], LowerPlot[iCountriesFocus], label="Lower Bound, $ρ_{lower}$", color='b')
plt.bar(Countries[iCountriesFocus], OptimalPlot[iCountriesFocus] , label="Best Estimation, $ρ_{sanctuary}$", bottom=LowerPlot[iCountriesFocus], color='g')
plt.bar(Countries[iCountriesFocus], UpperPlot[iCountriesFocus], label="Upper Bound, $ρ_{upper}$", bottom=[LowerPlot[i] + OptimalPlot[i] for i in iCountriesFocus], color='r')

plt.xlabel('Countries',fontsize=1.25*fs)
plt.ylabel('$M.$ $fascicularis$ Abundance (thousands)',fontsize=1.25*fs)
plt.title('Abundance Estimation of $M.$ $fascicularis$ across High Confidence Countries',fontsize=1.25*fs)
plt.legend()
plt.show()


iCountriesFocus = [1,4,5,7]

plt.figure(figsize=large)
plt.bar(Countries[iCountriesFocus], LowerPlot[iCountriesFocus], label="Lower Bound, $ρ_{lower}$", color='b')
plt.bar(Countries[iCountriesFocus], OptimalPlot[iCountriesFocus] , label="Best Estimation, $ρ_{sanctuary}$", bottom=LowerPlot[iCountriesFocus], color='g')
plt.bar(Countries[iCountriesFocus], UpperPlot[iCountriesFocus], label="Upper Bound, $ρ_{upper}$", bottom=[LowerPlot[i] + OptimalPlot[i] for i in iCountriesFocus], color='r')

plt.xlabel('Countries',fontsize=1.25*fs)
plt.ylabel('$M.$ $fascicularis$ Abundance (thousands)',fontsize=1.25*fs)
plt.title('Abundance Estimation of $M.$ $fascicularis$ across Lower Confidence Countries',fontsize=1.25*fs)
plt.legend()
plt.show()



'''Plots total result curves'''
fig, ax = plt.subplots(figsize=(9,4))
ax.tick_params(direction='in')
ax.set_xlabel("Habitat Preference",fontsize=1.25*fs)
ax.set_ylabel("Expected Number of LTMs per Spot\n(individuals)",fontsize=1.25*fs)
plt.title(f"LTM Population according to Habitat Preference",fontsize=1.25*fs)

MegaMeanZipSpline = np.load("MegaMeanZipSpline0.0.npy")
MeanMeanZipSpline = np.mean(MegaMeanZipSpline[1:], axis=0)
plt.plot(ColGrad[1:,-1][::-1], MeanMeanZipSpline[1:], 'b', linewidth = 0.2*fs, label = "Lower Bound, $ρ_{lower}$")

plt.legend()

plt.savefig(f"C:\\...\\Plots\\Lower.png", dpi=600, bbox_inches='tight')

plt.show()
np.mean(MeanMeanZipSpline)

fig, ax = plt.subplots(figsize=(9,4))
ax.tick_params(direction='in')
ax.set_xlabel("Habitat Preference",fontsize=1.25*fs)
ax.set_ylabel("Expected Number of LTMs per Spot\n(individuals)",fontsize=1.25*fs)
plt.title(f"LTM Population according to Habitat Preference",fontsize=1.25*fs)

MegaMeanZipSpline = np.load("MegaMeanZipSpline0.523813.npy")
MeanMeanZipSpline = np.mean(MegaMeanZipSpline[1:], axis=0)
plt.plot(ColGrad[1:,-1][::-1], MeanMeanZipSpline[1:], 'g', linewidth = 0.2*fs, label = "Best Estimation, $ρ_{opt}$")

plt.legend()

plt.savefig(f"C:\\...\\Plots\\Best.png", dpi=600, bbox_inches='tight')

plt.show()
np.mean(MeanMeanZipSpline)

fig, ax = plt.subplots(figsize=(9,4))
ax.tick_params(direction='in')
ax.set_xlabel("Habitat Preference",fontsize=1.25*fs)
ax.set_ylabel("Expected Number of LTMs per Spot\n(individuals)",fontsize=1.25*fs)
plt.title(f"LTM Population according to Habitat Preference",fontsize=1.25*fs)

MegaMeanZipSpline = np.load("MegaMeanZipSpline0.999999.npy")
MeanMeanZipSpline = np.mean(MegaMeanZipSpline[1:], axis=0)
plt.plot(ColGrad[1:,-1][::-1], MeanMeanZipSpline[1:], 'r', linewidth = 0.2*fs, label = "Upper Bound, $ρ_{upper}$")

plt.legend()

plt.savefig(f"C:\\...\\Plots\\Upper.png", dpi=600, bbox_inches='tight')

plt.show()
np.mean(MeanMeanZipSpline)
