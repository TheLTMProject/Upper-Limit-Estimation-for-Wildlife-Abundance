import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.cm as cm
import decimal as dm
dm.getcontext().prec = 100
import os

wdir = "C:\\...\\wdir"
os.chdir(wdir)

Main = "C:\\...\\LTM\\"

'''Group Size Distributions'''
def LognormalGroupSize(n,Nmean,Nsd,ProvRt): 
    '''With the mean and varience of a population distribution (random variable n),
    generates the lognormal distribution of that population.
    Scaled by exponenent of 39.16, optimized as per KSWS parametrization.''' 
    m = np.log( Nmean**2 / np.sqrt( Nmean**2 + Nsd**2 ) )
    s = np.sqrt( np.log( 1 + ( Nsd**2 / Nmean**2 ) ) )
    
    return ( (1/n)*(1/s)*(1/np.sqrt(2*np.pi))*np.exp(-(np.log(n)-m)**2 / (2*(s**2))) )**(1/ProvRt)

def Lognormal_P_of_N(Nmin,Nmax,Nmean,Nmean2,Nsd,Nsd2,ProvRt):
    '''Given individuals' range, mean and varience, and total number of groups observed,
    generates the GroupSizeCount array'''
    P_of_N_NonProvisioned = []
    for n in range(Nmin,Nmax+1):
        P_of_N_NonProvisioned.append(LognormalGroupSize(n,Nmean,Nsd,ProvRt) )
        
    P_of_N_Provisioned = []
    for n in range(Nmin,Nmax+1):
        P_of_N_Provisioned.append(ProvRt*LognormalGroupSize(n,Nmean2,Nsd2,ProvRt))
        
    P_of_N = [x+y for (x,y) in zip(P_of_N_NonProvisioned,P_of_N_Provisioned)]
    P_of_N = P_of_N/sum(P_of_N)
    P_of_N = np.insert(P_of_N, (Nmin)*[0], [0])
    P_of_N = [dm.Decimal(P_of_N[n]) for n in range(len(P_of_N))]
    
    return P_of_N


'''Group size distribution data, 
        needs groups size range (smallest and largest number of individuals in all groups), 
            and a ratio between group sizes (non-provisioned vs provisioned)'''
Nmin = 5 #Smallest number of individuals in a group
Nmax = 200 #Largest number of individuals in a group
ProvRt = 0.025092712 #Proportion of non-provisioned to provisioned groups. 
                        #Optimized according to KSWS ~ 1/40 groups are provisioned

# Generates two distribution for two group sizes, provisioned and non-provisioned
Nmean, Nmean2 = 20, 170 #Mean number of individuals in a group
Nsd, Nsd2 = 15, 30 #Standard deviation = sqrt(Varience) of individuals in a group

P_of_N = Lognormal_P_of_N(Nmin,Nmax,Nmean,Nmean2,Nsd,Nsd2,ProvRt)

'''Inquisitiveness data. Must be between 0 and 1:
        0 => No inquisitiveness = All signals represent different individuals
        1 => Full inquisitiveness = All signals are repetition and represent the same individual
        -- with little modification, this can be adjusted so variable rho relates to age of signal
                (Equivalent to making individuals more or less inquisitive as they age)'''

rho =  0.523813 #KSWS parametrized value
rho0 = 0.000000 #Minimum Inquisitiveness => 1 signal = 1 monkey
rho1 = 0.999999 #Maximum Inquisitiveness => n signals = 1 monkey
Rho = [rho0, rho, rho1]


'''Builds a matrix Prob_NgivenS so that Prob_NgivenS[n][s] = Probability of {N = n | S = s}
    according to model inputs (Inquisitiveness factor, rho).'''
N = [dm.Decimal(n) for n in range(0, Nmax+1)]
S = [dm.Decimal(s) for s in range(0, 801)]

Prob_NgivenS = np.zeros((201, 801))
exp = dm.Decimal( 1 - rho )

for s in S[1:]:
    if s%100 == 0:
        print(100*s/800)
    sigma = dm.Decimal(s) * exp
    Norm = np.sum([ P_of_N[int(n)] * (n**sigma) for n in N[1:] ])
    Prob_NgivenS[:, int(s)] = [(P_of_N[int(n)] * (n**sigma) / Norm) for n in N]

np.save("Prob_NgivenS"+str(rho)+".npy",Prob_NgivenS)
Prob_NgivenS = np.load("Prob_NgivenS"+str(rho)+".npy")
print(np.shape(Prob_NgivenS))

'''Computes the expectation values for all possible signals'''
N = range(0,len(Prob_NgivenS[:,0]))
S = range(0,len(Prob_NgivenS[0]))
Exp_NgivenS = []

for s in S:
    Sum = np.sum([n*Prob_NgivenS[n,s] for n in N])
    Exp_NgivenS.append(Sum)
    # print("Expected value of N given S; < N | S =",s,"> =",Sum)
np.save("Exp_NgivenS"+str(rho)+".npy",Exp_NgivenS)
Exp_NgivenS = np.load("Exp_NgivenS"+str(rho)+".npy")
(np.shape(Exp_NgivenS))


'''Plots Prob_NgivenS'''
#Defines plot parameters
N = range(0,len(Prob_NgivenS[:,0]))
S = range(0,len(Prob_NgivenS[0]))
fs = 10 #fontsize
fig, ax = plt.subplots(figsize=(5,3.5))
ax.tick_params(direction='in')

#Generates x and y values for the plot
PlotProb_NgivenS = 100*np.transpose(Prob_NgivenS) #TRANSPOSED ONLY FOR PLOTTING PURPOSES
N_values = range(len(PlotProb_NgivenS[0]))

# Create the plot
num_colors = len(PlotProb_NgivenS)
Colors = cm.coolwarm(np.linspace(0, 1, num_colors))
for i in range(len(PlotProb_NgivenS)):
    if i in [1,20,40,60,120,240,480,720]:
        plt.plot(N_values, PlotProb_NgivenS[i], color=Colors[i],label="%i sigs"%(i), linewidth=0.05*fs)
    else:
        plt.plot(N_values, PlotProb_NgivenS[i], color=Colors[i], linewidth=0.05*fs)

#Sets limits, labels and title
ax.set_xticks(range(0,len(N)+1,10),labels=range(0,len(N)+1,10),fontsize=1.25*fs)
ax.set_xlabel("Number of Individuals, N = n",fontsize=1.25*fs)
ax.set_ylabel("P{N|S}",fontsize=1.25*fs)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.tick_params(axis='x', labelsize=0.65*fs), ax.tick_params(axis='y', labelsize=fs) 
ax.legend(fontsize=0.75*fs)
ax.set_title("Conditional Probability of N individuals given S signals",fontsize=1.25*fs)

plt.savefig(Main + "Prob_NgivenS"+str(rho)+".png", dpi=600, bbox_inches='tight')
plt.show() # Show the plot

'''Plots the expectation values for each possible signal from data.'''
#Defines plot parameters
fs = 10 #fontsize
fig, ax = plt.subplots(figsize=(5,3.5))
ax.tick_params(direction='in')

# Create the plot
PlotExp_NgivenS = Exp_NgivenS[:202]
num_colors = len(PlotExp_NgivenS)
Colors = cm.coolwarm(np.linspace(0, 1, num_colors))
for i in range(1,201):
    if i in [1,10,20,30,40,50,60,120,200]:
        plt.scatter(i, PlotExp_NgivenS[i], color=Colors[i], s = 0.5*fs)
        plt.vlines(i, ymin=0, ymax=PlotExp_NgivenS[i], color=Colors[i], label="%i sigs"%(i), linewidth=0.1*fs)
    else:
        plt.scatter(i, PlotExp_NgivenS[i], color=Colors[i], s = 0.5*fs)
        plt.vlines(i, ymin=0, ymax=PlotExp_NgivenS[i], color=Colors[i], linewidth=0.1*fs)

        
#Sets limits, labels and title
ax.set_xlim(0, len(PlotExp_NgivenS)-1)
ax.set_xticks(range(0,len(PlotExp_NgivenS),20),labels=range(0,len(PlotExp_NgivenS),20))
ax.set_xticks(range(0,201,10),labels=range(0,201,10))
ax.tick_params(axis='x', labelsize=0.5*fs) 
ax.tick_params(axis='y', labelsize=fs)  
ax.set_ylim(0,1.1*max(PlotExp_NgivenS))
ax.set_xlabel("Number of Signals, S = sigs",fontsize=1.25*fs)
ax.set_ylabel("<N|S>",fontsize=1.25*fs)
ax.legend(fontsize=0.75*fs)
ax.set_title("Conditional Expectation of N given S",fontsize=1.25*fs)

# Show the plot
plt.savefig(Main + "Exp_NgivenS"+str(rho)+".png", dpi=600, bbox_inches='tight')
plt.show()



KSWSLatLonSigT = np.load("KSWSLatLonSigT.npy",allow_pickle=True) #Loads specific data for a region, as produced by DataTreatment.py
CTLatLonSigT = np.load("CTLatLonSigT.npy",allow_pickle=True)

'''Estimates the population of a specific data set.'''
Population = 0
LatLonSigT = CTLatLonSigT

for e in LatLonSigT:
    Population += Exp_NgivenS[e[2]]
print("From these",sum(LatLonSigT[:,2]),"signals, spread across",len(LatLonSigT),"locations,\n"+
      "we estimate a population of", int(np.round(Population)),"individuals:",Population)

'''Based on the given inquisitiveness rho and group size distribution,
    computes the percentage of LTMs that are represented explicitly in the data'''

IndivsTotal = 0
for i in range(1,Nmax+1):
    IndivsTotal += P_of_N[i]*i
IndivsPartcip = 0
for i in range(1,Nmax+1):
    IndivsPartcip += P_of_N[i]*dm.Decimal(i*(i**(1-rho) / i))
print("For a rho =",np.round(rho,3),"and the given distribution of group sizes",
      "\napproximately",(np.round(float(100*IndivsPartcip/IndivsTotal),2)),"% of signals represent singular individuals.")
