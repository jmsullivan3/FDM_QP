
# coding: utf-8

# In[2]:

get_ipython().magic('pylab inline')
get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pylab import *
from Laplacians import norm


# In[3]:

def Cuts(dataDM):
    #density cuts
    #cutting regions around max and min density
    #rn using 1e3 particles around
    rMaxSearch = np.linspace(5,50,10)
    #rMaxSearch = np.linspace(1,10,50)
    rMinSearch = np.linspace(50,500,50)
    #rMaxSearch = np.linspace(10,1000,10)
    #rMinSearch = np.linspace(100,1000,10)
    nthreshMin=1e3
    nthreshMax=1e3
    
    idmMax=np.argmax(dataDM[:,7])
    idmMin=np.argmin(dataDM[:,7][1:])
    #igasMax=np.argmax(dataGas[:,8])
    #igasMin=np.argmin(dataGas[:,8])

    for scale in rMinSearch:        
        nnMin=0
        partMin=[]
        radMin= []
        for j in range(dataDM.shape[0]):
            if(idmMin != j):
                rMin = norm([dataDM[idmMin,0],dataDM[idmMin,1],dataDM[idmMin,2]],                            [dataDM[j,0],dataDM[j,1],dataDM[j,2]])
                if(rMin<=scale):
                    nnMin += 1
                    partMin.append(dataDM[j,:])
                    radMin.append(rMin)
                if(nnMin>=nthreshMin):
                    #print('reached nn. nn = ', nnMin)
                    #print('min cut size = ', rMin)
                    break
        if(nnMin>=nthreshMin):
            #print('break',nnMin)
            break
    
    for scale in rMaxSearch:        
        nnMax=0
        partMax=[]
        radMax = []
        for j in range(dataDM.shape[0]):
            if(idmMax != j):
                rMax = norm([dataDM[idmMax,0],dataDM[idmMax,1],dataDM[idmMax,2]],                            [dataDM[j,0],dataDM[j,1],dataDM[j,2]])
                if(rMax<=scale):
                    nnMax += 1
                    partMax.append(dataDM[j,:])
                    radMax.append(rMax)
                    #print(nnMax)
                if(nnMax>=nthreshMax):
                    #print('reached nn. nn = ', nnMax)
                    #print('max cut size = ', rMax)
                    #hSmooth[i] = r
                    break
        if(nnMax>=nthreshMax):
            break
        


    partMax=np.array(partMax)
    radMax=np.array(radMax)
    partMin=np.array(partMin)
    radMin=np.array(radMin)

    return partMin,radMin,rMinSearch, partMax,radMax,rMaxSearch



# In[ ]:



