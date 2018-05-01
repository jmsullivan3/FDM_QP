
# coding: utf-8

# In[ ]:

get_ipython().magic('pylab inline')
get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pylab import *


# In[ ]:

def Densities(kernel, cutDataMin, hminSmooth, cutDataMax, hmaxSmooth):
    #calculating density field
    rhoMin=np.zeros(cutDataMin.shape[0])
    rhoMax=np.zeros(cutDataMax.shape[0])

    #not going to deal with this for now, but may come back to gravitational potential for direct comparison
    #in fluid equation
    Phi_rhoMin=np.zeros(rhoMin.shape[0])
    Phi_rhoMax=np.zeros(rhoMax.shape[0])
    G=4.301e-9*((1e-3)**2)*(1/(3.086e16)) #Msun^-1 pc^3 s^-2

    for i in range(rhoMin.shape[0]):
        for j in range(rhoMin.shape[0]):
            rhoMin[i] += cutDataMin[:,6][j]*kernel([cutDataMin[i,0],cutDataMin[i,1],cutDataMin[i,2]],                                                   [cutDataMin[j,0],cutDataMin[j,1],cutDataMin[j,2]],                                                   hminSmooth[i]) #this is m_j * W_ij
            #Phi_rhoMin[i] += -G*cutDataMin[:,6][i]*cutDataMin[:,6][j]*((diff_dist(cutDataMin[i,0],cutDataMin[i,1],cutDataMin[i,2], cutDataMin[j,0],cutDataMin[j,1],cutDataMin[j,2])**2  + cutDataMin[:,8][j]**2)**(-1/2))

    for i in range(rhoMax.shape[0]):
        for j in range(rhoMax.shape[0]):
            rhoMax[i] += cutDataMax[:,6][j]*kernel([cutDataMax[i,0],cutDataMax[i,1],cutDataMax[i,2]],                                                   [cutDataMax[j,0],cutDataMax[j,1],cutDataMax[j,2]],                                                   hmaxSmooth[i]) 
            #Phi_rhoMax[i] += -G*cutDataMax[:,6][i]*cutDataMax[:,6][j]*((diff_dist(cutDataMax[i,0],cutDataMax[i,1],cutDataMax[i,2],cutDataMax[j,0],cutDataMax[j,1],cutDataMax[j,2])**2 + cutDataMax[:,8][j]**2)**(-1/2))
    return rhoMin, rhoMax

