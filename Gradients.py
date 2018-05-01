
# coding: utf-8

# In[ ]:

get_ipython().magic('pylab inline')
get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pylab import *


# In[ ]:

def Gradients(del_kernel,delsq_kernel,cutDataMin,rhoMin,cutDataMax,rhoMax,hminSmooth,hmaxSmooth):
    #now for del and delsq rho

    del_rhoMin=np.zeros(rhoMin.shape[0])
    del_rhoMax=np.zeros(rhoMax.shape[0])
    delsq_rhoMin=np.zeros(rhoMin.shape[0])
    delsq_rhoMax=np.zeros(rhoMax.shape[0])

    #min region
    for i in range(del_rhoMin.shape[0]):
        for j in range(del_rhoMin.shape[0]):
            if(i!=j and rhoMin[i]!=0.0 and rhoMin[j]!=0.0):
                del_rhoMin[i] += cutDataMin[:,6][j]*del_kernel([cutDataMin[i,0],cutDataMin[i,1],cutDataMin[i,2]],                                                            [cutDataMin[j,0],cutDataMin[j,1],cutDataMin[j,2]],                                                            hminSmooth[i])*(rhoMin[j]-rhoMin[i])/np.sqrt(                                                            rhoMin[j]*rhoMin[i])
                delsq_rhoMin[i] += cutDataMin[:,6][j]*delsq_kernel([cutDataMin[i,0],cutDataMin[i,1],cutDataMin[i,2]],                                                                [cutDataMin[j,0],cutDataMin[j,1],cutDataMin[j,2]],                                                                hminSmooth[i])*(rhoMin[j]-rhoMin[i])/np.sqrt(                                                                rhoMin[j]*rhoMin[i]) - (np.absolute(del_rhoMin[i])**2)/(rhoMin[i])
    #now max region
    for i in range(del_rhoMax.shape[0]):
        for j in range(del_rhoMax.shape[0]):
            if(i!=j and rhoMax[i]!=0.0 and rhoMax[j]!=0.0):
                del_rhoMax[i] += cutDataMax[:,6][j]*del_kernel([cutDataMax[i,0],cutDataMax[i,1],cutDataMax[i,2]],                                                            [cutDataMax[j,0],cutDataMax[j,1],cutDataMax[j,2]],                                                            hmaxSmooth[i])*(rhoMax[j]-rhoMax[i])/np.sqrt(                                                            rhoMax[j]*rhoMax[i])
                delsq_rhoMax[i] += cutDataMax[:,6][j]*delsq_kernel([cutDataMax[i,0],cutDataMax[i,1],cutDataMax[i,2]],                                                                [cutDataMax[j,0],cutDataMax[j,1],cutDataMax[j,2]],                                                                hmaxSmooth[i])*(rhoMax[j]-rhoMax[i])/np.sqrt(                                                                rhoMax[j]*rhoMax[i]) - (np.absolute(del_rhoMax[i])**2)/(rhoMax[i])
    return del_rhoMin, delsq_rhoMin, del_rhoMax, delsq_rhoMax

