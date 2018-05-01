
# coding: utf-8

# In[ ]:

get_ipython().magic('pylab inline')
get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pylab import *


# In[ ]:

def QPressure(del_kernel,parth_kernel,rhoMin,rhoMax,del_rhoMin,del_rhoMax,delsq_rhoMin,delsq_rhoMax,hminSmooth,hmaxSmooth,cutDataMin,cutDataMax):
    #QP accel
    accelQ_rhoMin=np.zeros(rhoMin.shape[0])
    accelQ_rhoMax=np.zeros(rhoMax.shape[0])
    Q_rhoMin=np.zeros(rhoMin.shape[0])
    Q_rhoMax=np.zeros(rhoMax.shape[0])

    #constants
    hbar = 6.582e-16 #in eV s
    #c = 2.99e10 #in cm/s
    c = 9.716e-15 #in Mpc/s
    Mpctopc=1e6
    mx = (1e-22)/((c*Mpctopc)**2) # in eV s^2 pc^-2
    #pctocm = 3.086e18 #1pc in cm
    prefact = (hbar**2)/(2*mx**2) #SHOULD be in pc
    h=0.67

    #min region
    for i in range(del_rhoMin.shape[0]):
        for j in range(del_rhoMin.shape[0]):
            if(i!=j and rhoMin[i]!=0.0 and rhoMin[j]!=0.0):
            
                fMin=1+ (hminSmooth[i]/(3*rhoMin[j]))*parth_kernel([cutDataMin[i,0],cutDataMin[i,1],cutDataMin[i,2]],                                                                       [cutDataMin[j,0],cutDataMin[j,1],cutDataMin[j,2]],                                                                       hminSmooth[i])*cutDataMin[:,6][j]
            
                accelQ_rhoMin[i] += prefact*(cutDataMin[:,6][j]/(fMin*rhoMin[j])                                               )*del_kernel([cutDataMin[i,0],cutDataMin[i,1],cutDataMin[i,2]],                                                            [cutDataMin[j,0],cutDataMin[j,1],cutDataMin[j,2]],                                                            hminSmooth[i])*(delsq_rhoMin[j]/(2*rhoMin[j])                                                            - (np.abs(del_rhoMin[j])**2)/(4*rhoMin[j]**2))
        if(i!=j and rhoMin[i]!=0.0 and rhoMin[j]!=0.0):        
            Q_rhoMin[i] = prefact*(delsq_rhoMin[i]/(2*rhoMin[i])                                                            - (np.abs(del_rhoMin[i])**2)/(4*rhoMin[i]**2))
  

    #max region
    for i in range(del_rhoMax.shape[0]):
        for j in range(del_rhoMax.shape[0]):
            if(i!=j and rhoMax[i]!=0.0 and rhoMax[j]!=0.0):            
                
                fMax=1+ (hmaxSmooth[i]/(3*rhoMax[j]))*parth_kernel([cutDataMax[i,0],cutDataMax[i,1],cutDataMax[i,2]],                                                                    [cutDataMax[j,0],cutDataMax[j,1],cutDataMax[j,2]],                                                                       hmaxSmooth[i])*cutDataMax[:,6][j]
            
                accelQ_rhoMax[i] += prefact*(cutDataMax[:,6][j]/(fMax*rhoMax[j])
                                               )*del_kernel([cutDataMax[i,0],cutDataMax[i,1],cutDataMax[i,2]],\
                                                            [cutDataMax[j,0],cutDataMax[j,1],cutDataMax[j,2]],\
                                                            hmaxSmooth[i])*(delsq_rhoMax[j]/(2*rhoMax[j])\
                                                            - (np.abs(del_rhoMax[j])**2)/(4*rhoMax[j]**2))
            
        if(i!=j and rhoMax[i]!=0.0 and rhoMax[j]!=0.0):
            Q_rhoMax[i] = prefact*(delsq_rhoMax[i]/(2*rhoMax[i])                                                        - (np.abs(del_rhoMax[i])**2)/(4*rhoMax[i]**2))            
            
    return Q_rhoMin,accelQ_rhoMin,Q_rhoMax,accelQ_rhoMax 

