
# coding: utf-8

# In[4]:

import numpy as np

# In[ ]:
def norm(pos1,pos2):
        return np.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2 + (pos1[2]-pos2[2])**2)
#smoothing
def Smoothing(cutDataMin,cutDataMax,rMinSearch,rMaxSearch):
    hminSmooth = np.zeros(shape=cutDataMin.shape[0])
    hmaxSmooth = np.zeros(shape=cutDataMax.shape[0])

    #MIN search
    for i in range(cutDataMin.shape[0]):
        for scale in rMinSearch:
            nn=0
            for j in range(cutDataMin.shape[0]):
                if(i != j):
                    rmin = norm([cutDataMin[i,0],cutDataMin[i,1],cutDataMin[i,2]],                                [cutDataMin[j,0],cutDataMin[j,1],cutDataMin[j,2]])
                    if(rmin<=scale):
                        nn += 1
                    if(nn>63. and nn < 65.):
                        hminSmooth[i] = rmin
                        break
            if(nn>63. and nn < 65.):
                break

    #MAX search
    for i in range(cutDataMax.shape[0]):
        for scale in rMaxSearch:
            nn=0
            for j in range(cutDataMax.shape[0]):
                if(i != j):
                    rmax = norm([cutDataMax[i,0],cutDataMax[i,1],cutDataMax[i,2]],                                [cutDataMax[j,0],cutDataMax[j,1],cutDataMax[j,2]])
                    if(rmax<=scale and rmax >0):
                        nn += 1
                    if(nn>63. and nn < 65. and rmax>0):
                        hmaxSmooth[i] = rmax
                        break
            if(nn>63. and nn < 65.):
                break
    return hminSmooth, hmaxSmooth

