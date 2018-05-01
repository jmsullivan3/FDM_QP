
# coding: utf-8

# In[1]:

get_ipython().magic('pylab inline')
get_ipython().magic('matplotlib inline')
import numpy as np
import numpy as np
import matplotlib
import sys
import matplotlib.pyplot as plt
from pylab import *
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from Kernels import *


# In[ ]:

#distance between two particles
def norm(pos1,pos2):
    #return np.sqrt((pos1-pos2)**2)
     return np.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2 + (pos1[2]-pos2[2])**2)

# In[ ]:

#naive approx (Nori and Baldi Eqn. 15)
def naiveLaplace(delsq_kernel,pos1,pos2,hsmooth,mass2,rho1,rho2):
    r = norm(pos1,pos2)
    for j in range(pos1.shape[0]):
        delsqrho1 += mass2[j]*delsq_window(pos1,pos2[j],hsmooth)

    return delsqrho1, delrho1


# In[ ]:

#brookshaw correction (Nori and Baldi Eqn. 18)
def brookLaplace(del_kernel,pos1,pos2,hsmooth,mass2,rho1,rho2):
    #passes kernel gradient option, necessary parameters
    r = norm(pos1,pos2)
    for j in range(pos1.shape[0]):
        delsqrho1 += -2*mass2[j]*del_kernel(pos1,pos2[j],hsmooth)*((rho2[j]-rho1)/rho2[j])/r
        #not sure if this expression is quite right, should double check
    return delsqrho1, delrho1


# In[ ]:

#Nori and Baldi correction (Nori and Baldi Eqn. 19)
#really this is eqn. 24
def nbLaplace(del_kernel,delsq_kernel,pos1,pos2,hsmooth,mass2,rho1,rho2):
    #i will be looped over somewhere else, here just looping over j
    #so this function returns the laplacian for ONE PARTICLE
    #pos1 is a SINGLE POSITION
    #pos2 is an ARRAY of positions
    #hsmooth is a SINGLE smoothing length
    #mass2 is an ARRAY of masses
    #rho1 is a single density
    #rho2 is an ARRAY of densities
    #kernel is the choice of kernel
    delrho1=np.zeros(pos1.shape[0])
    delsqrho1=np.zeros(pos1.shape[0])
    for j in range(pos1.shape[0]):
        delrho1 += mass2[j]*del_kernel(pos1,pos2[j],hsmooth)*(rho2[j]-rho1)/np.sqrt(rho2[j]*rho1)
        delsqrho1 += mass2[j]*delsq_kernel(pos1,pos2[j],hsmooth)*(rho2[j]-rho1)/np.sqrt(rho2[j]*rho1)                     - (np.absolute(delrho1)**2)/rho1

    return delsqrho1, delrho1

