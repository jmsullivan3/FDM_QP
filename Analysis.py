
# coding: utf-8

# In[1]:

get_ipython().magic('pylab inline')
get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pylab import *
import sys


# In[2]:

#importing my functions in an organized manner
sys.path.insert(0, '/Users/jamiesullivan/Desktop/FDM/prj_FDM')
from Kernels import *
from Laplacians import *
from Cuts import *
from Smoothing import *
from Densities import *
from Gradients import *
from QPressure import *


# In[3]:

#main
#specify DM type
dmType='cdm'
#options are 'cdm', 'fdm21', 'fdm22'

#specify kernel via index
kernelChoice='cubic_bspline'

kernelList = [[cubic_bspline,del_cubic_bspline,lap_cubic_bspline,parth_cubic_bspline], #cubic bspline ->0
              [gaussian,del_gaussian,lap_gaussian,parth_gaussian], #gaussian ->1
              [quartic_bspline,del_quartic_bspline,lap_quartic_bspline,parth_quartic_bspline], #quartic bspline ->2
              [quintic_bspline,del_quintic_bspline,lap_quartic_bspline,parth_quartic_bspline], #quintic bspline ->3
              [poly6,del_poly6,lap_poly6,parth_poly6], #poly6 bspline ->4
              [wend_C2,del_wend_C2,lap_wend_C2,parth_wend_C2], #wendland C2 ->5
              [wend_C4,del_wend_C4,lap_wend_C4,parth_wend_C4], #wendland C4 ->6
              [wend_C6,del_wend_C6,lap_wend_C6,parth_wend_C6]] #wendland C6 ->7

kernelIndex ={'cubic_bspline':0,'gaussian':1,'quartic_bspline':2,'quintic_bspline':3,'poly6':4,              'wend_C2':5,'wend_C4':6,'wend_C6':7}

kernel = kernelList[kernelIndex.get(kernelChoice)][0]
del_kernel = kernelList[kernelIndex.get(kernelChoice)][1]
delsq_kernel = kernelList[kernelIndex.get(kernelChoice)][2]
parth_kernel = kernelList[kernelIndex.get(kernelChoice)][3]
#options are cubic_bspline, gaussian, quartic_bspline, quintic_bspline, poly6, wend_C2, wend_C4, wend_C6
print(kernelChoice)


#specify laplacian

#options are naiveLaplace, brookLaplace, nbLaplace


# In[4]:

#import data
#specify sim run
dataDM = np.loadtxt(dmType+'_dm.txt',skiprows=1)


# In[5]:

#cut function call
cutDataMin,radMin,rMinSearch,cutDataMax,radMax,rMaxSearch = Cuts(dataDM)


# In[6]:

#smoothing function call
hMin,hMax = Smoothing(cutDataMin,cutDataMax,rMinSearch,rMaxSearch)


# In[7]:

#density function call
rhoMin,rhoMax = Densities(kernel, cutDataMin, hMin, cutDataMax, hMax)


# In[8]:

#density gradient and laplacian function call
del_rhoMin, delsq_rhoMin, del_rhoMax, delsq_rhoMax = Gradients(del_kernel,delsq_kernel,cutDataMin,rhoMin,                                                             cutDataMax,rhoMax,hMin,hMax)


# In[9]:

#QP function call
Q_rhoMin,accelQ_rhoMin,Q_rhoMax,accelQ_rhoMax  = QPressure(del_kernel,parth_kernel,rhoMin,rhoMax,                                                 del_rhoMin,del_rhoMax,delsq_rhoMin,delsq_rhoMax,                                                 hMin,hMax,cutDataMin,cutDataMax)#*(1e12)*(1/.67**2) #conversion factor


# In[ ]:

np.savetxt('Qmin.data',Q_rhoMin)
np.savetxt('Qmax.data',Q_rhoMax)

