
# coding: utf-8

# In[20]:

import numpy as np
import numpy as np
import matplotlib
import sys
import matplotlib.pyplot as plt
from pylab import *
import matplotlib.cm as cm
from matplotlib.colors import LogNorm


# In[21]:

#default kernel - cubic B spline (oldstyle)
def default(x1,y1,z1,x2,y2,z2,h):
    #this function takes in the 3D positions of two particles, as well as the smoothing length for the first particle
    #and returns the value of the window function for the first particle (used to determine the density)
    r = diff_dist(x1,y1,z1,x2,y2,z2)
    prefact=(8/(np.pi*h**3))
    if(r/h <=.5 and r/h >0):
        return prefact*(1-6*(r/h)**2 + 6*(r/h)**3)
    elif(r/h>.5 and r/h<1):
        return prefact*2*(1-r/h)**3
    else:
        return 0.0

def del_default(x1,y1,z1,x2,y2,z2,h):
    r = diff_dist(x1,y1,z1,x2,y2,z2)
    prefact=(48/(np.pi*h**6))
    if(r/h <=.5 and r/h >0):
        return prefact*r*(-2*h + 3*r)
    elif(r/h>.5 and r/h<1):
        return prefact*-(h-r)**2
    else:
        return 0.0

def delsq_default(x1,y1,z1,x2,y2,z2,h):
    r = diff_dist(x1,y1,z1,x2,y2,z2)
    prefact=(96/(np.pi*h**6))
    if(r/h <=.5 and r/h >0):
        return prefact*-(h - 3*r)
    elif(r/h>.5 and r/h<1):
        return prefact*(h-r)
    else:
        return 0.0

def partial_h_default(x1,y1,z1,x2,y2,z2,h):
    r = diff_dist(x1,y1,z1,x2,y2,z2)
    if(r/h <=.5 and r/h >0):
        return -(24/(np.pi*h**7))*(h**3-10*h*r**2 +12*r**3)
    elif(r/h>.5 and r/h<1):
        return -(48/(np.pi*h**7))*(h-2*r)*(h-r)**2
    else:
        return 0


# In[22]:

#distance between two particles
def norm(pos1,pos2):
    return np.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2 + (pos1[2]-pos2[2])**2)


# In[23]:

#cubic B-spline (default)
def cubic_bspline(pos1,pos2,h):
#pos1 and pos2 are arrays of 3 elements, the X,Y,Z coords of the particles
#hsmooth is the smoothing length for the first particle
#returns kernel function value for pos1
    r = norm(pos1,pos2)
    prefact=(8/(np.pi*h**3))
    if(r/h <=.5 and r/h >0):
        return prefact*(1-6*(r/h)**2 + 6*(r/h)**3)
    elif(r/h>.5 and r/h<=1):
        return prefact*2*(1-r/h)**3
    else:
        return 0.0

def del_cubic_bspline(pos1,pos2,h):
    r = norm(pos1,pos2)
    prefact=(48/(np.pi*h**6))
    if(r/h <=.5 and r/h >0):
        return prefact*r*(-2*h + 3*r)
    elif(r/h>.5 and r/h<1):
        return prefact*-(h-r)**2
    else:
        return 0.0

def lap_cubic_bspline(pos1,pos2,h):
    r = norm(pos1,pos2)
    prefact=(96/(np.pi*h**6))
    if(r/h <=.5 and r/h >0):
        return prefact*-(h - 3*r)
    elif(r/h>.5 and r/h<1):
        return prefact*(h-r)
    else:
        return 0.0

#partial with respect to h, required for NB laplacian later
def parth_cubic_bspline(pos1,pos2,h):
    r = norm(pos1,pos2)
    if(r/h <=.5 and r/h >0):
        return -(24/(np.pi*h**7))*(h**3-10*h*r**2 +12*r**3)
    elif(r/h>.5 and r/h<1):
        return -(48/(np.pi*h**7))*(h-2*r)*(h-r)**2
    else:
        return 0.0


# In[24]:

#Gaussian (should be slow bc order O(N^2))
def gaussian(pos1,pos2,h):
    r = norm(pos1,pos2)
    prefact=(8*np.sqrt(np.pi))/((np.sqrt(np.pi)*h)**(3))
    return prefact*np.exp(-(9*r/h)**2)

#gaussian gradient
def del_gaussian(pos1,pos2,h):
    r = norm(pos1,pos2)
    prefact=(8*np.sqrt(np.pi))/((np.sqrt(np.pi)*h)**(3))
    return -prefact*(h**-2)*r*18*np.exp(-(9*r/h)**2)

#gaussian laplacian
def lap_gaussian(pos1,pos2,h):
    r = norm(pos1,pos2)
    prefact=(8*np.sqrt(np.pi))/((np.sqrt(np.pi)*h)**(3))
    return -prefact*(h**-4)*18*(h**2 - 18*r**2)*np.exp(-(9*r/h)**2)

#gaussian partial h
def parth_gaussian(pos1,pos2,h):
    r = norm(pos1,pos2)
    prefact=(8*np.sqrt(np.pi))/((np.sqrt(np.pi)*h)**(3))
    return -prefact*(h**-3)*3*(h**2 - 6*r**2)*np.exp(-(9*r/h)**2)


# In[31]:

#quartic B spline
def quartic_bspline(pos1,pos2,h):
#see cubic_bspline comments
#defined on 0 to 1.25 (to achieve same normalization)
    r = norm(pos1,pos2)
    prefact=(256/23)*(1/(20*np.pi))
    if(r/h <=.25 and r/h >0):
        return prefact*((2.5-2*(r/h))**4 -5*(1.5-2*(r/h))**4 +10*(.5-2*(r/h))**4)
    elif(r/h>.25 and r/h<=.75):
        return prefact*((2.5-2*(r/h))**4 -5*(1.5-2*(r/h))**4)
    elif(r/h>.75 and r/h<=1.25):
        return prefact*((2.5-2*(r/h))**4)
    else:
        return 0.0

#quartic B spline gradient
def del_quartic_bspline(pos1,pos2,h):
    r = norm(pos1,pos2)
    prefact=(256/23)
    if(r/h <=.25 and r/h >0):
        return prefact*(6/(5*np.pi))*(h**-4)*r*(-5*(h**2) + 16*r**2)
    elif(r/h>.25 and r/h<=.75):
        return prefact*(1/(10*np.pi))*(h**-4)*(5*(h**3) -120*(h**2)*r + 240*h*r**2 - 128*r**3)
    elif(r/h>.75 and r/h<=1.25):
        return -prefact*(1/(20*np.pi))*(h**-4)*(5*h-4*r)**3
    else:
        return 0.0

#quartic B spline laplacian
def lap_quartic_bspline(pos1,pos2,h):
    r = norm(pos1,pos2)
    prefact=(256/23)
    if(r/h <=.25 and r/h >0):
        return prefact*(6/(5*np.pi))*(h**-4)*(5*(h**2) - 48*r**2)
    elif(r/h>.25 and r/h<=.75):
        return -prefact*(1/(5*np.pi))*(h**-4)*12*(5*(h**2) -20*h*r + 16*r**2)
    elif(r/h>.75 and r/h<=1.25):
        return prefact*(1/(5*np.pi))*(h**-4)*3*(5*h-4*r)**2
    else:
        return 0.0

#quartic B spline partial h
def parth_quartic_bspline(pos1,pos2,h):
    r = norm(pos1,pos2)
    prefact=(256/23)
    if(r/h <=.25 and r/h >0):
        return prefact*(6/(5*np.pi))*(h**-5)*(r**2)*(5*h**2 - 16*r**2)
    elif(r/h>.25 and r/h<=.75):
        return prefact*(1/(10*np.pi))*(h**-5)*r*(-5*h**3 +120*(h**2)*r - 240*h*r**2 + 128*r**3)
    elif(r/h>.75 and r/h<=1.25):
        return prefact*(1/(20*np.pi))*(h**-5)*r*(5*h-4*r)**3
    else:
        return 0.0


# In[32]:

#quintic B spline
def quintic_bspline(pos1,pos2,h):
#see cubic_bspline comments
#defined on 0 to 1.5 (to achieve same normalization)
    r = norm(pos1,pos2)
    prefact=(160/11)*(1/(120*np.pi))
    if(r/h <=.5 and r/h >0):
        return prefact*((3-2*(r/h))**5 -6*(2-2*(r/h))**5 +15*(1.-2*(r/h))**5)
    elif(r/h>0.5 and r/h<=1.0):
        return prefact*((3-2*(r/h))**5 -6*(2-2*(r/h))**5)
    elif(r/h>1.0 and r/h<=1.5):
        return prefact*((3-2*(r/h))**5)
    else:
        return 0.0

#def quintic B spline gradient
def del_quintic_bspline(pos1,pos2,h):
    r = norm(pos1,pos2)
    prefact=(160/11)
    if(r/h <=.5 and r/h >0):
        return -prefact*(4/(3*np.pi))*(h**-5)*(3*r*h**3 -12*h*r**3 +10*r**4)
    elif(r/h<=1.0 and r/h>.5):
        return prefact*(1/(60*np.pi))*(1/h)*(6-5*(3-2*r/h)**4)
    elif(r/h>1.0 and r/h<=1.5):
        return -prefact*(1/(12*np.pi))*(h**-5)*(3*h-2*r)**4
    else:
        return 0.0

#quintic B spline laplacian
def lap_quintic_bspline(pos1,pos2,h):
    r = norm(pos1,pos2)
    prefact=(160/11)
    if(r/h <=.5 and r/h >0):
        return -prefact*(4/(3*np.pi))*(h**-5)*(3*h**3 -36*h*r**2 +40*r**3)
    elif(r/h<=1.0 and r/h>.5):
        return prefact*(2/(3*np.pi))*(h**-5)*(3*h-2*r)**3
    elif(r/h>1.0 and r/h<=1.5):
        return prefact*(2/(3*np.pi))*(h**-5)*(3*h-2*r)**3
    else:
        return 0.0

#quintic B spline partial h
def parth_quintic_bspline(pos1,pos2,h):
    r = norm(pos1,pos2)
    prefact=(160/11)
    if(r/h <=.5 and r/h >0):
        return prefact*(4/(3*np.pi))*(h**-6)*(3*h**3 -12*h*r**2 +10*r**3)
    elif(r/h<=1.0 and r/h>.5):
        return prefact*(1/(60*np.pi))*(h**-2)*r*(-6+5*(3-2*r/h)**4)
    elif(r/h>1.0 and r/h<=1.5):
        return prefact*(1/(12*np.pi))*(h**65)*r*(3*h-2*r)**4
    else:
        return 0.0


# In[33]:

#poly6
def poly6(pos1,pos2,h):
    r = norm(pos1,pos2)
    prefact = 2*4/(np.pi*h**8)
    if(r/h <=1.0):
        return prefact*(h**2 - r**2)**3
    else:
        return 0.0

#poly6 gradient
def del_poly6(pos1,pos2,h):
    r = norm(pos1,pos2)
    prefact = -2*24/(np.pi*h**8)
    if(r/h <=1.0):
        return prefact*r*(h**2 - r**2)**2
    else:
        return 0.0

#poly6 laplacian
def lap_poly6(pos1,pos2,h):
    r = norm(pos1,pos2)
    prefact = -2*48/(np.pi*h**8)
    if(r/h <=1.0):
        return prefact*(h**2 - r**2)*(h**2 - 3*r**2)
    else:
        return 0.0

#poly6 h partial
def parth_poly6(pos1,pos2,h):
    r = norm(pos1,pos2)
    if(r/h <=1.0):
        return -2*(8/np.pi)*(h**-9)*(h**2 - 4*r**2)*(h**2 - r**2)**2
    else:
        return 0.0


# In[28]:

#Wendland Functions - C2
def wend_C2(pos1,pos2,h):
    #defined on 0 to 1
    r = norm(pos1,pos2)
    prefact = (128/21)*(21/(16*np.pi))
    if(r/h <=1.0):
        return prefact*(((1-(r/h))**4)*(1+4*(r/h)))
    else:
        return 0.0

#C2 gradient
def del_wend_C2(pos1,pos2,h):
    r = norm(pos1,pos2)
    prefact = (128/21)
    if(r/h <=1.0):
        return -prefact*(105/(4*np.pi))*(h**-5)*r*(h-r)**3
    else:
        return 0.0

#C2 laplacian
def lap_wend_C2(pos1,pos2,h):
    r = norm(pos1,pos2)
    prefact = (128/21)
    if(r/h <=1.0):
        return -prefact*(105/(4*np.pi))*(h**-5)*(h-4*r)*(h-r)**2
    else:
        return 0.0

#C2 partial h
def parth_wend_C2(pos1,pos2,h):
    r = norm(pos1,pos2)
    prefact = (128/21)
    if(r/h <=1.0):
        return prefact*(105/(4*np.pi))*(h**-6)*(r**2)*(h-r)**3
    else:
        return 0.0


# In[34]:

#Wendland Functions - C4
def wend_C4(pos1,pos2,h):
    r = norm(pos1,pos2)
    prefact = (2048/495)*(495/(256*np.pi))
    if(r/h <=1.0):
        return prefact*(((1-(r/h))**6)*(1+6*(r/h)+(35/3)*(r/h)**2))
    else:
        return 0.0

#C4 gradient
def del_wend_C4(pos1,pos2,h):
    r = norm(pos1,pos2)
    prefact = (2048/495)
    if(r/h <=1.0):
        return -prefact*(1155/32)*(h**-8)*r*(h+5*r)*(h-r)**5
    else:
        return 0.0

#C4 laplacian
def lap_wend_C4(pos1,pos2,h):
    r = norm(pos1,pos2)
    prefact = (2048/495)
    if(r/h <=1.0):
        return -prefact*(1155/32)*(h**-8)*(h**2 + 4*h*r - 35*r**2)*(h-r)**4
    else:
        return 0.0

#C4 partial h
def parth_wend_C4(pos1,pos2,h):
    r = norm(pos1,pos2)
    prefact = (2048/495)
    if(r/h <=1.0):
        return prefact*(1155/32)*(h**-9)*(r**2)*(h+5*r)*(h-r)**5
    else:
        return 0.0


# In[35]:

#Wendland Functions - C6
def wend_C6(pos1,pos2,h):
    r = norm(pos1,pos2)
    prefact = (4096/1365)*(1365/(512*np.pi))
    if(r/h <=1.0):
        return prefact*(((1-(r/h))**8)*(1 + 8*(r/h) + 25*(r/h)**2 + 32*(r/h)**3))
    else:
        return 0.0


#C6 gradient
def del_wend_C6(pos1,pos2,h):
    r = norm(pos1,pos2)
    prefact = (4096/1365)
    if(r/h <=1.0):
        return -prefact*(15015/(256*np.pi))*(h**-11)*r*(h**2 + 7*h*r +16*r**2)*(h-r)**7
    else:
        return 0.0

#C6 laplacian
def lap_wend_C6(pos1,pos2,h):
    r = norm(pos1,pos2)
    prefact = (4096/1365)
    if(r/h <=1.0):
        return -prefact*(15015/(256*np.pi))*(h**-11)*(h**3 + 6*r*h**2 -15*h*r**2 -160*r**3)*(h-r)**6
    else:
        return 0.0

#C6 partial h
def parth_wend_C6(pos1,pos2,h):
    r = norm(pos1,pos2)
    prefact = (4096/1365)
    if(r/h <=1.0):
        return prefact*(15015/(256*np.pi))*(h**-12)*(r**2)*(h**2 + 7*h*r +16*r**2)*(h-r)**7
    else:
        return 0.0


# In[ ]:



