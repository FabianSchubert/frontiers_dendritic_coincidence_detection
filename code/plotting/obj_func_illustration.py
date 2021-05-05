#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import seaborn as sns
sns.set()
from stdParams import *
from src.plottools import gen_mesh_ax
from src.neuronmodel import *
plt.style.use('mpl_style.mplstyle')
import os
import sys

p = np.linspace(-2.,2.,100)
d = np.linspace(-2.,2.,100)

P,D = np.meshgrid(p,d)

p_ax = gen_mesh_ax(p)
d_ax = gen_mesh_ax(d)

def L(p,d):
    return (.5*(1.-alpha)*((d-thetad)>0.)*(np.maximum(0.,p-thetap1)-np.maximum(0.,-thetap1))
            -((alpha+1.)/2.-alpha**2.)*((d-thetad)<=0.)*np.maximum(0.,p-thetap0))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(P,D,L(P,D),cmap=cm.coolwarm,linewidth=0.2,rstride=6,cstride=6)

ax.set_xlabel(r'$I_{\rm p}$')
ax.set_ylabel(r'$I_{\rm d}$')
ax.set_zlabel(r'$L$')

plt.show()
