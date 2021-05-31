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

############################
xtest = np.random.normal(0.,0.5,(int(1e5)))
ytest = np.random.normal(0.,0.5,(int(1e5)))
print(L(xtest,ytest).mean())
print(L(xtest,xtest).mean())
############################



fig = plt.figure(figsize=(FIG_WIDTH,FIG_WIDTH*0.45))
ax3d = fig.add_subplot(121, projection='3d')
ax2d = fig.add_subplot(122)

ax3d.plot_surface(P,D,L(P,D),linewidth=0.2,rstride=6,cstride=6,cmap=sns.color_palette("rocket", as_cmap=True))

ax3d.set_xlabel(r'$I_p$')
ax3d.set_ylabel(r'$I_d$')
ax3d.set_zlabel(r'$\mathcal{L}_{\rm p}$')

ax3d.set_facecolor((0.,0.,0.,0.))

pcm = ax2d.pcolormesh(p_ax,d_ax,L(P,D))#,cmap=cm.coolwarm)
ax2d.set_xlabel(r'$I_p$')
ax2d.set_ylabel(r'$I_d$')

plt.colorbar(pcm, ax = ax2d)

ax3d.set_title(r'A',loc="left",fontweight="bold")
ax2d.set_title(r'B',loc="left",fontweight="bold")

fig.tight_layout(h_pad=0.,w_pad=3.5,pad=0.1)

fig.savefig(os.path.join(PLOT_DIR,"obj_func.pdf"))
fig.savefig(os.path.join(PLOT_DIR,"obj_func.png"),dpi=600)

plt.show()
