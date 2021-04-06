#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from stdParams import *
from src.plottools import gen_mesh_ax
plt.style.use('mpl_style.mplstyle')
import os
import sys

files = os.listdir(os.path.join(DATA_DIR,"classification_dimension_scaling"))

data = []

for file in files:
    data.append(np.load(os.path.join(DATA_DIR,"classification_dimension_scaling/"+file)))

perf = []
dim = []
s = []

for dat in data:
    perf.append(dat["perf"])
    dim.append(dat["distract_dimension"])
    s.append(dat["distract_scaling"])

for k in range(1,len(perf)):
    if(not(np.array_equal(perf[k].shape,perf[k-1].shape))):
        print("perf arrays do not match!")
        sys.exit()
    if(not(np.array_equal(dim[k],dim[k-1]))):
        print("dim arrays do not match!")
        sys.exit()
    if(not(np.array_equal(s[k],s[k-1]))):
        print("scaling arrays do not match!")
        sys.exit()

perf = np.array(perf)
dim = dim[0]
s = s[0]

n_total_samples = perf.shape[0]*perf.shape[3]

perf_flatten = np.ndarray((perf.shape[1],perf.shape[2],perf.shape[4],n_total_samples))

for k in range(perf.shape[0]):
    for l in range(perf.shape[3]):
        perf_flatten[:,:,:,k*perf.shape[3]+l] = perf[k,:,:,l,:]

perf = perf_flatten

dim_ax = gen_mesh_ax(dim)
s_ax = gen_mesh_ax(s)

n_sweep = perf.shape[2]

perf_mean = perf.mean(axis=3)

fig, ax = plt.subplots(1,3,gridspec_kw={'width_ratios': [15,15,1]}, figsize=(FIG_WIDTH,FIG_WIDTH*0.45))

pc0 = ax[0].pcolormesh(s_ax,dim_ax,perf_mean[:,:,0].T,rasterized=True,vmin=0.5,vmax=1.)
pc1 = ax[1].pcolormesh(s_ax,dim_ax,perf_mean[:,:,1].T,rasterized=True,vmin=0.5,vmax=1.)

ax[0].set_xlabel(r'$s$')
ax[1].set_xlabel(r'$s$')

ax[0].set_ylabel(r'$N_{dist}$')
ax[1].set_ylabel(r'$N_{dist}$')

#ax[0].set_xlim([0.1,5.])
#ax[1].set_xlim([0.1,5.])

ax[0].set_title("Compartment Model",loc="right")
ax[1].set_title("Point Model",loc="right")

plt.colorbar(pc1,cax=ax[2])

fig.tight_layout(pad=0.2)

fig.savefig(os.path.join(PLOT_DIR,"classification_dimension_scaling.pdf"))
fig.savefig(os.path.join(PLOT_DIR,"classification_dimension_scaling.png"),dpi=600)

plt.show()
