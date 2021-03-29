#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from stdParams import *
from src.plottools import gen_mesh_ax
plt.style.use('mpl_style.mplstyle')
import os

data = np.load(os.path.join(DATA_DIR,"distraction_scaling_dimension.npz"))

rho = data["align_corrcoef"]
dim = data["distract_dimension"]
s = data["distract_scaling"]

dim_ax = gen_mesh_ax(dim)
s_ax = gen_mesh_ax(s)

n_sweep = rho.shape[2]

rho_mean = rho.mean(axis=2)

fig, ax = plt.subplots(1,3,gridspec_kw={'width_ratios': [15,15,1]}, figsize=(FIG_WIDTH,FIG_WIDTH*0.45))

pc0 = ax[0].pcolormesh(s_ax,dim_ax,rho_mean[:,:,0].T,rasterized=True,vmin=0.,vmax=1.)
pc1 = ax[1].pcolormesh(s_ax,dim_ax,rho_mean[:,:,1].T,rasterized=True,vmin=0.,vmax=1.)

ax[0].set_xlabel(r'$s$')
ax[1].set_xlabel(r'$s$')

ax[0].set_ylabel(r'$N_{dist}$')
ax[1].set_ylabel(r'$N_{dist}$')

ax[0].set_xlim(right=4.)
ax[1].set_xlim(right=4.)

ax[0].set_title("Compartment Model",loc="right")
ax[1].set_title("Point Model",loc="right")

plt.colorbar(pc1,cax=ax[2])

fig.tight_layout(pad=0.2)

fig.savefig(os.path.join(PLOT_DIR,"corr_dimension_scaling.pdf"))
fig.savefig(os.path.join(PLOT_DIR,"corr_dimension_scaling.png"),dpi=600)

plt.show()
