#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from stdParams import *
plt.style.use('mpl_style.mplstyle')
import os

data = np.load(os.path.join(DATA_DIR,"distraction_dimension.npz"))

rho = data["align_corrcoef"]
dim = data["distract_dimension"]

rho_mean = rho.mean(axis=1)
rho_err = rho.std(axis=1)/rho.shape[1]**.5
rho_top = rho_mean + rho_err
rho_bottom = rho_mean - rho_err

fig, ax = plt.subplots(1,1,figsize=(FIG_WIDTH,0.6*FIG_WIDTH))

ax.fill_between(dim,rho_bottom[:,0],rho_top[:,0],alpha=0.5,color=COLORS[0])
ax.fill_between(dim,rho_bottom[:,1],rho_top[:,1],alpha=0.5,color=COLORS[1])

ax.plot(dim,rho_mean[:,0],c=COLORS[0])
ax.plot(dim,rho_mean[:,1],c=COLORS[1])

ax.set_ylabel(r'$\rho[I_p,I_d]$')
ax.set_xlabel(r'$N$')

fig.savefig(os.path.join(PLOT_DIR,"pd_corr_dimension.pdf"))

plt.show()
