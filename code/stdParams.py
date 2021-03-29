#! /usr/bin/env python3

import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']

DATA_DIR = os.path.join(os.path.dirname(__file__),"../data")
PLOT_DIR = os.path.join(os.path.dirname(__file__),"../plots")

FIG_WIDTH = 5.5

thetap0 = 0.
thetap1 = -1.
thetad = 0.
alpha = 0.3
thetay = (1.+alpha)/2.
mu_w = 5e-4
mu_b = 1e-3
mu_n = 1e-4
mu_av = 5e-3
VI_pt = 0.25
VI_dt = 0.25
I_pt = 0.
I_dt = 0.
