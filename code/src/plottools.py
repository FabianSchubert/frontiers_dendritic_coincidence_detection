#! /usr/bin/env python3
import numpy as np

def gen_mesh_ax(x):
    x_mesh = np.append(x,2.*x[-1] - x[-2])
    x_mesh -= .5 * (x_mesh[1] - x_mesh[0])
    return x_mesh
    
def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar
