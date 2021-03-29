#! /usr/bin/env python3
import numpy as np

def gen_mesh_ax(x):
    x_mesh = np.append(x,2.*x[-1] - x[-2])
    x_mesh -= .5 * (x_mesh[1] - x_mesh[0])
    return x_mesh
