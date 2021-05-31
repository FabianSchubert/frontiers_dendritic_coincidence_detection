#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors as mcolors
import seaborn as sns
sns.set()
plt.style.use('mpl_style.mplstyle')
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def cc(arg):
    return mcolors.to_rgba(arg, alpha=1.)

x = np.ndarray((8,3))

for k in range(2):
     for l in range(2):
         for m in range(2):
             x[k*4+l*2+m] = np.array([k,l,m])
             
verts = []

'''
array([[0., 0., 0.],0
       [0., 0., 1.],1
       [0., 1., 0.],2
       [0., 1., 1.],3
       [1., 0., 0.],4
       [1., 0., 1.],5
       [1., 1., 0.],6
       [1., 1., 1.]]7)
'''

verts.append([x[0],x[4],x[6],x[2]])
verts.append([x[0],x[4],x[5],x[1]])
verts.append([x[0],x[2],x[3],x[1]])
verts.append([x[7],x[3],x[1],x[5]])
verts.append([x[7],x[3],x[2],x[6]])
verts.append([x[7],x[5],x[4],x[6]])

poly = Poly3DCollection(verts, facecolors = cc('b'))

poly.set_alpha(0.6)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#ax.plot_trisurf(poly,shade=True)

ax.add_collection3d(poly)

plt.show()
