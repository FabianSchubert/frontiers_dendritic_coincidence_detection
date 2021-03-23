#! /usr/bin/env python3

import numpy as np
from stdParams import *
from src.neuronmodel import *
from tqdm import tqdm

N_sweep_distraction_scaling = 20
N_sweep_distraction_dimension = 9
N_samples = 1
distract_scaling = np.linspace(1.,6.,N_sweep_distraction_scaling)
distract_dimension = np.arange(1,N_sweep_distraction_dimension+1)

perf = np.ndarray((N_sweep_distraction_scaling,
                    N_sweep_distraction_dimension,
                    N_samples,2))

N_p = 10
N_out = 2

T = int(2e5)
T_test = int(1e4)
t_ax = np.arange(T)

stdMainDir = .25
distMainDir = 2.

modes = ["comp","point"]

for mode in tqdm(modes):
    for s in tqdm(range(N_sweep_distraction_scaling),leave=False):
        for n in tqdm(range(N_sweep_distraction_dimension),leave=False):
            for i in tqdm(range(N_samples),leave=False):
                
                
                patterns = np.random.normal(0.,1.,(T,N_p))
                patterns[:,0] *= stdMainDir
                patterns[:,1:distract_dimension[n]+1] *= distract_scaling[s]
                patterns[:,distract_dimension[n]+1:] *= 0.
                patterns[:,0] += (1.*(np.random.rand(T) <= 0.5) - 0.5) * distMainDir

                
                w_p = np.ones((T,N_out,N_p))
                
                n_d = np.ones((T,N_out))
                n_p = np.ones((T,N_out))
                b_d = np.zeros((T,N_out))
                b_p = np.zeros((T,N_out))
                
                x_d = np.ndarray((T,2))
                x_d[:,0] = 1.*(patterns[:,0] < 0.)
                x_d[:,1] = 1.*(patterns[:,0] > 0.)
                
                #generate random orthonormal basis via qr-decomposition
                Q,R = np.linalg.qr(np.random.normal(0.,1.,(N_p,N_p)))
                #Transform x_p with orthogonal basis.
                x_p = (Q @ patterns.T).T
                
                I_p = np.ndarray((T,N_out))
                I_d = np.ndarray((T,N_out))
                
                x_p_av = np.ndarray((T,N_p))
                
                I_p_av = np.ndarray((T,N_out))
                I_d_av = np.ndarray((T,N_out))
                
                y = np.ndarray((T,N_out))
                
                y_squ_av = np.ndarray((T,N_out))
                y_av = np.ndarray((T,N_out))
                
                #### Init values
                                
                w_p[0] = (w_p[0].T / np.linalg.norm(w_p[0],axis=1)).T
                                
                I_p[0] = n_p[0] * (w_p[0] @ x_p[0]) - b_p[0]
                I_d[0] = n_d[0] * x_d[0] - b_d[0]
                
                x_p_av[0] = 0.
                
                I_p_av[0] = 0.
                I_d_av[0] = 0.
                
                if(mode == "comp"):
                    y[0] = psi(I_p[0],I_d[0])
                else:
                    y[0] = phi(I_p[0] + I_d[0])
                    
                y_squ_av[0] = y[0]**2.
                y_av[0] = y[0]

                for t in tqdm(range(1,T),disable=False,leave=False):
                    
                    if(mode=="comp"):
                        #w_p[t] = w_p[t-1] + mu_w * (x_p[t-1] - x_p_av[t-1]) * y[t-1]*(y[t-1]-thetay)
                        w_p[t] = w_p[t-1] + mu_w * np.outer(y[t-1]-y_av[t-1], x_p[t-1] - x_p_av[t-1])
                    else:
                        #w_p[t] = w_p[t-1] + mu_w * (x_p[t-1] - x_p_av[t-1]) * y[t-1]*(y[t-1]-y_squ_av[t-1])
                        w_p[t] = w_p[t-1] + mu_w * np.outer(y[t-1]-y_av[t-1], x_p[t-1] - x_p_av[t-1])
                    
                    w_p[t] = (w_p[t].T / np.linalg.norm(w_p[t],axis=1)).T
                    
                    b_p[t] = b_p[t-1] + mu_b * (I_p[t-1] - I_pt)
                    b_d[t] = b_d[t-1] + mu_b * (I_d[t-1] - I_dt)
                    
                    n_p[t] = n_p[t-1] + mu_n * (VI_pt - (I_p[t-1] - I_p_av[t-1])**2.)
                    n_d[t] = n_d[t-1] + mu_n * (VI_dt - (I_d[t-1] - I_d_av[t-1])**2.)
                    
                    I_p[t] = n_p[t] * (w_p[t] @ x_p[t]) - b_p[t]
                    I_d[t] = n_d[t] * x_d[t] - b_d[t]
                    
                    x_p_av[t] = (1.-mu_av)*x_p_av[t-1] + mu_av*x_p[t]
                    
                    I_p_av[t] = (1.-mu_av)*I_p_av[t-1] + mu_av*I_p[t]
                    I_d_av[t] = (1.-mu_av)*I_d_av[t-1] + mu_av*I_d[t]
                    
                    
                    if(mode == "comp"):
                        y[t] = psi(I_p[t],I_d[t])
                    else:
                        y[t] = phi(I_p[t] + I_d[t])
                        
                    y_squ_av[t] = (1.-mu_av)*y_squ_av[t-1] + mu_av * y[t]**2.
                    y_av[t] = (1.-mu_av)*y_av[t-1] + mu_av * y[t]
                
                
                patterns_test = np.random.normal(0.,1.,(T_test,N_p))
                patterns_test[:,0] *= stdMainDir
                patterns_test[:,1:distract_dimension[n]+1] *= distract_scaling[s]
                patterns_test[:,distract_dimension[n]+1:] *= 0.
                patterns_test[:,0] += (1.*(np.random.rand(T_test) <= 0.5) - 0.5) * distMainDir
                
                lab_test = 1.*(patterns_test[:,0] > 0.)
                
                x_p_test = (Q @ patterns_test.T).T
                                
                I_p_test = n_p[-1] * (w_p[-1] @ x_p_test.T).T - b_p[-1]
                
                pred = np.argmax(I_p_test,axis=1)
                perf[s,n,i,modes.index(mode)] = (1.*(pred == lab_test)).mean()

np.savez(os.path.join(DATA_DIR,"classification_scaling_dimension.npz"),
         perf = perf,
         distract_scaling = distract_scaling,
         distract_dimension = distract_dimension)
