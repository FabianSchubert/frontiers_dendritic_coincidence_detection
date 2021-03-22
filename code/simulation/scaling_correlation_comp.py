#! /usr/bin/env python3

import numpy as np
from stdParams import *
from src.neuronmodel import *
from tqdm import tqdm

N_sweep_distraction = 20
N_samples = 2
distract_scaling = np.linspace(1.,6.,N_sweep_distraction)

align_corrcoef = np.ndarray((N_sweep_distraction,N_samples,2))

N_p = 10

T = int(5e5)
t_ax = np.arange(T)

modes = ["comp","point"]

for mode in tqdm(modes):
    for s in tqdm(range(N_sweep_distraction),leave=False):
        for i in tqdm(range(N_samples),leave=False):
            a = np.random.rand(1,N_p)

            a /= np.linalg.norm(a)

            #####
            # This just generates a random vector and subtracts the projection of the a vector,
            # which makes vector a and vector a_orth orthogonal. a_orth is then normalized to unit length
            # -> Gram Schmidt
            a_orth = np.random.rand(1,N_p)
            a_orth = a_orth - (a_orth @ a.T)*a/(a @ a.T)
            a_orth /= np.linalg.norm(a_orth)
            
            sequences = np.random.rand(T,N_p)
            
            w_p = np.ones((T,1,N_p))
            
            n_d = np.ones((T))
            n_p = np.ones((T))
            b_d = np.zeros((T))
            b_p = np.zeros((T))
            
            x_d = (a @ sequences.T).T
            
            x_p = np.array(sequences)
            x_p = x_p + ((x_p @ a_orth.T)*(distract_scaling[s] - 1.)) @ a_orth
            
            I_p = np.ndarray((T))
            I_d = np.ndarray((T))
            
            x_p_av = np.ndarray((T,N_p))
            
            I_p_av = np.ndarray((T))
            I_d_av = np.ndarray((T))
            
            y = np.ndarray((T))
            
            y_squ_av = np.ndarray((T))
            y_av = np.ndarray((T))
            
            #### Init values
            
            w_p[0] /= np.linalg.norm(w_p[0])
            
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
                    w_p[t] = w_p[t-1] + mu_w * (x_p[t-1] - x_p_av[t-1]) * y[t-1]*(y[t-1]-thetay)
                else:
                    #w_p[t] = w_p[t-1] + mu_w * (x_p[t-1] - x_p_av[t-1]) * y[t-1]*(y[t-1]-y_squ_av[t-1])
                    w_p[t] = w_p[t-1] + mu_w * (x_p[t-1] - x_p_av[t-1]) * (y[t-1]-y_av[t-1])
                
                w_p[t] /= np.linalg.norm(w_p[t])
                
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
            
            align_corrcoef[s,i,modes.index(mode)] = np.corrcoef(a[0],w_p[-1,0])[1,0]

np.savez(os.path.join(DATA_DIR,"distraction_scaling.npz"),
         align_corrcoef = align_corrcoef,
         distract_scaling = distract_scaling)
         
os.system("shutdown /s /t 1")
