#! /usr/bin/env python3

import numpy as np
from stdParams import thetap0, thetap1, thetad, alpha, gd, gp

def phi(x):
    return (np.tanh(2.*x) + 1.)/2.
    
def psi(p,d):
    return (phi(gp*(p-thetap0))*(1.-phi(gd*(d-thetad)))*alpha
            + phi(gp*(p-thetap1))*phi(gd*(d-thetad)))
    
