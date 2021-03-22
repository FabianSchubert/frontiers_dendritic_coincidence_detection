#! /usr/bin/env python3

import numpy as np
from stdParams import thetap0, thetap1, thetad, alpha

def phi(x):
    return (np.tanh(2.*x) + 1.)/2.
    
def psi(p,d):
    return (phi(p-thetap0)*(1.-phi(d-thetad))*alpha
            + phi(p-thetap1)*phi(d-thetad))
    
