import numpy as numpy
import pandas as pd
import sys
import os

import scipy
from scipy.stats import qmc, norm
from scipy.integrate import quad
from scipy.optimize import minimize

###Parameters - - Cosmology### (7 params)
#Omega m = 0.3156, prior = flat(0.1,0.6)
#Sigma_8 = 0.831, prior = flat(0.6,0.95)
#n_s = 0.9645, prior = flat(0.85,1.06)
#w0 = -1.0, prior = flat(-2.0,0.0)
#wa = 0.0, prior = flat(-2.5,2.5)
#Omega_b = 0.0492, flat(0.04,0.055)
#h0 = 0.6727, flat(0.6,0.76)

###Parameters - - Other### (10 params)
#b_g^i = 1.3+ix0.1, flat(0.8,3.0) (galaxy bias)

###Parameters --- IA### (2,3, or 4)
#A_IA = 5.95, flat(0.0,10)
#beta_IA = 1.1, flat(-4.0,6.0)
#eta_IA (high-z) = 0.49, flat(-10.0,10.0)
#eta_IA = 0.0, flat(-1.0,1.0)
#(If using 2, instead use below)
#a = 0.5, flat(-5,5)
#eta = 0, flat(-5,5)

#What do A1, A2, B_TA correspond to?

###Parameters --- Baryons### (3)
#Q1 = 0.0, gauss(0.0,16.0)
#Q2 = 0.0, gauss(0.0,5.0)
#Q3 = 0.0, gauss(0.0,0.8)

#photo-z (2x10)
#delta z^i x10^2 = 0, gauss(0,0.2)

#shear calibration (10)
#m_i x10^2 = 0, gauss(0,0.5)

sampler = qmc.LatinHypercube(d=6)
sample = sampler.random(n=100)/groups/behroozi/hbowden/.ipynb_checkpoints