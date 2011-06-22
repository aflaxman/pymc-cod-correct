""" Class for wrangling data
"""

import pylab as pl
import pymc as mc
import numpy as np

# simple models for some non-uniformly distributed subsets of the plane
def simulate_data(N):
    """ Create an Nx2 matrix of simulated data"""
    true_csmf = [.3, .7]
    tau = [.2**-2, .05**-2]
    X = [mc.rnormal(true_csmf[0], tau[0], size=N),
         mc.rnormal(true_csmf[1], tau[1], size=N)]  # TODO: find a cooler way to do this
    return pl.array(X).T
	
def sim_data(N,true_csmf,true_csmf_sd):
	""" 
	Create an NxJ matrix of simulated data (J is determined by the length 
	  of true_csmf). 
	
	true_csmf - a list of true cause fractions (must sum to one)
	true_csmf_sd - a list of the standard deviations corresponding to the true csmf's
	"""
	assert sum(true_csmf)==1, 'The sum of elements of true_csmf must equal 1' 
	assert len(true_csmf)==len(true_csmf_sd), 'The length of true_csmf and true_csmf_sd must be the same'
	X = []
	for i in range(len(true_csmf)):
		x = mc.rnormal(mu=true_csmf[i], tau=true_csmf_sd[i]**-2, size=N) # this can (does) give cause fractions outside [0,1]
		if i == 0: 
			X = x
		else: 
			X = np.vstack((X,x))
	return X.T
	
