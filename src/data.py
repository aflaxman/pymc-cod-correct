""" Class for wrangling data
"""

import pylab as pl
import pymc as mc
import numpy as np
from math import log, exp

# logit and inverse logit functions 
def logit(x):
    return log(x/(1-x))

def invlogit(x):
    return exp(x)/(1+exp(x))

# simple models for some non-uniformly distributed subsets of the plane
def sim_data(N, true_csmf=[.3, .7], true_csmf_sd=[.2, .05]):
    """ 
    Create an NxJ matrix of simulated data (J is determined by the length 
    of true_csmf). 
    
    true_csmf - a list of true cause fractions (must sum to one)
    true_csmf_sd - a list of the standard deviations corresponding to the true csmf's
    """

    assert sum(true_csmf)==1, 'The sum of elements of true_csmf must equal 1' 
    assert len(true_csmf)==len(true_csmf_sd), 'The length of true_csmf and true_csmf_sd must be the same'

    ## transform the mean and sd to logit space 
    transformed_csmf = [logit(i) for i in true_csmf]
    transformed_var = []
    for i in range(len(true_csmf)):
        # TODO: verify that this actually the appropriate equation; the standard deviation of the data returned doesn't match what is being provided in the arguments
        transformed_var.append( (1/(true_csmf[1]*(1-true_csmf[1])))**2 * true_csmf_sd[i]**2 )

    ## draw from a normal distribution 
    X = []
    for i in range(len(transformed_csmf)):
        x = mc.rnormal(mu=transformed_csmf[i], tau=transformed_var[i]**-1, size=N)
        if i == 0: 
            X = x
        else: 
            X = np.vstack((X,x))

    ## back transform the simulated values
    Y = np.ones(np.shape(X))
    for i in range(np.shape(X)[0]):
        for j in range(np.shape(X)[1]):
            Y[i,j] = invlogit(X[i,j])
    return Y.T

