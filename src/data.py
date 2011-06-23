""" Class for wrangling data
"""

import pylab as pl
import pymc as mc
import numpy as np

# simple models for some non-uniformly distributed subsets of the plane
def sim_data(N, true_csmf=[.3, .7], true_csmf_sd=[.2, .05]):
    """ 
    Create an NxJ matrix of simulated data (J is determined by the length 
    of true_csmf). 
    
    true_csmf - a list of true cause fractions (must sum to one)
    true_csmf_sd - a list of the standard deviations corresponding to the true csmf's
    """

    assert pl.allclose(sum(true_csmf), 1), 'The sum of elements of true_csmf must equal 1' 
    assert len(true_csmf)==len(true_csmf_sd), 'The length of true_csmf and true_csmf_sd must be the same'
    J = len(true_csmf)

    ## transform the mean and sd to logit space 
    transformed_csmf = mc.logit(true_csmf)
    transformed_var = []
    for pi_i, sigma_pi_i in zip(true_csmf, true_csmf_sd):
        # TODO: verify that this actually the appropriate equation; the standard deviation of the data returned doesn't match what is being provided in the arguments
        transformed_var.append( (1/(pi_i*(1-pi_i)))**2 * sigma_pi_i**2 )

    ## draw from distribution 
    X = mc.rnormal(mu=transformed_csmf, tau=pl.array(transformed_var)**-1, size=(N,J))

    ## back transform the simulated values
    Y = mc.invlogit(X)

    return Y

