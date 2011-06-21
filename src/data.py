""" Class for wrangling data
"""

import pylab as pl
import pymc as mc

# simple models for some non-uniformly distributed subsets of the plane
def simulate_data(N):
    """ Create an Nx2 matrix of simulated data"""
    true_csmf = [.3, .7]
    tau = [.2**-2, .05**-2]
    X = [mc.rnormal(true_csmf[0], tau[0], size=N),
         mc.rnormal(true_csmf[1], tau[1], size=N)]  # TODO: find a cooler way to do this
    return pl.array(X).T
