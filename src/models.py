""" Module for setting up statistical models
"""

import pylab as pl
import pymc as mc
import numpy as np

# model goes here
def bad_model(X):
    """ Results in a matrix with shape matching X, but all rows sum to 1"""
    return X / pl.outer(X.sum(axis=1), np.ones((1,np.shape(X)[1])))

def latent_dirichlet(X):
    pi_ = mc.Dirichlet('pi_', theta=[1., 1.])
    pi = mc.CompletedDirichlet('pi', pi_)
    tau = mc.Uniform('tau', lower=0., upper=1.e6, value=[1., 1.])  # TODO: take initial value based on standard deviation of each column of X
    obs = [mc.Normal('obs_%d'%i, mu=pi, tau=tau, value=X[i,:], observed=True) for i in range(len(X))]  # TODO: consider if other ways of doing this are faster, and if the problem with the non-list mc.Normal was due to shape of pi.value
    return vars()
