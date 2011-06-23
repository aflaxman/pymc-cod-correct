""" Module for setting up statistical models
"""

import pylab as pl
import pymc as mc
import numpy as np

# model goes here
def bad_model(X):
    """ Results in a matrix with shape matching X, but all rows sum to 1"""
    N, J = X.shape
    return X / pl.outer(X.sum(axis=1), np.ones(J))

def latent_dirichlet(X):
    N, J = X.shape

    pi_ = mc.Dirichlet('pi_', theta=pl.ones(J))
    @mc.deterministic
    def pi(pi_=pi_):
        pi = pl.zeros(len(pi_)+1)
        pi[0:len(pi_)] = pi_
        pi[len(pi_)] = 1. - pi_.sum()
        return pi

    tau = mc.Uniform('tau', lower=0., upper=1.e6, value=X.std(axis=0)**-2)
    @mc.deterministic
    def diag_tau(tau=tau):
        return np.diag(tau)

    obs = mc.MvNormal('obs', mu=pi, tau=diag_tau, value=X, observed=True) 
    return vars()
