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
    pi = mc.CompletedDirichlet('pi', pi_)[0]  # TODO: see about patching PyMC so [0] is unneeded
    tau = mc.Uniform('tau', lower=0., upper=1.e6, value=pl.ones(J))

    @mc.observed
    def obs(pi=pi, tau=tau, value=X):
        logp = 0.

        for i in range(len(pi)):
            logp += mc.normal_like(X[:,i], pi[i], tau[i])
        return logp

    tau = mc.Uniform('tau', lower=0., upper=1.e6, value=X.std(axis=0)**-2)
    @mc.deterministic
    def diag_tau(tau=tau):
        return np.diag(tau)
    obs = mc.MvNormal('obs', mu=pi, tau=diag_tau, value=X, observed=True) 
    return vars()
