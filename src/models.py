""" Module for setting up statistical models
"""

import pylab as pl
import pymc as mc

# model goes here
def bad_model(X):
    """ Results in a matrix with shape matching X, but all rows sum to 1"""
    N, J = X.shape
    return X / pl.outer(X.sum(axis=1), pl.ones(J))

def latent_dirichlet(X):
    N, J = X.shape

    pi_ = mc.Dirichlet('pi_', theta=pl.ones(J))
    @mc.deterministic
    def pi(pi_=pi_):
        J = len(pl.atleast_1d(pi_))+1
        pi = pl.zeros(J)
        pi[0:(J-1)] = pi_
        pi[J-1] = 1. - pi_.sum()
        return pi

    tau = mc.Uniform('tau', lower=0., upper=1.e20, value=X.std(axis=0)**-2)
    @mc.deterministic
    def diag_tau(tau=tau):
        return pl.diag(tau)

    @mc.potential
    def obs(pi=pi, tau=tau, X=X):
        i = pl.floor(pl.rand()*len(X))
        return mc.normal_like(X[i,:], pi, tau)
    return vars()
