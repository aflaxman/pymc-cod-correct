""" Module for setting up statistical models
"""

import pylab as pl
import pymc as mc

# model goes here
def bad_model(X):
    """ Results in a matrix with shape matching X, but all rows sum to 1"""
    N, J = X.shape
    Y = X / pl.outer(pl.array(X).sum(axis=1), pl.ones(J))
    return Y.view(pl.recarray) 

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

    @mc.potential
    def obs(pi=pi, tau=tau, X=X):
        N = len(X)
        logp_i = pl.array([mc.normal_like(X[i,:], pi, tau) for i in range(N)])
        #return log(sum(exp(logp_i - log(N)))) # better to use flib.logsum
        return mc.flib.logsum(logp_i - pl.log(N))
                       
    return vars()

def fit_latent_dirichlet(X, iter=1000, burn=500, thin=5): 
    vars = latent_dirichlet(X)
    m = mc.MCMC(vars) #, db='txt', dbname=dir + '/latent_dirichlet')
    m.sample(iter, burn, thin, verbose=1)
    pi = m.pi.trace()
    return pi.view(pl.recarray)
    
