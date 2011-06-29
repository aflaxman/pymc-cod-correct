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

    pi_ = mc.Dirichlet('pi_', theta=pl.ones(J), value=pl.ones(J-1)/J)
    @mc.deterministic
    def pi(pi_=pi_):
        J = len(pl.atleast_1d(pi_))+1
        pi = pl.zeros(J)
        pi[0:(J-1)] = pi_
        pi[J-1] = 1. - pl.atleast_1d(pi_).sum()
        return pi

    alpha = mc.Exponential('alpha', beta=1., value=1./X.mean())

    tau = mc.Uniform('tau', lower=1.**-2, upper=.01**-2, value=(pl.ones(J)*.1)**-2)

    #@mc.potential
    #def obs(pi=pi, tau=tau, X=X):
    #    N = len(X)
    #    logp_i = pl.array([mc.normal_like(X[i,:], pi, tau) for i in range(N)])
    #    return mc.flib.logsum(logp_i - pl.log(N))
        
    @mc.observed
    def X_obs(pi=pi, tau=tau, alpha=alpha, value=X):
        N = len(value)
        logp_i = pl.array([mc.normal_like(value[i,:]*alpha, pi, tau) for i in range(N)])
        return mc.flib.logsum(logp_i - pl.log(N))        
    
    return vars()

def fit_latent_dirichlet(X, iter=1000, burn=500, thin=5): 
    vars = latent_dirichlet(X)
    m = mc.MCMC(vars) #, db='txt', dbname=dir + '/latent_dirichlet')
    m.sample(iter, burn, thin, verbose=1)
    pi = m.pi.trace()
    return pi.view(pl.recarray)

