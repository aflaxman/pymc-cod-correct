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

    pi_ = mc.Dirichlet('pi_', theta=pl.ones(J),
                       value=pl.array(X).mean(0)[:-1])
    @mc.deterministic
    def pi(pi_=pi_):
        J = len(pl.atleast_1d(pi_))+1
        pi = pl.zeros(J)
        pi[0:(J-1)] = pi_
        pi[J-1] = 1. - pl.atleast_1d(pi_).sum()
        return pi

    alpha = 1 #mc.Exponential('alpha', beta=1., value=1./X.mean())

    tau = mc.Uniform('tau', lower=1.**-2, upper=.0001**-2,
                     value=pl.array(X).std(0)**-2)
        
    @mc.observed
    def X_obs(pi=pi, tau=tau, alpha=alpha, value=X):
        N = len(value)
        logp_i = pl.array([mc.normal_like(value[i,:]*alpha, pi, tau) for i in range(N)])
        return mc.flib.logsum(logp_i - pl.log(N))        
    
    return vars()

def fit_latent_dirichlet(X, iter=1000, burn=500, thin=5): 
    vars = latent_dirichlet(X)
    #m = mc.MAP([vars['pi_'], vars['X_obs']])
    #m.fit(verbose=1)
    
    m = mc.MCMC(vars) #, db='txt', dbname=dir + '/latent_dirichlet')
    m.sample(iter, burn, thin, verbose=1)
    pi = m.pi.trace()

    print 'mean: ', pl.floor(m.pi.stats()['mean']*100.+.5)/100.
    print 'ui:\n', pl.floor(m.pi.stats()['95% HPD interval']*100.+.5)/100.
    acorr5 = pl.dot((pi - pi.mean(0))[:-5].T, (pi - pi.mean(0))[5:]) / pl.dot((pi - pi.mean(0))[:].T, (pi - pi.mean(0))[:])
    print 'acorrs:', pl.diag(pl.floor(acorr5*1000.+.5)/1000.)

    return m, pi.view(pl.recarray)

