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
    """ TODO: describe this function"""
    N, T, J = X.shape

    pi_ = []
    for t in range(T):
        mu_pi_t = (pl.mean(X[:,t,:], 0) / pl.mean(X[:,t,:], 0).sum())[:-1]
        pi_.append(mc.Dirichlet('pi_%d'%t, theta=pl.ones(J),
                                value=mu_pi_t))

    @mc.deterministic
    def pi(pi_=pi_):
        pi = pl.zeros((T, J))
        for t in range(T):
            pi[t, 0:(J-1)] = pi_[t]
            pi[t, J-1] = 1. - pl.atleast_1d(pi_[t]).sum()
        return pi

    tau = mc.Uniform('tau', lower=1.**-2, upper=.0001**-2,
                     value=[pl.array(X[:,t,:]).std(0)**-2 for t in range(T)])
        
    @mc.observed
    def X_obs(pi=pi, tau=tau, value=X):
        logp = pl.zeros(N)
        for n in range(N):
            logp[n] = mc.normal_like(value[n].ravel(), pi.ravel(), tau.ravel())
        return mc.flib.logsum(logp - pl.log(N))        
    
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

def pretty_array(X, digits):
    return str(X)

