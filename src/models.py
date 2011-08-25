""" Module for setting up statistical models
"""

import pylab as pl
import pymc as mc

def bad_model(X):
    """ Results in a matrix with shape matching X, but all rows sum to 1"""
    N, T, J = X.shape
    Y = pl.zeros_like(X)
    for t in range(T):
        Y[:,t,:] = X[:,t,:] / pl.outer(pl.array(X[:,t,:]).sum(axis=1), pl.ones(J))
    return Y.view(pl.recarray) 

def new_bad_model(F):
    """ Results in a matrix with shape matching X, but all rows sum to 1"""
    N, T, J = F.shape
    pi = pl.zeros_like(F)
    for t in range(T):
        u = F[:,t,:].var(axis=0)
        u /= pl.sqrt(pl.dot(u,u))
        F_t_par = pl.dot(pl.atleast_2d(pl.dot(F[:,t,:], u)).T, pl.atleast_2d(u))
        F_t_perp = F[:,t,:] - F_t_par
        for n in range(N):
            alpha = (1 - F_t_perp[n].sum()) / F_t_par[n].sum()
            pi[n,t,:] = F_t_perp[n,:] + alpha*F_t_par[n,:]
    return pi

def latent_simplex(X):
    """ TODO: describe this function"""
    N, T, J = X.shape

    alpha = []
    for t in range(T):
        alpha_t = []
        for j in range(J):
            mu_alpha_tj = pl.mean(X[:,t,j]) / pl.mean(X[:,t,:], 0).sum()
            alpha_t.append(mc.Normal('alpha_%d_%d'%(t,j), mu=0., tau=1., value=pl.log(mu_alpha_tj)))
        alpha.append(alpha_t)

    @mc.deterministic
    def pi(alpha=alpha):
        pi = pl.zeros((T, J))
        for t in range(T):
            pi[t] = pl.reshape(pl.exp(alpha[t]), J) / pl.sum(pl.exp(alpha[t]))
        return pi

    @mc.observed
    def X_obs(pi=pi, value=X.mean(0), sigma=X.std(0), pow=2):
        """ TODO: experiment with different values of pow, although
        pow=2 seems like a fine choice based on our limited
        experience."""
        return -((pl.absolute(pi - value) / sigma)**pow).sum()
    
    return vars()

def latent_simplex_v2(X):
    """ TODO: describe this function"""
    N, T, J = X.shape

    alpha = []
    for t in range(T):
        alpha_t = []
        for j in range(J):
            mu_alpha_tj = pl.mean(X[:,t,j]) / pl.mean(X[:,t,:], 0).sum()
            alpha_t.append(mc.Normal('alpha_%d_%d'%(t,j), mu=0., tau=1., value=pl.log(mu_alpha_tj)))
        alpha.append(alpha_t)

    @mc.deterministic
    def pi(alpha=alpha):
        pi = pl.zeros((T, J))
        for t in range(T):
            pi[t] = pl.reshape(pl.exp(alpha[t]), J) / pl.sum(pl.exp(alpha[t]))
        return pi

    sigma = [[mc.Normal('sigma_%d_%d'%(t,j), mu=X[:,t,j].std(), tau=.01**-2,
                      value=X[:,t,j].std()) for j in range(J)] for t in range(T)]
        
    @mc.observed
    def X_obs(pi=pi, sigma=sigma, value=X):
        logp = mc.normal_like(pl.array(value).ravel(), 
                              (pl.ones([N,J*T])*pl.array(pi).ravel()).ravel(), 
                              (pl.ones([N,J*T])*pl.array(sigma).ravel()).ravel()**-2)
        return logp
        
        logp = pl.zeros(N)
        for n in range(N):
            logp[n] = mc.normal_like(pl.array(value[n]).ravel(),
                                     pl.array(pi+beta).ravel(),
                                     pl.array(sigma).ravel()**-2)
        return mc.flib.logsum(logp - pl.log(N))
    
    return vars()
    
def fit_latent_simplex(X, iter=10000, burn=5000, thin=5): 
    vars = latent_simplex(X)

    m = mc.MAP([vars['alpha'], vars['X_obs']])
    m.fit(method='fmin_powell', verbose=0)
    #print vars['pi'].value
    
    m = mc.MCMC(vars)
    for alpha_t in m.alpha:
        m.use_step_method(mc.AdaptiveMetropolis, alpha_t)

    m.sample(iter, burn, thin, verbose=0)
    pi = m.pi.trace()

    print 'mean: ', pl.floor(m.pi.stats()['mean']*100.+.5)/100.
    #print 'ui:\n', pl.floor(m.pi.stats()['95% HPD interval']*100.+.5)/100.

    return m, pi.view(pl.recarray)
    
def fit_latent_simplex_v2(X, iter=20000, burn=10000, thin=10): 
    vars = latent_simplex_v2(X)

    m = mc.MAP([vars['alpha'], vars['X_obs']])
    m.fit(method='fmin_powell', verbose=1)
    print vars['pi'].value

    m = mc.MAP([vars['sigma'], vars['X_obs']])
    m.fit(method='fmin_powell', verbose=1)
    print [['%.2f'%sigma_tj.value for sigma_tj in sigma_t] for sigma_t in vars['sigma']]

    m = mc.MCMC(vars)

    for alpha_t, beta_t, sigma_t in zip(m.alpha, m.beta, m.sigma):
        m.use_step_method(mc.AdaptiveMetropolis, alpha_t + [beta_t])
        #m.use_step_method(mc.AdaptiveMetropolis, sigma_t)

    m.sample(iter, burn, thin, verbose=1)
    pi = m.pi.trace()

    print 'mean: ', pl.floor(m.pi.stats()['mean']*100.+.5)/100.
    print 'ui:\n', pl.floor(m.pi.stats()['95% HPD interval']*100.+.5)/100.

    return m, pi.view(pl.recarray)    

