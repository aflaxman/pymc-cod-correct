# matplotlib backend setup
import matplotlib
matplotlib.use("AGG") 


import networkx as nx
import pylab as pl
import numpy as np

# put messy matplotlib code here so I don't have to look at it if I
# don't want to

# plot of simulated data
def plot_sim_data(X, i=0, j=1): 
    if np.concatenate(X).max() > 1 or np.concatenate(X).min() < 0: 
        print 'Warning: data outside [0,1] is not shown'
    pl.clf()
    pl.xlabel('csmf (cause %d)' %(i+1))
    pl.ylabel('csmf (cause %d)' %(j+1))	
    pl.plot(X[:,i], X[:,j], 'g.')
    if np.shape(X)[1]==2: 
        pl.plot([1,0],[0,1])
    pl.axis([0,1,0,1])


# lattice style plot of simulated data
def plot_all_sim_data(X, color='b'): 
    if np.concatenate(X).max() > 1 or np.concatenate(X).min() < 0: 
        print 'Warning: data outside [0,1] is not shown'
    #pl.clf()
    J = np.shape(X)[1]
    for i in range(J): 
        for j in range(i+1,J):
            # plot for upper triangle
            p = (J-1)*i+j
            pl.subplot(J-1,J-1,p)
            pl.plot(X[:,j], X[:,i], linestyle='None', marker='.', color=color)
            pl.axis([0,1,0,1])
            if J == 2: 
                pl.plot([1,0],[0,1])
            if j == i+1:
                pl.ylabel('csmf (cause %d)' %(i+1))
                pl.xlabel('csmf (cause %d)' %(j+1))

def plot_F_and_pi(F, pi, causes, title=''):
    N, T, J = F.shape
    pl.figure(figsize=(.5*T, 1.*J))

    left = 2./(T+5.)
    right = 1-.05/T
    bottom = 1/(T+5.)
    top = 1-.05/T

    xmax=F.max()

    dj = (top-bottom)/J
    dt = (right-left)/T

    ax = {}
    for jj, j in enumerate(sorted(range(J), key=lambda j: pi[:,:,j].mean())):
        for t in range(T):
            pl.axes([left + t*dt, bottom + jj*dj, dt, dj])
            pl.plot(pl.randn(N), F[:, t, j], 'b.', alpha=.5, zorder=-100)

            pi[:,t,j].sort()
            below = pi[:, t, j].mean() - pi[:,t,j][.025*N]
            above = pi[:,t,j][.975*N] - pi[:, t, j].mean()
            pl.errorbar([0], pi[:, t, j].mean(), [[below], [above]],
                        fmt='gs', ms=10, mew=1, mec='white', linewidth=3, capsize=10,
                        zorder=100)
            pl.text(-2.75, xmax*.9,
                    '%.2f\n%.2f\n%.2f'%(F[:,t,j].mean(), pi[:,t,j].mean(), F[:,t,j].mean() - pi[:,t,j].mean()),
                    va='top', ha='left')
            pl.xticks([])
            if jj == 0:
                pl.xlabel(t+1980)
                pl.text(-5.75, xmax*.9,
                    'in:\nout:\nres:'%(F[:,t,j].mean(), pi[:,t,j].mean(), F[:,t,j].mean() - pi[:,t,j].mean()),
                    va='top', ha='left')

            if t > 0:
                pl.yticks([])
            else:
                pl.ylabel(causes[j])

            pl.axis([-3, 3, 0, xmax])
    if title:
        pl.figtext(.01, .99, title, va='top', ha='left')
