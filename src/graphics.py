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
    pl.figure(figsize=(T, 2*J))

    left = .3
    right = .95
    bottom = .1
    top = .95

    xmax=F.max()

    dj = (top-bottom)/J
    dt = (right-left)/T

    ax = {}
    for jj, j in enumerate(sorted(range(J), key=lambda j: pi[:,:,j].mean())):
        for t in range(T):
            pl.axes([left + t*dt, bottom + jj*dj, dt, dj])
            pl.plot(pl.randn(N), F[:, t, j], 'b.', alpha=.5, zorder=-100)

            pl.errorbar([0], pi[:, t, j].mean(), 1.96*pi[:, t, j].std(),
                        fmt='gs', ms=10, mew=1, mec='white', linewidth=3, capsize=10,
                        zorder=100)
            pl.xticks([])
            if jj == 0:
                pl.xlabel(t+1980)

            if t > 0:
                pl.yticks([])
            else:
                pl.ylabel(causes[j])

            pl.axis([-3, 3, 0, xmax])
    if title:
        pl.figtext(.01, .99, title, va='top', ha='left')
    pl.savefig('/home/j/Project/Models/cod-correct/t.png')
