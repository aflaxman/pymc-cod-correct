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
def plot_all_sim_data(X): 
    if np.concatenate(X).max() > 1 or np.concatenate(X).min() < 0: 
        print 'Warning: data outside [0,1] is not shown'
    pl.clf()
    J = np.shape(X)[1]
    for i in range(J): 
        for j in range(J):
            # plot for upper triangle
            p = J*i+j+1
            pl.subplot(J,J,p)
            pl.plot(X[:,j], X[:,i], 'g.')
            pl.axis([0,1,0,1])
            if J == 2 and i != j: 
                pl.plot([1,0],[0,1])
            if j == 0:
                pl.ylabel('csmf (cause %d)' %(i+1))
            if i == (J-1):
                pl.xlabel('csmf (cause %d)' %(j+1))
