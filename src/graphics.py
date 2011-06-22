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
	
	
