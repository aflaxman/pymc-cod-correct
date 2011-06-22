import networkx as nx
import pylab as pl
import numpy as np

# put messy matplotlib code here so I don't have to look at it if I
# don't want to

# plot of simulated data
def plot_sim_data(X, index=[0,1]): 
    if np.concatenate(X).max() > 1 or np.concatenate(X).min() < 0: 
        print 'Warning: data outside [0,1] is not shown'
    pl.clf()
    pl.axis([0,1,0,1])
    pl.xlabel('csmf (cause %d)' %(index[0]+1))
    pl.ylabel('csmf (cause %d)' %(index[1]+1))	
    pl.plot(X[:,index[0]], X[:,index[1]], 'g.')
    if np.shape(X)[1]==2: 
        pl.plot([1,0],[0,1])
	
	
