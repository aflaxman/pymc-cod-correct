""" Module for setting up statistical models
"""

import pylab as pl
import pymc as mc

# model goes here
def bad_model(X):
    """ Results in a matrix with shape matching X, but all rows sum to 1"""
    return X / pl.outer(X.sum(axis=1), [1,1])
