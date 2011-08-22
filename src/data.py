""" Class for wrangling data
"""

import random 
import pylab as pl
import pymc as mc
import csv 
import os

def rec2csv_2d(Y, fname):
    """
    write a 2-dimensional recarray to a csv file
    """
    
    pl.rec2csv(pl.np.core.records.fromarrays(Y.T), fname)

def csv2array(fname): 
    """
    write a csv file to a 2-dimensional array (assumes all variables are numeric). 
    This allows for reading a csv into an array formatted such that means and other
    numeric functions are possible along the rows and columns. 
    """
    
    Y = []
    read = csv.reader(open(fname))
    read.next()
    for row in read: 
        Y.append(row)
    return pl.array(Y, dtype='f')

def sim_data(N, true_cf=[[.3, .6, .1],
                           [.3, .5, .2]],
             true_std=[[.2, .05, .05], 
                       [.3, 0.1, 0.1]],
             sum_to_one=True):
    """ 
    Create an NxTxJ matrix of simulated data (T is determined by the length 
    of true_cf, J by the length of the elements of true_cf). 

    true_cf - a list of lists of true cause fractions (each must sum to one)
    true_std - a list of lists of the standard deviations corresponding to the true csmf's 
             for each time point. Can either be a list of length J inside a list of length
             1 (in this case, the same standard deviation is used for all time points) or 
             can be T lists of length J (in this case, the a separate standard deviation 
             is specified and used for each time point). 
    """

    if sum_to_one == True: 
        assert pl.allclose(pl.sum(true_cf, 1), 1), 'The sum of elements of true_cf must equal 1' 
    T = len(true_cf)
    J = len(true_cf[0])
    
    ## if only one std provided, duplicate for all time points 
    if len(true_std)==1 and len(true_cf)>1: 
        true_std = [true_std[0] for i in range(len(true_cf))]    

    ## transform the mean and std to logit space
    transformed_std = []
    for t in range(T): 
        pi_i = pl.array(true_cf[t])
        sigma_pi_i = pl.array(true_std[t])
        transformed_std.append( ((1/(pi_i*(pi_i-1)))**2 * sigma_pi_i**2)**0.5 )
            
    ## find minimum standard deviation (by cause across time) and draw from this 
    min = pl.array(transformed_std).min(0)
    common_perturbation = [pl.ones([T,J])*mc.rnormal(mu=0, tau=min**-2) for n in range(N)]
    
    ## draw from remaining variation 
    tau=pl.array(transformed_std)**2 - min**2
    tau[tau==0] = 0.000001
    additional_perturbation = [[mc.rnormal(mu=0, tau=tau[t]**-1) for t in range(T)] for n in range(N)]

    result = pl.zeros([N, T, J])
    for n in range(N):
        result[n, :, :] = [mc.invlogit(mc.logit(true_cf[t]) + common_perturbation[n][t] + additional_perturbation[n][t]) for t in range(T)]

    return result

def sim_data_for_validation(N,
                            true_cf=[[0.1, 0.3, 0.6],
                                     [0.2, 0.3, 0.5]],
                            true_std=[[.2, .05, .05], 
                                      [.3, 0.1, 0.1]], 
                            std_bias=[1.,1.,1.]):
    """
    Input
    -----
    true_cf  - a list of lists of true cause fractions (each must sum to one).
    true_std - a list of lists of the standard deviations corresponding to the true csmf's 
             for each time point. Can either be a list of length J inside a list of length
             1 (in this case, the same standard deviation is used for all time points) or 
             can be T lists of length J (in this case, the a separate standard deviation 
             is specified and used for each time point). This is meant to capture how
             variable estimates of the true cause fraction will be (i.e. causes that
             are more difficult to estimate will be more variable and therefore will 
             have greater uncertainty).
    std_bias - a list of length J giving the bias for the standard deviations for each 
             cause (as a multiplier: i.e. 0.9 would imply that we will underestimate
             the standard deviation by 10% on average while 1.1 would imply that we
             will overestimate the standard deviation by 10% on average). 
    
    Output
    -----
    N JxT draws from an 'estimated' distribution for the specified causes 
    """

    if len(true_std)==1 and len(true_cf)>1: 
        true_std = [true_std[0] for i in range(len(true_cf))]
    
    est_cf = sim_data(1, true_cf, true_std)[0]
    est_error = est_cf - true_cf
    est_std = true_std*mc.runiform(pl.array(std_bias)*0.9, pl.array(std_bias)*1.1)
    sims = sim_data(N, est_cf, est_std, sum_to_one=False)
    return sims

def logit_normal_draw(cf_mean, std, N, J):
    std = pl.array(std)
    if mc.__version__ == '2.0rc2': # version on Omak 
        X = [mc.invlogit(mc.rnormal(mu=cf_mean, tau=std**-2)) for n in range(N)]
        Y = pl.array(X)
    else: 
        X = mc.rnormal(mu=cf_mean, tau=std**-2, size=(N,J))
        Y = mc.invlogit(X)
    return Y
    
def sim_cod_data(N, cf_rec): 
    """ 
    Create an NxJ matrix of simulated data (J is the number of causes and is determined
    by the length of cf_mean). 

    N - the number of simulations    
    cf_rec - a recarray containing: 
        cause - a list of causes 
        est - the estimates of the cause fractions
        lower - the lower bound of the cause fractions
        upper - the upper bound of the cause fractions 
    """

    # logit the mean and bounds and approximate the standard deviation in logit space
    cf_mean = mc.logit(cf_rec.est)
    cf_lower = mc.logit(cf_rec.lower)
    cf_upper = mc.logit(cf_rec.upper)
    std = (cf_upper - cf_lower)/(2*1.96)

def get_cod_data(dir = '/home/j/Project/Causes of Death/Under Five Deaths/CoD Correct Input Data/v02_prep_USA', 
                 causes = ['HIV', 'Injuries', 'Measles'], age = 'Under_Five', iso3 = 'USA', sex = 'M'): 
    csvs = {}
    sim_length = {}
    for c in causes: 
        csvs[c] = csv.reader(open('%s/%s+%s+%s+%s.csv' % (dir, iso3, c, age, sex)))
        names = csvs[c].next()
        sim_length[c] = len(names)-9
    
    cf = pl.zeros((1000, 32, len(causes)))
    for j in range(len(causes)): 
        cause = causes[j]; print(cause)
        sims = sim_length[cause]
        for t in range(32):
            temp = csvs[cause].next()[2:(sims+3)] 
            envelope = float(temp[-1])
            deaths = temp[0:sims]
            if sims < 1000: 
                deaths += random.sample(deaths, (1000-sims))
            if sims > 1000: 
                deaths = random.sample(deaths, 1000)
            cf[:,t,j] = pl.array(deaths, dtype='f')/(envelope*pl.ones(1000))   
    return cf


def get_cod_data_all_causes(iso3='USA', age_group='1_4', sex='F'):
    """ TODO: write doc string for this function"""
    print 'loading', iso3, age_group, sex
    import glob
    
    cause_list = []
    fpath = '/home/j/Project/Causes of Death/Under Five Deaths/CoD Correct Input Data/v02_prep_%s/%s+*+%s+%s.csv' % (iso3, iso3, age_group, sex)
    fnames = glob.glob(fpath)

    # initialize input distribution array
    N = 990  # TODO: get this from the data files
    T = 32  # TODO: get this from the data files
    J = len(fnames)
    F = pl.zeros((N, T, J))

    # fill input distribution array with data from files
    for j, fname in enumerate(sorted(fnames)):
        cause = fname.split('+')[1]  # TODO: make this less brittle and clearer
        print 'loading cause', cause
        F_j = pl.csv2rec(fname)

        for n in range(N):
            F[n, :, j] = F_j['ensemble_d%d'%(n+1)]/F_j['envelope']

        assert not pl.any(pl.isnan(F)), '%s should have no missing values' % fname
        cause_list.append(cause)
    
    print 'loading complete'
    return F, cause_list



