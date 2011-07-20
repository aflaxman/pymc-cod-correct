""" Class for wrangling data
"""

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
    min = pl.array(transformed_std).min(0) - 0.0000001
    common_perturbation = [pl.ones([T,J])*mc.rnormal(mu=0, tau=min**-2) for n in range(N)]

    ## draw from remaining variation 
    additional_perturbation = [mc.rnormal(mu=0, tau=(pl.array(transformed_std)**2 - min**2)**-1) for n in range(N)]

    result = pl.zeros([N, T, J])
    for n in range(N):
        result[n, :, :] = mc.invlogit(mc.logit(true_cf) + common_perturbation[n] + additional_perturbation[n])

    return result

def sim_data_for_validation(N,
                            true_cf=[[0.1, 0.3, 0.6],
                                     [0.2, 0.3, 0.5]],
                            true_std=[[.2, .05, .05], 
                                      [.3, 0.1, 0.1]]):
    """
    Input
    -----
    true_cf - a list of lists of true cause fractions (each must sum to one)
    true_std - a list of lists of the standard deviations corresponding to the true csmf's 
             for each time point. Can either be a list of length J inside a list of length
             1 (in this case, the same standard deviation is used for all time points) or 
             can be T lists of length J (in this case, the a separate standard deviation 
             is specified and used for each time point). This is meant to capture how
             variable estimates of the true cause fraction will be (i.e. causes that
             are more difficult to estimate will be more variable and therefore will 
             have greater uncertainty)
    
    Output
    -----
    N draws from an 'estimated' distribution for the specified causes 
    """

    if len(true_std)==1 and len(true_cf)>1: 
        true_std = [true_std[0] for i in range(len(true_cf))]
    
    est_cf = sim_data(1, true_cf, true_std)[0]
    est_error = est_cf - true_cf

    est_std = true_std # TODO: consider less correlated relationship
    
    sims = sim_data(N, est_cf, est_std, sum_to_one=False)
    return sims
    
def get_cod_data(level=1, keep_age = '20', keep_iso3 = 'USA', keep_sex = 'female', keep_year='2010'):
    """ Get data from CoDMod output on J drive
    Input 
    -----
    level : int, between 1 and ?
      level of causes to obtain data for (level 1 = A, B, C; each successive level contains more subdivisions) 
    keep_age : string
      the beginning of the age group to obtain data for (0, 0.01, 0.1, 1, 5, 10, ..., 80) 
    keep_sex : string
      male or female 
    keep_year : string
      in the range (1980, 2010) 
    
    Results
    -------
    lists of causes, mean estimates, lower bound estimates, upper bound estimates
    """
    # Currently this will only select causes at a given level: it needs to also select causes at higher levels that don't have children at the current level
 
    if os.name == 'nt':
        root = "J:/"
    else:
        root = "/home/j/"
    
    # get cause list; keep only causes from the specified level; keep only observations that include the specified sex
    cause_list = pl.csv2rec(root + 'Project/Causes of Death/CoDMod/Models/bigbang/covariate selection/bigbang_inputs.csv')
    cause_level = pl.array([len(cause_list['cause'][i].split('.')) for i in range(pl.shape(cause_list)[0])])
    if level == 1: 
        cause_list = cause_list[(cause_list.cause == 'A')|(cause_list.cause == 'B')|(cause_list.cause == 'C')]
    else: 
        cause_list = cause_list[(cause_level==(level-1))&(cause_list.cause != 'A')&(cause_list.cause != 'B')&(cause_list.cause != 'C')]
    cause_list = cause_list[(cause_list.sex == keep_sex)|(cause_list.sex == 'both')]

    # read in data; only retain data for the country, age, and year specified 
    count = 0 
    num = pl.shape(cause_list)[0]
    d_cause = []
    d_deaths_mean = []
    d_deaths_lower = []
    d_deaths_upper = []
    d_envelope = []
    for cause, start_age, end_age in zip(cause_list.cause, cause_list.start_age, cause_list.end_age):
        count += 1 
        print str(count) + ' of ' + str(num)
        if start_age == 'Early Neonatal': start_age = 'Early_Neonatal' 
        if start_age == 'Post Neonatal': start_age = 'Post_Neonatal' 
        if end_age == 'Early Neonatal': end_age = 'Early_Neonatal' 
        if end_age == 'Post Neonatal': end_age = 'Post_Neonatal' 
        print '  ' + cause + ' - ' + start_age + ' - ' + end_age
        
        # this is to correct for a temporary (according to Kyle) inconsistency in the cause list and the folder structure. 
        if level == 1 and cause == 'B': 
            start_age = 'Post_Neonatal'
        if level == 2 and cause == 'B02':
            if start_age == '1': end_age = '14'
            if start_age == '25': start_age = '15'

        # this is a temporary fix to get around the issue of what model to use. 
        if (cause == 'A04' or cause == 'A07' or cause == 'B02' or cause == 'B05'): 
            model = 'bb3' 
        elif (cause == 'A12'):
            model = 'Archives/bb'
        else: 
            model = 'bb'
 
        csvdata = csv.reader(open(root + 'Project/Causes of Death/CoDMod/Models/' + cause + '/' + model + '_' + keep_sex + '_' + start_age + '_to_' + end_age + '/Results/deaths_country.csv'), delimiter=",", quotechar='"')
        names = pl.array(csvdata.next())
        iso3_row = pl.where(names == 'iso3')[0][0] ## WHYYYYYY
        age_row = pl.where(names == 'age')[0][0] 
        year_row = pl.where(names == 'year')[0][0]
        cause_row = pl.where(names == 'cause')[0][0] 
        mean_row = pl.where(names == 'final_deaths_mean')[0][0] 
        lower_row = pl.where(names == 'final_deaths_lower')[0][0]
        upper_row = pl.where(names == 'final_deaths_upper')[0][0] 
        envelope_row = pl.where(names == 'envelope')[0][0]

        for row in csvdata:
            if row[iso3_row] == keep_iso3 and row[age_row] == keep_age and row[year_row] == keep_year: 
                d_cause.append(row[cause_row])
                d_deaths_mean.append(row[mean_row])
                d_deaths_lower.append(row[lower_row])
                d_deaths_upper.append(row[upper_row])
                d_envelope.append(row[envelope_row])
    d_cause = pl.array(d_cause)
    cf_mean = pl.array(d_deaths_mean, dtype='f') / pl.array(d_envelope, dtype = 'f')
    cf_lower = pl.array(d_deaths_lower, dtype='f') / pl.array(d_envelope, dtype = 'f')
    cf_upper = pl.array(d_deaths_upper, dtype='f') / pl.array(d_envelope, dtype = 'f')

    cf_rec = pl.np.core.records.fromarrays([d_cause,cf_mean, cf_lower, cf_upper], names='cause,est,lower,upper')

    return cf_rec

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

    # draw from distribution and back transform the simulated values
    J = len(cf_mean)
    return logit_normal_draw(cf_mean, std, N, J)


    


