""" Class for wrangling data
"""

import pylab as pl
import pymc as mc
import csv 
import os

def array2csv(Y, fname):
    """
    write a 2-dimensional recarray to a csv file
    """
    
    pl.rec2csv(pl.np.core.records.fromarrays(Y.T), fname)

def sim_data(N, true_csmf=[.3, .7], true_csmf_sd=[.2, .05], sum_to_one=True):
    """ 
    Create an NxJ matrix of simulated data (J is determined by the length 
    of true_csmf). 
    
    true_csmf - a list of true cause fractions (must sum to one)
    true_csmf_sd - a list of the standard deviations corresponding to the true csmf's
    """

    if sum_to_one == True: 
        assert pl.allclose(sum(true_csmf), 1), 'The sum of elements of true_csmf must equal 1' 
        assert len(true_csmf)==len(true_csmf_sd), 'The length of true_csmf and true_csmf_sd must be the same'
    J = len(true_csmf)

    ## transform the mean and sd to logit space 
    transformed_csmf = mc.logit(true_csmf)
    transformed_var = []
    for pi_i, sigma_pi_i in zip(true_csmf, true_csmf_sd):
        # TODO: verify that this actually the appropriate equation; the standard deviation of the data returned doesn't match what is being provided in the arguments
        transformed_var.append( (1/(pi_i*(1-pi_i)))**2 * sigma_pi_i**2 )

    ## draw from distribution
    return my_draw(true_csmf, true_csmf_sd, N, J)

def my_draw(cf_mean, std, N, J):
    std = pl.array(std)
    if mc.__version__ == '2.0rc2': # version on Omak 
        X = mc.rnormal(mu=cf_mean, tau=std**-2, size=N)  
        Y = mc.invlogit(X).reshape(N,J) 
    else: 
        X = mc.rnormal(mu=cf_mean, tau=std**-2, size=(N,J))
        Y = mc.invlogit(X)
    return Y

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
    return my_draw(cf_mean, std, N, J)



















