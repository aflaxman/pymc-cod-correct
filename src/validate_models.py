import pymc as mc 
import pylab as pl 

import data
import graphics
import models 
reload(models)

def calc_quality_metrics(pred_cf, true_cf): 
    """ 
    Calculate the CSMF accuracy, aboslute error, and relative error for the 
        provided true and predicted CSMFs
    """
    
    pred_cf = pl.array(pred_cf)
    true_cf = pl.array(true_cf)
    csmf_accuracy = 1. - pl.sum(pl.absolute(pred_cf - true_cf)) / (2*(1-min(true_cf))) 
    abs_err = pl.absolute(pred_cf-true_cf)
    rel_err = pl.absolute(pred_cf - true_cf)/true_cf
    all = pl.np.core.records.fromarrays([abs_err, rel_err, pl.ones(len(pred_cf))*csmf_accuracy], names=['abs_err','rel_err','csmf_accuracy'])
    return all
    
def validate_once(true_cf = pl.ones(3)/3.0, true_std = 0.01*pl.ones(3)):
    """
    Generate a set of simulated estimates for the provided true cause fractions; Fit the bad model and 
        the latent dirichlet model to this data and calculate quality metrics. 
    """ 
    
    # generate simulation data
    X = data.sim_data_for_validation(1000, true_cf, true_std)

    # fit bad model, calculate 95% HPD region and fit metrics 
    bad_model = models.bad_model(X)
    bad_model_metrics = calc_quality_metrics(bad_model.mean(0), true_cf)
    bad_model_hpd = mc.utils.hpd(X, 0.05)
    
    # fit latent dirichlet model, calculate 95% HPD region and fit metrics 
    m, latent_dirichlet = models.fit_latent_dirichlet(X) # TODO: Need to find the appropriate settings here
    latent_dirichlet_metrics = calc_quality_metrics(latent_dirichlet.mean(0), true_cf)
    latent_dirichlet_hpd = mc.utils.hpd(X, 0.05)

    return bad_model_metrics, latent_dirichlet_metrics, bad_model_hpd, latent_dirichlet_hpd 
    

def write_validate_once(metrics, hpd, model, dir, num): 
    """
    metrics - output from validate_once 
    model - model name 
    fpath - the filepath 
    num - a numeric to identify the file (this is to allow for making multiple files for the same model) 
    """
    
    pl.rec2csv(metrics, '%s/metrics_%s_%i.csv' % (dir, model, num))
    data.array2csv(hpd, '%s/hpd_%s_%i.csv' % (dir, model, num))

def combine_output(model, dir, nums):
    """
    """

    abs_err = pl.zeros(3, dtype='f').view(pl.recarray) # TODO: 3 is the number of causes: need to figure out how to do this with an indefinite number of causes...
    rel_err = pl.zeros(3, dtype='f').view(pl.recarray)
    csmf_accuracy = []
    for num in nums: 
        metrics = pl.csv2rec('%s/metrics_%s_%i.csv' % (dir, model, num))
        abs_err = pl.vstack((abs_err, metrics.abs_err))
        rel_err = pl.vstack((rel_err, metrics.rel_err))
        csmf_accuracy.append(metrics.csmf_accuracy[0])
    abs_err = abs_err[1:,]
    rel_err = rel_err[1:,]
    return abs_err, rel_err, csmf_accuracy
    
def calc_coverage(true_cf, model, dir, nums):
    """
    """
    J = len(true_cf)
    hpd = pl.zeros((J, len(nums)), dtype='f')

    for count in range(len(nums)): 
        hpds = pl.csv2rec('%s/hpd_%s_%i.csv' % (dir, model, nums[count]))
        for cause in range(J):
            hpd[cause,count] = true_cf[cause] > hpds[cause][0] and true_cf[cause] < hpds[cause][1]      
    return hpd

def run_all_sequentially(dir, true_cf=[0.3, 0.3, 0.4], true_std=[0.01, 0.01, 0.01], reps=range(5)): 
    """
    """
    for i in reps: 
        bm, lm, bh, lh = validate_once(true_cf, true_std)
        write_validate_once(bm, bh, 'bad_model', dir, i)    
        write_validate_once(lm, lh, 'latent_dirichlet', dir, i)  
    b_abs_err, b_rel_err, b_csmf_accuracy = combine_output('bad_model', dir, reps)
    l_abs_err, l_rel_err, l_csmf_accuracy = combine_output('latent_dirichlet', dir, reps)    
    b_hpd = calc_coverage(true_cf, 'bad_model', dir, reps)
    l_hpd = calc_coverage(true_cf, 'latent_dirichlet', dir, reps)
    # not sure what to do with all this output yet... 
    return b_abs_err, b_rel_err, b_csmf_accuracy, b_hpd, l_abs_err, l_rel_err, l_csmf_accuracy, l_hpd
 




