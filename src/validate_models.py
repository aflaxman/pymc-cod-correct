import pymc as mc 
import pylab as pl 

import data
import graphics
import models 
reload(models)

def calc_coverage(true_cf, preds):
    """
    """
    J = len(true_cf)
    hpd = mc.utils.hpd(preds, 0.05)
    covered = [hpd[cause][0] < true_cf[cause] < hpd[cause][1] for cause in range(J)]
    return pl.array(covered)
    
def calc_quality_metrics(true_cf, preds): 
    """ 
    Calculate the CSMF accuracy, aboslute error, and relative error for the 
        provided true and predicted CSMFs
    """
    
    pred_cf = pl.array(preds.mean(0))
    true_cf = pl.array(true_cf)
    
    csmf_accuracy = 1. - pl.sum(pl.absolute(pred_cf - true_cf)) / (2*(1-min(true_cf))) 
    abs_err = pl.absolute(pred_cf-true_cf)
    rel_err = pl.absolute(pred_cf - true_cf)/true_cf
    coverage = calc_coverage(true_cf, preds)
    all = pl.np.core.records.fromarrays([abs_err, rel_err, pl.ones(len(pred_cf))*csmf_accuracy, coverage], names=['abs_err','rel_err','csmf_accuracy','coverage'])
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
    bad_model_metrics = calc_quality_metrics(true_cf, bad_model)
    
    # fit latent dirichlet model, calculate 95% HPD region and fit metrics 
    m, latent_dirichlet = models.fit_latent_dirichlet(X) # TODO: Need to find the appropriate settings here
    latent_dirichlet_metrics = calc_quality_metrics(true_cf, latent_dirichlet)

    return bad_model_metrics, latent_dirichlet_metrics

def combine_output(model, dir, reps):
    """
    """

    abs_err = pl.zeros(3, dtype='f').view(pl.recarray) # TODO: 3 is the number of causes: need to figure out how to do this with an indefinite number of causes...
    rel_err = pl.zeros(3, dtype='f').view(pl.recarray)
    coverage = pl.zeros(3, dtype='f').view(pl.recarray)
    csmf_accuracy = []
    for i in range(reps): 
        metrics = pl.csv2rec('%s/metrics_%s_%i.csv' % (dir, model, i))
        abs_err = pl.vstack((abs_err, metrics.abs_err))
        rel_err = pl.vstack((rel_err, metrics.rel_err))
        coverage = pl.vstack((coverage, metrics.coverage))
        csmf_accuracy.append(metrics.csmf_accuracy[0])
    abs_err = abs_err[1:,]
    rel_err = rel_err[1:,]
    coverage = coverage[1:,]
    return abs_err, rel_err, csmf_accuracy, coverage

def run_all_sequentially(dir, true_cf=[0.3, 0.3, 0.4], true_std=[0.01, 0.01, 0.01], reps=5): 
    """
    """
    for i in range(reps): 
        bad_model_metrics, latent_dirichlet_metrics = validate_once(true_cf, true_std)
        pl.rec2csv(bad_model_metrics, '%s/metrics_bad_model_%i.csv' % (dir, i)) 
        pl.rec2csv(latent_dirichlet_metrics, '%s/metrics_latent_dirichlet_%i.csv' % (dir, i))
    b_abs_err, b_rel_err, b_csmf_accuracy, b_coverage = combine_output('bad_model', dir, reps)
    l_abs_err, l_rel_err, l_csmf_accuracy, l_coverage = combine_output('latent_dirichlet', dir, reps)    
    # not sure what to do with all this output yet... 
    return b_abs_err, b_rel_err, b_csmf_accuracy, b_coverage, l_abs_err, l_rel_err, l_csmf_accuracy, l_coverage
 




