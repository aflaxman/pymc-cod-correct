import pymc as mc 
import pylab as pl 
import os
import subprocess
import csv

import data
import graphics
import models 
reload(models)

def calc_coverage(true_cf, preds):
    """
    Calculates the 95% hpd region for each cause from the provided model output; Returns an array 
    the same length as true_cf which contains 1 for each cause that is included in the 95% hpd 
    region and 0 elsewhere.
    """
    
    T, J = true_cf.shape
    hpd = mc.utils.hpd(preds, 0.05)
    covered = [[hpd[t][cause][0] < true_cf[t][cause] < hpd[t][cause][1] for cause in range(J)] for t in range(T)]
    return pl.array(covered)

def calc_quality_metrics(true_cf, true_std, preds): 
    """ 
    Calculate the CSMF accuracy, absolute error, and relative error for the 
    provided true and predicted CSMFs.
    """
    
    T, J = pl.array(true_cf).shape
    pred_cf = pl.array(preds.mean(0))
    true_cf = pl.array(true_cf)
    
    csmf_accuracy = 1. - pl.absolute(pred_cf-true_cf).sum(1) / (2*(1-true_cf.min(1))) 
    abs_err = pl.absolute(pred_cf-true_cf)
    rel_err = pl.absolute(pred_cf-true_cf)/true_cf
    coverage = calc_coverage(true_cf, preds)
    all = pl.np.core.records.fromarrays([true_cf.ravel(), pl.array([true_std for t in range(T)]).ravel(), 
                                         abs_err.ravel(), rel_err.ravel(), pl.array([[i for j in range(J)] for i in csmf_accuracy]).ravel(),
                                         coverage.ravel(), pl.array([[j for j in range(J)] for t in range(T)]).ravel(),
                                         pl.array([[t for j in range(J)] for t in range(T)]).ravel()],
                                        names=['true_cf', 'true_std', 'abs_err','rel_err','csmf_accuracy','coverage','cause','time'])
    return all
    
def validate_once(true_cf = [pl.ones(3)/3.0, pl.ones(3)/3.0], true_std = 0.01*pl.ones(3), save=False, dir='', i=0):
    """
    Generate a set of simulated estimates for the provided true cause fractions; Fit the bad model and 
    the latent simplex model to this simulated data and calculate quality metrics. 
    """ 
    
    # generate simulation data
    X = data.sim_data_for_validation(1000, true_cf, true_std)

    # fit bad model, calculate fit metrics 
    bad_model = models.bad_model(X)
    bad_model_metrics = calc_quality_metrics(true_cf, true_std, bad_model)
    
    # fit latent simplex model, calculate fit metrics 
    m, latent_simplex = models.fit_latent_simplex(X)
    latent_simplex_metrics = calc_quality_metrics(true_cf, true_std, latent_simplex)
    
    # fit other version of latent simplex model, calculate fit metrics
    m, latent_simplex_v2 = models.fit_latent_simplex_v2(X)
    latent_simplex_v2_metrics = calc_quality_metrics(true_cf, true_std, latent_simplex_v2)
    
    # either write results to disk or return them 
    if save: 
        pl.rec2csv(bad_model_metrics, '%s/metrics_bad_model_%i.csv' % (dir, i)) 
        pl.rec2csv(latent_simplex_metrics, '%s/metrics_latent_simplex_%i.csv' % (dir, i))
        pl.rec2csv(latent_simplex_v2_metrics, '%s/metrics_latent_simplex_v2_%i.csv' % (dir, i))
    else: 
        return bad_model_metrics, latent_simplex_metrics, latent_simplex_v2_metrics

def combine_output(J, T, model, dir, reps, save=False):
    """
    Combine output on absolute error, relative error, csmf_accuracy, and coverage from from
    multiple runs of validate_once. Either saves the output to the disk, or returns arays
    for each. 
    """

    cause = pl.zeros(J*T, dtype='f').view(pl.recarray)
    time = pl.zeros(J*T, dtype='f').view(pl.recarray)
    abs_err = pl.zeros(J*T, dtype='f').view(pl.recarray) 
    rel_err = pl.zeros(J*T, dtype='f').view(pl.recarray)
    coverage = pl.zeros(J*T, dtype='f').view(pl.recarray)
    csmf_accuracy = pl.zeros(J*T, dtype='f').view(pl.recarray)

    for i in range(reps): 
        metrics = pl.csv2rec('%s/metrics_%s_%i.csv' % (dir, model, i))
        cause = pl.vstack((cause, metrics.cause))
        time = pl.vstack((time, metrics.time))
        abs_err = pl.vstack((abs_err, metrics.abs_err))
        rel_err = pl.vstack((rel_err, metrics.rel_err))
        coverage = pl.vstack((coverage, metrics.coverage))
        csmf_accuracy = pl.vstack((csmf_accuracy, metrics.csmf_accuracy))

    cause = cause[1:,]
    time = time[1:,]    
    abs_err = abs_err[1:,]
    rel_err = rel_err[1:,]
    coverage = coverage[1:,]
    csmf_accuracy = csmf_accuracy[1:,]

    mean_abs_err = abs_err.mean(0)
    median_abs_err =  pl.median(abs_err, 0)
    mean_rel_err = rel_err.mean(0)
    median_rel_err = pl.median(rel_err, 0)
    mean_csmf_accuracy = csmf_accuracy.mean(0)
    median_csmf_accuracy = pl.median(csmf_accuracy, 0)
    mean_coverage_bycause = coverage.mean(0)
    mean_coverage = coverage.reshape(reps, T, J).mean(0).mean(1)
    percent_total_coverage = (coverage.reshape(reps, T, J).sum(2)==3).mean(0)
    mean_coverage = pl.array([[i for j in range(J)] for i in mean_coverage]).ravel()
    percent_total_coverage = pl.array([[i for j in range(J)] for i in percent_total_coverage]).ravel()

    models = pl.array([[model for j in range(J)] for i in range(T)]).ravel()
    true_cf = metrics.true_cf
    true_std = metrics.true_std

    all = pl.np.core.records.fromarrays([models, cause[0], time[0], true_cf, true_std, mean_abs_err, median_abs_err, mean_rel_err, median_rel_err, 
                                         mean_csmf_accuracy, median_csmf_accuracy, mean_coverage_bycause, mean_coverage, percent_total_coverage], 
                                        names=['model', 'cause', 'time', 'true_cf', 'true_std', 'mean_abs_err', 'median_abs_err', 
                                         'mean_rel_err', 'median_rel_err', 'mean_csmf_accuracy', 'median_csmf_accuracy', 
                                         'mean_covearge_bycause', 'mean_coverage', 'percent_total_coverage'])   
    
    if save: 
        pl.rec2csv(all, '%s/%s_summary.csv' % (dir, model)) 
    else: 
        return all
    
def clean_up(model, dir, reps):
    """
    Delete temporary files produced when running validate_once multiple times.
    """
    
    for i in range(reps):
        os.remove('%s/metrics_%s_%i.csv' % (dir, model, i))

def run_all_sequentially(dir='../data', true_cf = [pl.ones(3)/3.0, pl.ones(3)/3.0], true_std = 0.01*pl.ones(3), reps=5): 
    """
    Runs validate_once multiple times (as sepcified by reps) for the given true_cf and 
    true_std. Combines the output and cleans up the temp files. This is all accomplished
    sequentially on the local machine. 
    """

    T, J = pl.array(true_cf).shape
    if os.path.exists(dir) == False: os.mkdir(dir)
    
    # repeatedly run validate_once and save output 
    for i in range(reps): 
        validate_once(true_cf, true_std, True, dir, i)

    # combine all output across repetitions 
    combine_output(J, T, 'bad_model', dir, reps, True)
    combine_output(J, T, 'latent_simplex', dir, reps, True)  
    combine_output(J, T, 'latent_simplex_v2', dir, reps, True)  
    
    # delete intermediate files 
    clean_up('bad_model', dir, reps)
    clean_up('latent_simplex', dir, reps)
    clean_up('latent_simplex_v2', dir, reps)

def run_on_cluster(dir='../data', true_cf = [pl.ones(3)/3.0, pl.ones(3)/3.0], true_std = 0.01*pl.ones(3), reps=5, tag=''):
    """
    Runs validate_once multiple times (as specified by reps) for the given true_cf and 
    true_std. Combines the output and cleans up the temp files. This accomplished in 
    parallel on the cluster. This function requires that the files cluster_shell.sh 
    (which allows for submission of a job for each iteration), cluster_validate.py (which
    runs validate_once for each iteration), and cluster_validate_combine.py (which 
    runs combine_output all exist. The tag argument allows for adding a string to the job 
    names so that this function can be run multiple times simultaneously and not have 
    conflicts between jobs with the same name. 
    """

    T, J = pl.array(true_cf).shape  
    if os.path.exists(dir) == False: os.mkdir(dir)

    # write true_cf and true_std to file
    data.rec2csv_2d(pl.vstack((true_std, true_cf)), '%s/truth.csv' % (dir))
    
    # submit all individual jobs to retrieve true_cf and true_std and run validate_once
    all_names = [] 
    for i in range(reps): 
        name = 'cc%s_%i' % (tag, i)
        call = 'qsub -cwd -N %s cluster_shell.sh cluster_validate.py %i "%s"' % (name, i, dir)
        subprocess.call(call, shell=True)
        all_names.append(name)
    
    # submit job to run combine_output and clean_up 
    hold_string = '-hold_jid %s ' % ','.join(all_names)
    call = 'qsub -cwd %s -N cc%s_comb cluster_shell.sh cluster_validate_combine.py %i "%s"' % (hold_string, tag, reps, dir)
    subprocess.call(call, shell=True)  

def compile_all_results (scenarios, dir='../data'):
    """
    Compiles the results across multiple scenarios produced by running run_on_cluster on each 
    one into a single sv file. The specified directory must be where where the results of 
    running run_on_cluster for each scenario are stored (each is a sub-directory named v0, v1, etc.)
    and is also where the output from this function will be saved.    
    """

    models = []
    causes = []
    time = []
    true_cf = []
    true_std = []
    mean_abs_err = []
    median_abs_err = []
    mean_rel_err = []
    median_rel_err = []
    mean_csmf_accuracy = []
    median_csmf_accuracy = []
    mean_coverage_bycause = []
    mean_coverage = []
    percent_total_coverage = []
    scenario = []

    for i in range(scenarios):
        for j in ['bad_model', 'latent_simplex', 'latent_simplex_v2']: 
            read = csv.reader(open('%s/v%s/%s_summary.csv' % (dir, i, j)))
            read.next()
            for row in read: 
                models.append(row[0])
                causes.append(row[1])
                time.append(row[2])
                true_cf.append(row[3])
                true_std.append(row[4])
                mean_abs_err.append(row[5])
                median_abs_err.append(row[6])
                mean_rel_err.append(row[7])
                median_rel_err.append(row[8])
                mean_csmf_accuracy.append(row[9])
                median_csmf_accuracy.append(row[10])
                mean_coverage_bycause.append(row[11])
                mean_coverage.append(row[12])
                percent_total_coverage.append(row[13])
                scenario.append(i)

    all = pl.np.core.records.fromarrays([scenario, models, time, true_cf, true_std, causes, mean_abs_err, median_abs_err, mean_rel_err, median_rel_err, 
                                         mean_csmf_accuracy, median_csmf_accuracy, mean_coverage_bycause, mean_coverage, percent_total_coverage], 
                                        names=['scenario', 'model', 'time', 'true_cf', 'true_std', 'cause', 'mean_abs_err', 'median_abs_err', 
                                         'mean_rel_err', 'median_rel_err', 'mean_csmf_accuracy', 'median_csmf_accuracy', 
                                         'mean_covearge_bycause', 'mean_coverage', 'percent_total_coverage'])
    pl.rec2csv(all, fname='%s/all_summary_metrics.csv' % (dir))  
    
def run_all_scenarios (truths, reps, dir='../data'): 
    """
    Runs run_on_cluster for each set of true cause fraction and standard deviation provided. This 
    function takes a list of pairs of lists, with the first element of each pair specifying the 
    true cause fractions and the second element of each pair specifying the corresponding 
    true standard deviations. This function creates a series of folders (one for each item in
    'truths') inside the specified directory.
    """

    scenarios = int(len(truths))
    J = len(truths[0][1])
    
    all_names = []
    for i in range(scenarios): 
        run_on_cluster(dir='%s/v%s' % (dir, i), true_cf = truths[i][0], true_std = truths[i][1], reps=reps, tag=str(i))
        all_names.append('cc%s_comb' % (i))
        
    hold_string = '-hold_jid %s ' % ','.join(all_names)
    call = 'qsub -cwd %s -N cc_compile cluster_shell.sh cluster_compile.py %i %i "%s"' % (hold_string, scenarios, J, dir)
    subprocess.call(call, shell=True)

    




