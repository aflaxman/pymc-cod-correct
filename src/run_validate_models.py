import validate_models as vm 
import pylab as pl 
import csv
import os 

reps = 10

truth = [(pl.ones(3)/3.0,[0.01, 0.01, 0.01]),     ## 0-2: different cause fractions, same (low) noise
         ([0.1, 0.1, 0.8],[0.01, 0.01, 0.01]), 
         ([0.1, 0.45, 0.45],[0.01, 0.01, 0.01]),
         (pl.ones(3)/3.0,[0.20, 0.20, 0.20]),     ## 3-5: different cause fractions, same (high) noise
         ([0.1, 0.1, 0.8],[0.20, 0.20, 0.20]), 
         ([0.1, 0.45, 0.45],[0.20, 0.20, 0.20]), 
         (pl.ones(3)/3.0,[0.01, 0.05, 0.10]),     ## 6-12: different cause fractions, different noise 
         ([0.1, 0.1, 0.8],[0.01, 0.05, 0.10]), 
         ([0.1, 0.1, 0.8],[0.01, 0.10, 0.05]), 
         ([0.1, 0.1, 0.8],[0.05, 0.10, 0.01]), 
         ([0.45, 0.45, 0.1],[0.01, 0.05, 0.10]), 
         ([0.45, 0.45, 0.1],[0.01, 0.10, 0.05]),
         ([0.45, 0.45, 0.1],[0.05, 0.10, 0.01])]

scenarios = pl.shape(truth)[0]         
cause_count = pl.shape(truth)[2]

for i in range(scenarios): 
    vm.run_on_cluster(dir='../data/v%s' % (i), true_cf = truth[i][0], true_std = truth[i][1], reps=reps, tag=str(i))

## the code that follows below can't be run until after everything submitted to the cluster above is finished 
models = []
causes = []
mean_abs_err = []
median_abs_err = []
mean_rel_err = []
median_rel_err = []
mean_csmf_accuracy = []
median_csmf_accuracy = []
mean_coverage_bycause = []
median_coverage_bycause = []
mean_coverage = []
percent_total_coverage = []
scenario = []

for i in range(scenarios):
    for j in ['bad_model', 'latent_dirichlet']: 
        read = csv.reader(open('../data/v%s/%s_summary.csv' % (i, j)))
        read.next()
        for row in read: 
            models.append(row[0])
            causes.append(row[1])
            mean_abs_err.append(row[2])
            median_abs_err.append(row[3])
            mean_rel_err.append(row[4])
            median_rel_err.append(row[5])
            mean_csmf_accuracy.append(row[6])
            median_csmf_accuracy.append(row[7])
            mean_coverage_bycause.append(row[8])
            median_coverage_bycause.append(row[9])
            mean_coverage.append(row[10])
            percent_total_coverage.append(row[11])
            scenario.append(i)

all = pl.np.core.records.fromarrays([scenario, models, causes, mean_abs_err, median_abs_err, mean_rel_err, median_rel_err, 
                                     mean_csmf_accuracy, median_csmf_accuracy, mean_coverage_bycause, median_coverage_bycause, mean_coverage, 
                                     percent_total_coverage], 
                                    names=['scenario', 'model', 'cause', 'mean_abs_err', 'median_abs_err', 
                                     'mean_rel_err', 'median_rel_err', 'mean_csmf_accuracy', 'median_csmf_accuracy', 
                                     'mean_covearge_bycause', 'median_coverage_bycause', 'mean_coverage', 
                                     'percent_total_coverage'])
pl.rec2csv(all, fname='../data/all_summary_metrics.csv')  
    
