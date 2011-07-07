import validate_models as vm 
import pylab as pl 
import os 

reps = 10

true_cfs = [pl.ones(3)/3.0,     ## 0-2: different cause fractions, same (low) noise
            [0.1, 0.1, 0.8], 
            [0.1, 0.45, 0.45],
            pl.ones(3)/3.0,     ## 3-5: different cause fractions, same (high) noise
            [0.1, 0.1, 0.8], 
            [0.1, 0.45, 0.45], 
            pl.ones(3)/3.0,     ## 6-12: different cause fractions, different noise 
            [0.1, 0.1, 0.8], 
            [0.1, 0.1, 0.8], 
            [0.1, 0.1, 0.8], 
            [0.45, 0.45, 0.1], 
            [0.45, 0.45, 0.1],
            [0.45, 0.45, 0.1]]

true_stds = [[0.01, 0.01, 0.01], 
             [0.01, 0.01, 0.01], 
             [0.01, 0.01, 0.01],
             [0.20, 0.20, 0.20], 
             [0.20, 0.20, 0.20], 
             [0.20, 0.20, 0.20], 
             [0.01, 0.05, 0.10], 
             [0.01, 0.05, 0.10], 
             [0.01, 0.10, 0.05], 
             [0.05, 0.10, 0.01], 
             [0.01, 0.05, 0.10], 
             [0.01, 0.10, 0.05], 
             [0.05, 0.10, 0.01]]

cause_count = len(true_cfs[0])
scenarios = len(true_cfs)

for i in range(scenarios): 
    vm.run_on_cluster(dir='../data/v%s' % (i), true_cf = true_cfs[i], true_std = true_stds[i], reps=reps, tag=str(i))

# ## this needs to move somewhere else: it can't be run until all the jobs submitted above are run. 
# for i in range(scenarios):
    # for j in ['bad_model', 'latent_dirichlet']: 
        # if i == 0 and j == 'bad_model': 
            # all = pl.csv2rec('../data/v%s/bad_%s_summary.csv' % (i, model))
        # else: 
            # temp = pl.csv2rec('../data/v%s/bad_%s_summary.csv' % (i, model))
            # all = pl.hstack((all, temp))
# ## this is not working
# #scenario = pl.array([i for i in range(scenarios) for j in range(2*cause_count)])
# #temp = pl.vstack((scenario, all))
# pl.rec2csv(all, '../data/all_results.csv')
