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

for i in range(len(true_cfs)): 
    vm.run_on_cluster(dir='../data/v%s' % (i), true_cf = true_cfs[i], true_std = true_stds[i], reps=reps, tag=str(i))

