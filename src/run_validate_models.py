import validate_models
reload(validate_models)
import pylab as pl

reps = 100

truths = [#### Effect of unequal cause fractions 
          ## equal, time-invariant cfs; equal, time-invariant unbiased stds
          [[pl.ones(3)/3 for i in range(10)],
           [[0.05, 0.05, 0.05]], 
           [1., 1., 1.]],
          ## unequal, time-invariant cfs; equal, time-invariant unbiased stds
          [[[0.1,0.2,0.7] for i in range(10)],
           [[0.05, 0.05, 0.05]],
           [1., 1., 1.]],
          
          #### Effect of unequal uncertainty 
          ## equal, time-invariant cfs; unequal, time-invariant unbiased stds
          [[pl.ones(3)/3 for i in range(10)],
           [[0.05, 0.25, 0.5]],
           [1., 1., 1.]],     
          ## unequal, time-invariant cfs; unequal, time-invariant unbiased stds (stds positively correlated with cfs)
          [[[0.1,0.2,0.7] for i in range(10)],
           [[0.05, 0.25, 0.5]],
           [1., 1., 1.]],
          ## unequal, time-invariant cfs; unequal, time-invariant unbiased stds (stds negatively correlated with cfs)
          [[[0.1,0.2,0.7] for i in range(10)],
           [[0.5, 0.25, 0.05]],
           [1., 1., 1.]],
        
          #### Effect of time-varying cause fractions (equal and constant uncertainty) 
          ## truth[5] 2 time-varying cfs; equal, time-invariant unbiased stds
          [[pl.ones(3)/3 + i*pl.array([-0.01, 0.01, 0]) for i in range(10)], 
           [[0.05, 0.05, 0.05]],
           [1., 1., 1.]],
          ## 2 more rapidly time-varying cfs; equal, time-invariant unbiased stds
          [[pl.ones(3)/3 + i*pl.array([-0.03, 0.03, 0]) for i in range(10)], 
           [[0.05, 0.05, 0.05]],
           [1., 1., 1.]], 
          ## 3 time-varying cfs; equal, time-invariant unbiased stds.  
          [[pl.ones(3)/3 + i*pl.array([-0.03, 0.015, 0.015]) for i in range(10)], 
           [[0.05, 0.05, 0.05]],
           [1., 1., 1.]], 
          
          #### Effect of time-varying cause fractions (unequal and constant uncertainty) 
          ## truth[8] 2 time-varying cfs; unequal, time-invariant unbiased stds
          [[pl.ones(3)/3 + i*pl.array([-0.01, 0.01, 0]) for i in range(10)], 
           [[0.05, 0.5, 0.25]],
           [1., 1., 1.]],
          ## 2 time-varying cfs; unequal, time-invariant unbiased stds
          [[pl.ones(3)/3 + i*pl.array([-0.01, 0.01, 0]) for i in range(10)], 
           [[0.5, 0.25, 0.05]],
           [1., 1., 1.]],        
          
          #### Effect of time-varying uncertainty (when cf is invariant) 
          ## unequal time-invariant cfs; unequal, time-varying unbiased stds
          [[[0.1,0.2,0.7] for i in range(10)],
           [pl.ones(3)/10+i*pl.array([-0.005, 0.02, 0]) for i in range(10)],
           [1., 1., 1.]], 
          ## 11) unequal time-invariant cfs; unequal, time-varying unbiased stds
          [[[0.7,0.2,0.1] for i in range(10)],
           [pl.ones(3)/10+i*pl.array([-0.005, 0.02, 0]) for i in range(10)],
           [1., 1., 1.]], 
          
          #### Effect of biased uncertainty
          ## unequal time-invariant cfs; unequal, time-invariant biased stds
          [[[0.1,0.2,0.7] for i in range(10)],
           [[0.05, 0.25, 0.5]],
           [1.5, 1., 1.]],
          ## unequal time-invariant cfs; unequal, time-invariant biased stds
          [[[0.1,0.2,0.7] for i in range(10)],
           [[0.05, 0.25, 0.5]],
           [1., 1.5, 1.]],
          ## unequal time-invariant cfs; unequal, time-invariant biased stds
          [[[0.1,0.2,0.7] for i in range(10)],
           [[0.05, 0.25, 0.5]],
           [1., 1., 1.5]],
          ## unequal time-invariant cfs; unequal, time-invariant biased stds
          [[[0.1,0.2,0.7] for i in range(10)],
           [[0.05, 0.05, 0.05]],
           [1.5, 1., 1.]],          
         ]
         
validate_models.run_all_scenarios(truths, reps, '../data') 


