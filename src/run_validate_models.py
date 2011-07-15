import validate_models
reload(validate_models)
import pylab as pl

reps = 5

# truths = [## base case: completely equal cause fractions, over an extended period of time. equal stds.
          # [[pl.ones(3)/3 for i in range(10)],
           # [0.05, 0.05, 0.05]],
          # ## unequal cause fractions that don't change over time. equal stds.
          # [[[0.1,0.2,0.7] for i in range(10)],
           # [0.05, 0.05, 0.05]],
          # ## unequal cause fractions that don't change over time. stds positively correlated with cf
          # [[[0.1,0.2,0.7] for i in range(10)],
           # [0.05, 0.1, 0.2]],
          # ## unequal cause fractions that don't change over time. stds negatively correlated with cf
          # [[[0.1,0.2,0.7] for i in range(10)],
           # [0.2, 0.1, 0.05]],
        
          # ## 2 changing cause fractions. equal stds. 
          # [[pl.ones(3)/3 + i*pl.array([-0.01, 0.01, 0]) for i in range(10)], 
           # [0.05, 0.05, 0.05]],
          # ## 2 more rapidly changing cause fractions. equal stds. 
          # [[pl.ones(3)/3 + i*pl.array([-0.03, 0.03, 0]) for i in range(10)], 
           # [0.05, 0.05, 0.05]], 
          # ## 3 changing cause fractions. equal stds.  
          # [[pl.ones(3)/3 + i*pl.array([-0.03, 0.015, 0.015]) for i in range(10)], 
           # [0.05, 0.05, 0.05]], 
          # ## 2 changing cause fractions. stds positively correlated with cf. 
          # [[pl.ones(3)/3 + i*pl.array([-0.01, 0.01, 0]) for i in range(10)], 
           # [0.05, 0.05, 0.05]],
           
          # ## 2 changing cause fractions. unequal stds. 
          # [[pl.ones(3)/3 + i*pl.array([-0.01, 0.01, 0]) for i in range(10)], 
           # [0.05, 0.05, 0.05]],
          # ## 2 changing cause fractions. unequal stds. 
          # [[pl.ones(3)/3 + i*pl.array([-0.01, 0.01, 0]) for i in range(10)], 
           # [0.05, 0.2, 0.1]],
          # ## 2 changing cause fractions. unequal stds. 
          # [[pl.ones(3)/3 + i*pl.array([-0.01, 0.01, 0]) for i in range(10)], 
           # [0.1, 0.2, 0.05]],           
 
         # ]
         
truths = [[[pl.ones(3)/3 for i in range(2)], [0.1,0.1,0.1]], [[pl.ones(3)/3 for i in range(2)], [0.1,0.1,0.1]]]

validate_models.run_all_scenarios(truths, reps, '../data') 


