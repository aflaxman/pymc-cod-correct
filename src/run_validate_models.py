import validate_models
reload(validate_models)
import pylab as pl

reps = 5

truths = [ [ [[0.1,0.1,0.8],
              [0.1,0.2,0.7]],
             [0.1,0.1,0.1] ], 
           [ [[0.1,0.1,0.8],
              [0.1,0.2,0.7]],
             [0.1,0.1,0.1] ] ] 

validate_models.run_all_scenarios(truths, reps, '../data')
