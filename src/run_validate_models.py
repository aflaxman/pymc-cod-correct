import validate_models
reload(validate_models)
import pylab as pl

reps = 50

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

validate_models.run_all_scenarios(truth, reps, '../data')
