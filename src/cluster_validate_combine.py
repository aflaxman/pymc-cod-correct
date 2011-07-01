import os 
import sys
import pylab as pl 

import validate_models

reps = int(sys.argv[1])
truth = pl.csv2rec('../data/truth.csv')

validate_models.combine_output(len(truth.true_cf), 'bad_model', '../data', reps, True)
validate_models.combine_output(len(truth.true_cf), 'latent_dirichlet', '../data', reps, True)

validate_models.clean_up('bad_model', '../data', reps)
validate_models.clean_up('latent_dirichlet', '../data', reps)
os.remove('../data/truth.csv')