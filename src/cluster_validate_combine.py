import os 
import sys
import pylab as pl 

import validate_models
reload(validate_models)

reps = int(sys.argv[1])
dir = str(sys.argv[2])
truth = pl.csv2rec('%s/truth.csv' % (dir))

validate_models.combine_output(len(truth.true_cf), 'bad_model', dir, reps, True)
validate_models.combine_output(len(truth.true_cf), 'latent_dirichlet', dir, reps, True)

validate_models.clean_up('bad_model', dir, reps)
validate_models.clean_up('latent_dirichlet', dir, reps)
os.remove('%s/truth.csv' %dir)