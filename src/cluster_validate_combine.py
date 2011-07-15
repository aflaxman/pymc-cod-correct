import os 
import sys
import pylab as pl 

import validate_models
reload(validate_models)
import data
reload(data)

reps = int(sys.argv[1])
dir = str(sys.argv[2])

truth = data.csv2array('%s/truth.csv' % (dir))
true_std = truth[0]
true_cf = truth[1:]
T, J = true_cf.shape

validate_models.combine_output(J, T, 'bad_model', dir, reps, True)
validate_models.combine_output(J, T, 'latent_simplex', dir, reps, True)

validate_models.clean_up('bad_model', dir, reps)
validate_models.clean_up('latent_simplex', dir, reps)
os.remove('%s/truth.csv' %dir)
