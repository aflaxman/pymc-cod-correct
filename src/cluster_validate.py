import os 
import sys
import pylab as pl 

import validate_models
reload(validate_models)
import data
reload(data)

i = int(sys.argv[1])
dir = str(sys.argv[2])

truth = data.csv2array('%s/truth.csv' % (dir))
true_std = truth[0]
true_cf = truth[1:]

validate_models.validate_once(true_cf, true_std, True, dir, i)
