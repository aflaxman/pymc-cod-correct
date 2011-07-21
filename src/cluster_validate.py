import os 
import sys
import pylab as pl 

import validate_models
reload(validate_models)
import data
reload(data)

i = int(sys.argv[1])
dir = str(sys.argv[2])

true_std = data.csv2array('%s/truth_std.csv' % (dir))
true_cf = data.csv2array('%s/truth_cf.csv' % (dir))
std_bias = data.csv2array('%s/truth_bias.csv' % (dir))[0]

validate_models.validate_once(true_cf, true_std, std_bias, True, dir, i)
