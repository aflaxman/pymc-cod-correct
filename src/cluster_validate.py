import os 
import sys
import pylab as pl 

import validate_models

i = int(sys.argv[1])
dir = str(sys.argv[2])
truth = pl.csv2rec('../data/truth.csv')

validate_models.validate_once(truth.true_cf, truth.true_std, True, dir, i)
