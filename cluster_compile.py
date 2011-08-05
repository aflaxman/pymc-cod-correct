import os 
import sys
import pylab as pl 

import validate_models
reload(validate_models)

scenarios = int(sys.argv[1])
dir = str(sys.argv[2])

validate_models.compile_all_results(scenarios, dir)
