import os 
import sys
import pylab as pl 

import validate_models
reload(validate_models)

scenarios = int(sys.argv[1])
cause_count = int(sys.argv[2])
dir = str(sys.argv[3])

validate_models.compile_all_results(scenarios, cause_count, dir)
