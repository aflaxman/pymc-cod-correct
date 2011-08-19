import os
import re
import data
import models
import sys
import pylab as pl 

outdir = '/home/j/Project/Causes of Death/Under Five Deaths/Cod Correct Output'
indir = '/home/j/Project/Causes of Death/Under Five Deaths/CoD Correct Input Data' 

age, iso3, sex = sys.argv[1:4]
full_dir = '%s/v02_prep_%s' % (indir, iso3)

causes = list(set([file.split('+')[1] for file in os.listdir(full_dir) if re.search(age, file)]))
causes.remove('HIV') # temporary until Miriam fixes the HIV files 
causes.remove('Tetanus') # temporary until Miriam reformats the Tetanus files 

cf = data.get_cod_data(full_dir, causes, age, iso3, sex)
m, pi = models.fit_latent_simplex(cf, 100, 50, 5) 
N, T, J = pi.shape

pi.shape = (N*T, J)
years = pl.array([t for s in range(N) for t in range(1980, 2012)])
sim = pl.array([s for s in range(N) for t in range(1980, 2012)])

output = pl.np.core.records.fromarrays(pi.T, names=causes)
output = pl.rec_append_fields(output, 'year', years)
output = pl.rec_append_fields(output, 'sim', sim)

pl.rec2csv(output, '%s/%s+%s+%s.csv' % (outdir, iso3, age, sex))
