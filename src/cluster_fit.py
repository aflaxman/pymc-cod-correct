import os
import re
import data
import models
import sys
import pylab as pl 
import scipy.stats.mstats as st

# set parameters 
outdir = '/home/j/Project/Causes of Death/Under Five Deaths/CoD Correct Output'
indir = '/home/j/Project/Causes of Death/Under Five Deaths/CoD Correct Input Data' 
age, iso3, sex = sys.argv[1:4]
full_dir = '%s/v02_prep_%s' % (indir, iso3)

# get cause list 
causes = list(set([file.split('+')[1] for file in os.listdir(full_dir) if re.search(age, file)]))
causes.remove('HIV') # temporary until Miriam fixes the HIV files 

# gather data and fit model 
cf = data.get_cod_data(full_dir, causes, age, iso3, sex)
m, pi = models.fit_latent_simplex(cf) 

# calculate summary measures
N, T, J = pi.shape
mean = pi.mean(0)
lower = pl.array([[st.mquantiles(pi[:,t,j], 0.025)[0] for j in range(J)] for t in range(T)])
upper = pl.array([[st.mquantiles(pi[:,t,j], 0.975)[0] for j in range(J)] for t in range(T)])

# format summary and save
output = pl.np.core.records.fromarrays(mean.T, names=['%s_mean' % c for c in causes])
output = pl.rec_append_fields(output, ['%s_lower' % c for c in causes], lower.T)
output = pl.rec_append_fields(output, ['%s_upper' % c for c in causes], upper.T)
pl.rec2csv(output, '%s/%s+%s+%s+summary.csv' % (outdir, iso3, age, sex))

# format all sims and save 
pi.shape = (N*T, J)
years = pl.array([t for s in range(N) for t in range(1980, 2012)])
sim = pl.array([s for s in range(N) for t in range(1980, 2012)])
output = pl.np.core.records.fromarrays(pi.T, names=causes)
output = pl.rec_append_fields(output, 'year', years)
output = pl.rec_append_fields(output, 'sim', sim)
pl.rec2csv(output, '%s/%s+%s+%s.csv' % (outdir, iso3, age, sex))
