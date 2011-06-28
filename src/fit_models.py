import os 
import csv
import pymc as mc 
import pylab as pl 
import scipy.stats.mstats as sp

os.chdir('C:/Users/ladwyer/pymc-cod-correct/src/') # TODO: make this STATA-ism go away!
import data
import graphics
import models 

years = range(1980, 2011)
isos = ['USA', 'ZMB'] 
sexes = ['female'] 
ages = ['20'] 

for iso in isos: 
    for sex in sexes: 
        for age in ages: 
            for year in years: 
            
                year = str(year)
               
                cf = data.get_cod_data(level=1, keep_age = age, keep_iso3 = iso, keep_sex = sex, keep_year= year)
                X = data.sim_cod_data(1000, cf)
                pl.rec2csv(pl.np.core.records.fromarrays(X.T), '../data/sim_cod_data_%s_%s.csv' % (iso, year))
                
                bad_model = models.bad_model(X) 
                pl.rec2csv(pl.np.core.records.fromarrays(bad_model.T), '../data/bad_model_%s_%s.csv' % (iso, year))

                latent_dirichlet = models.fit_latent_dirichlet(X, 100, 50, 5)
                pl.rec2csv(pl.np.core.records.fromarrays(latent_dirichlet.T), '../data/latent_dirichlet_%s_%s.csv' % (iso, year))

