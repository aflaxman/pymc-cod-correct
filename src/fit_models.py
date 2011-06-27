import os 
import csv
import pymc as mc 
import pylab as pl 
import scipy.stats.mstats as sp

os.chdir('C:/Users/ladwyer/pymc-cod-correct/src/') 
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
                dir = 'C:/Users/ladwyer/pymc-cod-correct/data/' + iso + '_' + sex + '_' + age + '_' + year 
                os.mkdir(dir)
                causes, mean, lower, upper = data.get_cod_data(level=1, keep_age = age, keep_iso3 = iso, keep_sex = sex, keep_year= year)
                X = data.sim_cod_data(1000, mean, lower, upper) 
                file = open(dir + '/cod_sims.txt', 'w')
                write = csv.writer(file, delimiter=';')
                [write.writerow(X[i,]) for i in range(pl.shape(X)[0])]
                file.close()
                #raw_estimates_cf = [mean, lower, upper]

                bad_model = models.bad_model(X)
                file = open(dir + '/bad_model.txt', 'w')
                write= csv.writer(file, delimiter=';')
                [write.writerow(bad_model[i,]) for i in range(pl.shape(bad_model)[0])]
                file.close()
                #bad_model_cf = [bad_model.mean(axis=0), pl.array(sp.mquantiles(bad_model, axis=0, prob=0.025))[0], pl.array(sp.mquantiles(bad_model, axis=0, prob=0.975))[0]]

                vars = models.latent_dirichlet(X)
                m = mc.MCMC(vars, db='txt', dbname=dir + '/latent_dirichlet')
                m.sample(1000000, 500000, 500, verbose=0) 
                #latent_dirichlet_cf = [m.pi.trace().mean(0), pl.array(sp.mquantiles(m.pi.trace(), axis=0, prob=0.025))[0], pl.array(sp.mquantiles(m.pi.trace(), axis=0, prob=0.975))[0]]


