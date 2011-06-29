import pymc as mc 
import pylab as pl 

import data
import graphics
import models 
reload(models)

# TODO: make the following code run repeatedly (possibly distributed on the cluster)

# generate simulation data
true_cf = pl.ones(3) / 3.
true_std = .01 * pl.ones(3)

X = data.sim_data_for_validation(1000, true_cf, true_std) # TODO: make a way to correlate true_std and est_std

# fit models to data
bad_model = models.bad_model(X)
m, latent_dirichlet = models.fit_latent_dirichlet(X)

# measure quality of fits
# TODO: measure quality of bad model
# TODO: select metrics for quality (CSMF Acc, MAE, RMSE, Absolute, Relative)
pred_cf = latent_dirichlet.mean(0)
csmf_accuracy = 1. - pl.sum(pl.absolute(pred_cf - true_cf)) / (2*(1-min(true_cf)))
abs_err = pl.absolute(pred_cf - true_cf)
print abs_err

# TODO: determine if 95% HPD interval includes truth
ui = mc.utils.hpd(latent_dirichlet, .05)
