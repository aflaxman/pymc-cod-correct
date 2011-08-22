import graphics
import data
import models

import pylab as pl

iso3='BWA'

F, causes = data.get_cod_data(iso3=iso3)
model, pi = models.fit_latent_simplex(F)

graphics.plot_F_and_pi(F, pi, causes, iso3)

pl.savefig('/home/j/Project/Models/cod-correct/%s.png'%iso3)
