import graphics
import data
import models

F, causes = data.get_cod_data(iso3='ZMB')
model, pi = models.fit_latent_simplex(F)

graphics.plot_F_and_pi(F, pi, causes)


