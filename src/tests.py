""" Tests """

# matplotlib will open windows during testing unless you do the following
import matplotlib
matplotlib.use("AGG") 
import pylab as pl
import pymc as mc

import models
import data
import graphics

class TestClass:
    def setUp(self):
        self.X = data.sim_data(10)

    def test_sim_data(self):
        sim_data = data.sim_data(10)
        assert sim_data.shape == (10,2), 'Should be 10x2 matrix of data (%s found)' % str(sim_data.shape)

        sim_data = data.sim_data(10, [.1, .4, .5], [.1, .1, .1])
        assert sim_data.shape == (10,3), 'Should be 10x3 matrix of data (%s found)' % str(sim_data.shape)

    def test_plot_sim_data(self):
        X = data.sim_data(10, [.1, .4, .5], [.1, .1, .1])
        graphics.plot_sim_data(X)

        assert list(pl.axis()) == [0., 1., 0., 1.], 'plot limits should be unit square, (%s found)' % str(pl.axis())

    def test_bad_model(self):
        X = data.sim_data(10)
        Y = models.bad_model(X)
        assert pl.all(Y.sum(axis=1) == 1), 'should be all ones, (%s found)' % str(Y)

        # test again for 10x3 dataset
        X = data.sim_data(10, [.1, .4, .5], [.1, .1, .1])
        Y = models.bad_model(X)
        assert pl.all(Y.sum(axis=1) == 1), 'should be all ones, (%s found)' % str(Y)

    def test_good_model(self):
        vars = models.latent_dirichlet(self.X)
        assert pl.sum(vars['pi'].value) <= 1.0, 'pi value should sum to at most 1, (%s found)' % sum(vars['pi'].value)
        m = mc.MCMC(vars)
        m.sample(10)
