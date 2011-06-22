""" Tests """

# matplotlib will open windows during testing unless you do the following
import matplotlib
matplotlib.use("AGG") 
import pylab as pl

import models
import data
import graphics

class TestClass:
    def setUp(self):
        pass

    def test_models(self):
        assert True, 'Write test, fail, write code, pass'

    def test_sim_data(self):
        sim_data = data.sim_data(10)
        assert sim_data.shape == (10,2), 'Should be 10x2 matrix of data (%s found)' % str(sim_data.shape)

        sim_data = data.sim_data(10, [.1, .4, .5], [.1, .1, .1])
        assert sim_data.shape == (10,3), 'Should be 10x3 matrix of data (%s found)' % str(sim_data.shape)

    def test_plot_sim_data(self):
        X = data.sim_data(10, [.1, .4, .5], [.1, .1, .1])
        graphics.plot_sim_data(X)

        assert list(pl.axis()) == [0., 1., 0., 1.], 'plot limits should be unit square, (%s found)' % str(pl.axis())
