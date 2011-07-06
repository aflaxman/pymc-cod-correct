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
     
    def test_sim_data_2(self): 
        sims = 10000 
        test1 = pl.zeros(3, dtype='f').view(pl.recarray)
        for i in range(sims): 
            temp = data.sim_data(1, [0.1,0.1,0.8], [0.01,0.01,0.01])
            test1 = pl.vstack((test1, temp))
        test1 = test1[1:,]
        test2 = data.sim_data(sims, [0.1,0.1,0.8], [0.01, 0.01, 0.01])
        diff = (test1.mean(0) - test2.mean(0))/test1.mean(0)
        assert pl.allclose(diff, 0, atol=0.01), 'should be close to zero, (%s found)' % str(diff)

    def test_get_cod_data(self): 
        cf = data.get_cod_data(level=1)
        assert len(cf.cause) == 3 and cf.cause.dtype == 'S1'
        assert len(cf.est) == 3 and cf.est.dtype == 'float32'
        assert len(cf.lower) == 3 and cf.lower.dtype == 'float32'
        assert len(cf.upper) == 3 and cf.upper.dtype == 'float32' 
        # this only tests that level 1 causes work; the function takes awhile to run at higher levels, so it may not be feasible to repeatedly test this at higher levels. 

    def test_sim_cod_data(self): 
        cf = data.get_cod_data(level=1)
        X = data.sim_cod_data(10, cf)
        assert pl.shape(X) == (10, 3)

    def test_sim_data_for_validation(self): 
        sim_data = data.sim_data_for_validation(10, [0.5, 0.5], [0.1, 0.1])
        assert sim_data.shape == (10,2), 'Should be 10x2 matrix of data (%s found)' % str(sim_data.shape)

        sim_data = data.sim_data_for_validation(10, [.1, .4, .5], [.1, .1, .1])
        assert sim_data.shape == (10,3), 'Should be 10x3 matrix of data (%s found)' % str(sim_data.shape)
        
    def test_plot_sim_data(self):
        X = data.sim_data(10, [.1, .4, .5], [.1, .1, .1])
        graphics.plot_sim_data(X)
        assert list(pl.axis()) == [0., 1., 0., 1.], 'plot limits should be unit square, (%s found)' % str(pl.axis())
        graphics.plot_all_sim_data(X)

    def test_bad_model(self):
        X = data.sim_data(10)
        Y = models.bad_model(X)
        assert pl.allclose(Y.sum(axis=1), 1), 'should be all ones, (%s found)' % str(Y.sum(axis=1))

        # test again for 10x3 dataset
        X = data.sim_data(10, [.1, .4, .5], [.1, .1, .1])
        Y = models.bad_model(X)
        assert pl.allclose(Y.sum(axis=1), 1), 'should be all ones, (%s found)' % str(Y.sum(axis=1))

    def test_good_model(self):
        vars = models.latent_dirichlet(self.X)
        assert pl.sum(vars['pi'].value) <= 1.0, 'pi value should sum to at most 1, (%s found)' % sum(vars['pi'].value)
        m = mc.MCMC(vars)
        m.sample(10)

if __name__ == '__main__':
    import nose
    nose.run()
    
