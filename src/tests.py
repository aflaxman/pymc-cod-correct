""" Tests """

# matplotlib will open windows during testing unless you do the following
import matplotlib
matplotlib.use("AGG") 

import models

class TestClass:
   def setUp(self):
      pass

   def test_models(self):
       assert True, 'Write test, fail, write code, pass'

   def test_sim_data(self):
      import data
      sim_data = data.sim_data(10)
      assert sim_data.shape == (10,2), 'Should be 10x2 matrix of data (%s found)' % str(sim_data.shape)
