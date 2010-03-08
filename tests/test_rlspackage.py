import sys
from numpy.testing import *
import scipy as sp
from scipy.io import loadmat
from numpy.testing import assert_array_almost_equal

# Remove this later ob
sys.path.append('../')
sys.path.append('../utils/')

from linearRLS import *
from non_linear_rls import *

# add tests for large arrays as Ha has suggested


#-------------------------------------------------------------------------------
class test_array_shapes:

    # setup here random elements and labels
    lambdas = sp.logspace(-6,6,50)
    samples = sp.random.random((10,10))
    labels = sp.asarray([[1,1,1,-1,-1, 1,1,1,-1,-1]])

    def test_lrls_wdim(self):
        """
        Test to check dimensions of weights for linear rls
        """
        w, loos = lrlsloo(self.samples, self.labels.T, self.lambdas)
        assert self.lambdas.shape[0] == w.shape[0]

    def test_lrls_loosdim(self):
        """
        Test to check dimensions of leave one out error for linear rls
        """
        w, loos = lrlsloo(self.samples, self.labels.T, self.lambdas)
        assert self.lambdas.shape[0] == loos.shape[0]

    
    def test_rls_wdim(self):
        """
        Test to check dimensions of weights for non-linear rls
        """
        w, loos = rlsloo(self.samples, self.labels.T, self.lambdas)
        assert self.lambdas.shape[0] == w.shape[0]

    def test_rls_loosdim(self):
        """
        Test to check dimensions of leave one out error for non- linear rls
        """
        w, loos = rlsloo(self.samples, self.labels.T, self.lambdas)
        assert self.lambdas.shape[0] == loos.shape[0]

class test_regression:

    # regression test with already stored matfiles

    # get stored samples and labels
    samples = loadmat('data/smp.mat')['samples']
    labels = loadmat('data/smp.mat')['labels'].T
    lambdas = sp.logspace(-6,6,30)

    # get stored output for linear and non-linear rls
    # computer from Ryan Rifkin's matlab code
    mat_lrls_w = loadmat('data/linear_rls.mat')['w']
    mat_lrls_loos = loadmat('data/linear_rls.mat')['loos']
    
    mat_rls_w = loadmat('data/non_linear_rls.mat')['w']
    mat_rls_loos = loadmat('data/non_linear_rls.mat')['loos']
    
    def test_linear_rls_w(self):
        """
        regression test for linear rls w
        """
        w, loos = lrlsloo(self.samples, self.labels.T, self.lambdas)
        assert_array_almost_equal(w, self.mat_lrls_w)
        
    def test_linear_rls_loos(self):
        """
        regression test for linear rls loos
        """
        w, loos = rlsloo(self.samples, self.labels.T, self.lambdas)
        assert_array_almost_equal(loos, self.mat_rls_loos)
        
    def test_non_linear_rls_w(self):
        """
        regression test for linear rls w
        """
        w, loos = rlsloo(self.samples, self.labels.T, self.lambdas)
        assert_array_almost_equal(w, self.mat_rls_w)
         
    def test_non_linear_rls_loos(self): 
        """
        regression test for linear rls loos
        """
        w, loos = rlsloo(self.samples, self.labels.T, self.lambdas)
        assert_array_almost_equal(loos, self.mat_rls_loos)



if __name__ == "__main__":
#    main()
    run_module_suite()
