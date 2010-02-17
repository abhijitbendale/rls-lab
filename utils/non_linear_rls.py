import scipy as sp
import scipy.linalg

#---------------------------------------------------------------------------------------
def rlsloo_ll1( V, D, Y, lambd):
	"""
	Computes cs and the actual LOO errors for a single value of lambda. (lambd)
	"""
        n = V.shape[0]
        cl = Y.shape[1]
        
        inner  = 1/(D + lambd)
        VtY = sp.dot(V.T, Y)

        # -- till here works fine

        # Because of signs of D are flipped (scipy.linalg.eig returns
        # flipped signs for complex part of the eigenvalues)
        in_dot = sp.ones((n,1)) * inner
        
        ViD = V * in_dot
        cs = sp.dot(ViD, VtY)
        dGi = sp.sum(ViD*V, axis = 1)

        #check matrix dimensions
        looerrs = cs/dGi
        cs = cs.transpose()
        return cs.ravel(), looerrs



#---------------------------------------------------------------------------------------
def rlsloo_ll(V, D, Y, lambdas=None):
	"""
	Input:
	V, D = eigenvectors and eigen values from eigen value decomposition
	lambdas = default used in our computation
	Output:
	cs = is a matrix of size representing function weights of lambda
	loos = total LOO error vector for nonlinear RLS with lambda
	
	"""

	n  = V.shape[0]
	cl = Y.shape[1]
	l = len(lambdas)

	cs = sp.zeros((l, cl, n))
	loos = sp.zeros((l,cl))
	loos[:] = sp.inf

	for i in range(l):
		#        print D, Y, lambdas[i]
		csll, looerrsll = rlsloo_ll1(V, D, Y, lambdas[i])
		cs[i][:][:] = csll
		loos[i][:] = sp.sqrt( sp.sum( looerrsll**2, axis=0) )
		
	return cs, loos

#---------------------------------------------------------------------------------------
def rlsloo(K, Y, lambdas=None):
	"""
	Nonlinear regularized least squares.
	Input:
	K = is an n by n symmetric kernel matrix.
	Y = Classes
	lamdas = default as used by rifkin lambdas = logspace(-6,6,30);

	Output:
	cs = is a matrix of size representing function weights of lambda
	loos = total LOO error vector for nonlinear RLS with lambda
	
	"""


    # This is where the problem is:
    # we are getting different signs
	D, V = scipy.linalg.eig(K)
	cs, loos = rlsloo_ll(V, D, Y, lambdas)
	return cs, loos
