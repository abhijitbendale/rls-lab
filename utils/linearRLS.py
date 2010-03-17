import sys
import os

import numpy as np
import scipy as sp
import scipy.io
import scipy.linalg

def lrlsloo(X, Y, lambdas=None):
    """
    Input:
    X = input matrix
    Y = labels
    lamdas = used as defaults like the ones mentioned in Rifkins code
    Output:
    w = weights
    loos = total LOO error vector for linear RLS with lambdas

    this is part of docstrings from rifkins code.
    --
    X is a data matrix whose ROWS are the data points.
    If we have n points in d dimensions, X is n by d.
    Y are the "labels".   Y is n by cl, where cl is the number
    of "classes" (the number of regression problems to be
    solved).
    lambdas is a vector of length l, containing the different
    regularization parameters to try.  DEFAULT: logspace(-6,6,30).

    """
#    U,S,V = sp.linalg.svd(X)

    # Ha Hong: if X has less samples than its dimensionality, don't do full SVD.
    if X.shape[0] < X.shape[1]:
        U,S0,V = sp.linalg.svd(X, full_matrices = False)
    else:
        U,S0,V = sp.linalg.svd(X)
    # Ha Hong: if X is not square, S needs to be in a matrix form (not vector!).
    S = sp.zeros(X.shape[0])
    S[:len(S0)] = S0

    S2 = S**2
    w, loos = lrlsloo_ll(X, U, S2, Y, lambdas)

    return w, loos
    
def lrlsloo_ll(X, U, S2, Y, lambdas=None):
    """
    A "lower level" function that does the work in lrlsloo
    Input:
    X = input matrix
    U, S2 = values from SVD
    Y = labels
    lambdas = used as defaults like the ones mentioned in Rifkins code
    Output:
    w = weights
    loos = total LOO error vector for linear RLS with lambdas
    """

    X_row, X_col = X.shape
    cl = Y.shape[1]
    l = len(lambdas)

    ws = sp.zeros((l, cl, X_col))
    loos = sp.zeros((l,cl))
    loos[:] = sp.inf

    for i in range(l):
        wsll, looerrsll = lrlsloo_ll1(X, U, S2, Y, lambdas[i])
        ws[i][:][:] = wsll
        loos[i][:] = sp.sqrt( sp.sum( looerrsll**2, axis=0) )

    return ws, loos

def lrlsloo_ll1(X, U, S2, Y, lambd):
    """
    Computes ws and the actual LOO errors for a single value of lambda.
    """

    n = X.shape[0]
    cl = Y.shape[1]
    
    UtY = sp.dot(U.transpose(), Y)
    
    inner0  = (1/(S2 + lambd) - 1/lambd)
    # Ha Hong: this might be better..
    inner = sp.zeros((n,n))
    for i, elem in enumerate(inner0):
        inner[i,i] = elem
    # Original: Uinner = U *( np.ones((n,1)) * inner)
    Uinner = sp.dot(U, inner)
    
    c = Y/lambd + sp.dot(Uinner,UtY)
    dGi = 1/lambd + sp.sum(Uinner*U, axis = 1)

    looerrs = c.ravel()/dGi
    ws = sp.dot(c.transpose(), X)
    return ws.ravel(), looerrs
