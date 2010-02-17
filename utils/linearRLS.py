import sys
import os

import numpy as np
import scipy as sp
import scipy.io
import scipy.linalg

def lrlsloo(X, Y, lambdas):

    U,S,V = sp.linalg.svd(X)
    S2 = S**2
    w, loos = lrlsloo_ll(X, U, S2, Y, lambdas)
    return w, loos
    
def lrlsloo_ll(X, U, S2, Y, lambdas):

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
        n = X.shape[0]
        cl = Y.shape[1]

        UtY = sp.dot(U.transpose(), Y)

        inner  = (1/(S2 + lambd) - 1/lambd)
        Uinner = U *( np.ones((n,1)) * inner)

        c = Y/lambd + sp.dot(Uinner,UtY)
        dGi = 1/lambd + sp.sum(Uinner*U, axis = 1)

        looerrs = c.ravel()/dGi
        ws = sp.dot(c.transpose(), X)
        return ws.ravel(), looerrs
