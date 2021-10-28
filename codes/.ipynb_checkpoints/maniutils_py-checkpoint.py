# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 14:18:38 2017

@author: jbobin
"""

import os
import sys
WrapLoc = os.environ['PYWRAP3_LOC']
sys.path.insert(1,WrapLoc)
import maniutils
import numpy as np

def LogSn(X1,X2):

    Nx = np.shape(X1)[0]
    Nz = np.shape(X1)[1]

    Ny = 1
    Niter = 1
    Step_size = 0.

    Out = maniutils.ManiMR(Nx,Ny,Nz,Niter,Step_size).LogSn(X1.astype('double'),X2.astype('double'))

    return Out

def ExpSn(X,V,Theta):

    Nx = np.shape(X)[0]
    Nz = np.shape(X)[1]

    Ny = 1
    Niter = 1
    Step_size = 0.

    Out = maniutils.ManiMR(Nx,Ny,Nz,Niter,Step_size).ExpSn(X.astype('double'),np.concatenate((V,Theta),axis=0).astype('double'))

    return Out

def FrechetMean(X,W,Niter=100,Step_size = 0.1):

    Nx = np.shape(X)[0]
    Ny = np.shape(X)[1]
    Nz = np.shape(X)[2]

    Out = maniutils.ManiMR(Nx,Ny,Nz,Niter,Step_size).FrechetMean(X.astype('double'),W.astype('double'))

    return Out

def FrechetRobustMean(X,W,Niter=100,Step_size = 0.1,mu=0.1):

    Nx = np.shape(X)[0]
    Ny = np.shape(X)[1]
    Nz = np.shape(X)[2]

    Out = maniutils.ManiMR(Nx,Ny,Nz,Niter,Step_size).FrechetRobustMean(X.astype('double'),W.astype('double'),np.array([mu]).astype('double'))

    return Out

def pFrechetMean(X,W,Ain=None,Niter=100,Step_size = 0.1):

    Nx = np.shape(X)[0]
    Ny = np.shape(X)[1]
    Nz = np.shape(X)[2]

    if Ain is None:
        Ain = X[:,:,0]

    Out = maniutils.ManiMR(Nx,Ny,Nz,Niter,Step_size).pFrechetMean(Ain.astype('double'),X.astype('double'),W.astype('double'))

    return Out

def pFrechetRobustMean(X,W,Ain=None,Niter=100,Step_size = 0.1,mu=0.1):

    Nx = np.shape(X)[0]
    Ny = np.shape(X)[1]
    Nz = np.shape(X)[2]

    if Ain is None:
        Ain = X[:,:,0]

    Out = maniutils.ManiMR(Nx,Ny,Nz,Niter,Step_size).pFrechetRobustMean(Ain.astype('double'),X.astype('double'),W.astype('double'),np.array([mu]).astype('double'))

    return Out


def TestFrechetMean(st=0.1,Nit=100,mu=0.1):

    import numpy as np
    import matplotlib.pyplot as plt

    U = np.tile(np.random.randn(10,1),(1,5))
    U = U/np.maximum(0.,np.linalg.norm(U,axis=0))
    V = np.tile(np.random.randn(10,1),(1,5))
    V = V/np.maximum(0.,np.linalg.norm(V,axis=0))

    Theta = np.zeros((5,1))
    Theta[0] = 0.2
    Theta[1] = 0.3
    Theta[2] = -0.15
    Theta[3] = 1.3
    Theta[4] = -0.35

    X = ExpSn(U,V,Theta.T)
    X = X/np.maximum(0.,np.linalg.norm(X,axis=0))
    Z = X.reshape((10,1,5))

    W = 1./5.*np.ones((5,))

    Fm = FrechetMean(Z,W,Niter=Nit,Step_size = st)
    Frm = FrechetRobustMean(Z,W,Niter=Nit,Step_size = st,mu=mu)

    plt.close()
    plt.plot(Fm,'--o',lw=4,markersize=10,alpha=0.5),plt.plot(Frm,'--s',lw=4,markersize=10,alpha=0.5),plt.plot(Z[:,0,:]),plt.show()

    return Z,W,Fm,Frm
