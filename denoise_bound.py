# Python port by Maxime Woringer, Apr. 2016
# This function implements the FISTA method for TV-based denoising problems
#
# Based on the paper Amir Beck and Marc Teboulle, "Fast Gradient-Based Algorithms
# for Constrained Total Variation Image Denoising and Deblurring Problems"
# -----------------------------------------------------------------------
# Copyright (2008): Amir Beck and Marc Teboulle
# FISTA is distributed under the terms of the GNU General Public License 2.0.
# ----------------------------------------------------------------------
# INPUT
# Xobs ..............................an observed noisy image.
# lam ........................ parameter
# l ..................................... lower bound on the pixels' values
# u ..................................... upper bound on the pixels' values
# pars.................................parameters structure
# pars.MAXITER ..................... maximum number of iterations (Default=100)
# pars.epsilon ..................... tolerance for relative error used in
#                                                       the stopping criteria (Default=1e-4)
# pars.print ..........................  1 if a report on the iterations is
#                                                       given, 0 if the  report is silenced
# pars.tv .................................. type of total variation
#                                                      penatly.  'iso' for isotropic (default)
#                                                      and 'l1' for nonisotropic
#  
# OUTPUT
# X_den ........................... The solution of the problem 
#                                            min{||X-Xobs||^2+2*lam*TV(X
#                                            ) : l <= X_{i,j} <=u} 
# iter .............................  Number of iterations required to get
#                                            an optimal solution (up to a tolerance)
# fun_all ......................   An array containing all the function values obtained 
#                                             during the iterations

## ==== Importations
from __future__ import print_function
import sys
import numpy as np

## ==== Helper functions
def Lforward(P1, P2):
    (m2,n2) = P1.shape
    (m1,n1) = P2.shape

    if n2 != (n1+1):
        raise TypeError('dimensions are not consistent')
    if m1 != (m2+1):
        raise TypeError('dimensions are not consistent')

    m=m2+1
    n=n2
    X = np.zeros((m,n))

    X[0:(m-1),:]=P1
    X[:,0:(n-1)]+=P2
    X[1:(m),:]-=P1
    X[:,1:(n)]-=P2
    
    return X

def Ltrans(X):
    (m,n)=X.shape
    P1 = X[0:(m-1),:]-X[1:(m),:]
    P2 = X[:,0:(n-1)]-X[:,1:(n)]
    return (P1,P2)

## ==== Main functions
def denoise_bound(Xobs, lam, l, u, pars={}):
    ## Define the Projection onto the box
    if len(Xobs.shape)!=2:
        raise ValueError("Xobs must have len(shape)==2")
    if l==-np.inf and u==np.inf:
        project = lambda x: x
    elif type(l)==float and u==np.inf:
        #print ("WARNING: projection not very well tested", file=sys.stderr)
        project = lambda x: (l<x)*x + l*(x<=l)
    elif type(u)==float and l==-np.Inf:
        #print ("WARNING: projection not very well tested", file=sys.stderr)
        project = lambda x: (x<u)*x + (x>=u)*u
    elif type(u) == float and type(l)==float and l<u:
        #print ("WARNING: projection not very well tested", file=sys.stderr)
        project = lambda x: ((l<x)&(x<u))*x + (x>=u)*u + l*(x<=l)
    else:
        raise TypeError('lower and upper bound l,u should satisfy l<u. Both should be float (not int)')

    ## Assigning parameres according to pars and/or default values
    if pars.has_key('MAXITER'):
        MAXITER = pars['MAXITER']
    else:
        MAXITER = 100
    if pars.has_key('epsilon'):
        epsilon = pars['epsilon']
    else:
        epsilon = 1e-4
    if pars.has_key('print'):
        prnt = pars['print']
    else:
        prnt = 1
    if pars.has_key('tv'):
        tv = pars['tv']
    else:
        tv = 'iso'

    ## Initialize objects
    (m,n)=Xobs.shape
    P1 = np.zeros((m-1,n))
    P2 = np.zeros((m,n-1))
    R1 = np.zeros((m-1,n))
    R2 = np.zeros((m, n-1))
    tk=1.
    tkp1=1.
    count=0.
    i=0
    D=np.zeros((m,n))
    fval=np.inf
    fun_all=[]

    if prnt:
        txt = """
***********************************
*Solving with FGP/FISTA**
***********************************
#iteration  function-value  relative-difference
---------------------------------------------------------------------
"""
        print (txt)

    ## ==== Main loop
    while (i<MAXITER) and (count<5):
        fold=fval
        i+=1 ## updating the iteration counter
        Dold = D ## Storing the old value of the current solution
        
        ## Computing the gradient of the objective function
        Pold1=P1
        Pold2=P2
        tk=tkp1
        D=project(Xobs-lam*Lforward(R1,R2))
        (Q1,Q2)=Ltrans(D)

        ## Taking a step towards minus of the gradient
        P1 = R1 + 1/(8*lam)*Q1
        P2 = R2 + 1/(8*lam)*Q2
        
        ## Peforming the projection step
        if tv=='iso':
            A = np.vstack((P1, np.zeros((n))))**2 + np.hstack((P2,np.zeros((m,1))))**2
            A[A<1]=1
            A=A**0.5
            P1 /= A[0:(m-1),:]
            P2 /= A[:,0:(n-1)]
        elif tv=='l1':
            PP1 = np.abs(P1.copy())
            PP2 = np.abs(P2.copy())
            PP1[PP1<1]=1
            PP2[PP2<1]=1
            P1 /= PP1
            P2 /= PP2
        else:
            raise InputError('unknown type of total variation. should be iso or l1')
        
        ## Updating R and t
        tkp1 = (1+ (1+4*tk**2)**0.5)/2
        R1 = P1+(tk-1)/tkp1 * (P1-Pold1)
        R2 = P2+(tk-1)/tkp1 * (P2-Pold2)

        re=np.linalg.norm(D-Dold, ord='fro')/np.linalg.norm(D, ord='fro')
        if re<epsilon:
            count+=1
        else:
            count=0

        C=Xobs-lam*Lforward(P1,P2)
        PC=project(C)
        fval = -np.linalg.norm(C-PC, 'fro')**2+np.linalg.norm(C, 'fro')**2
        if fun_all != []:
            fun_all = np.vstack((fun_all,fval))
        else:
            fun_all = fval

        if prnt and fval>fold:
            print ("{}\t{}\t{}*".format(i,fval, np.linalg.norm(D-Dold,'fro')/np.linalg.norm(D,'fro')) )
        elif prnt:
            print ("{}\t{}\t{}".format(i,fval, np.linalg.norm(D-Dold,'fro')/np.linalg.norm(D,'fro')) )
        X_den=D
        iter=1
    return (X_den,iter,fun_all)
