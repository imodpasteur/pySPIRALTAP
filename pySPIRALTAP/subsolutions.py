# Python port by Maxime Woringer, Jun. 2016
# This module implements in Python several algoritms coming from the `denoise` library
#
# Based on the paper Amir Beck and Marc Teboulle, "Fast Gradient-Based Algorithms
# for Constrained Total Variation Image Denoising and Deblurring Problems"
# -----------------------------------------------------------------------
# Copyright (2008): Amir Beck and Marc Teboulle
# FISTA is distributed under the terms of the GNU General Public License 2.0.
# ----------------------------------------------------------------------

# ==== Importations
from __future__ import print_function
import numpy as np
import sys

# ==== Error & helper functions
def todo():
    """Not implemented error function"""
    print('ERROR: This function is not yet implemented, please be patient!', file=sys.stderr)
    raise NotImplementedError


# ==== l2-l1 denoising
def constrainedl2l1denoise(y, W, WT, tau, mu, miniter, maxiter,
                           stopcriterion, tolerance,
                           verbose = 0):
    """
    Put some documentation here

    For now, the tolerance is the relative duality gap, and is the only
    convergence criterion implemented
    Also, in the future it would be good to output the number of nonzeros
    in theta, even though we output a solution in x.
    May be worthwhile to clean this up and keep as a separate function,
    however this would entail coding such that it checks the inputs and
    outputs and such...
    """

    if 0:
        print("W identity?:", np.all(y==W(y)))
        print("y:", y.shape, y.sum())
        print("W:", W(y).shape)
        print("WT:", WT(y).shape)
        print("tau:", tau)
        print("mu:", mu)
        print("miniter:", miniter)
        print("maxiter:", maxiter)
        print("stopcriterion:", stopcriterion)
        print("tolerance:", tolerance)
        np.savetxt('./dev/y.txt', y)
        print('Saved `y` input file')
        
    gamma = np.zeros(y.shape) #y.size
    lamb = gamma.copy() ## Renamed lambda to lamb as lambda is a reserved keyword in Python
    WTlamb = WT(lamb)
    y = WT(y)
    iter = 1
    converged = False

    while (iter <= miniter) or (iter <= maxiter) and not converged:
        gamma = -y - WTlamb
        gamma[gamma < -tau] = -tau
        gamma[gamma > tau] = tau
        lamb = -W(y+gamma)-mu
        lamb[lamb < 0] = 0
        WTlamb = WT(lamb)
        theta = y + gamma + WTlamb

        ## Check for convergence
        if iter >= miniter: ## no need to check if miniter not reached
            if stopcriterion == 0: ## Just exhaust maxiter
                converged = False
            elif stopcriterion == 1:
                primal_obj = 0.5*((theta-y)**2).sum() + tau*np.abs(theta).sum()
                dual_obj = -0.5*(theta**2).sum() + 0.5*(y**2).sum() - mu*lamb.sum()
                rel_duality_gap = np.abs(primal_obj-dual_obj)/max(-primal_obj,dual_obj);
                if verbose: ## display some stuff
                    duality_gap = primal_obj - dual_obj
                    print('l1Den: It={}, PObj={}, DObj={}, DGap={}, RDGap={}'.format(iter,
                                                                                     primal_obj,
                                                                                     dual_obj,
                                                                                     duality_gap,
                    rel_duality_gap))
                if rel_duality_gap <= tolerance or rel_duality_gap == np.inf:
                    converged = True
        iter += 1
    return np.abs(W(theta)) # Output
