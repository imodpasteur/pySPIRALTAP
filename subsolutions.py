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

    gamma = np.zeros(y.size)
    lamb = gamma.copy() ## Renamed lambda to lamb as lambda is a reserved keyword in Python
    WTlamb = WT(lamb)
    y = WT(y)
    iter = 1
    converged = False

    while (iter <= miniter) or (iter <= maxiter) and not converged:
        ## Assess the values of the Matlab code please. Use OCtave :-)

        iter += 1
    
    # while (iter <= miniter) || ((iter <= maxiter) && not(converged))
    #     %disp(['Subiter = ',num2str(iter)])
    #     gamma       = min( max( -tau, -y - WTlamb), tau);
    #     lamb      = max( -W(y + gamma) - mu, 0);
    #     WTlamb    = WT(lamb);
    #     theta       = y + gamma + WTlamb;

    #     % Check for convergence
    #     if iter >= miniter % no need to check if miniter not reached
    #         switch stopcriterion
    #             case 0
    #                 % Just exhaust maxiter
    #                 converged = 0; #=False
    #             case 1
    #                 primal_obj = sum( (theta(:)-y(:)).^2)./2 + tau.*sum(abs(theta(:)));
    #                 dual_obj = -sum( theta(:).^2)./2 + sum(y(:).^2)./2-mu.*sum(lamb(:));
    #                 % Need this for what's in verbose:
    #                 % duality_gap = primal_obj - dual_obj;
    #                 rel_duality_gap = abs(primal_obj-dual_obj)/max(-primal_obj,dual_obj);
    #                 if verbose
    #                     % display some stuff
    #                     % fprintf('l1Den: It=%4d, PObj=%13.5e, DObj=%13.5e, DGap=%13.5e, RDGap=%13.5e\n', iter,primal_obj,dual_obj,duality_gap,rel_duality_gap)
    #                 end
    #                 if (rel_duality_gap <= tolerance) || isinf(rel_duality_gap)
    #                     converged = 1; =True
    #                 end
    #         end
    #     end
    #     iter = iter + 1;
    # end    

    todo() ## Raise NotImplemented...
    return np.abs(W(theta)) # Outpur
