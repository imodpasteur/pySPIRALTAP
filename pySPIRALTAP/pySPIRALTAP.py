#-*-coding:utf-8-*-
# Python version by Maxime Woringer, Apr. 2016.
# =============================================================================
# =        SPIRAL:  Sparse Poisson Intensity Reconstruction Algorithms        =
# =                                Version 1.0                                =
# =============================================================================
# =    Copyright 2009, 2010                                                   =
# =    Zachary T. Harmany*, Roummel F. Marcia**, Rebecca M. Willett*          =
# =        *  Department of Electrical and Computer Engineering               =
# =           Duke University                                                 =
# =           Durham, NC 27708, USA                                           =
# =       **  School of Natural Sciences                                      =
# =           University of California, Merced                                =
# =           Merced, CA 95343, USA                                           =
# =                                                                           =
# =    Corresponding author: Zachary T. Harmany (zth@duke.edu)                =
#
# =============================================================================
# =                               Documentation                               =
# =============================================================================
# Syntax:
#   [x, optionalOutputs] = SPIRALTAP(y, A, tau, optionalInputs)
# 
#   More details and supporting publications are 
#   available on the SPIRAL Toolbox homepage
#   http://drz.ac/code/spiraltap/
# 
# =============================================================================
# =                                  Inputs                                   =
# =============================================================================
# Required Inputs:
#   y               Degraded observations.  For this documenation, we say
#                   that y has m total elements.
#
#   A               Sensing / observation matrix.  For this documentation, 
#                   we say that A is an m x n matrix.  A could also be a
#                   function call A() that computes matrix-vector products
#                   such that A(x) = A*x where x has n total elements.  In 
#                   this case one must also specify a function call AT() 
#                   using the 'AT' option that computes matrix-vector 
#                   products with the adjoint of A such that AT(x) = A'*x.  
#
#   tau             Regularization parameter that trades off the data fit
#                   (negative log-likelihood) with the regularization.
#                   The regularization parameter can either be a
#                   nonnegative real scalar or (for all methods except the
#                   total variation penalty) have n nonnegative real 
#                   elements which allows for nonuniform penalization 
#                   schemes. 
#           
# Optional Inputs:
# If one were to only input y, A, and tau into the algorithm, there are 
# many necessary assumptions as to what to do with the inputs.  By default
# SPIRAL assumes that:
#   - y contains Poisson realizations of A*f, where f is the true underlying
#     signal (to be estimated),
#   - the penalty is the l_1 norm of x (i.e., we promote sparsity in the 
#     canonical basis.
# This default behavior can be modified by providing optional inputs.
#  
# =============================================================================
# =                                  Outputs                                  =
# =============================================================================
# Required Outputs:
#   x               Reconstructed signal.  For this documentation, we assume
#                   x has n total elements.  That is, it is of size compatable
#                   with the given A matrix/function call.  
# 
# Optional Outputs:
#   The optional outputs are in the following order:
#       optionalOutputs = [itern, objective, reconerror, cputime, solutionpath]
#
#   itern            The total number of iternations performed by the 
#                   algorithm.  Clearly this number will be between miniter
#                   and maxiter and will depend on the chosen stopping
#                   criternia. 
#
#   objective       The evolution of the objective function with the number
#                   of iternations.  The initial value of the objective
#                   function is stored in objective(1), and hence the 
#                   length of objective will be itern + 1.
#
#   reconerror      The evolution of the specified error metric with the
#                   number of iternations.  The reconstruction error can
#                   only be computed if the true underlying signal or image
#                   is provided using the 'TRUTH' option.  The error
#                   corresponding to the initial value is stored in
#                   reconerror(1), and hence the length of reconerror will
#                   be itern + 1.
#                   
#   cputime         Keeps track of the total elapsed time to reach each
#                   iternation.  This provides a better measure of the
#                   computational cost of the algorithm versus counting
#                   the number of iternations.  The clock starts at time
#                   cputime(1) = 0 and hence the length of cputime will
#                   also be itern + 1.
#
#   solutionpath    Provides a record of the intermediate iternates reached
#                   while computing the solution x.  Both the "noisy" step
#                   solutionpath.step and the "denoised" iternate
#                   solutionpath.iternate are saved.  The initialialization
#                   for the algorithm is stored in solutionpath(1).iternate.
#                   Since there is no corresponding initial step, 
#                   solutionpath(1).step contains all zeros.  Like all 
#                   the other output variables, the length of solutionpath
#                   will be itern + 1.
#

# ==== Importations
from __future__ import print_function
import sys, time, datetime
import denoise_bound
import subsolutions
import haarApprox
import numpy as np

# ==== Error & helper functions
def todo():
    """Not implemented error function"""
    print('ERROR: This function is not yet implemented, please be patient!', file=sys.stderr)
    raise NotImplementedError

def check_input(expr, inp, chk, varname):
    """Simple wrapper to test the validity of an input
    
    Arguments:
    - expr: the assessed expression (must be TRUE or FALSE)
    - inp: the input value. It should have a way to be converted to a `str`.
    - varname (str): the name of the input variable
    - chk (str): a human-readable string of acceptable(s) inputs

    Returns: None
    """
    if not expr:
        raise TypeError("ERROR (Invalid setting): '{}'={}. '{}' must be {}".format(varname, inp, varname, chk))

    
def computegrad(y, Ax, AT, noisetype, logepsilon):
    """Compute the gradient"""
    if noisetype.lower() == 'poisson':
        return AT(1 - (y/(Ax + logepsilon)))
    elif noisetype.lower() == 'gaussian':
        return AT(Ax - y)
    else:
        raise TypeError("ERROR: undefined 'noisetype' in computegrad")

def computeobjective(x,y,Ax,tau,noisetype,logepsilon,penalty,WT):
    """Compute the objective function"""
    ## (1) Compute log-likelyhood
    if noisetype.lower()=='poisson':
        precompute = y*np.log(Ax + logepsilon)
        objective = Ax.sum() - precompute.sum()
    elif noisetype.lower()=='gaussian':
        objective = ((y-Ax)**2).sum()/2
        
    ## (2) Compute penalty
    if penalty.lower=='canonical':
        objective += np.abs(tau*x).sum()
    elif penalty.lower=='onb':
        WTx = WT(x)
        objective += np.abs(tau*WTx).sum()
    elif penalty.lower=='rdp':
        todo() # not implemented in the Matlab version
    elif penalty.lower=='rdp-ti':
        todo() # not implemented in the Matlab version
    elif penalty.lower=='tv':
        objective += tau*tlv(x,'l1') ## tlv comes from the toolbox
    return objective

## =====================================
## = Denoising Subproblem Computation: =
## =====================================
def computesubsolution(step, tau, alpha, penalty, mu, W, WT,
                       subminiter, submaxiter, substopcriterion, subtolerance):
    """Denoising subproblem computation"""
    if penalty.lower() == 'canonical':
        out = step - tau/alpha + mu
        out[out<0]=0
        return out
    elif penalty.lower() == 'tv':
        pars = {'print':0, 'tv':'l1', 'MAXITER':submaxiter, 'epsilon' : subtolerance}
        if tau>0:
            out = denoise_bound.denoise_bound(step.reshape((step.shape[0],-1)),
                                              tau/alpha,
                                              float(-1*mu), np.inf,
                                              pars)[0]
            return out
        else:
            return step*(step>0)        
    elif penalty.lower() == 'onb' : ## Signal not sparse in the direct basis.
        return subsolutions.constrainedl2l1denoise(step, W, WT, tau/alpha, mu,
                                                   subminiter, submaxiter,
                                                   substopcriterion, subtolerance)
    elif penalty.lower() == 'rdp':
        todo()
        return haarApprox.haarTVApprox2DNN_recentered(step,tau/alpha,-mu) ## To be implemented
    elif penalty.lower() == 'rdp-ti':
        todo()
        return haarApprox.haarTIApprox2DNN_recentered(step,tau/alpha,-mu)

# % =====================================
# % = Termination Criternia Computation: =
# % =====================================
def checkconvergence(itern,miniter,stopcriterion,tolerance, dx, x, cputime, objective):
    converged = 0
    if itern >= miniter: # no need to check if miniter not yet exceeded
        if stopcriterion == 1: # Simply exhaust the maximum iternation budget
            converged = 0
        elif stopcriterion == 2: # Terminate after a specified CPU time (in seconds)
            converged = cputime >= tolerance
        elif stopcriterion == 3: # Relative changes in iternate
            converged = dx.sum()**2/x.sum()**2 <= tolerance**2
        elif stopcriterion == 4: # relative changes in objective
            converged = np.abs(objective[itern]-objective[itern-1])/abs(objective[itern-1]) <=tolerance
        elif stopcriterion == 5: # complementarity condition
            todo() # not implemented in the Matlab version
        elif stopcriterion == 6: # Norm of lagrangian gradient
            todo() # not implemented in the Matlab version
            
    return converged 

def tlv(X,typ):
    """This function computes the total variation of an input image X
    This function comes from the 'denoise' toolbox, ie:
    > We also use the FISTA algorithm of Beck and Teboulle for constrained Total 
    > Variation denoising.  This toolbox is in the 'denoise' directory and is also
    > available online:
    W http://ie.technion.ac.il/~becka/papers/tv_fista.zip
    """
    (m,n)=X.shape
    P1 = X[0:(m-1),:]-X[1:(m),:]
    P2 = X[:,0:(n-1)]-X[:,1:(n)]

    if typ=='iso':
        D = np.zeros((m,n))
        D[0:(m-1),:]=P1**2
        D[:,0:(n-1)]=D[:,0:(n-1)]+P2**2
        return np.sqrt(D).sum()
    elif typ=='l1':
        return np.abs(P1).sum()+np.abs(P2).sum()
    else:
        raise InputError('Invalid total variation type. Should be either "iso" or "l1"')

# ==== Main functions
def SPIRALTAP(y, A, tau,
              verbose=0, converged=0, itern=1,     # Generic
              AT=[] ,truth=[], initialization=[], # Generic
              warnings=1, recenter=0, mu=0,       # Generic
              noisetype='Poisson', logepsilon=1e-10, sqrty=[],        # Poisson noise
              penalty='Canonical', W=[], WT=[], subminiter = 1,       # Penalization scheme
              submaxiter=50, substopcriterion=0, subtolerance = 1e-5, # Penalization scheme
              alphamethod=1, monotone = 1,                            # Choice of alpha
              alphainit=1, alphamin=1e-30, alphamax=1e30,             # Barz-Bor Scheme
              acceptdecrease=0.1, acceptpast=10, acceptmult=2,        # Acceptance criterion
              stopcriterion=1, miniter=5, maxiter=100, tolerance=1e-6,# Termination criterion
              saveobjective=False, computereconerror=False, reconerrortype=0, # Output parameters
              savecputime=False, savesolutionpath=False, savereconerror=False,    # Output parameters
              **kwargs):
    """
    Main SPIRALTAP function

    Returns: 
      - x
      - varargout ## Please document how it should work.
    """
    
    ## ==== Input parameters
    if not 'acceptalphamax' in kwargs: ## Python3 compatible expression.
        acceptalphamax=alphamax

    ## ==== Check the validity of the inputs
    
    ## NOISETYPE:  For now only two options are available 'Poisson' and 'Gaussian'.
    check_input(type(noisetype)==str and noisetype.lower() in ('poisson', 'gaussian'),
                noisetype, "either 'Gaussian' or 'Poisson'", 'noisetype')

    ## PENALTY:  The implemented penalty options are 'Canonical, 'ONB', 'RDP', 'RDP-TI','TV'.
    check_input(type(penalty)==str and penalty.lower() in ('canonical','onb','rdp','rdp-ti','tv'), penalty, "only 'Canonical', 'ONB', 'RDP', 'RDP-TI', or 'TV'", 'penalty')
    
    ## VERBOSE:  Needs to be a nonnegative integer.
    check_input(type(verbose) == int and verbose >= 0, verbose, "a nonnegative integer", 'verbose')

    ## LOGEPSILON:  Needs to be nonnegative, usually small but that's relative.
    check_input(logepsilon >= 0, logepsilon, "a nonnegative integer", 'logepsilon')

    ## TOLERANCE:  Needs to be nonnegative, usually small but that's relative.
    check_input(tolerance >= 0, tolerance, "a nonnegative integer", 'tolerance')

    ## SUBTOLERANCE:  Needs to be nonnegative, usually small but that's relative.
    check_input(subtolerance >= 0, subtolerance, "a nonnegative integer", 'subtolerance')

    ## MINITER and MAXITER:  Need to check that they are nonnegative integers and
    ## that miniter <= maxiter todo
    check_input(miniter>0 and maxiter>0, (miniter, maxiter), "positive integers", "('MINITER', 'MAXITER')")
    check_input(miniter <= maxiter, (miniter, maxiter), "like 'MINITER'<='MAXITER'", "('MINITER', 'MAXITER')")
    check_input(subminiter <= submaxiter, (subminiter, submaxiter), "like 'SUBMINITER'<='SUBMAXITER'", "('MINITER', 'MAXITER')")

    # Matrix dimensions
    # AT:  If A is a matrix, AT is not required, but may optionally be provided.
    # If A is a function call, AT is required.  In all cases, check that A and AT
    # are of compatable size.  When A (and potentially AT) are given
    # as matrices, we convert them to function calls for the remainder of the code
    # Note: I think that it suffices to check whether or not the quantity
    # dummy = y + A(AT(y)) is able to be computed, since it checks both the
    # inner and outer dimensions of A and AT against that of the data y
    if hasattr(A, '__call__'): # A is a function call, so AT is required
        if AT==[]:
            raise TypeError("Parameter ''AT'' not specified.  Please provide a method to compute A''*x matrix-vector products.")
        else:
            try:
                dummy = y + A(AT(y).copy()).copy()
                if not hasattr(AT, '__call__'):
                    raise TypeError('AT should be provided as a function handle, because so is A')
            except:
                raise TypeError('Size incompatability between ''A'' and ''AT''.')
    else: # A is a matrix
        Aorig = A.copy()
        A = lambda x: Aorig.dot(x)
        if AT==[]: # A is a matrix, and AT not provided.
            AT = lambda x: Aorig.transpose().dot(x)
        else: # A is a matrix, and AT provided, we need to check
            if hasattr(AT, '__call__'): # A is a matrix, AT is a function call
                try: 
                    dummy = y + A(AT(y).copy()).copy()
                except:
                    raise TypeError('Size incompatability between ''A'' and ''AT''.')
            else: #A and AT are matrices
                AT = lambda x: Aorig.transpose().dot(x)
                
    # TRUTH:  Ensure that the size of truth, if given, is compatible with A and
    # that it is nonnegative.  Note that this is irrespective of the noisetype
    # since in the Gaussian case we still model the underlying signal as a
    # nonnegative intensity.
    if truth != []:
        try:
            dummy = truth + AT(y)
            truth = np.array(truth, dtype='float64') # Convert to float
        except:
            raise TypeError("The size of ''TRUTH'' is incompatible with the given, sensing matrix ''A''.")
        if truth.min() < 0:
            raise ValueError("The size of ''TRUTH'' is incompatable with the given sensing matrix ''A''.")
    check_input(type(saveobjective)==bool, saveobjective, "a boolean (True or False)", 'saveobjective')
    check_input(type(savereconerror)==bool, savereconerror, "a boolean (True or False)", 'savereconerror')
    check_input(not ((savesolutionpath and truth==[]) or (savereconerror and truth==[])), '', "The option to save the reconstruction error ''SAVERECONERROR'' can only be used if the true signal ''TRUTH'' is provided.", 'savereconerror')
    check_input(type(savecputime)==bool, savecputime, "a boolean (True or False)", 'savecputime')
    check_input(type(savesolutionpath)==bool, savesolutionpath, "a boolean (True or False)", 'savecputime')
    
    ## ==== Initialize method-dependent parameters
    ## Things to check and compute that depend on NOISETYPE:
    if noisetype.lower() == 'poisson':
        if (y.round()!=y).sum()!=0 or y.min() < 0:
            raise ValueError("The data ''Y'' must contain nonnegative integer counts when ''NOISETYPE'' = ''Poisson''")
        # Maybe in future could check to ensure A and AT contain nonnegative
        # elements, but perhaps too computationally wasteful
        sqrty = np.sqrt(y) # Precompute useful quantities:
        if recenter: # Ensure that recentering is not set
            todo() # not implemented in the Matlab version

    ## Things to check and compute that depend on PENALTY:
    if penalty.lower() == 'canonical':
        pass
    elif penalty.lower() == 'tv': ## Cannot have a vectorized tau (yet)
            if type(tau) != float:
                raise TypeError('A vector regularization parameter ''TAU'' cannot be used in conjuction with the TV penalty.')
    elif penalty.lower() == 'onb': ## There is a sparsifying basis
        ## Already checked for valid subminiter, submaxiter, and subtolerance
        ## Check for valid substopcriterion
        ## Need to check for the presense of W and WT
        ## Further checks to ensure we have both W and WT defined and that
        ## the sizes are compatable by checking if y + A(WT(W(AT(y)))) can
        ## be computed
        check_input(W != [], 'W', "provided as a method to compute W*x matrix-vector products.", 'W')
        if hasattr(W, '__call__'): # W is a function call
            check_input(WT != [], 'WT', "provided as a method to compute W*x matrix-vector products.", 'WT') ## Fail if WT is not specified and W is a function call
            if hasattr(WT, '__call__'): # WT is a function call
                try: 
                    dummy = y + A(WT(W(AT(y))))
                except:
                    raise TypeError("Size incompatability between ''W'' and ''WT''.")
            else: ## W is a function call, WT is a matrix
                try: 
                    dummy = y + A(WT*W(AT(y)))
                except:
                    raise TypeError("Size incompatability between ''W'' and ''WT''.")
                WTorig = WT.copy()
                WT = lambda x: WTorig.dot(x)
        else: ## W is a matrix
            if WT==[]: ## W is a matrix, and WT not provided.
                Worig = W.copy()
                WT = lambda x: Worig.T.dot(x) # Just define function calls.
                W = lambda x: Worig.dot(x)
            else: # W is a matrix, and WT provided, we need to check
                if hasattr(WT, '__call__'): ## WT is a function call
                    try:
                        dummy = y + A(WT(W.dot(AT(y))))
                    except Exception :
                        raise TypeError("Size incompatability between ''W'' and ''WT''.")
                    Worig = W.copy()
                    W = lambda x: Worig.dot(x) ## Define W as a function call
                else: # W and WT are matrices
                    try:
                        dummy = y + A(WT.dot(W.dot(AT(y))))
                    except:
                        raise TypeError("Size incompatability between ''W'' and ''WT''.")
                    Worig = W.copy()
                    WT = lambda x: Worig.T.dot(x) ## Define W and WT as function calls
                    W  = lambda x: Worig.dot(x)
    elif penalty.lower() in ('rdp', 'rdp-ti'): # todo
        if monotone: # cannot enforce monotonicity (yet)
            raise NotImplemented("Explicit computation of the objective function cannot be performed when using the RDP penalty. Therefore monotonicity cannot be enforced. Invalid option ''MONOTONE'' = True for ''PENALTY'' = {}".format(penalty))
        if saveobjective: # Cannot compute objective function (yet)
            raise NotImplemented("Explicit computation of the objective function cannot be performed when using the RDP penalty. Invalid option ''SAVEOBJECTIVE'' = True for ''PENALTY'' = {}".format(penalty))
    else:
        raise TypeError('Unrecognized penalty')

    ## ==== check that initialization is a scalar or a vector
    if initialization == []: ## set initialization
        xinit = AT(y)
    else:
        xinit = initialization
    
    if recenter:
        print ("WARNING: This part of the code has not been debugged", file=sys.stderr)
        Aones = A(np.ones_like(xinit)).copy()
        meanAones(Aones.mean())
        meany = y.mean()
        y -= meany
        mu = meany/meanAones
        # Define new function calls for 'recentered' matrix
        print('recentering')
        A = lambda x: A(x) - meanAones*x.sum()/xinit.size
        AT = lambda x: AT(x) - meanAones*x.sum()/xinit.size
        xinit = xinit - mu # Adjust Initialization

    ## ==== Prepare for running the algorithm (Assumes that all parameters above are valid)
    ## Initialize Main Algorithm
    x = xinit
    Ax = A(x).copy()
    alpha = alphainit 
    Axprevious = Ax
    xprevious = x
    grad = computegrad(y, Ax, AT, noisetype, logepsilon)

    ## Prealocate arrays for storing results
    cputime = np.zeros((maxiter+1))
    objective = np.zeros((maxiter+1))

    if saveobjective:
        objective[itern-1] = computeobjective(x,y,Ax,tau,noisetype,logepsilon,penalty,WT)
    else:
        objective[itern-1] = computeobjective(x,y,Ax,tau,noisetype,logepsilon,penalty,WT) # MW: this must be initialized anyway
    if savereconerror:
        reconerror = np.zeros((maxiter+1))
        if reconerrortype == 0: # RMS error
            normtrue = (np.array(truth, dtype='float64')**2).sum()**0.5
            computereconerror = lambda x: ((x+mu-truth)**2).sum()**0.5/normtrue
        elif reconerrortype == 1:
            normtrue = np.abs(truth).sum()
            computereconerror = lambda x: np.abs(x+mu-truth).sum()/normtrue
        reconerror[itern-1] = computereconerror(xinit)
    if savesolutionpath:
        todo()
        #     % Note solutionpath(1).step will always be zeros since having an 
        #     % 'initial' step does not make sense
        #     solutionpath(1:maxiter+1) = struct('step',zeros(size(xinit)),...
        #         'iternate',zeros(size(xinit)));
        #     solutionpath(1).iternate = xinit;


    if verbose>0:
        txt = """
===================================================================
= Beginning SPIRAL Reconstruction    @ {} =
=   Noisetype: {}               Penalty: {}           =
=   Tau:       {}                 Maxiter: {}                 =
===================================================================
"""
        txt = txt.format(datetime.datetime.now(), noisetype, penalty, tau, maxiter)
        print(txt)
    
    tic=time.time() # Start clock for calculating computation time.
    
    ## =============================
    ## = Begin Main Algorithm Loop =
    ## =============================
    while (itern <= miniter) or ((itern <= maxiter) and not converged):
        ## ==== Compute solution
        if alphamethod == 0: # Constant alpha throughout all iternations.
            # If convergence criternia requires it, compute dx or dobjective
            dx = xprevious
            step = xprevious - grad/alpha
            x = computesubsolution(step, tau, alpha, penalty, mu, W, WT,
                                   subminiter, submaxiter, substopcriterion, subtolerance)
            dx = x - dx
            Ax = A(x).copy()
        elif alphamethod == 1: # Barzilai-Borwein choice of alpha
            if monotone: #do acceptance criterion.
                past = np.arange(max(itern-1-acceptpast,0),itern)
                maxpastobjective =  objective[past].max()
                accept=0
                while accept==0:
                    ## Compute the step and perform gaussian denoising subproblem
                    dx = xprevious
                    step = xprevious - np.array(grad/alpha, dtype='float64').reshape(xprevious.shape)
                    #x = x.reshape((-1,1))
                    x = computesubsolution(step, tau, alpha, penalty, mu, W, WT,
                                           subminiter, submaxiter, substopcriterion, subtolerance).reshape(x.shape)
                    dx = x - dx
                    Adx = Axprevious
                    Ax = A(x).copy()
                    Adx = Ax - Adx
                    normsqdx = (dx**2).sum()
                    ## Compute the resulting objective
                    objective[itern] = computeobjective(x,y,Ax,tau,
                                                        noisetype,logepsilon,penalty,WT)
                    if (objective[itern] <= (maxpastobjective-acceptdecrease*alpha/2*normsqdx)) or (alpha >= acceptalphamax):
                        accept = 1
                    acceptalpha=alpha #Keep value for displaying
                    alpha = acceptmult*alpha
            else: #just take bb setp, no enforcing monotonicity.
                dx = xprevious
                step = xprevious - grad/alpha
                x = computesubsolution(step,tau,alpha,penalty,mu,
                                       W,WT,subminiter,submaxiter,substopcriterion, subtolerance)
                dx = x - dx
                Adx = Axprevious
                Ax = A(x).copy()
                Adx = Ax - Adx
                normsqdx = (dx**2).sum()
                if saveobjective:
                    objective[itern] = computeobjective(x, y, Ax, tau, noisetype,
                                                       logepsilon, penalty,WT);
        ## ==== Calculate Output Quantities
        if savecputime:
            cputime[itern] = time.time()-tic
        if savereconerror:
            reconerror[itern] = computereconerror(x)
        if savesolutionpath:
            print("ERROR: this option is not implemented 'savesolutionpath'", file=sys.stderr)
            #solutionpath(itern).step = step
            #solutionpath(itern).iterate = x
        ## Needed for next iternation and also termination criternia
        Ax=Ax#.reshape((-1,1)) # Reshape before computing gradient
        grad = computegrad(y,Ax,AT,noisetype,logepsilon)
        converged = checkconvergence(itern,miniter,stopcriterion,tolerance,
                                     dx, x, cputime[itern], objective)

        ## ==== Display progress
        if verbose > 0 and itern % verbose == 0:
            txt = 'Iter: {}, ||dx||%%: {}, Alph: {}'.format(itern,
                                                        100*np.linalg.norm(dx)/np.linalg.norm(x),
                                                            alpha)
            ## use of np.linalg.norm could probably be removed
            if monotone and alphamethod==1:
                txt += ', Alph Acc: {}'.format(acceptalpha)
            if savecputime:
                txt += ', Time: {}'.format(cputime[itern])
            if saveobjective:
                txt += ', Obj: {}, dObj%%: {}'.format(objective[itern],
                            100*np.abs(objective[itern]-objective[itern-1])/np.abs(objective[itern-1]))
            if savereconerror:
                txt += ', Err: {}'.format(reconerror[itern])
            print(txt)

        ## ==== Prepare for next iternation
        ## Update alpha
        if alphamethod == 0:
            pass # do nothing, constant alpha
        elif alphamethod == 1: # BB method
            # Adx is overwritten at top of iternation, so this is an ok reuse
            if noisetype.lower() == 'poisson':
                Adx = Adx*sqrty/(Ax + logepsilon)
            elif noisetype.lower() == 'gaussian':
                pass # No need to scale Adx
            gamma = (Adx**2).sum()
            if gamma == 0:
                alpha = alphamin
            else:
                alpha = gamma/normsqdx
                alpha = min(alphamax, max(alpha, alphamin))
                
        ## ==== Store current values as previous values for next iternation
        xprevious = x
        Axprevious = Ax
        itern += 1
    ## ===========================
    ## = End Main Algorithm Loop =
    ## ===========================
        
    ## ==== Post process the output
    ## Add on mean if recentered (if not mu == 0);
    x = x + mu;

    ## Determine what needs to be in the variable output and
    ## crop the output if the maximum number of iternations were not used.
    ## Note, need to subtract 1 since itern is incremented at the end of the loop
    itern = itern - 1;
    varargout = {'iterations':itern}

    if saveobjective:
        varargout['objective'] = objective[0:itern]
        #varargout.append(objective[0:itern]) ## Useless bounds 1:itern+1 ???
    if savereconerror:
        varargout['reconerror']=reconerror[0:itern]
        #varargout.append(reconerror[0:itern])
    if savecputime:
        varargout['cputime']=cputime[0:itern]
        #varargout.append(cputime[0:itern])
    if savesolutionpath:
        varargout['solutionpath']=solutionpath[0:itern]
        #varargout.append(solutionpath[0:itern])

    if verbose > 0:
        txt = """
===================================================================
= Completed SPIRAL Reconstruction    @ {} =
=   Noisetype: {}               Penalty: {}           =
=   Tau:       {}                 Maxiter: {}                 =
===================================================================
"""
        txt = txt.format(datetime.datetime.now(), noisetype, penalty, tau, maxiter)
        print (txt)
    return (x, varargout)
