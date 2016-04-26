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
#       optionalOutputs = [iter, objective, reconerror, cputime, solutionpath]
#
#   iter            The total number of iterations performed by the 
#                   algorithm.  Clearly this number will be between miniter
#                   and maxiter and will depend on the chosen stopping
#                   criteria. 
#
#   objective       The evolution of the objective function with the number
#                   of iterations.  The initial value of the objective
#                   function is stored in objective(1), and hence the 
#                   length of objective will be iter + 1.
#
#   reconerror      The evolution of the specified error metric with the
#                   number of iterations.  The reconstruction error can
#                   only be computed if the true underlying signal or image
#                   is provided using the 'TRUTH' option.  The error
#                   corresponding to the initial value is stored in
#                   reconerror(1), and hence the length of reconerror will
#                   be iter + 1.
#                   
#   cputime         Keeps track of the total elapsed time to reach each
#                   iteration.  This provides a better measure of the
#                   computational cost of the algorithm versus counting
#                   the number of iterations.  The clock starts at time
#                   cputime(1) = 0 and hence the length of cputime will
#                   also be iter + 1.
#
#   solutionpath    Provides a record of the intermediate iterates reached
#                   while computing the solution x.  Both the "noisy" step
#                   solutionpath.step and the "denoised" iterate
#                   solutionpath.iterate are saved.  The initialialization
#                   for the algorithm is stored in solutionpath(1).iterate.
#                   Since there is no corresponding initial step, 
#                   solutionpath(1).step contains all zeros.  Like all 
#                   the other output variables, the length of solutionpath
#                   will be iter + 1.
#

# ==== Importations
from __future__ import print_function
import sys, time, datetime
import numpy as np

# ==== Error & helpter functions
def todo():
    """Not implemented error function"""
    print('ERROR: This function is not yet implemented, please be patient!', file=sys.stderr)

## TODO: write stopif function
    
def computegrad(y, Ax, AT, noisetype, logepsilon):
    """Compute the gradient"""
    if noisetype.lower() == 'poisson':
        return AT(1 - (y/(Ax + logepsilon)))
    elif noisetype.lower() == 'gaussian':
        return AT(Ax - y)
    else:
        print ("ERROR: undefined 'noisetype' in computegrad", file=sys.stderr)

# % ==========================
# % = Objective Computation: =
# % ==========================
# function objective = computeobjective(x,y,Ax,tau,noisetype,logepsilon,...
#     penalty,varargin)
# % Perhaps change to varargin 
# % 1) Compute log-likelihood:
# switch lower(noisetype)
#     case 'poisson'
#         precompute = y.*log(Ax + logepsilon);
#         objective = sum(Ax(:)) - sum(precompute(:));
#     case 'gaussian'
#         objective = sum( (y(:) - Ax(:)).^2)./2;
# end
# % 2) Compute Penalty:
# switch lower(penalty)
#     case 'canonical'
#         objective = objective + sum(abs(tau(:).*x(:)));
# 	case 'onb' 
#     	WT = varargin{1};
#         WTx = WT(x);
#         objective = objective + sum(abs(tau(:).*WTx(:)));
# 	case 'rdp'
#         todo
#     case 'rdp-ti'
#         todo
#     case 'tv'
#         objective = objective + tau.*tlv(x,'l1');
# end
# end

# % =====================================
# % = Denoising Subproblem Computation: =
# % =====================================
def computesubsolution(step, tau, alpha, penalty, mu, W, WT,
                       subminiter, submaxiter, substopcriterion, subtolerance):
    """Denoising subproblem computation"""
    if penalty.lower() == 'canonical':
        out = step - tau/alpha + mu
        out[out<0]=0
        return out
        return np.max(step - tau/alpha + mu, 0.0) ## previous method
    else:
        todo() ## Only partially implemented, see below.
# function subsolution = computesubsolution(step,tau,alpha,penalty,mu,varargin)
#     switch lower(penalty)
#         case 'canonical'
#             subsolution = max(step - tau./alpha + mu, 0.0);
#         case 'onb'
#             % if onb is selected, varargin must be such that
#             W                   = varargin{1};
#             WT                  = varargin{2};
#             subminiter          = varargin{3};
#             submaxiter          = varargin{4};
#             substopcriterion    = varargin{5};
#             subtolerance        = varargin{6};
                                   
#             subsolution = constrainedl2l1denoise(step,W,WT,tau./alpha,mu,...
#                 subminiter,submaxiter,substopcriterion,subtolerance);
#         case 'rdp'
#             subsolution = haarTVApprox2DNN_recentered(step,tau./alpha,-mu);
#         case 'rdp-ti'
#             subsolution = haarTIApprox2DNN_recentered(step,tau./alpha,-mu);
#         case 'tv'
#             subtolerance        = varargin{6};
#             submaxiter          = varargin{4};
#             % From Becca's Code:
#             pars.print = 0;
#             pars.tv = 'l1';
#             pars.MAXITER = submaxiter;
#             pars.epsilon = subtolerance; % Becca used 1e-5;
#             if tau>0
#                 subsolution = denoise_bound(step,tau./alpha,-mu,Inf,pars);
#             else
#                 subsolution = step.*(step>0);
#             end
#     end           
# end

# % =====================================
# % = Termination Criteria Computation: =
# % =====================================
def checkconvergence(iter,miniter,stopcriterion,tolerance, dx, x, cputime, objective):
    converged = 0
    if iter >= miniter: # no need to check if miniter not yet exceeded
        if stopcriterion == 1: # Simply exhaust the maximum iteration budget
            converged = 0
        elif stopcriterion == 2: # Terminate after a specified CPU time (in seconds)
            converged = cputime >= tolerance
        elif stopcriterion == 3: # Relative changes in iterate
            converged = dx.sum()**2/x.sum()**2 <= tolerance**2
        elif stopcriterion == 4: # relative changes in objective
            converged = np.abs(objective[iter]-objective[iter-1])/abs(objective[iter-1]) <=tolerance
        elif stopcriterion == 5: # complementarity condition
            todo()
        elif stopcriterion == 6: # Norm of lagrangian gradient
            todo()
            
    return converged 

# ==== Main functions
def SPIRALTAP(y, A, tau,
              verbose=0, converged=0, iter=1,     # Generic
              AT=[] ,truth=[], initialization=[], # Generic
              warnings=1, recenter=0, mu=0,       # Generic
              noisetype='Poisson', logepsilon=1e-10, sqrty=[],        # Poisson noise
              penalty='Canonical', W=[], WT=[], subminiter = 1,       # Penalization scheme
              submaxiter=50, substopcriterion=0, subtolerance = 1e-5, # Penalization scheme
              alphamethod=1, monotone = 1,                            # Choice of alpha
              alphainit=1, alphamin=1e-30, alphamax=1e30,             # Barz-Bor Scheme
              acceptdecrease=0.1, acceptpast=10, acceptmult=2,        # Acceptance criterion
              stopcriterion=1, miniter=5, maxiter=100, tolerance=1e-6,# Termination criterion
              saveobjective=0, computereconerror=0, reconerrortype=0, # Output parameters
              savecputime=0, savesolutionpath=0, savereconerror=0,    # Output parameters
              **kwargs):
    """
    Main SPIRALTAP function

    Returns: 
      - x
      - varargout (?)
    """
    #% Add a path to the denoising methods folder
    #spiraltapdir = which('SPIRALTAP');
    #[spiraltapdir dummy] = fileparts(spiraltapdir);
    #path([spiraltapdir,'/denoise'],path)
    
    ## ==== Input parameters
    if not kwargs.has_key('acceptalphamax'):
        acceptalphamax=alphamax

    ## ==== Check the validity of the inputs
    print ("WARNING: so far, the validity of the input is not checked", file=sys.stderr)


    ## NOISETYPE:  For now only two options are available 'Poisson' and 'Gaussian'.
    if not type(noisetype)==str or noisetype.lower() not in ('poisson', 'gaussian'):
        raise TypeError("ERROR (Invalid setting): 'noisetype'={}. 'noisetype' must be either 'Gaussian' or 'Poisson'".format(noisetype))

    ## PENALTY:  The implemented penalty options are 'Canonical, 'ONB', 'RDP', 'RDP-TI','TV'.
    if not type(penalty)==str or penalty.lower() not in ('canonical','onb','rdp','rdp-ti','tv'):
        raise TypeError("Invalid setting ''PENALTY'' = {}. The parameter ''PENALTY'' may only be ''Canonical'', ''ONB'', ''RDP'', ''RDP-TI'', or ''TV''.".format(penalty))

    ## VERBOSE:  Needs to be a nonnegative integer.
    if type(verbose) != int or verbose<0:
        raise TypeError("The parameter ''VERBOSE'' is required to be a nonnegative integer.  The setting ''VERBOSE'' = {} is invalid".format(verbose))
    
    ## LOGEPSILON:  Needs to be nonnegative, usually small but that's relative.
    if logepsilon < 0:
        raise TypeError("The parameter ''LOGEPSILON'' is required to be a nonnegative integer.  The setting ''LOGEPSILON'' = {} is invalid".format(logepsilon))

    ## TOLERANCE:  Needs to be nonnegative, usually small but that's relative.
    if tolerance <0:
        raise TypeError("The parameter ''TOLERANCE'' is required to be a nonnegative integer.  The setting ''TOLERANCE'' = {} is invalid".format(tolerance))

    ## SUBTOLERANCE:  Needs to be nonnegative, usually small but that's relative.
    if subtolerance <0:
        raise TypeError("The parameter ''SUBTOLERANCE'' is required to be a nonnegative integer.  The setting ''SUBTOLERANCE'' = {} is invalid".format(subtolerance))

    ## MINITER and MAXITER:  Need to check that they are nonnegative integers and
    ## that miniter <= maxiter todo
    if miniter <= 0 or maxiter <= 0:
        raise TypeError("The numbers of iterations ''MINITER'' = {} and ''MAXITER'' = {} should be non-negative.".format(miniter, maxiter))
    if miniter > maxiter:
        raise TypeError("The minimum number of iterations ''MINITER'' = {} exceeds the maximum number of iterations ''MAXITER'' = {}.".format(miniter, maxiter))

    if subminiter > submaxiter:
         raise TypeError("The minimum number of subproblem iterations ''SUBMINITER'' = {} exceeds the maximum number of subproblem iterations ''SUBMAXITER'' = {}".format(subminiter, subbmaxiter))

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
                dummy = y + A(AT(y))
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
                    dummy = y + A(AT(y))
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
        except:
            raise TypeError("The size of ''TRUTH'' is incompatible with the given, sensing matrix ''A''.")
        if truth.min() < 0:
            raise ValueError("The size of ''TRUTH'' is incompatable with the given sensing matrix ''A''.")

    print("WARNING: Not all input parameters are validated so far", file=sys.stderr)
    # % SAVEOBJECTIVE:  Just a binary indicator, check if not equal to 0 or 1.
    # if (numel(saveobjective) ~= 1)  || (sum( saveobjective == [0 1] ) ~= 1)
    #     error(['The option to save the objective evolution ',...
    #         'SAVEOBJECTIVE'' ',...
    #         'must be a binary scalar (either 0 or 1).'])
    # end     
    # % SAVERECONERROR:  Just a binary indicator, check if not equal to 0 or 1.
    # % If equal to 1, truth must be provided.
    # if (numel(savereconerror) ~= 1)  || (sum( savereconerror == [0 1] ) ~= 1)
    #     error(['The option to save the reconstruction error ',...
    #         'SAVERECONERROR'' ',...
    #         'must be a binary scalar (either 0 or 1).'])
    # end
    # if savesolutionpath && isempty(truth)
    #     error(['The option to save the reconstruction error ',...
    #         '''SAVERECONERROR'' can only be used if the true signal ',...
    #         '''TRUTH'' is provided.'])
    # end
    # % SAVECPUTIME: Just a binary indicator, check if not equal to 0 or 1.
    # if (numel(savecputime) ~= 1)  || (sum( savecputime == [0 1] ) ~= 1)
    #     error(['The option to save the computation time ',...
    #         'SAVECPUTIME'' ',...
    #         'must be a binary scalar (either 0 or 1).'])
    # end
    # % SAVESOLUTIONPATH: Just a binary indicator, check if not equal to 0 or 1.
    # if (numel(savesolutionpath) ~= 1)  || (sum( savesolutionpath == [0 1] ) ~= 1)
    #     error(['The option to save the solution path ',...
    #         'SAVESOLUTIONPATH'' ',...
    #         'must be a binary scalar (either 0 or 1).'])
    # end

    ## ==== Initialize method-dependent parameters
    ## Things to check and compute that depend on NOISETYPE:
    if noisetype.lower() == 'poisson':
        if (y.round()!=y).sum()!=0 or y.min() < 0:
            raise ValueError("The data ''Y'' must contain nonnegative integer counts when ''NOISETYPE'' = ''Poisson''")
        # Maybe in future could check to ensure A and AT contain nonnegative
        # elements, but perhaps too computationally wasteful
        sqrty = np.sqrt(y) # Precompute useful quantities:
        if recenter: # Ensure that recentering is not set
            todo()

    ## Things to check and compute that depend on PENALTY:
    if penalty.lower() == 'canonical':
        pass
    elif penalty.lower() == 'onb':
        todo()
    # switch lower(penalty)
    #     case 'canonical'

    #     case 'onb' 
    #         % Already checked for valid subminiter, submaxiter, and subtolerance
    #         % Check for valid substopcriterion 
    #         % Need to check for the presense of W and WT
    #         if isempty(W)
    #             error(['Parameter ''W'' not specified.  Please provide a ',...
    #                 'method to compute W*x matrix-vector products.'])
    #         end
    #         % Further checks to ensure we have both W and WT defined and that
    #         % the sizes are compatable by checking if y + A(WT(W(AT(y)))) can
    #         % be computed
    #         if isa(W, 'function_handle') % W is a function call, so WT is required
    #             if isempty(WT) % WT simply not provided
    #                 error(['Parameter ''WT'' not specified.  Please provide a ',...
    #                     'method to compute W''*x matrix-vector products.'])
    #             else % WT was provided
    #         if isa(WT, 'function_handle') % W and WT are function calls
    #             try dummy = y + A(WT(W(AT(y))));
    #             catch exception; 
    #                 error('Size incompatability between ''W'' and ''WT''.')
    #             end
    #         else % W is a function call, WT is a matrix        
    #             try dummy = y + A(WT*W(AT(y)));
    #             catch exception
    #                 error('Size incompatability between ''W'' and ''WT''.')
    #             end
    #             WT = @(x) WT*x; % Define WT as a function call
    #         end
    #     end
    # else
    #     if isempty(WT) % W is a matrix, and WT not provided.
    #         AT = @(x) W'*x; % Just define function calls.
    #         A = @(x) W*x;
    #     else % W is a matrix, and WT provided, we need to check
    #         if isa(WT, 'function_handle') % W is a matrix, WT is a function call            
    #             try dummy = y + A(WT(W*AT(y)));
    #             catch exception
    #                 error('Size incompatability between ''W'' and ''WT''.')
    #             end
    #             W = @(x) W*x; % Define W as a function call
    #         else % W and WT are matrices
    #             try dummy = y + A(WT(W*(AT(y))));
    #             catch exception
    #                 error('Size incompatability between ''W'' and ''WT''.')
    #             end
    #             WT = @(x) WT*x; % Define A and AT as function calls
    #             W = @(x) W*x;
    #         end
    #     end
    # end
    # 	case 'rdp'
    #         %todo
    #         % Cannot enforce monotonicity (yet)
    #         if monotone
    #             error(['Explicit computation of the objective function ',...
    #                 'cannot be performed when using the RDP penalty.  ',...
    #                 'Therefore monotonicity cannot be enforced.  ',...
    #                 'Invalid option ''MONOTONIC'' = 1 for ',...
    #                 '''PENALTY'' = ''',penalty,'''.']);
    #         end
    #         % Cannot compute objective function (yet)
    #         if saveobjective
    #             error(['Explicit computation of the objective function ',...
    #                 'cannot be performed when using the RDP penalty.  ',...
    #                 'Invalid option ''SAVEOBJECTIVE'' = 1 for ',...
    #                 '''PENALTY'' = ''',penalty,'''.']);
    #         end

    #     case 'rdp-ti'
    #         % Cannot enforce monotonicity
    #         if monotone
    #             error(['Explicit computation of the objective function ',...
    #                 'cannot be performed when using the RDP penalty.  ',...
    #                 'Therefore monotonicity cannot be enforced.  ',...
    #                 'Invalid option ''MONOTONIC'' = 1 for ',...
    #                 '''PENALTY'' = ''',penalty,'''.']);
    #         end
    #         % Cannot compute objective function 
    #         if saveobjective
    #             error(['Explicit computation of the objective function ',...
    #                 'cannot be performed when using the RDP-TI penalty.  ',...
    #                 'Invalid option ''SAVEOBJECTIVE'' = 1 for ',...
    #                 '''PENALTY'' = ''',penalty,'''.']);
    #         end

    #     case 'tv'
    #         % Cannot have a vectorized tau (yet)
    #         if (numel(tau) ~= 1)
    #             error(['A vector regularization parameter ''TAU'' cannot be ',...
    #                 'used in conjuction with the TV penalty.']);
    #         end
    # end

    ## ==== check that initialization is a scalar or a vector
    if initialization == []: ## set initialization
        xinit = AT(y)
    else:
        xinit = initialization
    
    if recenter:
        print ("WARNING: This part of the code has not been debugged", file=sys.stderr)
        Aones = A(np.ones_like(xinit))
        meanAones(Aones.mean())
        meany = y.mean()
        y -= meany
        mu = meany/meanAones
        # Define new function calls for 'recentered' matrix
        A = lambda x: A(x) - meanAones*x.sum()/xinit.size
        AT = lambda x: AT(x) - meanAones*x.sum()/xinit.size
        xinit = xinit - mu # Adjust Initialization
        print ("WARNING: This part of the code has not been debugged", file=sys.stderr)

    ## ==== Check for validity of output parameters (Matlab specific code)
    # % Check if there are too many or not enough
    # if (nargout == 0) && warnings
    #         disp('Warning:  You should reconsider not saving the output!');
    #         pause(1);
    # end
    # if (nargout < (2 + saveobjective + savereconerror ...
    #         + savecputime + savesolutionpath)) && warnings
    #     disp(['Warning:  Insufficient output parameters given to save ',...
    #         'the full output with the given options.']);
    # end
    # if nargout > (2 + saveobjective + savereconerror ...
    #         + savecputime + savesolutionpath)
    #         error('Too many output arguments specified for the given options.')
    # end

    ## ==== Prepare for running the algorithm (The below assumes that all parameters above are valid)
    ## Initialize Main Algorithm
    x = xinit
    Ax = A(x)
    alpha = alphainit
    Axprevious = Ax
    xprevious = x
    grad = computegrad(y, Ax, AT, noisetype, logepsilon)

    ## Prealocate arrays for storing results
    # Initialize cputime and objective empty anyway (avoids errors in subfunctions):
    #cputime = []
    cputime = np.zeros((maxiter+1))
    objective = np.zeros((maxiter+1))

    if saveobjective:
        print("ERROR: this part of the code is not implemented yet", file=sys.stderr)
        objective[iter-1] = computeobjective(x,y,Ax,tau,noisetype,logepsilon,penalty,WT)
    if savereconerror:
        reconerror = np.zeros((maxiter+1))
        if reconerrortype == 0: # RMS error
            normtrue = (truth**2).sum()**0.5
            computereconerror = lambda x: ((x+mu-truth)**2).sum()**0.5/normtrue
        elif reconerrortype == 1:
            normtrue = np.abs(truth).sum()
            computereconerror = lambda x: np.abs(x+mu-truth).sum()/normtrue
        reconerror[iter-1] = computereconerror(xinit)
    if savesolutionpath:
        pass
        #     % Note solutionpath(1).step will always be zeros since having an 
        #     % 'initial' step does not make sense
        #     solutionpath(1:maxiter+1) = struct('step',zeros(size(xinit)),...
        #         'iterate',zeros(size(xinit)));
        #     solutionpath(1).iterate = xinit;


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
    while (iter <= miniter) or ((iter <= maxiter) and not converged):
        ## ==== Compute solution
        if alphamethod == 0: # Constant alpha throughout all iterations.
            # If convergence criteria requires it, compute dx or dobjective
            dx = xprevious
            step = xprevious - grad/alpha
            x = computesubsolution(step, tau, alpha, penalty, mu, W, WT,
                                   subminiter, submaxiter, substopcriterion, subtolerance)
            dx = x - dx
            Ax = A(x)
        elif alphamethod == 1: # Barzilai-Borwein choice of alpha
            todo() ## not implemented, see below

        ## ==== Calculate Output Quantities
        if savecputime:
            cputime[iter] = time.time()-tic
        if savereconerror:
            reconerror[iter] = computereconerror(x)
        if savesolutionpath:
            print("ERROR: this option is not implemented 'savesolutionpath'", file=sys.stderr)
            solutionpath(iter).step = step
            solutionpath(iter).iterate = x

        ## Needed for next iteration and also termination criteria
        grad = computegrad(y,Ax,AT,noisetype,logepsilon)
        converged = checkconvergence(iter,miniter,stopcriterion,tolerance,
                                     dx, x, cputime[iter], objective)

        ## ==== Display progress
        if verbose > 0 and iter % verbose == 0:
            txt = 'Iter: {}, ||dx||%%: {}, Alph: {}'.format(iter,
                                                        100*np.linalg.norm(dx)/np.linalg.norm(x),
                                                            alpha)
            ## use of np.linalg.norm could probably be removed
            if monotone and alphamethod==1:
                txt += ', Alph Acc: {}'.format(acceptalpha)
            if savecputime:
                txt += ', Time: {}'.format(cputime[iter])
            if saveobjective:
                txt += ', Obj: {}, dObj%%: {}'.format(objective[iter],
                            100*np.abs(objective[iter]-objective[iter-1])/np.abs(objective[iter-1]))
            if savereconerror:
                txt += ', Err: {}'.format(reconerror[iter])
            print(txt)

        ## ==== Prepare for next iteration
        ## Update alpha
        if alphamethod == 0:
            pass # do nothing, constant alpha
        elif alphamethod == 1: # BB method
            # Adx is overwritten at top of iteration, so this is an ok reuse
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
                
        ## ==== Store current values as previous values for next iteration
        xprevious = x
        Axprevious = Ax
        iter += 1
    ## ===========================
    ## = End Main Algorithm Loop =
    ## ===========================
        
    #         case 1 % Barzilai-Borwein choice of alpha
    #             if monotone 
    #                 % do acceptance criterion.
    #                 past = (max(iter-1-acceptpast,0):iter-1) + 1;
    #                 maxpastobjective = max(objective(past));
    #                 accept = 0;
    #                 while (accept == 0)

    #                     % --- Compute the step, and perform Gaussian 
    #                     %     denoising subproblem ----
    #                     dx = xprevious;
    #                     step = xprevious - grad./alpha;
    #                     x = computesubsolution(step,tau,alpha,penalty,mu,...
    #                         W,WT,subminiter,submaxiter,substopcriterion,...
    #                         subtolerance);
    #                     dx = x - dx;
    #                     Adx = Axprevious;
    #                     Ax = A(x);
    #                     Adx = Ax - Adx;
    #                     normsqdx = sum( dx(:).^2 );

    #                     % --- Compute the resulting objective 
    #                     objective(iter + 1) = computeobjective(x,y,Ax,tau,...
    #                         noisetype,logepsilon,penalty,WT);

    #                     if ( objective(iter+1) <= (maxpastobjective ...
    #                             - acceptdecrease*alpha/2*normsqdx) ) ...
    #                             || (alpha >= acceptalphamax);
    #                         accept = 1;
    #                     end
    #                     acceptalpha = alpha;  % Keep value for displaying
    #                     alpha = acceptmult*alpha;
    #                 end
    #             else 
    #                 % just take bb setp, no enforcing monotonicity.
    #                 dx = xprevious;
    #                 step = xprevious - grad./alpha;
    #                 x = computesubsolution(step,tau,alpha,penalty,mu,...
    #                     W,WT,subminiter,submaxiter,substopcriterion,...
    #                     subtolerance);
    #                 dx = x - dx;
    #                 Adx = Axprevious;
    #                 Ax = A(x);
    #                 Adx = Ax - Adx;
    #                 normsqdx = sum( dx(:).^2 );
    #                 if saveobjective
    #                     objective(iter + 1) = computeobjective(x,y,Ax,tau,...
    #                         noisetype,logepsilon,penalty,WT);
    #                 end

    #             end
    #     end

    ## ==== Post process the output
    ## Add on mean if recentered (if not mu == 0);
    x = x + mu;

    ## Determine what needs to be in the variable output and
    ## crop the output if the maximum number of iterations were not used.
    ## Note, need to subtract 1 since iter is incremented at the end of the loop
    iter = iter - 1;
    varargout = [iter];

    if saveobjective:
        varargout.append(objective[0:iter]) ## Useless bounds 1:iter+1 ???
    if savereconerror:
        varargout.append(reconerror[0:iter])
    if savecputime:
        varargout.append(cputime[0:iter])
    if savesolutionpath:
        varargout.append(solutionpath[0:iter])

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