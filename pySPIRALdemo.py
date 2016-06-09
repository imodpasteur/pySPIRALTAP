#-*-coding:utf-8-*-
# SPIRAL: Sparse Poisson Intensity Reconstruction Algorithms
# Demonstration Code Version 1.0
# Matlab version by Zachary T. Harmany (zth@duke.edu)
# Python version by Maxime Woringer, Apr. 2016

#Included here are three demonstrations, changing the varialbe 'demo' 
# to 1, 2, or 3 selects among three simulations.  Details of each can be 
# found below.

# ==== importations
from __future__ import print_function
try:
    import rwt
except Exception:
    raise ImportError("the 'rwt' (Rice Wavelet Toolbox) package could not be loaded. It can be installed from https://github.com/ricedsp/rwt/")

import pySPIRALTAP, sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io # to import.mat files
from conv2 import conv2

# ==== variables
demo = 2

# ==== Demo 1
if demo == 1:
    """
    # ============================================================================= 
    # =                              Demonstration 1                              =
    # ============================================================================= 
    # Description:  One dimensional compressed sensing example penalizing the 
    # sparsity (l1 norm) of the coefficients in the canonical basis.  Here the
    # true signal f is of length 100,000 with 1,500 nonzero entries yielding a 
    # sparsity of 1.5%.  We take 40,000 compressive measurements in y. The 
    # average number of photons per measurement is 15.03, with a maximum of 145.
    # We run SPIRAL until the relative change in the iterates falls below
    # a tolerance of 1x10^-8, up to a maximum of 100 iterations (however only 
    # 37 iterations are required to satisfy the stopping criterion).  
    # 
    # Output:  This demonstration automatically displays the following:
    # Figure 1:   Simulation setup (true signal, true detector intensity, 
    #             observed counts),
    # Figure 2:   Reconstructed signal overlayed ontop of true signal,
    # Figure 3:   RMSE error evolution versus iteration and compute time, and
    # Figure 4:   Objective evolution versus iteration and compute time.
    """
    
    # ==== Load example data: 
    # f = True signal
    # A = Sensing matrix
    # y ~ Poisson(Af)
    rf=scipy.io.loadmat('./demodata/canonicaldata.mat')
    f,y,Aorig = (rf['f'], rf['y'], rf['A']) # A Stored as a sparse matrix

    ## Setup function handles for computing A and A^T:
    AT = lambda x: Aorig.transpose().dot(x)
    A = lambda x: Aorig.dot(x)
    
    # ==== Set regularization parameters and iteration limit:
    tau   = 1e-6
    maxiter = 100
    tolerance = 1e-8
    verbose = 10

    # ==== Simple initialization:  
    # AT(y) rescaled to a least-squares fit to the mean intensity
    finit = y.sum()*AT(y).size/AT(y).sum()/AT(np.ones_like(y)).sum() * AT(y)
    
    # ==== Run the algorithm:
    ## Demonstrating all the options for our algorithm:
    resSPIRAL = pySPIRALTAP.SPIRALTAP(y,A,tau,
                                      AT=AT,
                                      maxiter=maxiter,
                                      miniter=5,
                                      penalty='canonical',
                                      noisetype='gaussian',
                                      initialization=finit,
                                      stopcriterion=3,
                                      tolerance=tolerance,
                                      alphainit=1,
                                      alphamin=1e-30,
                                      alphamax=1e30,
                                      alphaaccept=1e30,
                                      logepsilon=1e-10,
                                      saveobjective=True,
                                      savereconerror=True,
                                      savesolutionpath=False,
                                      truth=f,
                                      verbose=verbose, savecputime=True)
    ## Deparse outputs
    fhatSPIRAL = resSPIRAL[0]
    parSPIRAL = resSPIRAL[1]
    iterationsSPIRAL = parSPIRAL['iterations']
    objectiveSPIRAL = parSPIRAL['objective']
    reconerrorSPIRAL = parSPIRAL['reconerror']
    cputimeSPIRAL = parSPIRAL['cputime']

    ## ==== Display Results:
    ## Problem Data:
    plt.figure(1)
    plt.subplot(311)
    plt.plot(f)
    plt.title('True Signal (f), Nonzeros = {}, Mean Intensity = {}'.format((f!=0).sum(), f.mean()))
    plt.ylim((0, 1.24*f.max()))

    plt.subplot(312)
    plt.plot(A(f))
    plt.title('True Detector Intensity (Af), Mean Intensity = {}'.format(A(f).mean()))

    plt.subplot(313)
    plt.plot(y)
    plt.title('Observed Photon Counts (y), Mean Count = {}'.format(y.mean()))

    ## Reconstructed Signals:
    plt.figure(2)
    plt.plot(f, color='blue')
    plt.plot(fhatSPIRAL, color='red')
    plt.xlabel('Sample number')
    plt.ylabel('Amplitude')
    plt.title('SPIRAL Estimate, RMS error = {}, Nonzero Components = {}'.format(np.linalg.norm(f-fhatSPIRAL)/np.linalg.norm(f), (fhatSPIRAL!=0).sum()))

    ## RMS Error:
    plt.figure(3)
    plt.subplot(211)
    plt.plot(range(iterationsSPIRAL), reconerrorSPIRAL, color='blue')
    plt.xlabel('Iteration')
    plt.ylabel('RMS Error')

    plt.subplot(212)
    plt.plot(cputimeSPIRAL, reconerrorSPIRAL, color='blue')
    plt.xlabel('CPU Time')
    plt.ylabel('RMS Error')
    plt.title('RMS Error Evolution (CPU Time)')

    ## Objective:
    plt.figure(4)
    plt.subplot(211)
    plt.plot(range(iterationsSPIRAL), objectiveSPIRAL)
    plt.xlabel('Iteration')
    plt.ylabel('Objective')
    plt.subplot(212)
    plt.plot(cputimeSPIRAL, objectiveSPIRAL)
    plt.xlabel('CPU Time')
    plt.ylabel('Objective')
    plt.title('Objective Evolution (CPU Time)')

    plt.show()

elif demo == 2:
    """
    % ============================================================================= 
    % =                              Demonstration 2                              =
    % ============================================================================= 
    % Description:  Here we consider an image deblurring example.  The true signal 
    % is a 128x128 Shepp-Logan phantom image with mean intensity 1.22x10^5.  The
    % true detector mean intensity is 45.8, and the observed photon count mean
    % is 45.8 with a maximum of 398.  Here we consider four penalization methods
    % - Sparsity (l1 norm) of coefficients in an orthonormal (wavelet) basis,
    % - Total variation of the image,
    % - Penalty based on Recursive Dyadic Partitions (RDPs), and
    % - Penalty based on Translationally-Invariant (cycle-spun) RDPs.
    % We run all the SPIRAL methods for a minimum of 50 iterations until the 
    % relative change in the iterates falls below a tolerance of 1x10^-8, up 
    % to a maximum of 100 iterations (however only ~70 iterations are required
    % to satisfy the stopping criterion for all the methods).  
    % 
    % Output:  This demonstration automatically displays the following:
    % Figure 1:   Simulation setup (true signal, true detector intensity, 
    %             observed counts),
    % Figure 2:   The objective evolution for the methods where explicit 
    %             computation of the objective is possible,
    % Figure 3:   RMSE error evolution versus iteration and compute time,
    % Figure 4:   The final reconstructions, and
    % Figure 5:   The the magnitude of the errors between the final 
    %             reconstructions and the true phantom image. 
    """
    # ==== Load example data
    # f = True signal
    # blur = Blur PSF
    # y ~ Poisson(Af)
    rf=scipy.io.loadmat('./demodata/imagedata.mat')
    f, blur, y = (np.float_(rf['f']), np.float_(rf['blur']), np.float_(rf['y']))

    A = lambda x: conv2(x, blur, 'same')
    AT = lambda x: conv2(x, blur, 'same')
    Af = A(f)

    # ==== Setup wavelet basis for l1-onb
    wav = rwt.daubcqf(2)[0]
    W = lambda x: rwt.idwt(x,wav)[0]
    WT = lambda x: rwt.dwt(x,wav)[0]

    # ==== Set regularization parameters and iteration limit:
    tauonb      = 1.0e-5
    tautv       = 3.0e-6
    taurdp      = 2.0e+0
    taurdpti    = 6.0e-1
    
    miniter = 50
    maxiter = 100
    stopcriterion = 3
    tolerance = 1e-8
    verbose = 10 

    # ==== Simple initialization: AT(y) rescaled to have a least-squares fit to the mean value
    finit = y.sum()*AT(y).size/AT(y).sum()/AT(np.ones_like(y)).sum() * AT(y)

    # ==== Run the algorithm, demonstrating all the options for our algorithm:
    resSPIRAL = pySPIRALTAP.SPIRALTAP(y, A, tauonb, penalty='onb', AT=AT, W=W, WT=WT,
                                      maxiter=maxiter, initialisation=finit, miniter=miniter,
                                      stopcriterion=stopcriterion, monotone=True, 
                                      saveobjective=True, savereconerror=True, savecputime=True, 
                                      savesolutionpath=False, truth=f, verbose=verbose)
    ## Deparse outputs
    fhatSPIRALonb = resSPIRAL[0]
    parSPIRAL = resSPIRAL[1]
    iterationsSPIRALonb = parSPIRAL['iterations']
    objectiveSPIRALonb = parSPIRAL['objective']
    reconerrorSPIRALonb = parSPIRAL['reconerror']
    cputimeSPIRALonb = parSPIRAL['cputime']

    resSPIRAL = pySPIRALTAP.SPIRALTAP(y, A, tautv, penalty='tv', AT=AT,
                                      maxiter=maxiter, initialisation=finit, miniter=miniter,
                                      stopcriterion=stopcriterion, tolerance=tolerance,
                                      monotone=True, saveobjective=True, savereconerror=True,
                                      savecputime=True, savesolutionpath=False, truth=f,
                                      verbose=verbose)
    ## Deparse outputs
    fhatSPIRALtv = resSPIRAL[0]
    parSPIRAL = resSPIRAL[1]
    iterationsSPIRALtv = parSPIRAL['iterations']
    objectiveSPIRALtv = parSPIRAL['objective']
    reconerrorSPIRALtv = parSPIRAL['reconerror']
    cputimeSPIRALtv = parSPIRAL['cputime']
    
    # resSPIRAL = pySPIRALTAP.SPIRALTAP(y, A, taurdp, penalty='rdp', AT=AT, maxiter=maxiter,
    #                                   initialisation=finit, miniter=miniter,
    #                                   stopcriterion=stopcriterion, tolerance=tolerance,
    #                                   monotone=False, saveobjective=False, savereconerror=True,
    #                                   savecputime=True, savesolutionpath=False, truth=f,
    #                                   verbose=verbose)
    # ## Deparse outputs
    # fhatSPIRALrdp = resSPIRAL[0]
    # parSPIRAL = resSPIRAL[1]
    # iterationsSPIRALrdp = parSPIRAL['iterations']
    # reconerrorSPIRALrdp = parSPIRAL['reconerror']
    # cputimeSPIRALrdp = parSPIRAL['cputime']

    # [fhatSPIRALrdpti, iterationsSPIRALrdpti, ...
    #  reconerrorSPIRALrdpti, cputimeSPIRALrdpti] ...
    # resSPIRAL = pySPIRALTAP.SPIRALTAP(y, A, taurdpti, penalty='rdp-ti', maxiter=maxiter,
    #                                   initialization=finit, miniter=miniter, AT=AT,
    #                                   stopcriterion=stopcriterion, tolerance=tolerance,
    #                                   monotone=False, saveobjective=False, savereconerror=True,
    #                                   savecputime=True, savesolutionpath=False, truth=f,
    #                                   verbose=verbose)
    # ## Deparse outputs
    # fhatSPIRALrdpti = resSPIRAL[0]
    # parSPIRAL = resSPIRAL[1]
    # iterationsSPIRALrdpti = parSPIRAL['iterations']
    # objectiveSPIRALtv = parSPIRAL['objective']
    # reconerrorSPIRALrdpti = parSPIRAL['reconerror']
    # cputimeSPIRALrdpti = parSPIRAL['cputime']
    print("WARNING: RDP-based reconstruction are not implemented yet" , file=sys.stderr)

    # ==== Display results
    # Problem data
    plt.figure()
    plt.subplot(131); plt.imshow(f, cmap='gray'); plt.title('True Signal (f)')
    plt.subplot(132); plt.imshow(Af, cmap='gray'); plt.title('True Detector Intensity (Af)')
    plt.subplot(133); plt.imshow(y, cmap='gray'); plt.title('Observed Photon Counts (y)')

    # Display Objectives for Monotonic Methods
    plt.figure()
    plt.subplot(121)
    plt.plot(range(iterationsSPIRALonb), objectiveSPIRALonb,
             label='ONB Objective Evolution (Iteration)')
    plt.plot(range(iterationsSPIRALtv), objectiveSPIRALtv,
            label='TV Objective Evolution (Iteration)')
    plt.xlabel('Iteration');plt.ylabel('Objective');plt.legend()
    plt.xlim((0, np.max((iterationsSPIRALonb, iterationsSPIRALtv))))

    plt.subplot(122)
    plt.plot(cputimeSPIRALonb, objectiveSPIRALonb, label='ONB Objective Evolution (CPU Time)')
    plt.plot(cputimeSPIRALtv, objectiveSPIRALtv, label='TV Objective Evolution (CPU Time)')
    plt.xlabel('CPU Time');plt.ylabel('Objective');plt.legend()
    
    # Display RMS Error Evolution for All Methods
    plt.subplot(121)
    plt.plot(range(iterationsSPIRALonb), reconerrorSPIRALonb, label='ONB')
    plt.plot(range(iterationsSPIRALtv), reconerrorSPIRALtv, label='TV')
    #plt.plot(range(iterationsSPIRALrdp), reconerrorSPIRALrdp, label='RDP')
    #plt.plot(range(iterationsSPIRALrdpti), reconerrorSPIRALrdpti, label='RDP-TI')
    plt.title('Error Evolution (Iteration)');plt.xlabel('Iteration');plt.ylabel('RMS Error')

    plt.subplot(122)
    plt.plot(cputimeSPIRALonb, reconerrorSPIRALonb, label='ONB')
    plt.plot(cputimeSPIRALtv, reconerrorSPIRALtv, label='TV')
    #plt.plot(cputimeSPIRALrdp), reconerrorSPIRALrdp, label='RDP')
    #plt.plot(cputimeSPIRALrdpti), reconerrorSPIRALrdpti, label='RDP-TI')
    plt.title('Error Evolution (CPU Time)');plt.xlabel('CPU Time');plt.ylabel('RMS Error')

    # Display Images for All Methods
    plt.figure()
    plt.subplot(121);plt.imshow(fhatSPIRALonb, cmap='gray')
    plt.title("ONB, RMS={}".format(reconerrorSPIRALonb[-1]))
    plt.subplot(122);plt.imshow(fhatSPIRALtv, cmap='gray')
    plt.title("TV, RMS={}".format(reconerrorSPIRALtv[-1]))
    #plt.subplot(223);plt.imshow(fhatSPIRALrdp, cmap='gray')
    #plt.title("RDP, RMS=".format(reconerrorSPIRALrdp[-1]))
    #plt.subplot(224);plt.imshow(fhatSPIRALrdpti, cmap='gray')
    #plt.title("RDP-TI, RMS=".format(reconerrorSPIRALrdpti[-1]))

    # Difference images
    diffSPIRALonb = np.abs(f-fhatSPIRALonb)
    diffSPIRALtv = np.abs(f-fhatSPIRALtv)
    #diffSPIRALrdp = np.abs(f-fhatSPIRALrdp)
    #diffSPIRALrdpti = np.abs(f-fhatSPIRALrdpti)

    plt.figure()
    plt.subplot(121);plt.imshow(diffSPIRALonb)
    plt.title("ONB, RMS={}".format(reconerrorSPIRALonb[-1]))
    plt.subplot(122);plt.imshow(diffSPIRALtv)
    plt.title("TV, RMS={}".format(reconerrorSPIRALtv[-1]))
    #plt.subplot(223);plt.imshow(diffSPIRALrdp)
    #plt.title("RDP, RMS=".format(reconerrorSPIRALrdp[-1]))
    #plt.subplot(224);plt.imshow(diffSPIRALrdpti)
    #plt.title("RDP-TI, RMS=".format(reconerrorSPIRALrdpti[-1]))    

    plt.show()
