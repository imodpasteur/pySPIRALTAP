#-*-coding:utf-8-*-
# SPIRAL: Sparse Poisson Intensity Reconstruction Algorithms
# Demonstration Code Version 1.0
# Matlab version by Zachary T. Harmany (zth@duke.edu)
# Python version by Maxime Woringer, Apr. 2016

#Included here are three demonstrations, changing the varialbe 'demo' 
# to 1, 2, or 3 selects among three simulations.  Details of each can be 
# found below.

# ==== importations
try:
    import rwt
except Exception:
    raise ImportError("the 'rwt' (Rice Wavelet Toolbox) package could not be loaded. It can be installed from https://github.com/ricedsp/rwt/")

import pySPIRALTAP
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
    wav = rwt.daubcqf(2)
    W = lambda x: rwt.midwt(x,wav)
    WT = lambda x: rwt.midwt(x,wav)

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
    #    [fhatSPIRALonb, iterationsSPIRALonb, objectiveSPIRALonb,...
    #        reconerrorSPIRALonb, cputimeSPIRALonb] ...
    pySPIRALTAP.SPIRALTAP(y, A, tauonb, penalty='onb', AT=AT, W=W, WT=WT, maxiter=maxiter,
                          initialisation=finit, miniter=miniter, stopcriterion=stopcriterion,
                          monotone=True, saveobjective=True, savereconerror=True,
                          savecputime=True, savesolutionpath=False, truth=f, verbose=verbose)
"""        
        % Run the algorithm:
        % Demonstrating all the options for our algorithm:
        [fhatSPIRALonb, iterationsSPIRALonb, objectiveSPIRALonb,...
            reconerrorSPIRALonb, cputimeSPIRALonb] ...
            = SPIRALTAP(y,A,tauonb,...
            'penalty','ONB',...
            'AT',AT,...
            'W',W,...
            'WT',WT,...
            'maxiter',maxiter,...
            'Initialization',finit,...
            'miniter',miniter,...
            'stopcriterion',stopcriterion,...
            'monotone',1,...
            'saveobjective',1,...
            'savereconerror',1,...
            'savecputime',1,...
            'savesolutionpath',0,...
            'truth',f,...
            'verbose',verbose);

        [fhatSPIRALtv, iterationsSPIRALtv, objectiveSPIRALtv,...
            reconerrorSPIRALtv, cputimeSPIRALtv] ...
            = SPIRALTAP(y,A,tautv,...
            'penalty','TV',...
            'AT',AT,...    
            'maxiter',maxiter,...
            'Initialization',finit,...
            'miniter',miniter,...
            'stopcriterion',stopcriterion,...
            'tolerance',tolerance,...
            'monotone',1,...
            'saveobjective',1,...
            'savereconerror',1,...
            'savecputime',1,...
            'savesolutionpath',0,...
            'truth',f,...
            'verbose',verbose);
        
        [fhatSPIRALrdp, iterationsSPIRALrdp, ...
            reconerrorSPIRALrdp, cputimeSPIRALrdp] ...
            = SPIRALTAP(y,A,taurdp,...
            'penalty','RDP',...
            'AT',AT,...
            'maxiter',maxiter,...
            'Initialization',finit,...
            'miniter',miniter,...
            'stopcriterion',stopcriterion,...
            'tolerance',tolerance,...
            'monotone',0,...
            'saveobjective',0,...
            'savereconerror',1,...
            'savecputime',1,...
            'savesolutionpath',0,...
            'truth',f,...
            'verbose',verbose);
        
        [fhatSPIRALrdpti, iterationsSPIRALrdpti, ...
            reconerrorSPIRALrdpti, cputimeSPIRALrdpti] ...
            = SPIRALTAP(y,A,taurdpti,...
            'penalty','RDP-TI',...
            'maxiter',maxiter,...
            'Initialization',finit,...
            'miniter',miniter,...
            'AT',AT,...
            'stopcriterion',stopcriterion,...
            'tolerance',tolerance,...
            'monotone',0,...
            'saveobjective',0,...
            'savereconerror',1,...
            'savecputime',1,...
            'savesolutionpath',0,...
            'truth',f,...
            'verbose',verbose);
        
        % ===== Display Results ====
        % Display Problem Data
        figure(1);
        clf
        subplot(1,3,1);
            imagesc(f);colormap gray;axis image
            title('True Signal (f)')
        subplot(1,3,2);
            imagesc(Af);colormap gray;axis image
            title('True Detector Intensity (Af)')
        subplot(1,3,3);
            imagesc(y),[0 max(Af(:))];colormap gray;axis image
            title('Observed Photon Counts (y)')
        
        % Display Objectives for Monotonic Methods
        figure(2)
        subplot(2,2,1)
            plot(0:iterationsSPIRALonb,objectiveSPIRALonb,'b')
            title({'ONB Objective Evolution (Iteration)',' '})
            xlabel('Iteration'); ylabel('Objective')
            xlim([0 iterationsSPIRALonb])
        subplot(2,2,2)
            plot(cputimeSPIRALonb, objectiveSPIRALonb,'b')
            title({'ONB Objective Evolution (CPU Time)',' '})
            xlabel('CPU Time'); ylabel('Objective')
            xlim([0 cputimeSPIRALonb(end)])
        subplot(2,2,3)
            plot(0:iterationsSPIRALtv, objectiveSPIRALtv,'r')
            title({'TV Objective Evolution (Iteration)',' '})
            xlabel('Iteration'); ylabel('Objective')
            xlim([0 iterationsSPIRALtv])
        subplot(2,2,4)
            plot(cputimeSPIRALtv, objectiveSPIRALtv,'r')
            title({'TV Objective Evolution (CPU Time)',' '})
            xlabel('CPU Time'); ylabel('Objective')
            xlim([0 cputimeSPIRALtv(end)])

        % Display RMS Error Evolution for All Methods
        figure(3)
        clf
        subplot(2,1,1)
            plot(0:iterationsSPIRALonb,reconerrorSPIRALonb,'b')
            hold on
            plot(0:iterationsSPIRALtv,reconerrorSPIRALtv,'r')
            plot(0:iterationsSPIRALrdp,reconerrorSPIRALrdp,'m')
            plot(0:iterationsSPIRALrdpti,reconerrorSPIRALrdpti,'g')
            legend('ONB','TV','RDP','RDP-TI')
            title('Error Evolution (Iteration)')
            xlabel('Iteration'); ylabel('RMS Error')
        subplot(2,1,2)
            plot(cputimeSPIRALonb,reconerrorSPIRALonb,'b')
            hold on
            plot(cputimeSPIRALtv,reconerrorSPIRALtv,'r')
            plot(cputimeSPIRALrdp,reconerrorSPIRALrdp,'m')
            plot(cputimeSPIRALrdpti,reconerrorSPIRALrdpti,'g')
            legend('ONB','TV','RDP','RDP-TI')
            title('Error Evolution (CPU Time)')
            xlabel('CPU Time'); ylabel('RMS Error')
        
        % Display Images for All Methods
        figure(4)
        subplot(2,2,1);
            imagesc(fhatSPIRALonb,[0 max(f(:))]);colormap gray;axis image
            title({'ONB', ['RMS = ',num2str(reconerrorSPIRALonb(end))]})
        subplot(2,2,2);
            imagesc(fhatSPIRALtv,[0 max(f(:))]);colormap gray;axis image
            title({'TV', ['RMS = ',num2str(reconerrorSPIRALtv(end))]})
        subplot(2,2,3);
            imagesc(fhatSPIRALrdp,[0 max(f(:))]);colormap gray;axis image
            title({'RDP', ['RMS = ',num2str(reconerrorSPIRALrdp(end))]})
        subplot(2,2,4);
            imagesc(fhatSPIRALrdpti,[0 max(f(:))]);colormap gray;axis image
            title({'RDP-TI', ['RMS = ',num2str(reconerrorSPIRALrdpti(end))]})
            
        % Difference images
        diffSPIRALonb = abs(f-fhatSPIRALonb);
        diffSPIRALtv = abs(f-fhatSPIRALtv);
        diffSPIRALrdp = abs(f-fhatSPIRALrdp);
        diffSPIRALrdpti = abs(f-fhatSPIRALrdpti);
        maxdiffSPIRAL = max([diffSPIRALonb(:);diffSPIRALtv(:);...
            diffSPIRALrdp(:);diffSPIRALrdpti(:)]);
        figure(5)
        subplot(2,2,1);
            imagesc(diffSPIRALonb,[0 maxdiffSPIRAL]);colormap jet;axis image
            title({'ONB', ['RMS = ',num2str(reconerrorSPIRALonb(end))]})
        subplot(2,2,2);
            imagesc(diffSPIRALtv,[0 maxdiffSPIRAL]);colormap jet;axis image
            title({'TV', ['RMS = ',num2str(reconerrorSPIRALtv(end))]})
        subplot(2,2,3);
            imagesc(diffSPIRALrdp,[0 maxdiffSPIRAL]);colormap jet;axis image
            title({'RDP', ['RMS = ',num2str(reconerrorSPIRALrdp(end))]})
        subplot(2,2,4);
            imagesc(diffSPIRALrdpti,[0 maxdiffSPIRAL]);colormap jet;axis image
            title({'RDP-TI', ['RMS = ',num2str(reconerrorSPIRALrdpti(end))]})
"""
