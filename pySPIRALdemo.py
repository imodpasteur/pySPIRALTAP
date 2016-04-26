#-*-coding:utf-8-*-
# SPIRAL: Sparse Poisson Intensity Reconstruction Algorithms
# Demonstration Code Version 1.0
# Matlab version by Zachary T. Harmany (zth@duke.edu)
# Python version by Maxime Woringer, Apr. 2016

#Included here are three demonstrations, changing the varialbe 'demo' 
# to 1, 2, or 3 selects among three simulations.  Details of each can be 
# found below.

# ==== importations
import pySPIRALTAP
import numpy as np
import scipy.io # to import.mat files

# ==== variables
demo = 1

# ==== Demo 1
if demo == 1:
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

    # ==== Load example data: 
    # f = True signal
    # A = Sensing matrix
    # y ~ Poisson(Af)
    rf=scipy.io.loadmat('./demodata/canonicaldata.mat')
    f,y,Aorig = (rf['f'], rf['y'], rf['A']) # A Stored as a sparse matrix

    #  % Setup function handles for computing A and A^T:
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
    # % Demonstrating all the options for our algorithm:
    # [fhatSPIRAL, iterationsSPIRAL, objectiveSPIRAL,...
    #     reconerrorSPIRAL, cputimeSPIRAL] ...
    #     = SPIRALTAP(y,A,tau,...
    #     'maxiter',maxiter,...
    #     'Initialization',finit,...
    #     'AT',AT,...
    #     'miniter',5,...
    #     'stopcriterion',3,...
    #     'tolerance',tolerance,...
    #     'alphainit',1,...
    #     'alphamin', 1e-30,...
    #     'alphamax', 1e30,...
    #     'alphaaccept',1e30,...
    #     'logepsilon',1e-10,...
    #     'saveobjective',1,...
    #     'savereconerror',1,...
    #     'savecputime',1,...
    #     'savesolutionpath',0,...
    #     'truth',f,...
    #     'verbose',verbose);
    fhatSPIRAL = pySPIRALTAP.SPIRALTAP(y,A,tau,
                                       AT=AT,
                                       verbose=1)
    
    # % Display Results:
    # % Problem Data:
    # figure(1); clf
    # subplot(3,1,1)
    #     stem(f)
    #     title(['True Signal (f), Nonzeros = ',num2str(sum(f ~= 0)),...
    #         ' Mean Intensity = ',num2str(mean(f))])
    #     ylim([0 1.25.*max(f)])
    # subplot(3,1,2)
    #     stem(A(f))
    #     title(['True Detector Intensity (Af), Mean Intensity = ',...
    #         num2str(mean(A(f)))])
    # subplot(3,1,3)
    #     stem(y)
    #     title(['Observed Photon Counts (y), Mean Count = ',...
    #         num2str(mean(y))])

    # % Reconstructed Signals:
    # figure(2); clf
    # stem(f,'b')
    # hold on
    # stem(fhatSPIRAL,'r')
    # hold off
    # xlabel('Sample Number'); ylabel('Amplitude')
    # title({'SPIRAL Estimate',['RMS Error = ',...
    #     num2str(norm(f(:) - fhatSPIRAL(:))./norm(f(:)))],...
    #     ['Nonzero Components = ',...
    #     num2str(sum(fhatSPIRAL~=0))]})

    # % RMS Error:
    # figure(3); clf
    # subplot(2,1,1)
    #     plot(0:iterationsSPIRAL,reconerrorSPIRAL,'b')
    #     xlabel('Iteration'); ylabel('RMS Error')
    #     title('RMS Error Evolution (Iteration)')
    # subplot(2,1,2)
    #     plot(cputimeSPIRAL, reconerrorSPIRAL,'b')
    #     xlabel('CPU Time'); ylabel('RMS Error')
    #     title('RMS Error Evolution (CPU Time)')

    # % Objective:
    # figure(4); clf
    # subplot(2,1,1)
    #     plot(0:iterationsSPIRAL,objectiveSPIRAL,'b')
    #         xlabel('Iteration'); ylabel('Objective')
    #     title('Objective Evolution (Iteration)')
    # subplot(2,1,2)
    #     plot(cputimeSPIRAL, objectiveSPIRAL,'b')
    #         xlabel('CPU Time'); ylabel('Objective')
    #     title('Objective Evolution (CPU Time)')
