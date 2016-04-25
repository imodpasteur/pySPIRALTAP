% =============================================================================
% =        SPIRAL:  Sparse Poisson Intensity Reconstruction Algorithms        =
% =                      Demonstration Code Version 1.0                       =
% =============================================================================
% =    Copyright 2009, 2010                                                   =
% =    Zachary T. Harmany*, Roummel F. Marcia**, Rebecca M. Willett*          =
% =        *  Department of Electrical and Computer Engineering               =
% =           Duke University                                                 =
% =           Durham, NC 27708, USA                                           =
% =       **  School of Natural Sciences                                      =
% =           University of California, Merced                                =
% =           Merced, CA 95343, USA                                           =
% =                                                                           =
% =    Corresponding Author: Zachary T. Harmany (zth@duke.edu)                =
% =============================================================================
% Included here are three demonstrations, changing the varialbe 'demo' 
% to 1, 2, or 3 selects among three simulations.  Details of each can be 
% found below.
clear;close all
demo = 1;

switch demo
    case 1
% ============================================================================= 
% =                              Demonstration 1                              =
% ============================================================================= 
% Description:  One dimensional compressed sensing example penalizing the 
% sparsity (l1 norm) of the coefficients in the canonical basis.  Here the
% true signal f is of length 100,000 with 1,500 nonzero entries yielding a 
% sparsity of 1.5%.  We take 40,000 compressive measurements in y. The 
% average number of photons per measurement is 15.03, with a maximum of 145.
% We run SPIRAL until the relative change in the iterates falls below
% a tolerance of 1x10^-8, up to a maximum of 100 iterations (however only 
% 37 iterations are required to satisfy the stopping criterion).  
% 
% Output:  This demonstration automatically displays the following:
% Figure 1:   Simulation setup (true signal, true detector intensity, 
%             observed counts),
% Figure 2:   Reconstructed signal overlayed ontop of true signal,
% Figure 3:   RMSE error evolution versus iteration and compute time, and
% Figure 4:   Objective evolution versus iteration and compute time.

        % Load example data: 
        % f = True signal
        % A = Sensing matrix
        % y ~ Poisson(Af)
        load demodata/canonicaldata
         % Setup function handles for computing A and A^T:
        AT  = @(x) A'*x;
        A   = @(x) A*x;
        % Set regularization parameters and iteration limit:
        tau   = 1e-6;
        maxiter = 100;
        tolerance = 1e-8;
        verbose = 10;
        
        % Simple initialization:  
        % AT(y) rescaled to a least-squares fit to the mean intensity
        finit = (sum(sum(y)).*numel(AT(y)))...
                    ./(sum(sum(AT(y))) .*sum(sum((AT(ones(size(y)))))))...
                    .*AT(y);
                
        % Run the algorithm:
        % Demonstrating all the options for our algorithm:
        [fhatSPIRAL, iterationsSPIRAL, objectiveSPIRAL,...
            reconerrorSPIRAL, cputimeSPIRAL] ...
            = SPIRALTAP(y,A,tau,...
            'maxiter',maxiter,...
            'Initialization',finit,...
            'AT',AT,...
            'miniter',5,...
            'stopcriterion',3,...
            'tolerance',tolerance,...
            'alphainit',1,...
            'alphamin', 1e-30,...
            'alphamax', 1e30,...
            'alphaaccept',1e30,...
            'logepsilon',1e-10,...
            'saveobjective',1,...
            'savereconerror',1,...
            'savecputime',1,...
            'savesolutionpath',0,...
            'truth',f,...
            'verbose',verbose);
        
        % Display Results:
        % Problem Data:
        figure(1); clf
        subplot(3,1,1)
            stem(f)
            title(['True Signal (f), Nonzeros = ',num2str(sum(f ~= 0)),...
                ' Mean Intensity = ',num2str(mean(f))])
            ylim([0 1.25.*max(f)])
        subplot(3,1,2)
            stem(A(f))
            title(['True Detector Intensity (Af), Mean Intensity = ',...
                num2str(mean(A(f)))])
        subplot(3,1,3)
            stem(y)
            title(['Observed Photon Counts (y), Mean Count = ',...
                num2str(mean(y))])
            
        % Reconstructed Signals:
        figure(2); clf
        stem(f,'b')
        hold on
        stem(fhatSPIRAL,'r')
        hold off
        xlabel('Sample Number'); ylabel('Amplitude')
        title({'SPIRAL Estimate',['RMS Error = ',...
            num2str(norm(f(:) - fhatSPIRAL(:))./norm(f(:)))],...
            ['Nonzero Components = ',...
            num2str(sum(fhatSPIRAL~=0))]})
       
        % RMS Error:
        figure(3); clf
        subplot(2,1,1)
            plot(0:iterationsSPIRAL,reconerrorSPIRAL,'b')
            xlabel('Iteration'); ylabel('RMS Error')
            title('RMS Error Evolution (Iteration)')
        subplot(2,1,2)
            plot(cputimeSPIRAL, reconerrorSPIRAL,'b')
            xlabel('CPU Time'); ylabel('RMS Error')
            title('RMS Error Evolution (CPU Time)')
        
        % Objective:
        figure(4); clf
        subplot(2,1,1)
            plot(0:iterationsSPIRAL,objectiveSPIRAL,'b')
	        xlabel('Iteration'); ylabel('Objective')
            title('Objective Evolution (Iteration)')
        subplot(2,1,2)
            plot(cputimeSPIRAL, objectiveSPIRAL,'b')
	        xlabel('CPU Time'); ylabel('Objective')
            title('Objective Evolution (CPU Time)')
            
    case 2
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

        % Load example data: 
        % f = True signal
        % blur = Blur PSF
        % y ~ Poisson(Af)
        load demodata/imagedata
        
        A = @(x) conv2(x,blur,'same');
        AT = @(x) conv2(x,blur,'same');
        Af = A(f);
                
        % Setup wavelet basis for l1-onb
        wav = daubcqf(2);
        W = @(x) midwt(x,wav);
        WT = @(x) mdwt(x,wav);
        
        % Set regularization parameters and iteration limit:
        tauonb      = 1.0e-5;
        tautv       = 3.0e-6;
        taurdp      = 2.0e+0;
        taurdpti    = 6.0e-1;
        
        miniter = 50;
        maxiter = 100;
        stopcriterion = 3;
        tolerance = 1e-8;
        verbose = 10;
        
        
        % Simple initialization:
        % AT(y) rescaled to have a least-squares fit to the 
        % mean value
        finit = (sum(sum(y)).*numel(AT(y)))...
            ./(sum(sum(AT(y))) .*sum(sum((AT(ones(size(y))))))).*AT(y);
        
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
          
    case 3
% ============================================================================= 
% =                              Demonstration 3                              =
% ============================================================================= 
% Description:  This demonstration is similar to Demonstration 2, but uses the
% 256x256 'Cameraman' image instead of the 128x128 Shepp-Logan phantom.  The
% true signal has a mean intensity of 1.19x10^5, the true detector mean 
% intensity is 44.4, and the observed photon count mean is 44.4 with a
% maximum of 111.  Due to the larger problem size, we need to alter the 
% termination criteria.  We run all the SPIRAL methods for a minimum of
% 50 iterations until the relative change in the iterates falls below a
% tolerance of 1x10^-8, up to a maximum of 300 iterations.
% 
% Output:  Like Demonstration 2, this demonstration automatically displays
% the following:
% Figure 1:   Simulation setup (true signal, true detector intensity, 
%             observed counts),
% Figure 2:   The objective evolution for the methods where explicit 
%             computation of the objective is possible,
% Figure 3:   RMSE error evolution versus iteration and compute time,
% Figure 4:   The final reconstructions, and
% Figure 5:   The the magnitude of the errors between the final 
%             reconstructions and the true phantom image. 

        % Load example data: 
        % f = True signal
        % blur = Blur PSF
        % y ~ Poisson(Af)
        load demodata/imagedata2
        % Need to generate data.
        
        A = @(x) conv2(x,blur,'same');
        AT = @(x) conv2(x,blur,'same');
        Af = A(f);
                
        % Setup wavelet basis for l1-onb
        wav = daubcqf(2);
        W = @(x) midwt(x,wav);
        WT = @(x) mdwt(x,wav);
        
        % Set regularization parameters and iteration limit:
        tauonb    = 2.10000e-05;
        tautv     = 1.40000e-05;
        taurdp    = 2.00000e+00;
        taurdpti  = 7.80000e-01;
       
        
        miniter = 50;
        maxiter = 100;
        stopcriterion = 3;
        tolerance = 1e-8;
        verbose = 10;
        
        
        % Simple initialization:
        % AT(y) rescaled to have a least-squares fit to the 
        % mean value
        finit = (sum(sum(y)).*numel(AT(y)))...
            ./(sum(sum(AT(y))) .*sum(sum((AT(ones(size(y))))))).*AT(y);
       % return
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
            
    otherwise
        error('Invalid demonstration number, choose 1, 2, or 3.')
end 
