## I have little information about this package.
## The most informative stuff I could derive comes from this page
## However, this does not provide clear licensing information.

## ==== Importations
import numpy as np

## ==== Helper functions
def LogLike(x,lambda,noiseType):
    """Returns log-likelihood"""
    # function L = logLike(x,lambda,noiseType)

    # switch strcmp(noiseType,'Poisson') 
    #   case 1
    #     L = (-lambda+x.*log(lambda+realmin)).*(lambda>0) + 0;
    #   case 0
    #     L = -(lambda-x).^2;
    # end
    # return;

## ==== Main Functions
def haarTIApprox2DNN_recentered(x,pen,mu=None):
    ## ==== Validate inputs
    if mu==None:
        noiseType='Poisson'
    else:
        noiseType='Gaussian'
        if np.any(np.isnan(x)) or np.any(x<0):
            raise TypeError('Invalid Poisson counts; check to make sure intensity non-negative.')

    (M,N) = x.shape
    L = np.log2(min(M,N))
        
    ## ==== Let's go
    xScalingPrev = x
    y = x.copy()
    y[x<mu] = mu

    wavelet_n = np.zeros([M,N,L])
    wavelet_m = np.zeros([M,N,L])
    wavelet_k = np.zeros([M,N,L])
    splitDecision = np.zeros([M,N,L]);

optProb = logLike(x,x,noiseType)-pen;
if strcmp(noiseType,'Poisson')
  mergeProb = zeros(size(x));
else
  mergeProb = -max(x,mu).^2;
end


for iL = 1:(L+1)
  dyadLen = 2^iL;
  % calculate current scaling coefficient
  xScaling = (xScalingPrev + circshift(xScalingPrev,[0,-dyadLen/2]) + ...
    circshift(xScalingPrev,[-dyadLen/2,0]) + ...
    circshift(xScalingPrev,[-dyadLen/2,-dyadLen/2]))/4;

  % log probability of merging
  mergeSum = mergeProb+circshift(mergeProb,[0,-dyadLen/2]) ...
    + circshift(mergeProb,[-dyadLen/2,0]) ...
    + circshift(mergeProb,[-dyadLen/2,-dyadLen/2]);
  if strcmp(noiseType,'Poisson')
    pMerge = mergeSum - ...
      (xScaling*dyadLen*dyadLen.*(1-log(xScaling*dyadLen*dyadLen+realmin)-log(0.25))).*(xScaling>0) - ...
      pen;
  else
    pMerge = mergeSum + max(xScaling,mu).^2*dyadLen*dyadLen - pen;
  end

  % log probability of splitting
  pSplit = optProb+circshift(optProb,[0,-dyadLen/2]) ...
    + circshift(optProb,[-dyadLen/2,0]) ...
    + circshift(optProb,[-dyadLen/2,-dyadLen/2]);
  
  % terms of merge log probability needed to calculated pMerge at next
  % scale
  if strcmp(noiseType,'Poisson')
    mergeProb = mergeSum + xScaling*dyadLen*dyadLen*log(0.25);
  else
    mergeProb = mergeSum;
  end
  
  % "wavelet" coefficients
  wavelet_n(:,:,iL) = (xScalingPrev + circshift(xScalingPrev,[-dyadLen/2,0]))/2-xScaling;
  wavelet_m(:,:,iL) = (xScalingPrev + circshift(xScalingPrev,[0,-dyadLen/2]))/2-xScaling;
  wavelet_k(:,:,iL) = (xScalingPrev + circshift(xScalingPrev,[-dyadLen/2,-dyadLen/2]))/2-xScaling;

  % decide whether to split or merge, save decision and associated log
  % probability
  splitDecision(:,:,iL) = (pSplit > pMerge).*1;
  optProb = pSplit.*splitDecision(:,:,iL) + pMerge.*(1-splitDecision(:,:,iL));
  
  xScalingPrev = xScaling;
  
end

% initial estimate is coarse scale scaling coefficients 
y = xScaling;
waveletScale = ones(M,N);
waveletScaleNext = ones(M,N);

for iL = (L+1):-1:1
  dyadLen = 2^iL;
  % if the split decision is zero, then the associated wavelet should be set
  % to zero.
  waveletScale = waveletScale.*splitDecision(:,:,iL);
  
  if iL > 1
    waveletScaleNext = waveletScaleNext - (0.25*(1.0-waveletScale));
    waveletScaleNext = circshift(circshift(waveletScaleNext,[0,-dyadLen/2]) ...
      - (0.25*(1.0-waveletScale)),[0,dyadLen/2]);
    waveletScaleNext = circshift(circshift(waveletScaleNext,[-dyadLen/2,0]) ...
      - (0.25*(1.0-waveletScale)),[dyadLen/2,0]);
    waveletScaleNext = circshift(circshift(waveletScaleNext,[-dyadLen/2,-dyadLen/2]) ...
      - (0.25*(1.0-waveletScale)),[dyadLen/2,dyadLen/2]);
  end
  
  % construct estimate based on wavelet coefficients and thresholds
  xD1 = (wavelet_n(:,:,iL) + wavelet_m(:,:,iL) + wavelet_k(:,:,iL)).*waveletScale;
  xD2 = circshift((wavelet_n(:,:,iL) - wavelet_m(:,:,iL) + wavelet_k(:,:,iL)).*waveletScale,[0,dyadLen/2]);
  xD3 = circshift((-wavelet_n(:,:,iL) + wavelet_m(:,:,iL) + wavelet_k(:,:,iL)).*waveletScale,[dyadLen/2,0]);
  xD4 = circshift((-wavelet_n(:,:,iL) - wavelet_m(:,:,iL) + wavelet_k(:,:,iL)).*waveletScale,[dyadLen/2,dyadLen/2]);

  xScaling = (y+xD1 + circshift(y,[0,dyadLen/2])-xD2 + circshift(y,[dyadLen/2,0])-xD3 + circshift(y,[dyadLen/2,dyadLen/2])+xD4)/4;

  y = xScaling;
  waveletScale = waveletScaleNext;
  waveletScaleNext = ones(M,N);
end
y = max(y,mu);
return;


function y = haarTVApprox2dNN_recentered(x,pen,mu);
%noiseType = 'Poisson';
noiseType = 'Gaussian';
if ((strcmp(noiseType,'Poisson')==1) & (any(isnan(x))==1))
  error('Invalid Poisson counts; check to make sure intensity non-negative.');
end
[M,N] = size(x);
L = log2(min(M,N));

xScalingPrev = x;
y = max(x,mu);
%y = x;

optProb = logLike(x,max(x,mu),noiseType)-pen;
if strcmp(noiseType,'Poisson')
  mergeProb = zeros(size(x));
else
  mergeProb = -max(x,mu).^2;
end


for iL = 1:L
  dyadLen = 2^iL;
  % calculate current scaling coefficient
  xScaling = dnsamp2((xScalingPrev + circshift(xScalingPrev,[0,-1]) + ...
    circshift(xScalingPrev,[-1,0]) + circshift(xScalingPrev,[-1,-1]))/4);
  
  % log probability of merging
  mergeSum = dnsamp2(mergeProb + circshift(mergeProb,[0,-1]) ...
    + circshift(mergeProb,[-1,0]) + circshift(mergeProb,[-1,-1]));
  if strcmp(noiseType,'Poisson')
    pMerge = mergeSum - ...
      (xScaling*dyadLen*dyadLen.*(1-log(xScaling*dyadLen*dyadLen+realmin)...
      -log(0.25))).*(xScaling>0) - ...
      pen;
    %     pMerge = mergeSum - xScaling*dyadLen*dyadLen.*(1-log(xScaling*dyadLen*dyadLen+realmin)-log(0.25)) - pen;
  else
    pMerge = mergeSum + max(xScaling,mu).^2*dyadLen*dyadLen - pen;
  end

  % log probability of splitting
  pSplit = dnsamp2(optProb+circshift(optProb,[0,-1]) ...
    + circshift(optProb,[-1,0]) + circshift(optProb,[-1,-1]));
  
  % terms of merge log probability needed to calculated pMerge at next
  % scale
  if strcmp(noiseType,'Poisson')
    mergeProb = mergeSum + xScaling*dyadLen*dyadLen*log(0.25);
  else
    mergeProb = mergeSum;
  end
  
  % decide whether to split or merge, save decision and associated log
  % probability
  splitDecision = (pSplit > pMerge).*1;
  optProb = pSplit.*splitDecision + pMerge.*(1-splitDecision);

  sDiL = kron(splitDecision,ones(dyadLen));
  mergeEst = max(kron(xScaling,ones(dyadLen)),mu);
  y = y.*sDiL + mergeEst.*(1-sDiL);
  
  xScalingPrev = xScaling;
  
end

return;
