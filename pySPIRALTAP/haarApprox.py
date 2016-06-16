## I have little information about this package.
## The most informative stuff I could derive comes from this page
## However, this does not provide clear licensing information.

## ==== Importations
from __future__ import print_function
import sys
import numpy as np

## ==== Helper functions
def todo():
    """Not implemented error function"""
    print('ERROR: This function is not yet implemented, please be patient!', file=sys.stderr)
    raise NotImplementedError

def logLike(x,lamb,noiseType):
    """Returns log-likelihood"""
    realmin =  np.finfo(np.float64).min
    if noiseType == 'Poisson':
        #     L = (-lambda+x.*log(lambda+realmin)).*(lambda>0) + 0;
        return (-lamb+x*np.log(lamb+realmin))*(lamb>0) + 0
    else:
        #     L = -(lambda-x).^2;
        return -(lamb-x)**2

def dnsamp2(x):
    """Downsamples by a factor 2 a 2D numpy array"""
    return x[::2,::2]

## ==== Main Functions
def haarTIApprox2DNN_recentered(x,pen,mu=None):
    ## ==== Various variables
    realmin =  np.finfo(np.float64).min
    
    ## ==== Validate inputs
    if mu==None:
        noiseType='Poisson'
        if np.any(np.isnan(x)) or np.any(x<0):
            raise TypeError('Invalid Poisson counts; check to make sure intensity non-negative.')
    else:
        noiseType='Gaussian'

    (M,N) = x.shape
    L = int(np.log2(min(M,N)))+1 ## Added +1 to debug
        
    ## ==== Let's go
    xScalingPrev = x
    y = x.copy()
    y[x<mu] = mu

    wavelet_n = np.zeros([M,N,L])
    wavelet_m = np.zeros([M,N,L])
    wavelet_k = np.zeros([M,N,L])
    splitDecision = np.zeros([M,N,L]);

    optProb = logLike(x,x,noiseType)-pen

    if noiseType=='Poisson':
        mergeProb = np.zeros(x.shape) # Might be x.shape
    else:
        mergeProb = x.copy()
        mergeProb[mergeProb<mu]=mu
        mergeProb = -1*mergeProb**2

    for iL in range(1,L+2-1): ## -1 to correct a bug in the code
        dyadLen = 2**iL
        ## Calculate scaling coefficient
        xScaling = 1/4.*(xScalingPrev +
                         np.roll(xScalingPrev, -dyadLen/2, 1) +
                         np.roll(xScalingPrev, -dyadLen/2, 0) +
                         np.roll(np.roll(xScalingPrev, -dyadLen/2, 0), -dyadLen/2, 1))

        ## Log probability of merging
        mergeSum = mergeProb + \
                   np.roll(mergeProb, -dyadLen/2, 1) + \
                   np.roll(mergeProb, -dyadLen/2, 0) + \
                   np.roll(np.roll(mergeProb, -dyadLen/2, 0), -dyadLen/2, 1)
        if noiseType == 'Poisson':
            pMerge = mergeSum - (xScaling*dyadLen*dyadLen *  (1-np.log(xScaling*dyadLen*dyadLen+realmin)-nplog(0.25))) * (xScaling>0) - pen
        else:
            pMerge = xScaling
            pMerge[pMerge<mu] = mu
            pMerge = pMerge**2 *dyadLen*dyadLen - pen + mergeSum
        
        ## Log probability of splitting
        pSplit = optProb + \
                   np.roll(optProb, -dyadLen/2, 1) + \
                   np.roll(optProb, -dyadLen/2, 0) + \
                   np.roll(np.roll(optProb, -dyadLen/2, 0), -dyadLen/2, 1)
        ## terms of merge log probability needed to calculated pMerge at next scale
        if noiseType == 'Poisson':
            mergeProb = mergeSum + xScaling*dyadLen*dyadLen*np.log(0.25)
        else:
            mergeProb = mergeSum

        ## Wavelets coefficients
        print(iL, wavelet_n.shape)
        wavelet_n[:,:,iL-1] = (xScalingPrev + np.roll(xScalingPrev,-dyadLen/2, 0))/2.-xScaling;
        wavelet_m[:,:,iL-1] = (xScalingPrev + np.roll(xScalingPrev,-dyadLen/2, 1))/2.-xScaling;
        wavelet_k[:,:,iL-1] = (xScalingPrev + np.roll(np.roll(xScalingPrev,-dyadLen/2, 0),
                                                    -dyadLen/2, 1))/2-xScaling
        
        ## decide whether to split or merge, save decision and associated log probability
        splitDecision[:,:,iL-1] = (pSplit > pMerge)*1
        optProb = pSplit * splitDecision[:,:,iL-1] + pMerge * (1-splitDecision[:,:,iL-1])
        xScalingPrev = xScaling

    ## initial estimate is coarse scale scaling coefficients 
    y = xScaling
    waveletScale = np.ones((M,N));
    waveletScaleNext = np.ones((M,N));

    for iL in range(L+1-1, 1-1, -1): ## -1 twice because of a bug
        dyadLen = 2**iL

        ## if the split decision is zero, then the associated wavelet should be set to zero.
        waveletScale = waveletScale * splitDecision[:,:,iL-1]

        if iL > 1:
            waveletScaleNext = waveletScaleNext - (0.25*(1.0-waveletScale))
            waveletScaleNext = np.roll(
                np.roll(waveletScaleNext, -dyadLen/2, 1) - (0.25*(1.0-waveletScale)),
                dyadLen/2, 1)
            waveletScaleNext = np.roll(
                np.roll(waveletScaleNext,-dyadLen/2,0) - (0.25*(1.0-waveletScale)),dyadLen/2,0)
            waveletScaleNext = np.roll(np.roll(
                np.roll(np.roll(waveletScaleNext,-dyadLen/2, 0) -dyadLen/2, 1)
                - (0.25*(1.0-waveletScale)), dyadLen/2, 0), dyadLen/2, 1)
        ## construct estimate based on wavelet coefficients and thresholds
        xD1 = (wavelet_n[:,:,iL-1] + wavelet_m[:,:,iL-1] + wavelet_k[:,:,iL-1]) * waveletScale
        xD2 = np.roll(
            (wavelet_n[:,:,iL-1] - wavelet_m[:,:,iL-1] +  wavelet_k[:,:,iL-1])*waveletScale,
            dyadLen/2, 1)
        xD3 = np.roll(
            (-wavelet_n[:,:,iL-1] + wavelet_m[:,:,iL-1] + wavelet_k[:,:,iL-1])*waveletScale,
            dyadLen/2, 0)
        xD4 = np.roll(np.roll(
            (-wavelet_n[:,:,iL-1] - wavelet_m[:,:,iL-1] + wavelet_k[:,:,iL-1])*waveletScale,
            dyadLen/2, 0), dyadLen/2, 1)

        xScaling = (y+xD1 +
                    np.roll(y,dyadLen/2,1)-xD2 +
                    np.roll(y,dyadLen/2,0)-xD3 +
                    np.roll(np.roll(y,dyadLen/2, 0), dyadLen/2, 1)+xD4)/4

        y = xScaling;
        waveletScale = waveletScaleNext;
        waveletScaleNext = np.ones((M,N));
        
    y[y<mu]=mu
    return y

def haarTVApprox2DNN_recentered(x,pen,mu=None):
    ## ==== Various variables
    realmin =  np.finfo(np.float64).min
    noiseType = 'Gaussian' # 'Poisson'

    if noiseType == 'Poisson':
        if np.any(np.isnan(x)) or np.any(x<0):
            raise TypeError('Invalid Poisson counts; check to make sure intensity non-negative.')

    (M,N) = x.shape
    L = int(np.log2(min(M,N))) ## Added +1 to debug

    xScalingPrev = x
    y = x.copy()
    y[x<mu] = mu

    optProb = logLike(x,y,noiseType)-pen

    if noiseType=='Poisson':
        mergeProb = np.zeros(x.shape) # Might be x.shape
    else:
        mergeProb = x.copy()
        mergeProb[mergeProb<mu]=mu
        mergeProb = -1*mergeProb**2

    for iL in range(1,L+2):
        dyadLen = 2**iL
        ## Calculate scaling coefficient
        xScaling = 1/4.*(xScalingPrev +
                         np.roll(xScalingPrev, -1, 1) +
                         np.roll(xScalingPrev, -1, 0) +
                         np.roll(np.roll(xScalingPrev, -1, 0), -1, 1))

        ## Log probability of merging
        mergeSum = mergeProb + \
                   np.roll(mergeProb, -dyadLen/2, 1) + \
                   np.roll(mergeProb, -dyadLen/2, 0) + \
                   np.roll(np.roll(mergeProb, -dyadLen/2, 0), -dyadLen/2, 1)
        if noiseType == 'Poisson':
            pMerge = mergeSum - (xScaling*dyadLen*dyadLen *  (1-np.log(xScaling*dyadLen*dyadLen+realmin)-nplog(0.25))) * (xScaling>0) - pen
        else:
            pMerge = xScaling
            pMerge[pMerge<mu] = mu
            pMerge = pMerge**2 *dyadLen*dyadLen - pen + mergeSum
            
        ## Log probability of merging
        mergeSum = dnsamp2(mergeProb + 
                   np.roll(mergeProb, -1, 1) + \
                   np.roll(mergeProb, -1, 0) + \
                   np.roll(np.roll(mergeProb, -1, 0), -1, 1))
        if noiseType == 'Poisson':
            pMerge = mergeSum - (xScaling*dyadLen*dyadLen *  (1-np.log(xScaling*dyadLen*dyadLen+realmin)-nplog(0.25))) * (xScaling>0) - pen
        else:
            pMerge = xScaling
            pMerge[pMerge<mu] = mu
            pMerge = pMerge**2 *dyadLen*dyadLen - pen + mergeSum
                           
        ## log probability of splitting
        pSplit = dnsamp2(optProb + \
                   np.roll(optProb, -1, 1) + \
                   np.roll(optProb, -1, 0) + \
                   np.roll(np.roll(optProb, -1, 0), -1, 1))

        ## terms of merge log probability needed to calculated pMerge at next scale
        if noiseType == 'Poisson':
            mergeProb = mergeSum + xScaling*dyadLen*dyadLen*np.log(0.25)
        else:
            mergeProb = mergeSum

        ## decide whether to split or merge, save decision and associated log probability
        splitDecision = (pSplit > pMerge)*1
        optProb = pSplit * splitDecision + pMerge * (1-splitDecision)

        sDiL = np.kron(splitDecision,np.ones(dyadLen))
        margeEst = np.kron(xScaling,np.ones(dyadLen))
        margEst[margeEst<mu]=mu

        y = y*sDiL + mergeEst*(1-sDiL);
        xScalingPrev = xScaling;

    return y
