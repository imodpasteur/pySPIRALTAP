#-*-coding:utf-8-*-
# Tools to work with compressed sensing libraries. This file includes simple and useful
#+functions in order to get compressed sensing algorithms to work.

# TODO
# - Create a Sphinx or readthedocs documentation
# - Get something that works.

# ==== Importations
from __future__ import print_function
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import sys

import matplotlib.gridspec as gridspec # subplots with different sizes
from matplotlib.ticker import NullFormatter # matrices with margins

from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
#from pySPIRALTAP import pySPIRALTAP
import pySPIRALTAP
import pyCSalgos

# ==== Measure & reconstruct
def measure(data, basis, gaussian=0, poisson=0):
    """Function computes the dot product <x,phi>
    for a given measurement basis phi
    
    Args:
    - data (n-size, numpy 1D array): the initial, uncompressed data
    - basis (nxm numpy 2D array): the measurement basis
    
    Returns:
    - A m-sized numpy 1D array to the dot product"""
    data = np.float_(data)
    
    if gaussian!=0 or poisson!=0: # Create the original matrix
        data = np.repeat([data], basis.shape[0], 0)
        #print data.shape

    if gaussian!=0: # Bruit
        data +=np.random.normal(scale=gaussian, size=data.shape)
    if poisson != 0:
        data = np.float_(np.random.poisson(np.abs(data)))
        
    if gaussian!=0 or poisson!=0:
        return np.diag((data).dot(basis.transpose()))
    else:
        return (data).dot(basis.transpose())

def reconstruct_1Dmagic(measure, basis):
    """Reconstruction with the L1 minimization provided by L1-magic"""
    x0 = basis.transpose().dot(measure)
    rec_l1 = pyCSalgos.l1min.l1eq_pd(x0, basis, [], measure, 1e-3);
    
    return rec_l1

def reconstruct_1Dlasso(measure, basis):
    """Reconstruction with L1 (Lasso) penalization
    the best value of alpha was determined using cross validation
    with LassoCV"""
    rgr_lasso = Lasso(alpha=4e-3)
    rgr_lasso.fit(basis, measure)
    rec_l1 = rgr_lasso.coef_#.reshape(l, l)
    
    return rec_l1

def reconstruct_1Dspiral(measure, basis):
    """Reconstruct using pySPIRALTAP and default parameters"""

    ## ==== Parameters
    tau   = 1e-6
    maxiter = 100
    tolerance = 1e-8
    verbose = 0 # To update

    ## ==== Create function handles
    y=measure.reshape((measure.size,1))
    AT = lambda x: basis.transpose().dot(x)
    A = lambda x: basis.dot(x)
    finit = y.sum()*AT(y).size/AT(y).sum()/AT(np.ones_like(y)).sum() * AT(y)

    #print ("WARNING: `finit` is not properly implemented", file=sys.stderr) 
    rec_l1 = pySPIRALTAP.SPIRALTAP(y,A,tau,
                                   AT=AT,
                                   maxiter=maxiter,
                                   miniter=5,
                                   penalty='canonical',
                                   noisetype='gaussian',
                                   #initialization=finit,
                                   stopcriterion=3,
                                   tolerance=tolerance,
                                   alphainit=1,
                                   alphamin=1e-30,
                                   alphamax=1e30,
                                   alphaaccept=1e30,
                                   logepsilon=1e-10,
                                   verbose=verbose)[0]
    #print ('ok')
    return rec_l1

# ==== Data generation algorithms
def add_noise(S, nmes, gaussian=0):
    """Applies gaussian and Poisson noise to a measured matrix
    
    Args:
    - S (numpy a rray): the input original signal
    - nmes (int): number of noisy measurements to perform
    - gaussian (float): intensity of the additive gaussian noise

    Returns:
    - Sn (list of numpy arrays): a list of noisy copies of the image
    """
    Sn = []
    for i in range(nmes):
        if gaussian > 0:
            Sn.append(np.abs(np.random.poisson(S))+np.abs(np.random.normal(0,gaussian,S.shape)))
        else:
            Sn.append(np.abs(np.random.poisson(S)))
    return Sn

def generate_1D(n, sparsity, noise=0):
    """This function generates a random vector with a 
    proportion of 1 is given by the sparsity parameter 
    and where an *additive gaussian noise* is added.
    
    Args:
     - n (int): size of the numpy array to generate
     - sparsity (float): fraction of zero-components of the vector (floored to the nearest possible value)
     - noise (float): standard deviation of a gaussian noise
     
     Returns:
     - a nx1 numpy array"""
    
    n_sig = int(math.floor((1-sparsity)*n))
    sd = np.random.random_integers(0,n-1,(n_sig))# signal
    ss = np.zeros((n))
    ss[sd] = 1
    if noise!=0: # Bruit
        sn=np.random.normal(scale=noise, size=(n))
    else:
        sn=np.zeros((n))
    
    return ss+sn

def gaussian_2D(n, mean, sigma):
    """A multivariate gaussian on a n-shaped array"""
    var = scipy.stats.multivariate_normal(mean=mean, cov=[[sigma,0],[0,sigma]])
    xv = []
    yv = []
    for i in range(n[0]):
        for j in range(n[1]):
            xv.append(i)
            yv.append(j)
    return var.pdf(zip(xv, yv)).reshape(n)

def generate_2D(n=(512, 200), gl=[(.4, 100, 1),(10, 10, 10)], I=int(500*(2*3.14)**.5),
                verbose=False):
    """Generates a 2D image with given number of spots of given variance

    Args:
    - n (2-tuple): size (x,z) of the numpy array to generate
    - gl (list of 3-uple): list of spots (variance, number of spots, rescaling intensity)
    - I (int): the global intensity of the signal

    Returns:
    - S (n-shaped array): array with the given spots and variance placed randomly"""

    S = np.zeros(n)
    gls = []
    for i in gl:
        if verbose:
            print ("signal of variance: ", i[0])
        for j in range(i[1]):
            mean = (np.random.randint(0, n[0]), np.random.randint(0, n[1]))
            S += i[2]*gaussian_2D(n, mean, i[0])
    S=np.int_(S*I)
    return S

# ==== Basis generation algorithms
def generate_bernoulli_basis(n,m, alpha):
    """Function generates a basis for random sampling of the 1D 
    data vector.
    
    Args:
    - n (int): size of the data
    - m (int): number of measurements
    - alpha (float): fraction of planes to be measured, parameter of the Bernoulli law"""
    
    return np.random.binomial(1, alpha, size=(m,n))

def generate_normal_basis(n,m, sigma=1):
    """Function generates a basis for random sampling of the 1D 
    data vector.
    
    Args:
    - n (int): size of the data
    - m (int): number of measurements
    - sigma (float): variance of the normal law"""
    
    return np.random.normal(1, sigma, size=(m,n))


def generate_fourier_basis(n,m, positive=True, sample=True):
    """Function generates a basis of cosines
    
    Args:
    - n (int): size of the data
    - m (int): number of measurements
    - positive (bool): tells if we should make sure that the all basis is positive (between 0 and 1)
    
    Returns:
    - a (mxn) matrix.
    """
    out = np.zeros((n,n))
    ran = np.array(range(n))
    out[0,:]=1 #0
    i=0
    while 2*i+1 < n:
        out[2*i+1,:]=np.sin(2*np.pi*ran*(i+1)/(n-1))
        if 2*i+2 >= n:
            break
        out[2*i+2,:]=np.cos(2*np.pi*ran*(i+1)/(n-1))
        i+=1
    if sample:
        out = out[np.random.choice(range(n), m, replace=False),:]
    else:
        out = out[0:(m-1),:]
    return (out+1)/2. ## Here we can create photons

# ==== Plotting functions
def plot_1D(data, sparsity="NA", noise="NA", measured=False):
    """Function plots the output from `generate_1D`. No parameters so far except the data
    
    Args:
    - data (numpy 1D array): output of `generate_1D`.
    
    Returns: None"""
    #plt.figure(figsize=(16, 4))
    plt.plot(data)
    plt.xlim((0,data.shape[0]))
    plt.title("Random data sample \n(sparsity: {}, noise: {})".format(sparsity, noise))
    plt.ylabel("Intensity (normalized)")
    if not measured:
        plt.xlabel("z (slice number)")
    else:
        plt.xlabel("measurement on the basis")

def plot_basis_old(basis, alpha="NA"):
    """Function plots the output from `generate_random_basis`. No parameters so far except the basis itself
    
    Args:
    - data (numpy 1D array): output of `generate_random_basis`.
    - alpha (float): for legend only.
    
    Returns: None"""
    
    #plt.figure(figsize=(16, 4))
    gs = gridspec.GridSpec(1, 2,width_ratios=[5,1])
    plt.subplot(gs[0])
    
    #plt.subplot(121)
    plt.imshow(basis, cmap=plt.cm.gray, interpolation='none')
    plt.title("Random basis \n(alpha: {})".format(alpha))
    plt.ylabel("Measurement index")
    plt.xlabel("z (slice number)")

    plt.subplot(gs[1])
    b_s = basis.shape
    sx=basis.sum(axis=1)
    plt.plot(sx,range(b_s[0]))
    plt.xlim((0,b_s[1]))
    plt.title("Number of illuminated planes")
    plt.xlabel("number of illuminated planes")
    
def plot_basis(basis):
    """Function plots the output from `generate_random_basis`. No parameters so far except the basis itself
    
    Args:
    - data (numpy 1D array): output of `generate_random_basis`.
    
    Returns: None"""

    nullfmt = NullFormatter()

    # definitions for the axes
    left, width = 0.12, 0.60
    bottom, height = 0.08, 0.60
    bottom_h =  0.16 + width 
    left_h = left - .63 
    rect_plot = [left_h, bottom, width, height]
    rect_x = [left_h, bottom_h, width, 0.2]
    rect_y = [left, bottom, 0.2, height]

    # start with a rectangular Figure
    plt.figure(2, figsize=(16, 5))

    axplot = plt.axes(rect_plot)
    axy = plt.axes(rect_y)

    # Plot the matrix
    axplot.pcolor(basis,cmap=plt.cm.gray)
    plt.ylabel("measurement")

    axplot.set_xlim((0, basis.shape[1]))
    axplot.set_ylim((0, basis.shape[0]))

    axplot.tick_params(axis='both', which='major', labelsize=10)

    # Plot time serie vertical
    axy.plot(basis.sum(axis=1),range(basis.shape[0]),color='k')
    axy.set_xlim((0,basis.shape[1]))
    axy.tick_params(axis='both', which='major', labelsize=10)
    plt.xlabel("number of illuminated planes")

def compare_1D(data, recon):
    """Function compares the original data and its reconstruction through Lasso penalization (L1)
    
    Args:
    - data (numpy 1D array): original data.
    - recon (numpy 1D array): reconstructed data (same length as data)
    
    Returns:
    - mean error between `data` and `recon`"""
    #plt.figure(figsize=(16, 8))
    plt.subplot(311)
    plt.plot(recon)
    plt.title("Reconstruction")
    
    plt.subplot(312)
    plt.plot(data)
    plt.title("Data")

    plt.subplot(313)
    plt.plot(np.abs(data-recon))
    plt.title("Absolute rror")
    plt.ylim((data.min(),data.max()))

    return np.abs(recon-data).mean()
