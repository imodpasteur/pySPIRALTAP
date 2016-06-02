## Tests for SPIRALTAP
## Various tests for pySPIRALTAP inputs
## by MW, May 2016
##+ some code is copy-pasted from cstools.py

from __future__ import print_function
import sys
#sys.path.append('..')
import pySPIRALTAP
#from pySPIRALTAP import pySPIRALTAP
import cstools
import numpy as np

## ==== Test inputs

## ==== Test outputs on 1D stuff
def test_matrix_input(m=512, mes=55, n=100, dat=25, seed=0):
    """Test if SPIRAL accepts the right type of parameters. 
    Testing for matrices and dimensions conservation"""
    
    # === Generate data
    f = cstools.generate_1D(m, n/float(m)) # Signal
    A = cstools.generate_fourier_basis(m, mes)
    y = cstools.measure(f,A)
    
    rec_l1 = run_spiral(y,A,f)
    assert rec_l1.shape == f.shape
    rec_l1b = run_spiral(y.reshape((-1,1)),A,f)
    assert rec_l1b.shape == (f.shape[0], 1)
    assert np.all(rec_l1 == rec_l1b.reshape(-1))

def test_are_we_saveobjective_agnostic():
    """Test whether the reconstruction is the same unrespective to the 
    'saveobjective flag'
    """
    
    ## Generate data
    np.random.seed(42)
    A = np.genfromtxt('./demodata/fourierbasis.csv') # Measurement matrix
    f = cstools.generate_1D(512, .95) # Signal
    y = cstools.measure(f,A)

    rec_save = pySPIRALTAP.SPIRALTAP(y,A,1e-6,
                                     maxiter=100,
                                     miniter=5,
                                     penalty='canonical',
                                     noisetype='gaussian',
                                     stopcriterion=3,
                                     tolerance=1e-8,
                                     alphainit=1,
                                     alphamin=1e-30,
                                     alphamax=1e30,
                                     alphaaccept=1e30,
                                     logepsilon=1e-10,
                                     saveobjective=True,
                                     savereconerror=False,
                                     savesolutionpath=False,
                                     truth=f,
                                     verbose=0, savecputime=False)[0]

    rec_nosave = pySPIRALTAP.SPIRALTAP(y,A,1e-6,
                                       maxiter=100,
                                       miniter=5,
                                       penalty='canonical',
                                       noisetype='gaussian',
                                       stopcriterion=3,
                                       tolerance=1e-8,
                                       alphainit=1,
                                       alphamin=1e-30,
                                       alphamax=1e30,
                                       alphaaccept=1e30,
                                       logepsilon=1e-10,
                                       saveobjective=False,
                                       savereconerror=False,
                                       savesolutionpath=False,
                                       truth=f,
                                       verbose=0,savecputime=False)[0]

    er = ((rec_save- rec_nosave)**2).sum()/(rec_save**2).sum()
    print("L2 error between the two reconstructions : ", er)
    assert er == 0 

    
def test_canonical_reconstruction(m=512, mes=55, n=100, dat=25, seed=0):
    """Test if the reconstruction is accurate, norm is l2 norm
    using the 'canonical' penalty of pySPIRALTAP

    Inputs:
    - m (int) : size of the vector to generate
    - mes (int) : number of measurements
    - n (int) : number of replicates to run
    - dat (int) : number of non-zero components
    """
    ## Initialize 
    A = np.genfromtxt('./demodata/fourierbasis.csv') ## Measurement matrix
    np.random.seed(42) # Allow for reproducible tests
    refs = [0.845297049572287, 0.802530575360916, 0.774075772492764,
            0.802787086734257, 0.809810937491642, 0.810120589034238,
            0.840299111810711, 0.853267503485126, 0.844573574577239,
            0.766039296833392, 0.822626044063411, 0.816361180374730] # maSPIRALTAP outputs

    for er_ref in refs:
        ## Generate data
        f = cstools.generate_1D(m, 1-dat/float(m)) # Signal
        y = cstools.measure(f,A)

        rec_l1 = run_spiral(y,A,f, penalty='canonical')
        assert rec_l1.shape == f.shape
        er = ((rec_l1-f)**2).sum()/(f**2).sum()
        if (er-er_ref) > 1e-8:
            print(np.where(f!=0))
        assert (er- er_ref) < 1e-8 # Compute quadratic error

def test_tv_reconstruction(m=512, mes=55, n=100, dat=25, seed=0):
    """Test if the reconstruction is accurate, norm is l2 norm
    using the total variation 'tv' penalty of pySPIRALTAP

    Inputs:
    - m (int) : size of the vector to generate
    - mes (int) : number of measurements
    - n (int) : number of replicates to run
    - dat (int) : number of non-zero components
    """
    ## Initialize 
    A = np.genfromtxt('./demodata/fourierbasis.csv') ## Measurement matrix
    np.random.seed(42) # Allow for reproducible tests
    refs = [0.845297049572287, 0.802530575360916, 0.774869261903731,
            0.802775321025487, 0.809918447832432, 0.810027761833191,
            0.839509617696185, 0.853253330528812, 0.844586950649805,
            0.767217921052847, 0.822639434444858, 0.815667604706261] # maSPIRALTAP outputs

    for er_ref in refs:
        ## Generate data
        f = cstools.generate_1D(m, 1-dat/float(m)) # Signal
        y = cstools.measure(f,A)

        rec_l1 = run_spiral(y,A,f, penalty='tv')
        assert rec_l1.shape == f.shape
        er = ((rec_l1-f)**2).sum()/(f**2).sum()
        if (er-er_ref) > 1e-8:
            print(np.where(f!=0))
        assert (er- er_ref) < 1e-6 # Test quadratic error

def test_onb_reconstruction(m=512, mes=55, n=100, dat=25, seed=0):
    """Check if the algorithm can proceed with the `onb` penalty

    Inputs:
    - m (int) : size of the vector to generate
    - mes (int) : number of measurements
    - n (int) : number of replicates to run
    - dat (int) : number of non-zero components
    """
        
    ## Initialize 
    A = np.genfromtxt('./demodata/fourierbasis.csv') ## Measurement matrix
    np.random.seed(42) # Allow for reproducible tests
    refs = [0, 0, 0,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0] # maSPIRALTAP outputs

    for er_ref in refs:
        ## Generate data
        f = cstools.generate_1D(m, 1-dat/float(m)) # Signal
        y = cstools.measure(f,A)
        W = np.eye(A.shape[1])
        
        rec_l1 = run_spiral(y,A,f, penalty='onb', W=W)
        assert rec_l1.shape == f.shape
        er = ((rec_l1-f)**2).sum()/(f**2).sum()
        if (er-er_ref) > 1e-8:
            print(np.where(f!=0))
        assert (er- er_ref) < 1e-6 # Test quadratic error

    
def run_spiral(y,Ao,f,finit=None, penalty='canonical', W=[]):
    """Simple wrapper for SPIRAL function. All the parameters are preset."""
    ## Set regularization parameters and iteration limit:
    tau   = 1e-6
    maxiter = 100
    tolerance = 1e-8
    verbose = 0

    ## Setup function handles for computing A and A^T:
    AT = lambda x: Ao.transpose().dot(x)
    A = lambda x: Ao.dot(x)

    if finit==None: ## rescaled least-square to initialize finit
        finit = y.sum()*AT(y).size/AT(y).sum()/AT(np.ones_like(y)).sum() * AT(y)
    
    return pySPIRALTAP.SPIRALTAP(y,A,tau,
                                 AT=AT,
                                 W=W,
                                 maxiter=maxiter,
                                 miniter=5,
                                 penalty=penalty,
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
                                 savereconerror=False,
                                 savesolutionpath=False,
                                 truth=f,
                                 verbose=verbose, savecputime=False)[0]

if __name__=='__main__':
    #test_matrix_input()
    test_are_we_saveobjective_agnostic()
    #test_canonical_reconstruction()
    #test_tv_reconstruction()
