pySPIRAL-TAP, a Python version of SPIRAL-TAP
--------------------------------------------

# Inspiration

Everything comes from the code by [Zachary Harmany](http://drz.ac) (harmany@wisc.edu).

Original source: http://drz.ac/code/spiraltap/ or https://github.com/zharmany/SPIRAL-TAP (might be more up-to-date). The algorithm is described in this paper:
> Z. T. Harmany, R. F. Marcia, and R. M. Willett, "This is SPIRAL-TAP: Sparse Poisson Intensity Reconstruction ALgorithms – Theory and Practice," IEEE Transactions on Image Processing, vol. 21, pp. 1084–1096, Mar. 2012.

# Disclaimer

2. *Code*: still experimental.

# Installation from pip (recommended)

```{shell}
pip install pySPIRALTAP
```

# Installation from the git repository
## Requirements
`pySPIRALTAP` requires the following dependencies:
- `numpy`
- `rwt`, the Rice Wavelet Toolbox, available on this page: https://github.com/ricedsp/rwt
- `scipy.io` (only to run the demo)
- `pytest` (only to run the tests)

## Installing `numpy`
Installation using your package manager (Debian/Ubuntu):

```{shell}
sudo apt-get install python-numpy
```

Alternatively, if you have `pip` installed, you can install `numpy` by typing the following:

```{shell}
pip install --user numpy
```

## Installing `rwt`
You can install the [Rice Wavelet Toolbox](https://github.com/ricedsp/rwt) by typing the following. Before that, make sure that you have `cmake` installed (`sudo apt-get install cmake` if you use a Debian-derived distribution):

```{shell}
git clone https://github.com/ricedsp/rwt.git
cd rwt/python
cmake .
sudo make install
```

And then test if the installation succeeded by typing:

```{shell}
python -c "import rwt"
```

If this returns nothing, the installation worked. In case it returns an error such as: 

```{python}
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ImportError: No module named rwt
```

This means that the installation failed. It can be because the installation was performed using a different version of Python that is not the default version. So far, I do not have information about how to perform this installation.

More information about the installation can be found in the `INSTALL` file, or on the `rwt` [webpage](https://github.com/ricedsp/rwt).

## Installing `pySPIRALTAP`

Installation can be performed using the following `shell` commands:

```{shell}
git clone https://gitlab.com/padouppadoup/pySPIRAL-TAP.git
cd pySPIRAL-TAP
sudo python setup.py install
```

# Usage/Example

## Demo examples
A simple working example can be run at (in the main folder):

```{shell}
python SPIRALdemo.py
```

This file contains one demo where a 1D signal is reconstructed using a $l1$ penalty. This demo uses `matplotlib` for plotting output.

Alternatively, one can play with the [Jupyter notebook](http://jupyter.org): `SPIRALdemo.ipynb`, that feature the same demo as `SPIRALdemo.py`, but in a more fancy format.

## Calling from a script
The pySPIRALTAP methods can be imported with `import pySPIRALTAP`.

## `SPIRALTAP` function parameters

Here is a canonical function call with many parameters exposed:

```{python}
    resSPIRAL = pySPIRALTAP.SPIRALTAP(y,A,              # y: measured signal, A: projection matrix
		                              1e-6,             # regularization parameter
                                      maxiter=100,      # min. number of iterations
                                      miniter=5,        # max. number of iterations
                                      penalty='canonical', # type of penalty to apply
                                      noisetype='gaussian',# form of the log-likelihood penalty
                                      stopcriterion=3,  # index of the termination criterion
                                      tolerance=1e-8,
                                      alphainit=1,
                                      alphamin=1e-30,
                                      alphamax=1e30,
                                      alphaaccept=1e30,
                                      logepsilon=1e-10,
                                      saveobjective=True,
                                      savereconerror=True,
                                      savesolutionpath=False,
                                      verbose=verbose, savecputime=True)
```

# Status
The methods based on RDP (*recursive dyadic partitions*) have not been implemented yet. Additionnally, the code has not been fully tested, although we did our best to provide a working product.

# License
- This software is released under the MIT license. See the `LICENSE` file for more details.
- The `denoise_bound` code is released under the GNU GPLv2 license and was written by Copyright (2008): Amir Beck and Marc Teboulle.
