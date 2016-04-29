pySPIRAL-TAP, a Python version of SPIRAL-TAP
--------------------------------------------

# Inspiration

Everything comes from the code by [Zachary Harmany](http://drz.ac) (harmany@wisc.edu).

Original source: http://drz.ac/code/spiraltap/. The algorithm is described in this paper:
> Z. T. Harmany, R. F. Marcia, and R. M. Willett, "This is SPIRAL-TAP: Sparse Poisson Intensity Reconstruction ALgorithms – Theory and Practice," IEEE Transactions on Image Processing, vol. 21, pp. 1084–1096, Mar. 2012.

# Disclaimer

1. *License*: I am unsure about the license of the code
2. *Code*: still experimental, many methods have not been fully tested.

# Install
Install can be performed using the following `shell` commands:

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
The pySPIRALTAP methods can be imported with `from pySPIRALTAP import pySPIRALTAP`.

# Status
The methods relying on the `rwt` method have not been implemented.
