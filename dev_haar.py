# Developping pySPIRALTAP's haar-related functions
# This small script is intended to check the byte-per-byte correspondence of the
#+haar-related functions compared to pySPIRALTAP.
#
# by MW, GPLv3+, Jun 2016

import pySPIRALTAP
import numpy as np

# ==== Parameters and loading stuff for haarTIApprox2DNN_recentered
x=np.loadtxt('save_x.mat', delimiter=',')
y=np.loadtxt('save_y.mat', delimiter=',')
pen = 0.6
mu = 0

pySPIRALTAP.haarApprox.haarTIApprox2DNN_recentered(x,pen,mu)
