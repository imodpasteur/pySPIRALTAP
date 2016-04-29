#%matplotlib inline
import denoise_bound
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
np.random.seed(314)

X=np.asarray(Image.open('cameraman.pgm'))
X=X/255.
Bobs=X+2e-2*np.random.randn(*X.shape)

#denoise_bound.denoise_bound(Bobs,0.02,-np.inf,np.inf, {'print':1})
denoise_bound.denoise_bound(X,0.02,-np.inf,np.inf, {'print':1, 'tv':'l1'})

