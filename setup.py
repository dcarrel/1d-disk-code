from consts import *
import numpy as np
import json, sys

XH   = 0.7
XHe  = 0.3
mu   = XH*MP + 4*XHe*MP
MBH  = 1e6*MSUN




## write out solver



## electrons, protons, hydrogen
Xi = np.array([XH, XH, XHe])
mi = np.array([MP, ME, 4*MP])
Xi_over_mi = Xi/np.array([MP, MP, 4*MP])




