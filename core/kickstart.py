# -*- coding: utf-8 -*-

# python extended
import numpy as np
from scipy.signal import gaussian
from scipy.special import legendre

def orthonormal(n=2, size=1024):
    """
    Return the first `n` orthonormal legendre
    polynoms weighted by a gaussian
    They are useful as kickstart arrays for a
    imaginary time evolution

    Params
    ------
    n : int
        The number of arrays
    size : int
        The number of points in each array
    """
    sg = np.linspace(-1, 1, size) # short grid
    g = gaussian(size, std=int(size/100)) # gaussian
    vls = [g*legendre(i)(sg) for i in range(n)]
    return np.array(vls, dtype=np.complex_)
