# -*- coding: utf-8 -*-

def in_segregation(x0, R, n, N=None):
    """
    return the actual indium concentration
    in th nth layer

    Params
    ------
    x0 : float
        the indium concentration between 0 and 1
    R : float
        the segregation coefficient
    n : int
        the current layer
    N : int
        number of layers in the well
    """
    if N:
        return x0*(1-R**N)*R**(n-N)
    return x0*(1-R**n)
