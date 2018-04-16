#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""

# python standard
import sys, os, time

# python extended
import numpy as np
import scipy.constants as cte
from scipy.integrate import simps
from scipy.sparse import diags
from scipy.linalg import inv
from scipy.signal import gaussian
from scipy.special import legendre
from scipy.spatial.distance import cdist

from scipy.fftpack import fft, ifft, fftfreq

# project
from core.evolution import imaginary

class InGaAsQW(object):
    """
    """
    def __init__(self, well_length=15.0, R=0.85, x=0.16, T=1.4):
        """class constructor

        Keyword Arguments:
            well_length {float} -- the well's intendend length in
                Angstrom(default: {15.0})
            R {float} -- the segregation coefficient (default: {0.85})
            x {float} -- indium concentration (default: {0.16})
        """
        self.well_length = well_length
        self.R = R
        self.x = x
        self.N = 512
        self.T = T
        self.dt = 5e-18
        self.precision = 1e-5

        # AU of interest
        self.au_l = cte.value('atomic unit of length')
        self.au_t = cte.value('atomic unit of time')
        self.au_e = cte.value('atomic unit of energy')

        # other units, relations, and constants of interest
        self.ev = cte.value('electron volt')
        self.c = cte.value('speed of light in vacuum')
        self.hbar_si = cte.value('Planck constant over 2 pi')
        self.me = cte.value('electron mass')
        self.au2ang = self.au_l / 1e-10
        self.au2ev = self.au_e / self.ev

        self._build_grid()

    def _build_grid(self):
        bulk_len = 300.0
        bulk_lattice = 5.65/2.0
        bulk_layers = np.ceil(bulk_len/bulk_lattice)
        bulk_len = bulk_layers * bulk_lattice

        # concentration of on each layer
        self.system_layers_x = [0.0] * int(bulk_layers)
        self.system_layers_a = [bulk_lattice] * int(bulk_layers)

        # find concetration inside the well
        well_actual_len = 0.0
        cur_layer = 1
        while well_actual_len < self.well_length:
            x = self._layer_x(cur_layer)
            a0 = 5.65*(1-x)+6.08*x
            well_actual_len += a0/2.0
            self.system_layers_x.append(x)
            self.system_layers_a.append(a0)
            cur_layer += 1

        # find concentration outside the well
        barrier_len = 0.0
        well_N = cur_layer-1
        while barrier_len < bulk_len:
            x = self._layer_x(cur_layer, N=well_N)
            a0 = 5.65*(1-x)+6.08*x
            barrier_len += a0/2.0
            self.system_layers_x.append(x)
            self.system_layers_a.append(a0)
            cur_layer += 1

        # that is how we know the whole size of the system
        self.system_length = L = bulk_len+well_actual_len+barrier_len

        # we know the system size, the number of layers, and the In
        # concentration on each layer, next step is to build z grid
        # and x(z)
        self.z_ang = np.linspace(0.0, L, self.N)

        def indium_x(z):
            position = 0.0
            for layer_x, layer_a in zip(self.system_layers_x,
                    self.system_layers_a):
                position += layer_a/2.0
                if z < position:
                    return layer_x
            return self.system_layers_x[-1]
        self.x_z = np.vectorize(indium_x)(self.z_ang)

        # now we have a z grid and the indium concentration on each
        # point
        gaas_a0 = 5.65
        def gap(x):
            a0 = 5.65*(1-x)+6.08*x
            a = -8.33*(1-x)-6.08*x
            b = -1.90*(1-x)-1.55*x
            c11 = 1.22*(1-x)+0.83*x
            c12 = 0.57*(1-x)+0.45*x
            me = 0.067*(1-0.426*x)
            mhh = 0.34*(1+0.117*x)
            epp = (gaas_a0-a0)/a0
            dehh = (2*a*(c11-c12)-b*(c11+2*c12))*epp/c11
            gap = 1.5192-1.5837*x+0.475*x**2
            return gap + dehh, me, mhh

        self.gap, self.me, self.mhh = np.vectorize(gap)(self.x_z)
        self.gap_c = 0.7 * self.gap
        self.gap_v = 0.3 * self.gap

        # transform to AU
        self.gap_au = self.gap / self.au2ev
        self.gap_c_au = self.gap_c / self.au2ev
        self.gap_v_au = self.gap_v / self.au2ev
        self.z_au = self.z_ang / self.au2ang
        self.dt_au = self.dt / self.au_t

    def _layer_x(self, n, N=0):
        if N > 0:
            return self.x*(1.0-self.R**N)*self.R**(n-N)
        return self.x*(1.0-self.R**n)

    def evolve_pe(self, nmax, hh=False):
        m = np.copy(self.mhh) if hh else np.copy(self.me)
        v = np.copy(self.gap_v_au) if hh else np.copy(self.gap_c_au)
        v = v - np.min(v)
        info = imaginary(z=self.z_au, v=v, m_eff=m, nmax=nmax, dt=self.dt_au, precision=self.precision)
        self.eigenvalues_ev = info['eigenvalues'] * self.au2ev
        self.eigenstates = info['eigenstates']

if __name__ == u'__main__':
    #import matplotlib.pyplot as plt
    #device = InGaAsQW(well_length=25.0, R=0.85, x=0.15)
    #plt.scatter(device.z_ang, device.gap_c)
    #plt.scatter(device.z_ang, device.gap_v)
    #plt.show()
    #sys.exit(0)
    l = 33.0
    R = 0.80
    x0 = 0.11
    #for R in np.linspace(0.7, 0.9, 9):
    #for l in [15, 27, 39, 51, 63]:
    for l in [33]:
        device = InGaAsQW(well_length=l, R=R, x=x0)
        device.evolve_pe(nmax=1)
        electron = device.eigenvalues_ev[0]
        device.evolve_pe(nmax=1, hh=True)
        heavy_hole = device.eigenvalues_ev[0]
        gap = np.min(device.gap)
        print("L=%f, R=%f, Ee=%.10f, Ehh=%.10f, Eg=%.10f, Epl=%.10f"%(l,R, electron, heavy_hole, gap, gap+electron+heavy_hole-0.007))




