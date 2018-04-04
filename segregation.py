#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import scipy.constants as cte
from scipy.sparse import diags
from scipy.linalg import inv

from band_structure_database import Alloy, Database

class InGaAsQW(object):
    """
    """
    def __init__(self, well_length=15.0, R=0.85, x=0.16):
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
        self.N = 2048
        self.T = 1.4
        self.dt = 1e-19
        
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
        self.au2ev = self.au_e / ev

        self._build_grid()

    def _build_grid(self):
        bulk_len = 300.0
        bulk = Database(Alloy.GaAs)
        bulk_lattice = bulk.parameters('a0')
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
            db = Database(Alloy.InGaAs, 1.0-x)
            a0 = db.parameters('a0')
            well_actual_len += a0
            self.system_layers_x.append(x)
            self.system_layers_a.append(a0)
            cur_layer += 1
        
        # find concentration outside the well
        barrier_len = 0.0
        well_N = cur_layer-1
        while barrier_len < bulk_len:
            x = self._layer_x(cur_layer, N=well_N)
            db = Database(Alloy.InGaAs, 1.0-x)
            a0 = db.parameters('a0')
            barrier_len += a0
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
                position += layer_a
                if z < position:
                    return layer_x
        self.x_z = np.vectorize(indium_x)(self.z_ang)

        # now we have a z grid and the indium concentration on each
        # point
        gaas = Database(Alloy.GaAs)
        gaas_a = gaas.parameters('a0')
        def gap(x):
            db = Database(Alloy.InGaAs, 1.0-x)
            lattice = db.parameters('a0')
            a = db.deformation_potentials('a')
            b = db.deformation_potentials('b')
            c11 = db.deformation_potentials('c11')
            c12 = db.deformation_potentials('c12')
            epp = (gaas_a-lattice) / lattice
            gap_0 = db.parameters('eg_0')
            gap_300 = db.parameters('eg_300')
            gap = gap_0 + (gap_300-gap_0)/300.0
            dehh = (2.0*a*(c11-c12)/c11-b*(c11+c12)/c11)*epp
            return gap + dehh, db.effective_masses('m_e'), \
                db.effective_masses('m_hh')
    
        self.gap, self.me, self.mhh = \
            tuple(zip(*np.vectorize(gap)(self.x_z)))
        self.gap_c = 0.7 * np.vectorize(gap)(self.x_z)
        self.gap_v = 0.3 * np.vectorize(gap)(self.x_z)

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
    
    def _crank_nicolson(self, n, imag=True):

        dz = self.z_au[1]-self.z_au[0]
        dt = (-1.0j if imag else 1.0) * self.dt_au
        
        m = np.copy(self.me)
        v = np.copy(self.gap_c_au)

        up_diag = np.zeros(self.N-2, dtype=np.complex_)
        down_diag = np.zeros(self.N-2, dtype=np.complex_)
        main_diag = np.zeros(self.N, dtype=np.complex_)
        
        a = -1.0j*dt/(16.0*dx**2)
        b = 1.0j * dt / 2.0

        for i in range(self.N - 2):
            up_diag[i] = a/m[i+1]
            down_diag[i] = a/m[i-1]
        
        main_diag[i] = 1.0+a/m[i-1]+a/m[i+1]+b*v[i]

        diagonais_2 = [diagonal_3, diagonal_4, diagonal_4]
        C = diags(diagonais_2, [0, -1, 1]).toarray()

if __name__ == u'__main__':
    import matplotlib.pyplot as plt
    device = InGaAsQW(well_length=25.0, R=0.85, x=0.15)
    plt.scatter(device.z_ang, device.gap_c)
    plt.scatter(device.z_ang, device.gap_v)
    plt.show()