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
        self.N = 256
        self.T = 1.4
        self.dt = 1e-18
        self.precision = 1e-4

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
        bulk = Database(Alloy.GaAs)
        bulk_lattice = bulk.parameters('a0')/2.0
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
            well_actual_len += a0/2.0
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
        gaas = Database(Alloy.GaAs)
        gaas_a = gaas.parameters('a0')
        def gap(x):
            lat = 5.65*(1-x)+6.08*x
            a = -8.33*(1-x)-6.08*x
            b = -1.9*(1-x)-1.55*x
            c11 = 1.22*(1-x)+0.83*x
            c12 = 0.57*(1-x)+0.45*x
            me = 0.067*(1-0.426*x)
            mhh = 0.34*(1+0.117*x)
            epp = (gaas_a-lat)/lat
            edhh = 2*a*(c11-c12)*epp/c11-b*(c11+2*c12)*epp/c11
            gap = 1.5192-1.5837*x+0.475*x**2

            return gap + edhh, me, mhh

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

    def _crank_nicolson(self, nmax=2, imag=True, hh=False):
        # renaming for facilitate use
        dz = self.z_au[1]-self.z_au[0]
        dt = (-1.0j if imag else 1.0) * self.dt_au

        m = np.copy(self.mhh) if hh else np.copy(self.me)
        v = np.copy(self.gap_v_au) if hh else np.copy(self.gap_c_au)
        v = v - np.min(v)

        # prepare diagonal arrays
        up_diag = np.zeros(self.N-2, dtype=np.complex_)
        down_diag = np.zeros(self.N-2, dtype=np.complex_)
        main_diag_b = main_diag_c = np.zeros(self.N, dtype=np.complex_)

        # set constants
        a = 1.0j*dt/(16.0*dz**2)
        b = -1.0j*dt/2.0

        # build up and down diagonals
        for i in range(self.N - 2):
            up_diag[i] = a/m[i+1]
            down_diag[i] = a/m[i+1]

        # build main diagonals
        for i in range(self.N):
            if i == 0:
                main_diag_c[i] = 1.0+(-a)*(1.0/m[i+1])+b*v[i]
                main_diag_b[i] = 1.0-(-a)*(1.0/m[i+1])-b*v[i]
            elif i == self.N-1:
                main_diag_c[i] = 1.0+(-a)*(1.0/m[i-1])+b*v[i]
                main_diag_b[i] = 1.0-(-a)*(1.0/m[i-1])-b*v[i]
            else:
                main_diag_c[i] = 1.0+(-a)*(1.0/m[i+1]+1.0/m[i-1])+b*v[i]
                main_diag_b[i] = 1.0-(-a)*(1.0/m[i+1]+1.0/m[i-1])-b*v[i]

        diagonals_c = [main_diag_c, down_diag , up_diag]
        diagonals_b = [main_diag_b, -down_diag, -up_diag]
        C = diags(diagonals_c, [0, -2, 2]).toarray()
        B = diags(diagonals_b, [0, -2, 2]).toarray()

        #import matplotlib.pyplot as plt
        #plt.matshow(np.abs(C))
        #plt.show()

        invB = inv(B)
        D = invB * C
        k = fftfreq(self.N, d=dz)
        exp_v2 = np.exp(- 0.5j * v * dt)
        exp_t = np.exp(- 0.5j * (2 * np.pi * k) ** 2 * dt / np.max(m))
        evolution_operator = lambda p: exp_v2*ifft(exp_t*fft(exp_v2*p))

        # kick start functions
        short_grid = np.linspace(-1, 1, self.N)
        g = gaussian(self.N, std=int(self.N/100))
        eigenstates = np.array([g*legendre(i)(short_grid) \
            for i in range(nmax)],dtype=np.complex_)
        eigenstates_last = np.ones(self.N, dtype=np.complex_)
        eigenvalues_ev = np.zeros(nmax)

        counters = np.zeros(nmax)
        timers = np.zeros(nmax)
        precisions = np.zeros(nmax)
        vectors_sqeuclidean = np.zeros(nmax)

        for s in range(nmax):
            eigenvalues_ev_last = 10.0

            while True:
                start_time = time.time()
                eigenstates[s] = D.dot(eigenstates[s])
                #eigenstates[s] = evolution_operator(eigenstates[s])
                counters[s] += 1

                # gram-shimdt
                for j in range(s):
                    proj = simps(eigenstates[s] * \
                        np.conjugate(eigenstates[j]), self.z_au)
                    eigenstates[s] -= proj * eigenstates[j]

                # normalize
                A = np.sqrt(simps(np.abs(eigenstates[s])**2, self.z_au))
                eigenstates[s] /= A
                timers[s] += time.time() - start_time

                if counters[s] % 1000 == 0:
                    # calculate eigenvalue
                    psi = np.copy(eigenstates[s])
                    h_psi = np.zeros(self.N-4, dtype=np.complex_)

                    h0 = -(0.125/dz**2)
                    for j in range(2,self.N-2):
                        h1 = (psi[j+2]-psi[j])/m[j+1]
                        h2 = (psi[j]-psi[j-2])/m[j-1]
                        h3 = v[j]*psi[j]
                        h_psi[j-2] = h0 * (h1-h2) + h3
                    psi = psi[2:-2]

                    # <Psi|H|Psi>
                    p_h_p = simps(psi.conj()*h_psi, self.z_au[2:-2])
                    p_h_p /= A**2

                    eigenvalues_ev[s] = p_h_p.real * self.au2ev # eV
                    print(eigenvalues_ev[s])

                    precisions[s] = np.abs(1.0-eigenvalues_ev[s] \
                        / eigenvalues_ev_last)
                    eigenvalues_ev_last = eigenvalues_ev[s]

                    if precisions[s] < self.precision:
                        XA = [eigenstates[s]]
                        XB = [eigenstates_last]
                        vectors_sqeuclidean[s] = \
                            cdist(XA, XB, 'sqeuclidean')[0][0]
                        break
                    else:
                        eigenstates_last = np.copy(eigenstates[s])

        self.eigenvalues_ev = eigenvalues_ev
        self.eigenstates = eigenstates

if __name__ == u'__main__':
    #import matplotlib.pyplot as plt
    #device = InGaAsQW(well_length=25.0, R=0.85, x=0.15)
    #plt.scatter(device.z_ang, device.gap_c)
    #plt.scatter(device.z_ang, device.gap_v)
    #plt.show()
    #sys.exit(0)
    #for R in np.linspace(0.7, 0.9, 5):
    R = 0.85
    for l in [15, 27, 39, 51, 63]:
        device = InGaAsQW(well_length=l, R=R, x=0.16)
        device._crank_nicolson(nmax=1)
        electron = device.eigenvalues_ev[0]
        device._crank_nicolson(nmax=1, hh=True)
        heavy_hole = device.eigenvalues_ev[0]
        gap_v = -device.gap_v
        gap_c = device.gap_c
        gap = gap_c-gap_v
        gap = np.min(gap)
        print("L=%f, R=%f, Ee=%.10f, Ehh=%.10f, Eg=%.10f, Epl=%.10f"%(l,R, electron, heavy_hole, gap, gap+electron+heavy_hole-0.007))




