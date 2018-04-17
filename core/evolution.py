# -*- coding: utf-8 -*-

# python standard
import time

# python extended
import numpy as np
from scipy.integrate import simps
from scipy.fftpack import fft, ifft, fftfreq
from scipy.spatial.distance import cdist
from scipy.sparse import diags
from scipy.linalg import inv

# project
from .kickstart import orthonormal

def eigenvalue_simple(z, v, psi, m):
    """
    Calculates the eigenvalues of an eigenstate
    
    Params
    ------
    z : list
        the grid points in atomic units
    v : list
        the potential in atomic units
    psi : list
        the wave function
    m : float
        the effective electron/hole mass
    """
    dz = z[1]-z[0]

    # second derivative
    der2 = (psi[:-2]-2*psi[1:-1]+psi[2:])/dz**2

    # <Psi|H|Psi>
    psi = psi[1:-1]
    psic = psi.conj()
    p_h_p = simps(psic*(-0.5*der2/m+v[1:-1]*psi), z[1:-1])

    # divide by <Psi|Psi> hope it is not necessary
    p_h_p /= simps(psic*psi, z[1:-1])

    return p_h_p.real

def eigenvalue_pdm(z, v, psi, m):
    """
    Calculates the eigenvalues of a eigenstate
    using position dependent mass
    
    Params
    ------
    z : list
        the grid points in atomic units
    v : list
        the potential in atomic units
    psi : list
        the wave function
    m : list
        the effective electron/hole mass in each point
    """
    N = len(z)
    dz = z[1]-z[0]
    h_psi = np.zeros(len(z)-4, dtype=np.complex_)

    h0 = -(0.125/dz**2)
    for j in range(2, N-2):
        h1 = (psi[j+2]-psi[j])/m[j+1]
        h2 = (psi[j]-psi[j-2])/m[j-1]
        h3 = v[j]*psi[j]
        h_psi[j-2] = h0 * (h1-h2) + h3
    psi = psi[2:-2]

    # <Psi|H|Psi>
    p_h_p = simps(psi.conj()*h_psi, z[2:-2])
    p_h_p /= simps(psi.conj()*psi, z[2:-2])

    return p_h_p.real

def imaginary(z, v, m_eff, nmax=1, dt=0.04, precision=1e-5, method='pe'):
    """
    This function generates the first `nmax` eigenvalues and
    eigenvectors of the potential specified by `z` and `v`
    under the effective mass `m_eff`
    
    Params
    ------
    z : list
        the grid points in atomic units
    v : list
        the potential in atomic units
    m_eff : list or float
        the effective electron/hole mass, it might
        be a float for the whole system or a list
        for each point
    nmax : int
        number of states to calculate
    dt : float
        the time step in atomic units
    precision : float
        the precision in percentage
    method : string
        it is the method to use:
        - 'pe' stands for pseudo-spectral
        - 'cn' stands for crank-nicolson
    """

    N = len(z)
    dz = z[1]-z[0]
    k = fftfreq(N, d=dz)
    dt *= -1.0j
    m = m_eff

    if method == 'pe':
        # split step
        exp_v2 = np.exp(- 0.5j * v * dt)
        exp_t = np.exp(- 0.5j * (2 * np.pi * k) ** 2 * dt / m)
        evolution_operator = lambda p: exp_v2*ifft(exp_t*fft(exp_v2*p))
        
        if not (type(m) is list or type(m) is np.ndarray):
            eigenvalue = eigenvalue_simple
        else:
            eigenvalue = eigenvalue_pdm

    elif method == 'cn':
        # crank nicolson
        if not (type(m) is list or type(m) is np.ndarray):
            m = np.ones(N) * m
            
        up_diag = np.zeros(N-2, dtype=np.complex_)
        down_diag = np.zeros(N-2, dtype=np.complex_)
        main_diag_b = main_diag_c = np.zeros(N, dtype=np.complex_)

        # set constants
        a = 1.0j*dt/(16.0*dz**2)
        b = -1.0j*dt/2.0

        # build up and down diagonals
        for i in range(N - 2):
            up_diag[i] = a/m[i+1]
            down_diag[i] = a/m[i+1]

        # build main diagonals
        for i in range(N):
            if i == 0:
                main_diag_c[i] = 1.0+(-a)*(1.0/m[i+1])+b*v[i]
                main_diag_b[i] = 1.0-(-a)*(1.0/m[i+1])-b*v[i]
            elif i == N-1:
                main_diag_c[i] = 1.0+(-a)*(1.0/m[i-1])+b*v[i]
                main_diag_b[i] = 1.0-(-a)*(1.0/m[i-1])-b*v[i]
            else:
                main_diag_c[i] = 1.0+(-a)*(1.0/m[i+1]+1.0/m[i-1])+b*v[i]
                main_diag_b[i] = 1.0-(-a)*(1.0/m[i+1]+1.0/m[i-1])-b*v[i]

        diagonals_c = [main_diag_c, down_diag , up_diag]
        diagonals_b = [main_diag_b, -down_diag, -up_diag]
        C = diags(diagonals_c, [0, -2, 2]).toarray()
        B = diags(diagonals_b, [0, -2, 2]).toarray()
        invB = inv(B)
        D = invB * C
        evolution_operator = lambda p: D.dot(p)
        eigenvalue = eigenvalue_pdm

    # kick start eigenstates
    eigenstates = orthonormal(nmax, size=N)
    eigenvalues = np.zeros(nmax)

    counters = np.zeros(nmax)
    timers = np.zeros(nmax)
    precisions = np.zeros(nmax)
    vectors_sqeuclidean = np.zeros(nmax)

    for s in range(nmax):
        last_ev = 1.0
        last_es = np.zeros(N, dtype=np.complex_)
        while True:
            start_time = time.time()
            eigenstates[s] = evolution_operator(eigenstates[s])
            counters[s] += 1

            # gram-shimdt
            for j in range(s):
                proj = simps(eigenstates[s] * \
                    np.conjugate(eigenstates[j]), z)
                eigenstates[s] -= proj * eigenstates[j]

            # normalize
            A = np.sqrt(simps(np.abs(eigenstates[s])**2, z))
            eigenstates[s] /= A
            timers[s] += time.time() - start_time

            if counters[s] % 1000 == 0:
                eigenvalues[s] = eigenvalue(z, v, eigenstates[s], m)

                # check precision
                precisions[s] = np.abs(1-eigenvalues[s]/last_ev)
                last_ev = eigenvalues[s]

                if precisions[s] < precision:
                    XA = [np.abs(eigenstates[s])**2]
                    XB = [np.abs(last_es)**2]
                    vectors_sqeuclidean[s] = cdist(XA, XB, 'sqeuclidean')[0][0]
                    break

                last_es = np.copy(eigenstates[s])

    return {
        'eigenvalues': eigenvalues,
        'eigenstates': eigenstates,
        'counters': counters,
        'timers': timers,
        'precisions': precisions,
        'squared_euclidean_dist': vectors_sqeuclidean
    }

