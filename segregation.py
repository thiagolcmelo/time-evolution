#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""

import numpy as np
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

    def _build_grid(self):
        bulk_len = 300.0
        bulk = Database(Alloy.GaAs)
        bulk_lattice = bulk.parameters('a0')
        bulk_layers = np.ceil(bulk_len/bulk_lattice)
        bulk_len = bulk_layers * bulk_lattice

        # concentration of on each layer
        self.system_layers_x = [0.0] * int(bulk_layers)

        # find concetration inside the well
        well_actual_length = 0.0
        cur_layer = 1
        while well_actual_length < self.well_length:
            x = self._layer_x(cur_layer)
            self.system_layers_x.append(x)
            db = Database(Alloy.InGaAs, 1.0-x)
            well_actual_length += db.parameters('a0')
            cur_layer += 1
        
        # find concentration outside the well
        barrier_length = 0.0
        well_N = cur_layer-1
        while barrier_length < bulk_len:
            x = self._layer_x(cur_layer, N=well_N)
            self.system_layers_x.append(x)
            db = Database(Alloy.InGaAs, 1.0-x)
            barrier_length += db.parameters('a0')
            cur_layer += 1

    def _layer_x(self, n, N=0):
        if N > 0:
            return self.x * (1.0 - self.R**N) * self.R ** (n-N)
        return self.x * (1.0 - self.R**n)

if __name__ == u'__main__':
    import matplotlib.pyplot as plt
    device = InGaAsQW(well_length=15)
    device._build_grid()

    plt.plot(range(len(device.system_layers_x)),device.system_layers_x)
    plt.show()