#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module provides an enum for better dealing with alloys names
and a database with alloys properties
"""

from enum import Enum

class Alloy(Enum):
    """
    An enum for labeling some alloys of interest
    
    Alloys under support currently are:
    - GaAs
    - AlAs
    - InAs
    - InP
    - GaP
    - AlGaAs
    - InGaAs
    - AlInAs
    """
    GaAs = 1
    AlAs = 2
    InAs = 3
    InP = 4
    GaP = 5
    AlGaAs = 6 # Al(x)Ga(1-x)As
    InGaAs = 7 # In(1-x)Ga(x)As
    AlInAs = 8 # Al(x)In(1-x)As

class Database:
    """
    This class provides parameters, deformation, and effective masses
    for some selected alloys, it is currently based on:

    Chuang, S. L. (1995). Physics of optoelectronic devices. New York: Wiley.
    Appendix K: tables K.2 and K.3
    """

    def __init__(self, alloy, concentration=1.0):
        """
        An instance of this class represents a single alloy, which must be
        informed in this constructor

        Parameters
        ----------
        alloy : Alloy
            the alloy which the instance is going to represent
        concentration : float
            some alloys have characteristic ratios between components, they
            are usually pointed as `x`

        Examples
        --------
        >>> from band_structure_database import Alloy, Database
        >>> algaas_03 = Database(Alloy.AlGaAs, 0.3)
        >>> print("Gap at 0K - %.2f eV" % algaas_03.parameters('eg_0'))
        Gap at 0K - 2.00 eV
        """
        self.alloy = alloy
        self.concentration = concentration

        # this is the default (currently the only one) source
        self.chuang_db = {
            Alloy.GaAs: {
                'parameters': {
                    'a0': 5.6533,
                    'eg_0': 1.519,
                    'eg_300': 1.424,
                    'eg_0_ind': None,
                    'eg_300_ind': None,
                    'delta': 0.34,
                    'ev_av': -6.92,
                    'optical_matrix': 25.7,
                    'parameter_ep': 25.0
                },
                'deformation_potentials': {
                    'a_c': -7.17,
                    'a_v': 1.16,
                    'a': -8.33,
                    'b': -1.7,
                    'd': -4.55,
                    'c11': 11.879,
                    'c12': 5.376,
                    'c44': 5.94
                },
                'effective_masses': {
                    'm_e': 0.067,
                    'm_hh': 0.5,
                    'm_lh': 0.087,
                    'm_hh_z': 0.333,
                    'm_lh_z': 0.094,
                    'gamma_1': 6.8,
                    'gamma_2': 1.9,
                    'gamma_3': 2.73,
                }
            },
            Alloy.AlAs: {
                'parameters': {
                    'a0': 5.66,
                    'eg_0': 3.13,
                    'eg_300': 3.03,
                    'eg_0_ind': 2.229,
                    'eg_300_ind': 2.168,
                    'delta': 0.28,
                    'ev_av': -7.49,
                    'optical_matrix': 21.1,
                    'parameter_ep': None
                },
                'deformation_potentials': {
                    'a_c': -5.64,
                    'a_v': 2.47,
                    'a': -8.11,
                    'b': -1.5,
                    'd': -3.4,
                    'c11': 12.5,
                    'c12': 5.34,
                    'c44': 5.42
                },
                'effective_masses': {
                    'm_e': 0.15,
                    'm_hh': 0.79,
                    'm_lh': 0.15,
                    'm_hh_z': 0.478,
                    'm_lh_z': 0.208,
                    'gamma_1': 3.45,
                    'gamma_2': 0.68,
                    'gamma_3': 1.29,
                }
            },
            Alloy.InAs: {
                'parameters': {
                    'a0': 6.0584,
                    'eg_0': 0.42,
                    'eg_300': 0.354,
                    'eg_0_ind': None,
                    'eg_300_ind': None,
                    'delta': 0.38,
                    'ev_av': -6.67,
                    'optical_matrix': 22.2,
                    'parameter_ep': None
                },
                'deformation_potentials': {
                    'a_c': -5.08,
                    'a_v': 1.0,
                    'a': -6.08,
                    'b': -1.8,
                    'd': -3.6,
                    'c11': 8.329,
                    'c12': 4.526,
                    'c44': 3.96
                },
                'effective_masses': {
                    'm_e': 0.023,
                    'm_hh': 0.4,
                    'm_lh': 0.026,
                    'm_hh_z': 0.0263,
                    'm_lh_z': 0.027,
                    'gamma_1': 20.4,
                    'gamma_2': 8.3,
                    'gamma_3': 9.1,
                }
            },
            Alloy.InP: {
                'parameters': {
                    'a0': 5.8688,
                    'eg_0': 1.424,
                    'eg_300': 1.344,
                    'eg_0_ind': None,
                    'eg_300_ind': None,
                    'delta': 0.11,
                    'ev_av': -7.04,
                    'optical_matrix': 20.7,
                    'parameter_ep': 16.7
                },
                'deformation_potentials': {
                    'a_c': -5.04,
                    'a_v': 1.27,
                    'a': -6.31,
                    'b': -1.7,
                    'd': -5.6,
                    'c11': 10.11,
                    'c12': 5.61,
                    'c44': 4.56
                },
                'effective_masses': {
                    'm_e': 0.077,
                    'm_hh': 0.6,
                    'm_lh': 0.12,
                    'm_hh_z': 0.606,
                    'm_lh_z': 0.121,
                    'gamma_1': 4.95,
                    'gamma_2': 1.65,
                    'gamma_3': 2.35,
                }
            },
            Alloy.GaP: {
                'parameters': {
                    'a0': 5.4505,
                    'eg_0': 2.78,
                    'eg_300': 1.344,
                    'eg_0_ind': 2.35,
                    'eg_300_ind': 2.27,
                    'delta': 0.08,
                    'ev_av': -7.4,
                    'optical_matrix': 22.2,
                    'parameter_ep': None
                },
                'deformation_potentials': {
                    'a_c': -7.14,
                    'a_v': 1.70,
                    'a': -8.83,
                    'b': -1.8,
                    'd': -4.5,
                    'c11': 14.05,
                    'c12': 6.203,
                    'c44': 7.033
                },
                'effective_masses': {
                    'm_e': 0.25,
                    'm_hh': 0.67,
                    'm_lh': 0.17,
                    'm_hh_z': 0.326,
                    'm_lh_z': 0.199,
                    'gamma_1': 4.05,
                    'gamma_2': 0.49,
                    'gamma_3': 1.25,
                }
            },
        }

        # specify a dummy order for some known ternary alloys
        self.ternary_order = {
            Alloy.AlGaAs: (Alloy.AlAs, Alloy.GaAs),
            Alloy.InGaAs: (Alloy.GaAs, Alloy.InAs),
            Alloy.AlInAs: (Alloy.AlAs, Alloy.InAs)
        }

        # P(A(x)B(1-x)C) = x P(AC) + (1-x) P(BC)
        if self.alloy in self.ternary_order:
            alloy_1, alloy_2 = self.ternary_order[self.alloy]
            self.chuang_db[self.alloy] = {}
            for property_type in self.chuang_db[alloy_1].keys():
                property_values = {}
                for property_name in \
                    self.chuang_db[alloy_1][property_type].keys():
                    value_alloy_1 = \
                        self.chuang_db[alloy_1][property_type][property_name]
                    value_alloy_2 = \
                        self.chuang_db[alloy_2][property_type][property_name]
                    property_values[property_name] = \
                        (value_alloy_1 and value_alloy_2 and \
                            concentration*value_alloy_1+\
                                (1.0-concentration)*value_alloy_2) or None
                self.chuang_db[self.alloy][property_type] = property_values

    def alloy_property(self, property_type, property_name):
        """
        this is just a shortcut for accessing the database

        Parameters
        ----------
        property_type : string
            the property type as in the database, possible values are:
            - `parameters`
            - `deformation_potentials`
            - `effective_masses`
        property_name : string
            the property name as in the database, example values are:
            - `eg_0`
            - `a_c`

        Returns
        -------
        property : float
            the value of the property

        Examples
        --------
        >>> from band_structure_database import Alloy, Database
        >>> algaas_03 = Database(Alloy.AlGaAs, 0.3)
        >>> gap = algaas_03.alloy_property('parameters', 'eg_0')
        >>> print("Gap at 0K - %.2f eV" % gap)
        Gap at 0K - 2.00 eV
        """
        return self.chuang_db[self.alloy][property_type][property_name]

    def parameters(self, parameter_name):
        """
        this is just a shortcut for using `alloy_property` method above

        Parameters
        ----------
        property_name : string
            the property name as in the database, example values are:
            - `eg_0`
            - `eg_300`

        Returns
        -------
        property : float
            the value of the parameter

        Examples
        --------
        >>> from band_structure_database import Alloy, Database
        >>> algaas_03 = Database(Alloy.AlGaAs, 0.3)
        >>> print("Gap at 0K - %.2f eV" % algaas_03.parameters('eg_0'))
        Gap at 0K - 2.00 eV
        """
        return self.alloy_property('parameters', parameter_name)

    def deformation_potentials(self, parameter_name):
        """
        this is just a shortcut for using `alloy_property` method above

        Parameters
        ----------
        property_name : string
            the property name as in the database, example values are:
            - `a_c`
            - `a_v`

        Returns
        -------
        property : float
            the value of the deformation parameters

        Examples
        --------
        >>> from band_structure_database import Alloy, Database
        >>> algaas_03 = Database(Alloy.AlGaAs, 0.3)
        >>> a_v = algaas_03.deformation_potentials('a_v')
        >>> print("Valence defformation potencial - %.2f eV" % a_v)
        Valence defformation potencial - 1.55 eV
        """
        return self.alloy_property('deformation_potentials', parameter_name)

    def effective_masses(self, parameter_name):
        """
        this is just a shortcut for using `alloy_property` method above

        Parameters
        ----------
        property_name : string
            the property name as in the database, example values are:
            - `m_e`
            - `m_hh`

        Returns
        -------
        property : float
            the value of the effective mass

        Examples
        --------
        >>> from band_structure_database import Alloy, Database
        >>> algaas_03 = Database(Alloy.AlGaAs, 0.3)
        >>> m_e = algaas_03.effective_masses('m_e')
        >>> print("Electron effective mass - %.3f" % m_e)
        Electron effective mass - 0.092
        """
        return self.alloy_property('effective_masses', parameter_name)
