#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 15:15:11 2023

@author: roncoroni
"""

import numpy as np

# check if cupy is available
try:
    import cupy as cp
except ImportError:
    cp = None

import copy

#%%

# class to read and store data from a pyscf calculation
class pyscf_data():
    """
    This class provides a convenient wrapper for storing and manipulating
    data extracted from a PySCF calculation. It supports conversion between
    NumPy (CPU) and CuPy (GPU) array formats for potential acceleration.
    """

    _FIELDS = ("mol", "mo_coeff", "mo_occ", "mo_energy", "e_tot", "nelec",
               "mo_coeff_del")
    _LAZY_FIELDS = ("mo_coeff", "mo_occ", "mo_energy", "mo_coeff_del")

    def __init__(self, calculator):
        """
        Initializes the pyscf_data object.

        Args:
            calculator (pyscf.gto.Mole, pyscf.scf.HF, etc.): A PySCF calculator
                object that has already been run. If None, an empty object is created.
        """

        self.mol       = calculator.mol
        self.mo_coeff  = calculator.mo_coeff
        self.mo_occ    = calculator.mo_occ
        self.mo_energy = calculator.mo_energy
        self.e_tot     = calculator.e_tot
        self.nelec     = calculator.nelec

        # Convert arrays in-place to ensure np.array
        if cp is not None:
            for attr_name in vars(self):
                attr_value = getattr(self, attr_name)
                if isinstance(attr_value, cp.ndarray):
                    setattr(self, attr_name, attr_value.get())

        self._is_gpu = False

        return

    @classmethod
    def from_arrays(cls, mol, e_tot, nelec, **arrays):
        data = cls.__new__(cls)
        data.mol   = mol
        data.e_tot = e_tot
        data.nelec = nelec
        for name, value in arrays.items():
            setattr(data, name, value)
        data._is_gpu = False
        return data

    @classmethod
    def from_h5_source(cls, mol, e_tot, nelec, path, key):
        data = cls.__new__(cls)
        data.mol        = mol
        data.e_tot      = e_tot
        data.nelec      = nelec
        data._h5_source = (path, key)
        data._is_gpu    = False
        return data

    def __getattr__(self, name):
        if name in type(self)._LAZY_FIELDS:
            source = self.__dict__.get("_h5_source")
            if source is not None:
                from pymbxas.io.h5 import read_lazy_field
                value = read_lazy_field(source, name)
                self.__dict__[name] = value
                return value
        raise AttributeError(name)

    def materialize(self):
        """Force every deferred array to be read from disk."""
        if self.__dict__.get("_h5_source") is None:
            return
        for name in type(self)._LAZY_FIELDS:
            getattr(self, name, None)
        self.__dict__.pop("_h5_source", None)
        return

    def copy(self):
        return copy.deepcopy(self)

    def to_cpu(self):
        """Converts all internal arrays to NumPy format"""

        result = self.copy()

        if not result._is_gpu:
            return result

        for attr_name in type(self)._FIELDS:
            attr_value = getattr(result, attr_name, None)
            if cp is not None and isinstance(attr_value, cp.ndarray):
                setattr(result, attr_name, attr_value.get())

        result._is_gpu = False

        return result

    def to_gpu(self):
        """Converts all internal arrays to CuPy format"""

        result = self.copy()

        if result._is_gpu:
            return result

        result.materialize()

        for attr_name in type(self)._FIELDS:
            attr_value = getattr(result, attr_name, None)
            if isinstance(attr_value, np.ndarray):
                setattr(result, attr_name, cp.asarray(attr_value))

        result._is_gpu = True

        return result
