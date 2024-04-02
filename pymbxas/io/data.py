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
    
    def copy(self):
        return copy.deepcopy(self)
    
    def to_cpu(self):
        """Converts all internal arrays to NumPy format"""
        
        result = self.copy()  # Create a placeholder instance
        
        if not result._is_gpu:
            return result

        # Iterate through attributes and convert CuPy arrays
        for attr_name in vars(self):
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, cp.ndarray):
                result.__setattr__(attr_name, attr_value.get())  
            else:
                result.__setattr__(attr_name, attr_value)
                
        result._is_gpu = False

        return result
    
    def to_gpu(self):
        """Converts all internal arrays to CuPy format"""
        
        result = self.copy()  # Create a placeholder instance
        
        if result._is_gpu:
            return result
        
        # Iterate through attributes and convert NumPy arrays
        for attr_name in vars(self):
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, np.ndarray):
                result.__setattr__(attr_name, cp.asarray(attr_value))  
            else:
                result.__setattr__(attr_name, attr_value)
                
        result._is_gpu = True
    
        return result