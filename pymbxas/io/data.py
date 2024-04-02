#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 15:15:11 2023

@author: roncoroni
"""

import numpy as np

try:
    import cupy as cp
except:
    pass # no GPU stuff

from pyscf import gto
from io import StringIO
from .logger import Logger
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
        
        if calculator is None:
            return
        
        self.mol       = calculator.mol
        self.mo_coeff  = calculator.mo_coeff
        self.mo_occ    = calculator.mo_occ
        self.mo_energy = calculator.mo_energy
        self.e_tot     = calculator.e_tot
        self.nelec     = calculator.nelec
        
        # Convert arrays in-place to ensure np.array
        for attr_name in vars(self):
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, cp.ndarray):
                setattr(self, attr_name, attr_value.get()) 
        
        return 
    
    def to_cpu(self):
        """Converts all internal arrays to NumPy format"""
        result = pyscf_data(None)  # Create a placeholder instance

        # Iterate through attributes and convert CuPy arrays
        for attr_name in vars(self):
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, cp.ndarray):
                result.__setattr__(attr_name, attr_value.get())  
            else:
                result.__setattr__(attr_name, attr_value)  

        return result
    
    def to_gpu(self):
        """Converts all internal arrays to CuPy format"""
        result = pyscf_data(None)  # Create a placeholder instance
    
        # Iterate through attributes and convert NumPy arrays
        for attr_name in vars(self):
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, np.ndarray):
                result.__setattr__(attr_name, cp.asarray(attr_value))  
            else:
                result.__setattr__(attr_name, attr_value)  
    
        return result