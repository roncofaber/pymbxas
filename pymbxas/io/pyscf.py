#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 15:15:11 2023

@author: roncoroni
"""

from pyscf import gto
from io import StringIO
from .logger import Logger
#%%

# convert an ase Atoms object to a mole object for pyscf
def ase_to_mole(structure, charge=0, spin=0, basis='def2-svpd',
                verbose=5, print_output=False):

    # generate atom list
    atom_list = []
    for ii in range(len(structure)):
        atom_list.append([
            structure.get_chemical_symbols()[ii],
            tuple(structure.get_positions()[ii])
            ])

    # create mole object
    mol = gto.Mole()

    # assign atoms
    mol.atom = atom_list

    # assign basis
    mol.basis = basis

    # assign charge and spin
    mol.charge = charge
    mol.spin   = spin

    # define output
    mol.verbose = verbose

    # log object
    mol.stdout = Logger(print_output)

    # build mole
    mol.build()

    return mol

class pyscf_data():
    
    def __init__(self, calculator):
            
        self.mo_coeff  = calculator.mo_coeff
        self.mo_occ    = calculator.mo_occ
        self.mo_energy = calculator.mo_energy
        self.e_tot     = calculator.e_tot
        
        return