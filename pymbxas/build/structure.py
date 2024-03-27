#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 15:29:31 2023

@author: roncoroni
"""

import numpy as np

import ase

from pyscf import gto
from pyscf.pbc import gto as pgto

from pymbxas.io.logger import Logger
from pymbxas.utils.check_keywords import check_pbc

try:
    import sea_urchin.alignement.align as ali
    has_SU = True
except:
    has_SU = False
#%%

# convert an ase Atoms object to a mole or cell object for pyscf
def ase_to_mole(structure, charge=0, spin=0, basis='def2-svpd', pbc=None,
                verbose=4, print_output=True, symmetry=False, **kwargs):

    # generate atom list to feed to object
    atom_list = []
    for ii in range(len(structure)):
        atom_list.append([
            structure.get_chemical_symbols()[ii],
            tuple(structure.get_positions()[ii])
            ])
    
    if pbc is None:
        pbc = check_pbc(pbc, structure)
    
    # periodic system
    if pbc:
        
        mol = pgto.Cell(
            atom  = atom_list,
            basis = basis,
            charge = charge,
            spin = spin,
            verbose = verbose,
            stdout = Logger(print_output),
            a = structure.get_cell().array,
            ke_cutoff = 100.0,
            symmetry = symmetry,
            )
    
    # non periodic system
    else:

        mol = gto.Mole(
            atom  = atom_list,
            basis = basis,
            charge = charge,
            spin = spin,
            verbose = verbose,
            stdout = Logger(print_output),
            max_memory = 0,
            unit = 'Angstrom',
            symmetry = symmetry,
            )

    mol.build()
    
    # overwrite intergrals if PBC with proper ones #TODO: check if needed and correct
    if pbc:
        mol.intor = mol.pbc_intor
    
    return mol

# convert a mol object to ase Atoms
def mole_to_ase(mol, units="Angstrom", **kwargs):
    
    structure = ase.Atoms(
        mol.elements,
        mol.atom_coords(unit=units),
        **kwargs
        )
    
    return structure

# rotate, translate, permute, inverse a structure
def rotate_structure(structure, rot, tr, perm, inv, rtype):
    assert has_SU, "Please install Sea Urchin to use this"
    return ali.align_structure(structure, rot, tr, perm, inv, rtype)
    