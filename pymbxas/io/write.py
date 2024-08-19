#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 12:01:06 2023

@author: roncofaber
"""

import os
# import pymbxas.build.input as inp
import numpy as np

# MOKIT stuff
try:
    from mokit.lib.py2fch import py2fch
    from mokit.lib.py2fch_direct import mol2fch
    is_mokit = True
except:
    is_mokit = False
    
# from mokit.lib.py2fch import py2fch
# from mokit.lib.py2fch_direct import mol2fch

#%%

def write_data_to_fchk(mol, mo_coeff=None, mo_occ=None, density=False,
                       mo_energy=None, oname="tmp.fchk",  center=False):
    
    # if not is_mokit:
    #     print("No MOKIT")
    #     return
    
    directory = os.path.dirname(oname)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        
    mol = mol.copy()
    
    if center:
        origin = np.mean([ii[1] for ii in mol.atom], axis=0)
        
        new_atom = mol.atom.copy()
        for cc, atom in enumerate(new_atom):
            new_atom[cc][1] = np.array(atom[1]) - origin
        
        mol.set(atom=new_atom)
        mol.build()
    
    if mo_coeff is None:
        mo_coeff = mol.intor_symmetric('int1e_ovlp')
    
    if len(mo_coeff) == 2:
        nbasis, norbit = mo_coeff[0].shape
    else:
        nbasis, norbit = mo_coeff.shape
        mo_coeff = np.array([mo_coeff, mo_coeff])
    
    if mo_energy is None:
        pass
    elif not len(mo_energy) == 2:
        mo_energy = np.array([mo_energy, mo_energy])
        
    # update mo_occ just to make sure
    if mo_occ is not None:
        nelec = mo_occ.sum(axis=1, dtype=int)
    else:
        nelec = list(mol.nelec)
    
    for cc, ne in enumerate(nelec):
        
        if ne > mo_coeff[cc].shape[1]:
            nelec[cc] = mo_coeff[cc].shape[1]
        
            # nelec[cc] = np.minimum(ne, mo_coeff.shape[cc+1])
    
    mol.nelec = nelec
        
    # shape doesn't match, means that probably the energy is useless
    if mo_energy is None:
        mo_energy = np.zeros((2, norbit))
    if mo_energy.shape[1] != norbit:
        mo_energy = np.zeros((2, norbit))
        
    if norbit > mo_coeff[0].shape[1]:
        print("Cut orbital number cause higher than allowed by IQmol")
        
        norbit_max = mo_coeff[0].shape[1]
    else:
        norbit_max = norbit
    
    # actual write
    mol2fch(mol, oname , True, mo_coeff[:,:,:norbit_max])
    for cc, spin in enumerate(["a", "b"]):
        py2fch(oname, nbasis, norbit_max, mo_coeff[cc][:,:norbit_max], spin,
               mo_energy[cc][:norbit_max], False, density)
    
    return
