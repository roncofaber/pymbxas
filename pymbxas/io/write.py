#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 12:01:06 2023

@author: roncofaber
"""

import os
import pymbxas.build.input as inp
import numpy as np

# MOKIT stuff
try:
    from mokit.lib.py2fch import py2fch
    from mokit.lib.py2fch_direct import mol2fch
    is_mokit = True
except:
    is_mokit = False

#%%

def write_mbxas_input(mbxas_parameters = {}, run_path = "."):

    # define MBXAS input file
    mbxas_f = run_path + "/INP_FILE"

    # copy default params and update with input ones
    mbxas_params = {
        "gridP"    : 100,
        "highE"    : 1127.5,
        "lowE"     : 20,
        "sigma"    : 0.3,
        "do_align" : False,
        "DoRIXS"   : False,

        "Gamma"         : "0+0.3j",
        "check_amp"     : True,
        "printsticks"   : True,
        "printspec"     : True,
        "Dodebug"       : False,
        "calc_abs"      : True,
        "printinteg"    : False,
        "input_file"    : "{}/qchem.input".format(run_path),
        "output_file"   : "{}/qchem.output".format(run_path),
        "printanalysis" : True
        }

    mbxas_params.update(mbxas_parameters)

    # write input file
    with open(mbxas_f, "w") as fout:
        for key, value in mbxas_params.items():
              fout.write("{} = {}\n".format(key, value))

    return


def write_qchem_job(molecule, charge, multiplicity,
                    qchem_params, run_path, occupation = None,
                    from_scratch = True):

    # make dir if not existent
    if not os.path.isdir(run_path):
        os.mkdir(run_path)

    # if from scratch write new file, otherwise append
    if from_scratch:
        write_mode = "w"
    else:
        write_mode = "a"

    # generate qchem input
    qchem_input = inp.make_qchem_input(molecule, charge,
                                       multiplicity, qchem_params,
                                       occupation=occupation)



    # write input in target path (append mode)
    with open(run_path + "qchem.input", write_mode) as fout:

        if write_mode == "a":
            fout.write("\n@@@\n\n")

        fout.write(qchem_input.get_txt())

    return


def write_data_to_fchk(mol, data, oname="tmp.fchk", mo_coeff=None, mo_occ=None,
                       density=False, mo_energy=None):
    
    if not is_mokit:
        print("No MOKIT")
        return
    
    directory = os.path.dirname(oname)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        
    mol = mol.copy()
    
    if mo_coeff is None:
        mo_coeff = data.mo_coeff
    
    if mo_energy is None:
        mo_energy = data.mo_energy
        
    if len(mo_coeff) == 2:
        nbasis, norbit = mo_coeff[0].shape
    else:
        nbasis, norbit = mo_coeff.shape
        mo_coeff = np.array([mo_coeff, mo_coeff])
        
    if not len(mo_energy) == 2:
        mo_energy = np.array([mo_energy, mo_energy])
        
    if mo_occ is None:
        mo_occ = data.mo_occ
        
    # update mo_occ just to make sure
    nelec = mo_occ.sum(axis=1, dtype=int)
    
    for cc, ne in enumerate(nelec):
        
        if ne > mo_coeff[cc].shape[1]:
            nelec[cc] = mo_coeff[cc].shape[1]
        
            # nelec[cc] = np.minimum(ne, mo_coeff.shape[cc+1])
    
    mol.nelec = nelec
        
    # shape doesn't match, means that probably the energy is useless
    if mo_energy.shape[1] != norbit:
        mo_energy = np.zeros((2, norbit))
        
    if norbit > data.mo_coeff[0].shape[1]:
        print("Cut orbital number cause higher than allowed by IQmol")
        
        norbit_max = data.mo_coeff[0].shape[1]
    else:
        norbit_max = norbit
        
    
    # actual write
    mol2fch(mol, oname , True, mo_coeff[:,:,:norbit_max])
    for cc, spin in enumerate(["a", "b"]):
        py2fch(oname, nbasis, norbit_max, mo_coeff[cc][:,:norbit_max], spin,
               mo_energy[cc][:norbit_max], False, density)
    
    return