#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 12:28:13 2023

@author: roncofaber
"""

import os

import pymbxas
from pymbxas.io.copy import copy_output_files
from pymbxas.build.input import make_qchem_input

from pyqchem import get_output_from_qchem

#%%

class Qchem_mbxas():
    
    def __init__(self, structure,
                 gs_params   = None,
                 fch_params  = None,
                 fch_occ     = None,
                 xch_params  = None,
                 xch_occ     = None,
                 scratch_dir = None,
                 run_calc    = True,
                 ):
        
        # initialize environment
        pymbxas.utils.environment.set_qchem_environment()
        
        # set up internal variables
        self.__nprocs = os.cpu_count()
        self.__pid    = os.getpid()
        self.__cdir   = os.getcwd()
        self.__sdir   = os.getcwd() if scratch_dir is None else scratch_dir
        self.__wdir   = "{}/pyqchem_{}/".format(os.getcwd(), self.__pid)
        
        # run MBXAS calculation
        if run_calc:
            self.run_calculations()
        
        return
     
    def run_calculations(self, structure, gs_params, fch_params, fch_occ,
                     xch_params, xch_occ):
        
        # delete scratch earlier if not XCH calc
        is_xch = True if xch_params is not None else False
        
        # GS input
        charge       = 0
        multiplicity = 1
        gs_input = make_qchem_input(structure, charge, multiplicity, gs_params)
        
        # run GS
        gs_output, gs_data = get_output_from_qchem(
            gs_input, processors = self.__nprocs, use_mpi = True,
            return_electronic_structure = True, scratch = self.__sdir,
            delete_scratch = False)
        
        # write output file #TODO change in the future to be more flexible
        with open("qchem.output", "w") as fout: 
            fout.write(gs_output)
        
        # update input with guess and run FCH
        # FCH input
        charge       = 1
        multiplicity = 2
        fch_params["scf_guess"] = gs_data["coefficients"]
        fch_input = make_qchem_input(structure, charge, multiplicity,
                                     fch_params, occupation=fch_occ)
          
        fch_output, fch_data = get_output_from_qchem(
            fch_input, processors = self.__nprocs, use_mpi = True,
            return_electronic_structure = True, scratch = self.__sdir,
            delete_scratch = not is_xch)
        
        # write input and output plus copy MOM files
        with open("qchem.input", "w") as fout:
            fout.write(fch_input.get_txt())
        with open("qchem.output", "a") as fout: 
            fout.write(fch_output)
        copy_output_files(self.__wdir, self.__cdir)
        
        # only run XCH if there is input
        if is_xch:
            
            charge       = 0
            multiplicity = 1
            xch_params["scf_guess"] = fch_data["coefficients"]
            xch_input = make_qchem_input(structure, charge, multiplicity,
                                         xch_params, occupation=xch_occ)
            
            xch_output, xch_data = get_output_from_qchem(
                xch_input, processors = self.__nprocs, use_mpi = True,
                return_electronic_structure = True, scratch = self.__sdir,
                delete_scratch = is_xch)
        
            # generate AlignDir directory #TODO change for more flex
            os.mkdir("AlignDir")
            
            # write XCH output file
            with open("AlignDir/align_calc.out", "w") as fout: 
                fout.write(xch_output)
        
        return