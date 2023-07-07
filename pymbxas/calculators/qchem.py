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
        
        # generate input objects
        gs_input, fch_input, xch_input = self.setup_inputs(
            structure, gs_params, fch_params, fch_occ, xch_params, xch_occ)
        
        self.gs_input  = gs_input
        self.fch_input = fch_input
        self.xch_input = xch_input
        
        if run_calc:
            self.run_calculations(gs_input, fch_input, xch_input)
        
        return
    
    def setup_inputs(self, structure, gs_params, fch_params, fch_occ,
                     xch_params, xch_occ):
        
        # GS input
        charge       = 0
        multiplicity = 1
        gs_input = make_qchem_input(structure, charge, multiplicity, gs_params)
        
        # FCH input
        charge       = 1
        multiplicity = 2
        fch_input = make_qchem_input(structure, charge, multiplicity,
                                     fch_params, occupation=fch_occ)
        
        # XCH input (only if specified)
        if xch_params is not None:
            charge       = 0
            multiplicity = 1
            xch_input = make_qchem_input(structure, charge, multiplicity,
                                         xch_params, occupation=xch_occ)
        else:
            xch_input = None
        
        return gs_input, fch_input, xch_input
    
    def run_calculations(self, gs_input, fch_input, xch_input):
        
        # delete scratch earlier if not XCH calc
        is_xch = True if xch_input is not None else False
        
        # run GS
        gs_output, gs_data = get_output_from_qchem(
            gs_input, processors = self.__nprocs, use_mpi = True,
            return_electronic_structure = True, scratch = self.__sdir,
            delete_scratch = False)
        
        # write output file #TODO change in the future to be more flexible
        with open("qchem.output", "w") as fout: 
            fout.write(gs_output)
        
        # update input with guess and run FCH
        fch_input.update_input({"scf_guess" : gs_data["coefficients"]})
          
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
        if xch_input is not None: 
            
            # update input with guess and run XCH
            xch_input.update_input({"scf_guess" : fch_data["coefficients"]})
            
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