#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 12:28:13 2023

@author: roncofaber
"""

import os

# good ol' numpy
import numpy as np

# self module utilities
import pymbxas
from pymbxas.io.copy import copy_output_files
from pymbxas.build.input import make_qchem_input

# pyqchem stuff
from pyqchem import get_output_from_qchem
from pyqchem.file_io import write_to_fchk

#%%

class Qchem_mbxas():

    def __init__(self,
                 structure,
                 charge,
                 multiplicity,
                 qchem_params = None,
                 excite_atom  = None,
                 fch_occ      = None,
                 do_xch       = True,
                 scratch_dir  = None,
                 print_fchk   = False,
                 run_calc     = True,
                 use_mpi      = False, # somewhat MPI is not working atm
                 use_boys     = True,  # use Boys localization or not
                 save_all     = False,
                 ):

        # initialize environment (set env variables)
        pymbxas.utils.environment.set_qchem_environment()

        # set up internal variables
        self.__is_mpi = use_mpi
        self.__nprocs = os.cpu_count()
        self.__pid    = os.getpid()
        self.__cdir   = os.getcwd()
        self.__sdir   = os.getcwd() if scratch_dir is None else scratch_dir
        self.__wdir   = "{}/pyqchem_{}/".format(os.getcwd(), self.__pid)
        self.__print_fchk = print_fchk
        self.__use_boys   = use_boys
        self.__save_all     = save_all
        # delete scratch earlier if not XCH calc
        self.__is_xch = True if do_xch is not None else False
        self.__ran_GS = False

        # check excite atom and modify accord.
        if isinstance(excite_atom, int):
            excite_atom = {
                "index"   : excite_atom,
                "channel" : "beta"
                }

        # store data
        self.structure    = structure
        self.charge       = charge
        self.multiplicity = multiplicity
        self.qchem_params = qchem_params
        self.excite_atom  = excite_atom
        self.fch_occ      = fch_occ

        # initialize empty stuff
        self.output = {}
        self.data   = {}

        # run MBXAS calculation
        if run_calc:
            self.run_all_calculations()

        return

    # Function to run all calc (GS, FCH, XCH) in sequence
    def run_all_calculations(self):

        # run ground state
        gs_output, gs_data = self.run_ground_state()

        # run FCH
        scf_guess = gs_data["localized_coefficients"] if self.__use_boys else gs_data["coefficients"]

        fch_output, fch_data  = self.run_fch(scf_guess) #TODO change only if Boys

        # only run XCH if there is input
        if self.__is_xch:
            xch_output, xch_data = self.run_xch(fch_data["coefficients"])

        return

    # run the GS calculation
    def run_ground_state(self):

        structure    = self.structure
        charge       = self.charge
        multiplicity = self.multiplicity
        qchem_params = self.qchem_params

        # GS input
        gs_input = make_qchem_input(structure, charge, multiplicity,
                             qchem_params, "gs", occupation = None)

        # run calculation
        gs_output, gs_data = get_output_from_qchem(
            gs_input, processors = self.__nprocs, use_mpi = self.__is_mpi,
            return_electronic_structure = True, scratch = self.__sdir,
            delete_scratch = False)

        # obtain number of electrons
        #TODO make a function that stores relevant output (but not too heavy stuff)
        self.n_alpha = gs_data["number_of_electrons"]["alpha"]
        self.n_beta  = gs_data["number_of_electrons"]["beta"]
        self.n_electrons = self.n_alpha + self.n_beta

        # store output
        self.output["gs"] = gs_output
        if self.__save_all:
            self.data["gs"] = gs_data

        # do boys postprocessing to understand orbital occupations
        if self.__use_boys:
            self.s_orbitals = self.__boys_postprocess(gs_data)

            to_eject = self.s_orbitals[self.excite_atom["channel"]][self.excite_atom["index"]]

            # overwrite occupation
            if self.fch_occ is None:
                self.fch_occ = {
                    "nelectrons" : self.n_alpha,
                    "eject"      : to_eject + 1,
                    "channel"    : self.excite_atom["channel"]
                    }

        # write output file
        #TODO change in the future to be more flexible
        with open("qchem.output", "w") as fout:
            fout.write(gs_output)

        if self.__print_fchk:
            write_to_fchk(gs_data, 'output_gs.fchk')

        # mark that GS has been run
        self.__ran_GS = True

        return gs_output, gs_data

    # run the FCH calculation
    def run_fch(self, scf_guess=None):

        assert self.__ran_GS, "Please run a GS calculation first."

        structure = self.structure
        charge = self.charge + 1 # +1 cause we kick out one lil electron
        multiplicity = abs(self.n_alpha - self.n_beta - 1) + 1
        qchem_params = self.qchem_params
        fch_occ = self.fch_occ

        # FCH input
        fch_input = make_qchem_input(structure, charge, multiplicity,
                                     qchem_params, "fch", occupation=fch_occ,
                                     scf_guess=scf_guess)

        # run calculation
        fch_output, fch_data = get_output_from_qchem(
            fch_input, processors = self.__nprocs, use_mpi = self.__is_mpi,
            return_electronic_structure = True, scratch = self.__sdir,
            delete_scratch = not self.__is_xch)

        # store output
        self.output["fch"] = fch_output
        if self.__save_all:
            self.data["fch"] = fch_data

        # write input and output
        with open("qchem.input", "w") as fout:
            fout.write(fch_input.get_txt())
        with open("qchem.output", "a") as fout:
            fout.write(fch_output)
        # copy MOM files in relevant directory
        copy_output_files(self.__wdir, self.__cdir)

        if self.__print_fchk:
            write_to_fchk(fch_data, 'output_fch.fchk')

        return fch_output, fch_data

    # run the XCH calculation
    def run_xch(self, scf_guess=None):

        assert self.__ran_GS, "Please run a GS calculation first."

        structure = self.structure
        charge = self.charge
        multiplicity = self.multiplicity
        qchem_params = self.qchem_params

        # xch occupation is always the same
        xch_occ = {
            "nelectrons" : self.n_alpha,
            "eject"      : self.n_alpha,
            "inject"     : self.n_alpha + 1, #nelec + 1
            "channel"    : self.fch_occ["channel"],
            }

        # XCH input
        xch_input = make_qchem_input(structure, charge, multiplicity,
                                     qchem_params, "xch", occupation=xch_occ,
                                     scf_guess=scf_guess)

        # run calculation
        xch_output, xch_data = get_output_from_qchem(
            xch_input, processors = self.__nprocs, use_mpi = self.__is_mpi,
            return_electronic_structure = True, scratch = self.__sdir,
            delete_scratch = self.__is_xch)

        # store output
        self.output["xch"] = xch_output
        if self.__save_all:
            self.data["xch"] = xch_data

        if self.__print_fchk:
            write_to_fchk(xch_data, 'output_xch.fchk')

        # generate AlignDir directory #TODO change for more flex
        os.mkdir("AlignDir")

        # write XCH output file
        with open("AlignDir/align_calc.out", "w") as fout:
            fout.write(xch_output)

        return xch_output, xch_data

    # do Boys postprocessing
    def __boys_postprocess(self, gs_electronic_structure):

        # get basis set information
        atom_coeffs, atom_labels, symbols, nbasis = self.__get_basis_set_info(
            gs_electronic_structure['basis'])

        s_orbitals = {}
        # iterate over channels and find 1s orbitals
        for channel in ["alpha", "beta"]:

            boys_coeff = np.array(
                gs_electronic_structure["localized_coefficients"][channel])

            s1_list = self.__find_1s_orbitals(boys_coeff, atom_coeffs, atom_labels, symbols)

            s_orbitals[channel] = s1_list

        return s_orbitals

    @staticmethod
    def __get_basis_set_info(basis):
        atom_coeffs = []
        atom_labels = []
        symbols     = []
        nbasis = 0
        for cc, atom in enumerate(basis['atoms']):
            istart = nbasis
            atom_label = []
            for shell in atom['shells']:
                nbasis += shell['functions']
                atom_label.extend(shell['functions']*[shell['shell_type']])

            atom_coeffs.append(np.array(range(istart, nbasis)))
            atom_labels.append(np.array(atom_label))
            symbols.append(atom["symbol"])

        return atom_coeffs, atom_labels, symbols, nbasis

    @staticmethod
    def __find_1s_orbitals(boys_coeff, atom_coeffs, atom_labels, symbols):

        symbol_list = np.concatenate([[cc]*len(atom_coeffs[cc]) for cc in range(len(symbols))])
        labels_list = np.concatenate(atom_labels)
        dominant_atoms = np.argmax(np.abs(boys_coeff), axis=1)

        orb_types   = labels_list[dominant_atoms]
        orb_symbols = symbol_list[dominant_atoms]

        found_elements = []
        s1_list = [None]*len(symbols)
        for idx, (orb_typ, orb_id) in enumerate(zip(orb_types, orb_symbols)):

            if orb_typ == "s" and orb_id not in found_elements:
                found_elements.append(orb_id)
                s1_list[orb_id] = idx
                # print("The 1s of {}-{} is {}".format(symbols[orb_id], orb_id, idx))

        return s1_list