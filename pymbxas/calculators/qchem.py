#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 12:28:13 2023

@author: roncofaber
"""

import os

import time

# good ol' numpy
import numpy as np

# self module utilities
import pymbxas
import pymbxas.utils.check_keywords as check
from pymbxas.io.copy import copy_output_files
from pymbxas.build.input import make_qchem_input
from pymbxas.utils.boys import find_1s_orbitals, calculate_boys_overlap

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
                 excitation   = None,
                 do_xch       = True,
                 target_dir   = None,
                 scratch_dir  = None,
                 print_fchk   = False,
                 run_calc     = True,
                 use_mpi      = False, # somewhat MPI is not working atm
                 use_boys     = True,  # use Boys localization or not
                 save_all     = False,
                 ):

        # set up internal variables
        self.__is_mpi = use_mpi
        self.__nprocs = os.cpu_count()
        self.__pid    = os.getpid()

        # store directories and path
        self.__cdir = os.getcwd() # current directory
        self.__tdir = os.getcwd() if target_dir is None \
            else os.path.abspath(target_dir) # target directory
        self.__sdir = self.__tdir if scratch_dir is None else scratch_dir # scratch directory
        self.__wdir = "{}/pyqchem_{}/".format(self.__sdir, self.__pid) # PyQCHEM work directory

        if not os.path.exists(self.__tdir):
            os.makedirs(self.__tdir)

        # initialize environment (set env variables)
        pymbxas.utils.environment.set_qchem_environment(self.__sdir)

        # verbose and printing
        self.__print_fchk = print_fchk
        self.__save_all   = save_all

        # calculation details
        self.__use_boys   = use_boys
        self.__is_xch = do_xch

        # check (#TODO in future allow to restart from a GS calculation)
        self.__ran_GS = False

        # store data
        self.structure    = structure
        self.charge       = charge
        self.multiplicity = multiplicity
        self.qchem_params = qchem_params
        self.excitation   = check.determine_excitation(excitation)

        # initialize empty stuff
        self.output = {}
        self.data   = {}

        # run MBXAS calculation
        if run_calc:
            self.run_all_calculations()

        return

    # Function to run all calc (GS, FCH, XCH) in sequence
    def run_all_calculations(self):

        start_time  = time.time()

        # run ground state
        print("Running ground state calculation.")
        gs_output, gs_data = self.run_ground_state()

        gs_time = time.time() - start_time

        print("Ground state calculation finished in {:.2f} s.".format(gs_time))

        print("Running excited state (FCH) calculation with {} orbitals as guess.".format(
            ["KS", "Boys"][self.__use_boys]))
        print("Exciting orbital #{}.".format(self.excitation["eject"]))
        # run FCH
        scf_guess = gs_data["localized_coefficients"] if self.__use_boys else gs_data["coefficients"]
        fch_output, fch_data  = self.run_fch(scf_guess) #TODO change only if Boys

        fch_time = time.time() - gs_time - start_time
        print("Excited state (FCH) calculation finished in {:.2f} s.".format(fch_time))

        # only run XCH if there is input
        xch_time = None,
        if self.__is_xch:
            print("Running excited state (XCH) calculation for energy alignment.")
            xch_output, xch_data = self.run_xch(fch_data["coefficients"])

            xch_time = time.time() - fch_time - gs_time - start_time

            print("Excited state (XCH) calculation finished in {:.2f} s.".format(xch_time))

        self.timings = {
            "gs"  : gs_time,
            "fch" : fch_time,
            "xch" : xch_time
            }

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

        # update with correct number of electrons
        self.excitation["nelectrons"] = [self.n_alpha, self.n_beta]

        # store output
        self.output["gs"] = gs_output
        self.basis_overlap = gs_data["overlap"]
        if self.__save_all:
            self.data["gs"] = gs_data

        # do boys postprocessing to understand orbital occupations
        if self.__use_boys:
            self.boys_coeffs = gs_data["localized_coefficients"]

        # find 1s orbitals
        self.s_orbitals  = find_1s_orbitals(gs_data, use_localized=self.__use_boys)
        # overwrite occupation if it's not been specified
        if not self.excitation["eject"]:
            to_eject = self.s_orbitals[
                self.excitation["channel"]][self.excitation["ato_idx"]]
            self.excitation["eject"] = to_eject + 1

        # write output file
        #TODO change in the future to be more flexible
        with open(self.__tdir + "/qchem.output", "w") as fout:
            fout.write(gs_output)

        if self.__print_fchk:
            write_to_fchk(gs_data, self.__tdir + "/output_gs.fchk")

        # mark that GS has been run
        self.__ran_GS = True

        return gs_output, gs_data

    # run the FCH calculation
    def run_fch(self, scf_guess=None):

        assert self.__ran_GS, "Please run a GS calculation first."

        structure    = self.structure
        charge       = self.charge + 1 # +1 cause we kick out one lil electron
        multiplicity = abs(self.n_alpha - self.n_beta - 1) + 1
        qchem_params = self.qchem_params
        excitation   = self.excitation

        # FCH input
        fch_input = make_qchem_input(structure, charge, multiplicity,
                                     qchem_params, "fch",
                                     occupation = excitation,
                                     scf_guess  = scf_guess)

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
        with open(self.__tdir + "/qchem.input", "w") as fout:
            fout.write(fch_input.get_txt())
        with open(self.__tdir + "/qchem.output", "a") as fout:
            fout.write(fch_output)
        # copy MOM files in relevant directory
        copy_output_files(self.__wdir, self.__tdir)

        if self.__print_fchk:
            write_to_fchk(fch_data, self.__tdir + "/output_fch.fchk")

        # calculate overlap if using BOYS
        if self.__use_boys:

            self.boys_overlap = calculate_boys_overlap(self.boys_coeffs,
                                                       fch_data["coefficients"],
                                                       self.basis_overlap)

            # overwrite files
            for channel, overlap in self.boys_overlap.items():
                np.savetxt(self.__tdir + "/{}_ovlp_gs_es.txt".format(
                    channel), overlap)

        return fch_output, fch_data

    # run the XCH calculation
    def run_xch(self, scf_guess=None):

        assert self.__ran_GS, "Please run a GS calculation first."

        structure = self.structure
        charge = self.charge
        multiplicity = self.multiplicity
        qchem_params = self.qchem_params

        # xch occupation is always the same
        channel = self.excitation["channel"]
        channel_idx = 0 if  channel == "alpha" else 1
        nelec = [self.n_alpha, self.n_beta][channel_idx]
        xch_occ = {
            "nelectrons" : [self.n_alpha, self.n_beta],
            "eject"      : nelec,
            "inject"     : nelec + 1, #nelec + 1
            "channel"    : channel,
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
            write_to_fchk(xch_data, self.__tdir + "/output_xch.fchk")

        # generate AlignDir directory #TODO change for more flex
        try:
            os.mkdir(self.__tdir + "/AlignDir")
        except OSError:
            pass

        # write XCH output file
        with open(self.__tdir + "/AlignDir/align_calc.out", "w") as fout:
            fout.write(xch_output)

        return xch_output, xch_data