#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 12:28:13 2023

@author: roncofaber
"""

import os
import dill
import time

# good ol' numpy
import numpy as np

# self module utilities
import pymbxas
import pymbxas.utils.check_keywords as check
from pymbxas.io.copy import copy_output_files
from pymbxas.build.input import make_qchem_input
from pymbxas.utils.orbitals import find_1s_orbitals, calculate_boys_overlap
from pymbxas.utils.environment import set_qchem_environment, get_qchem_version_from_output
from pymbxas.utils.basis import get_basis_set_info
import pymbxas.CleaRIXS.read_modules as rd
import pymbxas.CleaRIXS.rixs_modules as rx

# pyqchem stuff
from pyqchem import get_output_from_qchem
from pyqchem.file_io import write_to_fchk
from pyqchem.parsers.basic import basic_parser_qchem

#%%

class Qchem_mbxas():

    def __init__(self,
                 structure    = None,
                 charge       = None,
                 multiplicity = None,
                 pkl_file     = None,
                 qchem_params = None,
                 excitation   = None,
                 do_xch       = True,
                 target_dir   = None,
                 scratch_dir  = None,
                 print_fchk   = True,
                 run_calc     = True,
                 use_mpi      = False, # somewhat MPI is not working atm
                 use_boys     = True,  # use Boys localization or not
                 save         = True,  # save object as pkl file
                 save_all     = True, # save all fchk output in the pkl
                 save_name    = "mbxas_obj.pkl", # name of saved file
                 save_path    = None, # path of saved object
                 keep_scratch = False,
                 ):

        # restart object from a pkl file
        if pkl_file is not None:
            self.__restart_from_pickle(pkl_file)

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
        set_qchem_environment(self.__sdir)

        if pkl_file is None:
            self.__initialize_from_scratch(structure, charge, multiplicity,
                                           qchem_params, excitation, use_boys,
                                           do_xch, print_fchk, save, save_name,
                                           save_path, save_all, keep_scratch,
                                           run_calc)

        return

    def __initialize_from_scratch(self, structure, charge, multiplicity,
                                  qchem_params, excitation, use_boys, do_xch,
                                  print_fchk, save, save_name, save_path,
                                  save_all, keep_scratch, run_calc):

        # calculation details
        self.__use_boys   = use_boys
        self.__is_xch = do_xch

        # output, verbose and printing
        self.__print_fchk = print_fchk
        self.__save_all = save_all
        self.__keep_scratch = keep_scratch

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

        if save:
            self.save_object(save_name, save_path)

        return

    # restart object from pkl file previously saved
    def __restart_from_pickle(self, pkl_file):

        # open previously generated gpw file
        with open(pkl_file, "rb") as fin:
            restart = dill.load(fin)

        self.__dict__ = restart.__dict__.copy()

        return

    # Function to run all calc (GS, FCH, XCH) in sequence
    def run_all_calculations(self):

        print(">>> Starting PyMBXAS <<<")

        start_time  = time.time()

        # run ground state
        print("> Ground state calculation: ", end = "", flush=True)
        gs_output, gs_data = self.run_ground_state()
        gs_time = time.time() - start_time
        print("finished in {:.2f} s.".format(gs_time))

        print("> Exciting orbital #{} - {} orbs.".format(self.excitation["eject"],
                                                    ["KS", "Boys"][self.__use_boys]))
        # run FCH
        print("> FCH calculation: ", end = "", flush=True)
        scf_guess = gs_data["localized_coefficients"] if self.__use_boys else gs_data["coefficients"]
        fch_output, fch_data  = self.run_fch(scf_guess) #TODO change only if Boys
        fch_time = time.time() - gs_time - start_time
        print("finished in {:.2f} s.".format(fch_time))

        tot_time = fch_time
        # only run XCH if there is input
        xch_time = None,
        if self.__is_xch:
            print("> XCH calculation: ", end = "", flush=True)
            xch_output, xch_data = self.run_xch(fch_data["coefficients"])
            xch_time = time.time() - fch_time - gs_time - start_time
            print("finished in {:.2f} s.".format(xch_time))
            tot_time += xch_time

        print("> MBXAS calculation: ", end = "", flush=True)
        self.run_mbxas()
        mbxas_time = time.time() - tot_time - gs_time - start_time
        print("finished in {:.2f} s.".format(mbxas_time))

        self.timings = {
            "gs"  : gs_time,
            "fch" : fch_time,
            "xch" : xch_time,
            "mbxas" : mbxas_time
            }

        print(">>> PyMBXAS finished successfully! <<<\n")

        return

    # run the GS calculation
    def run_ground_state(self):

        # check if GS was already performed, if so: skip
        if self.__ran_GS:
            print("GS already ran with this configuration. Skipping.")
            return self.output["gs"], self.data["gs"]

        # get system settings
        structure    = self.structure
        charge       = self.charge
        multiplicity = self.multiplicity
        qchem_params = self.qchem_params

        # GS input
        gs_input = make_qchem_input(structure, charge, multiplicity,
                             qchem_params, "gs", occupation = None,
                             use_boys=self.__use_boys)

        # run calculation
        gs_output, gs_data = get_output_from_qchem(
            gs_input, processors = self.__nprocs, use_mpi = self.__is_mpi,
            return_electronic_structure = True, scratch = self.__sdir,
            delete_scratch = False)

        # parse output (#TODO in future we won't need to print the output so it can be done directly)
        gs_data.update(basic_parser_qchem(gs_output))

        # read basis set information and store them
        atom_coeffs, atom_labels, symbols, nbasis, indexing = get_basis_set_info(gs_data["basis"])
        self.basis = {
            "atom_coefficients" : atom_coeffs,
            "atom_labels"       : atom_labels,
            "symbols"           : symbols,
            "nbasis"            : nbasis,
            "indexing"          : indexing,
            }

        # old version, loc coeff need to be swapped and override qchem_order
        gs_data = self.__check_basis_indexing(gs_data, gs_output, self.basis["indexing"])

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
            delete_scratch = not (self.__is_xch or self.__keep_scratch))

        # parse output (#TODO in future we won't need to print the output so it can be done directly)
        fch_data.update(basic_parser_qchem(fch_output))

        # old version, loc coeff need to be swapped and override qchem_order
        fch_data = self.__check_basis_indexing(fch_data, fch_output, self.basis["indexing"])

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
            delete_scratch = not self.__keep_scratch)

        # parse output (#TODO in future we won't need to print the output so it can be done directly)
        xch_data.update(basic_parser_qchem(xch_output))

        # old version, loc coeff need to be swapped and override qchem_order
        xch_data = self.__check_basis_indexing(xch_data, xch_output, self.basis["indexing"])

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

    def run_mbxas(self):

        # Hartree to eV
        Ha = 27.211407953

        # select excitation channel
        channel = self.excitation["channel"]

        # TODO check if this works:
        # CI_Expansion = rd.Unres_CISreader(output_file, check_amp)
        # E_f = []
        # E_f.append(0)
        # for i in CI_Expansion:
        #  E_f.append(i[1])

        # get excitation info
        nocc = self.data["fch"]["number_of_electrons"][channel]
        core_ind_gs = self.excitation["eject"]

        # read matrices
        full_mom_matrix, full_gs_matrix, full_ovlp_matrix = rd.read_full_matrices(
            nocc, core_ind_gs, path=self.__tdir, channel=channel)

        # read energy of unoccupied states #TODO check consistency
        ener = self.data["fch"]["mo_energies"][channel][nocc+1:]

        # align energies
        if self.__is_xch:
            ener += self.data["xch"]["scf_energy"] - self.data["gs"]["scf_energy"] - np.min(ener)
        else:
            ener += self.data["fch"]["scf_energy"] #TODO check this

        ener = ener*Ha # convert energy to eV

        # calculate overlap and store absorption
        xi = (np.array(full_ovlp_matrix)).T
        norb = min(xi.shape)

        absorption = []
        for ixyz in [0, 1, 2]:
            chb_xmat = np.array(full_mom_matrix)[ixyz, :]
            absorption.append(rx.AbsCalc(ixyz, xi, nocc, norb, chb_xmat))

        self.mbxas = {
            "absorptions" : absorption,
            "energies"    : ener
            }

        return

    @staticmethod
    def __check_basis_indexing(data, output, indices):

        # override ordering to be consistent (53.0 can be overwritten by Boys)
        data["coefficients"]["qchem_order"] = indices

        if "localized_coefficients" not in data:
            return data

        data["localized_coefficients"]["qchem_order"] = indices

        if get_qchem_version_from_output(output) < 6:

            reverse_indices = [list(indices).index(j) for j in range(len(indices))]

            for channel in ["alpha", "beta"]:
                data['localized_coefficients'][channel] = \
                    np.array(data['localized_coefficients'][channel])[:, reverse_indices].tolist()

            data['localized_coefficients']["qchem_order"] = indices

        return data

    # broaden spectrum to plot it
    @staticmethod
    def broadened_spectrum(x, energies, intensities, sigma):

        def gaussian_broadening(x, sigma):
            return np.exp(-x**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

        broadened_spec = np.zeros_like(x)
        for energy, intensity in zip(energies, intensities**2):
            broadened_spec += intensity * gaussian_broadening(x - energy, sigma)

        return broadened_spec

    # function to get MBXAS spectra
    def get_mbxas_spectra(self, sigma=0.3, npoints=3001, tol=0.01):

        min_E = np.min(self.mbxas["energies"])
        max_E = np.max(self.mbxas["energies"])

        dE = max_E - min_E

        energy = np.linspace(min_E - tol*dE, max_E + tol*dE, npoints)

        spectra = self.broadened_spectrum(energy, self.mbxas["energies"],
                                          np.mean(self.mbxas["absorptions"], axis=0),
                                          sigma)

        return energy, spectra


    # save object as pkl file
    def save_object(self, oname=None, save_path=None):

        if oname is None:
            oname = self.savename

        if save_path is None:
            path = self.__tdir
        else:
            path = save_path

        if oname.endswith(".pkl"):
            oname =  path + "/" + oname
        else:
            oname = path + "/" + oname.split(".")[-1] + ".pkl"

        with open(oname, 'wb') as fout:
            dill.dump(self, fout)

        print("Saved everything as {}".format(oname))
        return