#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 18:13:29 2023

@author: roncoroni
"""

import os
import dill
import time

# good ol' numpy
import numpy as np

# self module utilities
import pymbxas
import pymbxas.utils.check_keywords as check
from pymbxas.io.pyscf import ase_to_mole, parse_pyscf_calculator
from pymbxas.utils.orbitals import find_1s_orbitals_pyscf
from pymbxas.utils.boys import do_boys_pyscf
from pymbxas.mbxas.mbxas import run_MBXAS_pyscf
from pymbxas.mbxas.broaden import get_mbxas_spectra
from pymbxas.io.cleanup import remove_tmp_files

# pyscf stuff
from pyscf import gto, scf, dft

# MOKIT stuff
try:
    from mokit.lib.py2fch_direct import fchk as write_to_fchk
    is_mokit = True
except:
    is_mokit = False

#%%

def s2i(string):
    if string == "beta":
        return 1
    elif string == "alpha":
        return 0
    else:
        raise "ERROR CHANNEL"

class PySCF_mbxas():

    def __init__(self,
                 structure    = None,
                 charge       = None,
                 spin         = None,
                 pkl_file     = None,
                 excitation   = None,
                 xc           = "b3lyp",
                 basis        = "def2-svpd",
                 do_xch       = True,
                 target_dir   = None,
                 verbose      = 5,
                 print_fchk   = True,
                 print_output = False,
                 run_calc     = True,
                 save         = True,  # save object as pkl file
                 save_all     = True, # save chkfile
                 save_name    = "pyscf_obj.pkl", # name of saved file
                 save_path    = None, # path of saved object
                 ):

        # restart object from a pkl file
        if pkl_file is not None:
            self.__restart_from_pickle(pkl_file)

        # store directories and path
        self.__cdir = os.getcwd() # current directory
        self.__tdir = os.getcwd() if target_dir is None \
            else os.path.abspath(target_dir) # target directory

        if not os.path.exists(self.__tdir):
            os.makedirs(self.__tdir)


        if pkl_file is None:
            self.__initialize_from_scratch(structure, charge, spin, excitation,
                                          xc, basis, do_xch, verbose, print_fchk,
                                          print_output, save_all)
        
        # run MBXAS calculation
        if run_calc:
            self.run_all_calculations()

        if save:
            self.save_object(save_name, save_path)
        
        return

    def __initialize_from_scratch(self, structure, charge, spin, excitation,
                                  xc, basis, do_xch, verbose, print_fchk,
                                  print_output, save_all):

        # output, verbose and printing
        self.__print_fchk   = print_fchk
        self.__print_output = print_output
        self.__verbose      = verbose
        self.__save_all     = save_all

        # check (#TODO in future allow to restart from a GS calculation)
        self.__ran_GS   = False
        self.__use_boys = False

        # store calculation details
        self.structure  = structure
        self.charge     = charge
        self.spin       = spin
        self.excitation = check.determine_excitation(excitation)
        self.xc         = xc
        self.basis      = basis
        self.__is_xch   = do_xch

        # initialize empty stuff
        self.output = {}
        self.data   = {}

        return

    # restart object from pkl file previously saved
    def __restart_from_pickle(self, pkl_file):

        # open previously generated gpw file
        with open(pkl_file, "rb") as fin:
            restart = dill.load(fin)

        self.__dict__ = restart.__dict__.copy()

        return

    # # Function to run all calc (GS, FCH, XCH) in sequence
    def run_all_calculations(self):

        print(">>> Starting PyMBXAS <<<")
        
        # change directory
        os.chdir(self.__tdir)

        start_time  = time.time()

        # run ground state
        print("> Ground state calculation: ", end = "", flush=True)
        gs_calc = self.run_ground_state()
        gs_time = time.time() - start_time
        print("finished in {:.2f} s.".format(gs_time))

        print("> Exciting orbital #{} - {} orbs.".format(self.excitation["eject"],
                                                    ["KS", "Boys"][self.__use_boys]))
        
        #TODO: allow to specify a list of atoms to excite so that only one GS
        # has to be run
        
        # run FCH
        print("> FCH calculation: ", end = "", flush=True)
        fch_calc = self.run_fch()
        fch_time = time.time() - gs_time - start_time
        print("finished in {:.2f} s.".format(fch_time))

        tot_time = fch_time
        # only run XCH if there is input
        xch_time = None,
        if self.__is_xch:
            print("> XCH calculation: ", end = "", flush=True)
            xch_calc = self.run_xch()
            xch_time = time.time() - fch_time - gs_time - start_time

            if xch_calc is None: #failed
                print("failed! - ignoring XCH")

            else:
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
        
        # go back where we were
        remove_tmp_files(self.__tdir)
        os.chdir(self.__cdir)

        return

    # run the GS calculation
    def run_ground_state(self):

        # check if GS was already performed, if so: skip
        if self.__ran_GS:
            print("GS already ran with this configuration. Skipping.")
            return self.output["gs"], self.data["gs"]

        # get system settings
        structure = self.structure
        charge    = self.charge
        spin      = self.spin
        basis     = self.basis
        xc        = self.xc
        excitation  = self.excitation
        
        channel = s2i(excitation["channel"])

        # generate molecule
        gs_mol = ase_to_mole(structure, charge, spin, basis,
                             verbose=self.__verbose, print_output=self.__print_output)

        # generate KS calculator
        gs_calc = dft.UKS(gs_mol, xc=xc)

        # Use chkfile to store calculation (if you want)
        if self.__save_all:
            gs_calc.chkfile = 'gs.chkfile'

        # run SCF #TODO: check how to change convergence parameters
        gs_calc.kernel()

        # check for degenerate delocalized orbitals and if necessary, do Boys
        s1_orbitals = self.__find_delocalized_orbitals(gs_mol, gs_calc,
                                                       excitation["ato_idx"],
                                                       check_deg = True)

        # if orbitals are delocalized, do Boys and overwrite MO coeff
        if not len(s1_orbitals[channel]) == 1:

            self.__use_boys = True
            mo_boys = do_boys_pyscf(gs_calc, s1_orbitals)
            gs_calc.mo_coeff = mo_boys

            # check again for delocalized orbitals
            s1_orbitals = self.__find_delocalized_orbitals(gs_mol, gs_calc,
                                                           excitation["ato_idx"],
                                                           check_deg = False)

            assert len(s1_orbitals[channel]) == 1

        # decide which orbital to eject
        self.excitation["eject"] = s1_orbitals[channel][0]

        # obtain number of electrons
        self.n_electrons = [gs_calc.nelec[0], gs_calc.nelec[1]]

        # store input/output
        self.output["gs"] = gs_calc.stdout.log.getvalue()
        self.data["gs"]   = parse_pyscf_calculator(gs_calc)

        # write fchk file if using mokit
        if self.__print_fchk and is_mokit:
            write_to_fchk(gs_calc, self.__tdir + "/output_gs.fchk", overwrite_mol=True)

        # mark that GS has been run
        self.__ran_GS = True

        return gs_calc

    # run the FCH calculation
    def run_fch(self, scf_guess=None):

        assert self.__ran_GS, "Please run a GS calculation first."

        structure  = self.structure
        charge     = self.charge + 1 # +1 cause we kick out one lil electron
        excitation = self.excitation
        basis      = self.basis
        xc         = self.xc
        channel    = s2i(excitation["channel"])
        spin       = self.spin - channel*2 + 1 # spin changes +1 if c=0, -1 else

        # Read MO coefficients and occupation number from chkfile
        if scf_guess is None:
            scf_guess = self.data["gs"].mo_coeff.copy()

        # determine occupation
        occupation = self.data["gs"].mo_occ.copy()

        # Assign initial occupation pattern --> kick orbital N
        occupation[channel][excitation["eject"]] = 0

        # change charge #TODO: test what happens to mol outside of here
        fch_mol = ase_to_mole(structure, charge=charge, spin=spin, basis=basis,
                              verbose=self.__verbose, print_output=self.__print_output)

        # Defnine new SCF calculator
        fch_calc = dft.UKS(fch_mol, xc=xc)

        # Use chkfile to store calculation (if you want)
        if self.__save_all:
            fch_calc.chkfile = 'fch.chkfile'

        # Construct new density matrix with new occupation pattern
        dm_u = fch_calc.make_rdm1(scf_guess, occupation)

        # Apply mom occupation principle
        fch_calc = scf.addons.mom_occ(fch_calc, scf_guess, occupation)

        # Start new SCF with new density matrix
        fch_calc.scf(dm_u)

        # store input/output
        self.output["fch"] = fch_calc.stdout.log.getvalue()
        self.data["fch"]   = parse_pyscf_calculator(fch_calc)

        if self.__print_fchk:
            write_to_fchk(fch_calc, self.__tdir + "/output_fch.fchk")

        return fch_calc

    # run the XCH calculation
    def run_xch(self, scf_guess=None):

        assert self.__ran_GS, "Please run a GS calculation first."

        structure  = self.structure
        charge     = self.charge
        excitation = self.excitation
        basis      = self.basis
        xc         = self.xc
        spin       = self.spin
        channel    = s2i(excitation["channel"])

        # Read MO coefficients and occupation number from chkfile
        if scf_guess is None:
            scf_guess = self.data["fch"].mo_coeff.copy()

        # Read MO coefficients and occpuation number from FCH calc
        occupation = self.data["fch"].mo_occ.copy()

        # Assign initial occupation pattern (re-add one lil electron in LUMO)
        occupation[channel][self.n_electrons[channel]] = 1

        # make XCH molecule
        xch_mol = ase_to_mole(structure, charge=charge, spin=spin, basis=basis,
                              verbose=self.__verbose, print_output=self.__print_output)

        # define new SCF calculator
        xch_calc = dft.UKS(xch_mol, xc=xc)

        # Use chkfile to store calculation (if you want)
        if self.__save_all:
            xch_calc.chkfile = 'xch.chkfile'

        # Construct new density matrix with new occupation pattern
        dm_u = xch_calc.make_rdm1(scf_guess, occupation)

        # Apply mom occupation principle
        xch_calc = scf.addons.mom_occ(xch_calc, scf_guess, occupation)

        # Start new SCF with new density matrix
        try:
            xch_calc.scf(dm_u)
            self.__xch_failed = False

        except: #calculation failed

            self.__is_xch = False
            self.__xch_failed = True

            return None

        # store input/output
        self.output["xch"] = xch_calc.stdout.log.getvalue()
        self.data["xch"]   = parse_pyscf_calculator(xch_calc)

        if self.__print_fchk:
            write_to_fchk(xch_calc, self.__tdir + "/output_xch.fchk")

        return xch_calc

    # run MBXAS from a set of pySCF calculations
    def run_mbxas(self):

        energies, absorption, mb_ovlp, dip_KS, b_ovlp = run_MBXAS_pyscf(
            self.data["gs"], self.data["fch"], s2i(self.excitation["channel"]),
            self.data["xch"])

        self.mbxas = {
            "energies"   : energies,
            "absorption" : absorption,
            "mb_overlap" : mb_ovlp,
            "dipole_KS"  : dip_KS,
            "basis_ovlp" : b_ovlp
            }

        return

    def get_mbxas_spectra(self, axis=None, sigma=0.3, npoints=3001, tol=0.01):

        energies    = self.mbxas["energies"]
        intensities = self.mbxas["absorption"]


        erange, spectras = get_mbxas_spectra(energies, intensities,
                                              sigma=sigma, npoints=npoints,tol=tol)

        if axis is None:
            spectras = np.mean(spectras, axis=0)
        else:
            spectras = spectras[axis]

        return erange, spectras

    @staticmethod
    def __find_delocalized_orbitals(mole, dft_calc, exc_atom, check_deg):

        s1_orbitals = []
        for ii in [0,1]:
            s1orb = find_1s_orbitals_pyscf(mole, dft_calc.mo_coeff[ii],
                                         dft_calc.mo_energy[ii], exc_atom,
                                         check_deg=check_deg)
            s1_orbitals.append(s1orb)

        return s1_orbitals

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