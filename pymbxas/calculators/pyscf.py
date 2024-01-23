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
# import pymbxas
from pymbxas.excitation import Excitation
import pymbxas.utils.check_keywords as check
from pymbxas.utils.auxiliary import as_list
from pymbxas.utils.indexing import atoms_to_indexes
from pymbxas.io.pyscf import pyscf_data
from pymbxas.build.structure import ase_to_mole
from pymbxas.build.input_pyscf import make_pyscf_calculator
from pymbxas.utils.orbitals import find_1s_orbitals_pyscf
from pymbxas.utils.boys import do_localization_pyscf
from pymbxas.mbxas.broaden import get_mbxas_spectra
from pymbxas.io.cleanup import remove_tmp_files
from pymbxas.io.write import write_data_to_fchk

# pyscf stuff
from pyscf.scf.addons import mom_occ
from pyscf import lo

#%%

class PySCF_mbxas():

    def __init__(self,
                 structure    = None,
                 charge       = None,
                 spin         = None,
                 pkl_file     = None,
                 excitation   = None,
                 xc           = "b3lyp",
                 basis        = "def2-svpd",
                 pbc          = None,
                 solvent      = None,
                 do_xch       = True,
                 target_dir   = None,
                 verbose      = 4,
                 print_fchk   = True,
                 print_output = False,
                 run_calc     = True,
                 save         = True,  # save object as pkl file
                 save_chk     = False, # save calculation as chkfile
                 save_name    = "pyscf_obj.pkl", # name of saved file
                 save_path    = None, # path of saved object
                 loc_type     = "ibo",
                 save_calc    = False, # store calculator in object (large size!)
                 ):
        
        # restart object from a pkl file and terminate
        if pkl_file is not None:
            self._restart_from_pickle(pkl_file)
            return
        
        # store directories and path
        self._cdir = os.getcwd() # current directory
        self._tdir = os.getcwd() if target_dir is None \
            else os.path.abspath(target_dir) # target directory

        if not os.path.exists(self._tdir):
            os.makedirs(self._tdir)


        if pkl_file is None:
            self._initialize_from_scratch(structure, charge, spin, excitation,
                                          xc, basis, pbc, solvent, do_xch,
                                          verbose, print_fchk, print_output,
                                          save, loc_type, save_name, save_path,
                                          save_chk, save_calc)
        
        return

    def _initialize_from_scratch(self, structure, charge, spin, excitation,
                                  xc, basis, pbc, solvent, do_xch, verbose,
                                  print_fchk, print_output, save, loc_type,
                                  save_name, save_path, save_chk, save_calc):

        # output, verbose and printing
        self._print_fchk   = print_fchk
        self._print_output = print_output
        self._verbose      = verbose
        self._save         = save
        self._save_chk     = save_chk
        self._print_all    = True # for the moment print all
        self._save_path    = save_path
        self._save_name    = save_name
        self._save_calc    = save_calc

        # check (#TODO in future allow to restart from a GS calculation)
        self._ran_GS   = False
        self._used_loc = False # becomes true if localization was needed

        # store calculation details
        self.structure = structure
        self.charge    = charge
        self.spin      = spin
        self.xc        = xc
        self.basis     = basis
        self.solvent   = solvent
        self.pbc       = check.check_pbc(pbc, structure)
        
        self._is_xch  = do_xch
        self.loc_type = loc_type

        # initialize empty stuff
        self.output      = {}
        self.data        = {}
        self.excitations = []
        self.excited_idxs = []
        
        # run everything
        self.run_all_calculations(excitation)
        
        return

    # Function to run all calc (GS, FCH, XCH) in sequence
    def run_all_calculations(self, excitation):
        
        self.print(">>> Starting PyMBXAS <<<")
        
        # change directory
        os.chdir(self._tdir)
        
        self.print(">> GS:", end="")
        
        # run ground state
        gt = self.run_ground_state()
        
        self.print(u" {:.1f} s [\u2713] <<".format(gt))
        
        # save object if needed
        if self._save:
            self.save_object(self._save_name, self._save_path)
        
        # run all specified excitations
        self.excite(excitation)
        
        # save object if needed
        if self._save:
            self.save_object(self._save_name, self._save_path)
            
        self.print("Saved everything as {}".format(self._save_name))
            
        # go back where we were
        os.chdir(self._cdir)
        
        self.print(">>> PyMBXAS finished successfully! <<<\n")
        
        # clean up mess
        remove_tmp_files(self._tdir)
        
        return
    
    # restart object from pkl file previously saved
    def _restart_from_pickle(self, pkl_file):

        # open previously generated gpw file
        with open(pkl_file, "rb") as fin:
            restart = dill.load(fin)

        self.__dict__ = restart.__dict__.copy()

        return
    
    # excite an atom or a list of atoms
    def _single_excite(self, ato_idx):
        
        if ato_idx in self.excited_idxs:
           return 
            
        self.excited_idxs.append(ato_idx)
        self.excitations.append(Excitation(ato_idx, self))
            
        return
    
    def excite(self, ato_idxs):
        
        to_excite = atoms_to_indexes(self.structure, ato_idxs)
        
        for ato_idx in to_excite:
            self._single_excite(ato_idx)
            
            # save object if needed
            if self._save:
                self.save_object(self._save_name, self._save_path)
        
        return
        

    # run the GS calculation
    def run_ground_state(self):
        
        # check if GS was already performed, if so: skip
        if self._ran_GS:
            self.print("GS already ran with this configuration. Skipping.")
            return
        
        start_time  = time.time()

        # get system settings
        structure = self.structure
        charge    = self.charge
        spin      = self.spin
        basis     = self.basis
        xc        = self.xc
        pbc       = self.pbc
        solvent   = self.solvent
        
        # generate molecule
        gs_mol = ase_to_mole(structure, charge, spin, basis=basis, pbc=self.pbc,
                             verbose=self._verbose, print_output=self._print_output)
        
        # generate KS calculator
        gs_calc = make_pyscf_calculator(gs_mol, xc, pbc=pbc, solvent=solvent,
                                        dens_fit=None, calc_name="gs",
                                        save=self._save_chk)

        # run SCF #TODO: check how to change convergence parameters
        gs_calc.kernel()
        
        # store input/output
        self.output = gs_calc.stdout.log.getvalue()
        self.data   = pyscf_data(gs_calc)
        self.mol    = gs_mol
        
        # store density object
        if self.pbc:
            # generate density fitter
            self.df_obj = gs_calc.with_df
        else:
            self.df_obj = None

        # obtain number of electrons
        self.n_electrons = [gs_calc.nelec[0], gs_calc.nelec[1]]

        # check if localization is needed and run it
        ato_list = [cc for cc, ato in enumerate(structure) if ato.symbol != "H"]
        
        mo_loc, self._used_loc = self._run_localization(gs_mol, gs_calc,
                                                        ato_list, self.loc_type)
        
        # update MO coeff if localization was used
        if self._used_loc:
            self.data.mo_coeff_del = self.data.mo_coeff.copy()
            self.data.mo_coeff = mo_loc
        
        # calculate LIVVOs
        self.data.mo_livvo = self._calculate_livvo()
        
        # write output fchk files if using mokit
        if self._print_fchk:
            self._print_fchk_files()
            
        # store calculator (cannot be saved as pkl anyway, just for on the fly)
        if self._save_calc:
            self.data.calc = gs_calc

        # mark that GS has been run
        self._ran_GS = True
        
        gs_time = time.time() - start_time

        return gs_time
    
    def _run_localization(self, mole, dft_calc, ato_idxs, loc_type):
        
        # check for degenerate delocalized orbitals and if necessary, do Boys
        s1_orbitals = []
        for ii in [0,1]:
            s1orb = find_1s_orbitals_pyscf(mole, dft_calc.mo_coeff[ii],
                                         dft_calc.mo_energy[ii],
                                         dft_calc.mo_occ[ii], as_list(ato_idxs),
                                         check_deg=True)
            s1_orbitals.append(s1orb)
        
        if len(s1_orbitals[1]) <= 1:
            return dft_calc.mo_coeff, False
        
        self.print(" loc: {}".format(s1_orbitals[1]), end="")
        
        # localize up to highest degenerate orbital #TEST
        if loc_type.endswith("m"):
            s1_orbitals = [list(range(np.max(orb))) for orb in s1_orbitals]
            
        used_loc = True
        mo_loc = do_localization_pyscf(dft_calc, s1_orbitals, loc_type)
        
        return mo_loc, used_loc
    
    def _calculate_livvo(self, locmethod='IBO', iaos=None, s=None,
              exponent=4, grad_tol=1e-8, max_iter=200, verbose=None): #TODO only for channel 1 (but should be same)
        
        mo_coeff = self.data.mo_coeff[1].copy()
        mo_occ   = self.data.mo_occ[1].copy()
        
        occ_idxs = np.where(mo_occ > 0)[0]
        uno_idxs = np.where(mo_occ == 0)[0]
        
        mo_vvo = lo.vvo.livvo(self.mol,
                              mo_coeff[:, occ_idxs], mo_coeff[:, uno_idxs],
                              locmethod=locmethod,
                              iaos=iaos, s=s,
                              exponent=exponent, grad_tol=grad_tol,
                              max_iter=max_iter, verbose=verbose)
        
        return mo_vvo
    
    def _print_fchk_files(self):
        
    # write MOsmo_occ
        if self._used_loc:
        
            write_data_to_fchk(self.mol,
                               self.data,
                               oname    = self._tdir + "/output_gs_del.fchk",
                               mo_coeff = self.data.mo_coeff_del
                               )
            
            write_data_to_fchk(self.mol,
                               self.data,
                               oname    = self._tdir + "/output_gs_loc.fchk",
                               mo_coeff = self.data.mo_coeff
                               )
        
        else:
            write_data_to_fchk(self.mol,
                               self.data,
                               oname    = self._tdir + "/output_gs.fchk",
                               mo_coeff = self.data.mo_coeff
                               )
        
        # write LIVVOs
        mo_vvo = np.zeros(self.data.mo_coeff.shape)
        for channel in [0,1]:
            
            shape = self.data.mo_livvo.shape[1]
            
            mo_vvo[channel][:,:shape] = self.data.mo_livvo
        
        write_data_to_fchk(self.mol,
                           self.data,
                           oname    = self._tdir + "/livvos.fchk",
                           mo_coeff = self.data.mo_livvo
                           )
            
        return
    
    def get_excited_orbital_character(self, ato_idx, livvo=None):
        
        if livvo is None:
            livvo = self.data.mo_livvo
            
        basis_ovlp = self.mol.intor("int1e_ovlp")
        
        ato_idxs = atoms_to_indexes(self.structure, ato_idx)
        
        projections = []
        for excitation in self.excitations:
            if excitation.ato_idx not in ato_idxs:
                continue
            
            data = excitation.data["fch"]
            
            # get unoccupied orbitals
            uno_idxs = np.where(data.mo_occ[excitation.channel] == 0)[0][1:]
            CMO      = data.mo_coeff[excitation.channel][:, uno_idxs]
            
            # calculate projection of CMOs on LIVVOs basis
            projection = (CMO.conj().T@basis_ovlp@livvo)**2
            
            projections.append(projection)
        
        if len(ato_idxs) == 1:
            return projections[0]
        else:
            return projections

    
    def get_mbxas_spectra(self, ato_idx, axis=None, sigma=0.02, npoints=3001, tol=0.01,
                          erange=None):
        
        ato_idxs = atoms_to_indexes(self.structure, ato_idx)
        
        energies    = []
        intensities = []
        for exc in self.excitations:
            if exc.ato_idx not in ato_idxs:
                continue
            energies.append(exc.mbxas["energies"])
            intensities.append(exc.mbxas["absorption"])
            
        energies = np.concatenate(energies)
        intensities = np.concatenate(intensities, axis=1)
        
        erange, spectras = get_mbxas_spectra(energies, intensities,
                                              sigma=sigma, npoints=npoints,
                                              tol=tol, erange=erange)

        if axis is None:
            spectras = np.mean(spectras, axis=0)
        else:
            spectras = spectras[axis]

        return erange, spectras

    # save object as pkl file
    def save_object(self, oname=None, save_path=None):
        
        if self._save_calc:
            
            tmp_calc = self.data.calc
            self.data.calc = None
            
            # self.print("Attention, cannot save pkl file if 'save_calc' is True.")

        if oname is None:
            oname = self._save_name

        if save_path is None:
            path = self._tdir
        else:
            path = save_path

        if oname.endswith(".pkl"):
            oname =  path + "/" + oname
        else:
            oname = path + "/" + oname.split(".")[-1] + ".pkl"

        with open(oname, 'wb') as fout:
            dill.dump(self, fout)
            
        if self._save_calc:
            self.data.calc = tmp_calc

        return
    
    def print(self, toprint, end="\n", flush=True):
        if self._print_all:
            print(toprint, end=end, flush=flush)