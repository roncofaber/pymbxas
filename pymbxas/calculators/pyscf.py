#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 18:13:29 2023

@author: roncoroni
"""

# os 'n similar
import os
import dill
import time
import logging

# good ol' numpy
import numpy as np

# self module utilities
import pymbxas
from pymbxas.calculators.excitation import Excitation
import pymbxas.utils.check_keywords as check
from pymbxas.utils.auxiliary import as_list, change_key
from pymbxas.utils.indexing import atoms_to_indexes
from pymbxas.io.data import pyscf_data
from pymbxas.io.logger import configure_logger
from pymbxas.build.structure import ase_to_mole
from pymbxas.build.input_pyscf import make_pyscf_calculator
from pymbxas.utils.orbitals import find_1s_orbitals_pyscf
from pymbxas.utils.boys import do_localization_pyscf
from pymbxas.mbxas.broaden import get_mbxas_spectra
from pymbxas.io.cleanup import remove_tmp_files
from pymbxas.io.write import write_data_to_fchk
from pymbxas import Spectra, Spectras

# ase
from ase import units
Ha = units.Ha

#%%

class PySCF_mbxas():
    
    def __init__(self,
                 structure    = None,
                 charge       = 0,
                 spin         = 0,
                 xc           = "b3lyp",
                 basis        = "def2-svpd",
                 pbc          = False,
                 solvent      = None,
                 calc_type    = "UKS",
                 do_xch       = True,
                 pkl_file     = None,
                 target_dir   = None,
                 verbose      = 4,
                 print_fchk   = True,
                 print_output = False,
                 save         = True,  # save object as pkl file
                 save_chk     = False, # save calculation as chkfile
                 save_name    = "pyscf_obj.pkl", # name of saved file
                 save_path    = None, # path of saved object
                 loc_type     = "ibo",
                 gpu          = False,
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
            self._initialize_from_scratch(structure, charge, spin,
                                          xc, basis, pbc, solvent, calc_type,
                                          do_xch, verbose, print_fchk,
                                          print_output, save, loc_type,
                                          save_name, save_path, save_chk, gpu)
        
        return

    def _initialize_from_scratch(self, structure, charge, spin,
                                  xc, basis, pbc, solvent, calc_type,
                                  do_xch, verbose, print_fchk, print_output,
                                  save, loc_type, save_name, save_path,
                                  save_chk, gpu):

        # output, verbose and printing
        self._output_settings = {
            "print_fchk"   : print_fchk,
            "print_output" : print_output,
            "verbose"      : verbose,
            "save"         : save,
            "save_chk"     : save_chk,
            "save_path"    : save_path,
            "save_name"    : save_name,
            "is_gpu"       : gpu,
            }
        
        # logger
        configure_logger(verbose)
        self.logger = logging.getLogger(__name__)  # Logger tied to this class

        # check (#TODO in future allow to restart from a GS calculation)
        self._ran_GS   = False
        self._used_loc = False # becomes true if localization was needed

        # store calculation details
        self.structure = structure
        
        self._parameters = {
            "charge"   : charge,
            "spin"     : spin,
            "xc"       : xc,
            "basis"    : basis,
            "solvent"  : solvent,
            "pbc"      : check.check_pbc(pbc, structure),
            "loc"      : loc_type,
            "xch"      : do_xch,
            "calc_type": calc_type,
            # ... add more parameters as needed
            }
        
        # initialize empty stuff
        self.output        = {}
        self.data          = {}
        self._excitations  = []
              
        return
    
    # run all pymbxas from scratch
    def kernel(self, excitation):
        
        
        header = """----------------------------------|
           |                                  |
           |>>>>>>   Starting PyMBXAS   <<<<<<|
           |                                  |
           |       ver {:>7} | {:<12} |
           |----------------------------------|
        """.format(pymbxas.__version__, pymbxas.__date__)
        
        self.logger.info(header)
        
        # change directory
        os.chdir(self._tdir)
        
        # run ground state
        self.run_ground_state()
        
        # save object if needed
        if self.oset["save"]:
            self.save_object()
        
        # run all specified excitations
        self.excite(excitation)
        
        # save object if needed
        if self.oset["save"]:
            self.save_object()
            
            self.logger.info("Saved everything as {}".format(self.oset["save_name"]))
            
        # go back where we were
        os.chdir(self._cdir)
        
        self.logger.info("PyMBXAS finished successfully!")
        
        # clean up mess
        remove_tmp_files(self._tdir)
        
        return

    # excite an atom or a list of atoms
    def excite(self, ato_idxs, channel=1):
        
        if not self._ran_GS:
            self.logger.error("Please run a GS calculation first.")
            return
            
        
        # convert into atom indexes
        to_excite = atoms_to_indexes(self.structure, ato_idxs)
        
        # iterate over the indexes 
        for ato_idx in to_excite:
            self._single_excite(ato_idx, channel)
            
            # save object if needed
            if self.oset["save"]:
                self.save_object()
        
        return
    
    # perform a single excitation 
    def _single_excite(self, ato_idx, channel):
         
        if ato_idx in self.excited_idxs:
            return 
            
        excitation = Excitation(self.structure, self.gs_data, ato_idx,
                                self.parameters, channel, self.df_obj,
                                self.oset, self.logger)
        
        self._excitations.append(excitation)
            
        return
        

    # run the GS calculation
    def run_ground_state(self, force=False):
        
        # check if GS was already performed, if so: skip
        if self._ran_GS and not force:
            self.logger.warn("GS already ran with this configuration. Skipping.")
            return
        self.logger.info("Started a new GS calculation")
        
        start_time  = time.time()

        # get system settings
        charge    = self.parameters["charge"]
        spin      = self.parameters["spin"]
        basis     = self.parameters["basis"]
        xc        = self.parameters["xc"]
        pbc       = self.parameters["pbc"]
        solvent   = self.parameters["solvent"]
        calc_type = self.parameters["calc_type"]
        
        # generate molecule
        gs_mol = ase_to_mole(self.structure, charge, spin, basis=basis, pbc=pbc,
                             verbose=self.oset["verbose"],
                             print_output=self.oset["print_output"],
                             is_gpu=self.oset["is_gpu"])
        
        # generate KS calculator
        gs_calc = make_pyscf_calculator(gs_mol, xc, pbc=pbc, solvent=solvent,
                                        calc_type=calc_type, dens_fit=None,
                                        calc_name="gs", save=self.oset["save_chk"],
                                        gpu=self.oset["is_gpu"])

        # run SCF #TODO: check how to change convergence parameters
        gs_calc.kernel()
        
        # store input/output
        self.output  = gs_calc.stdout.log.getvalue()
        self.gs_data = pyscf_data(gs_calc)
        self.mol     = gs_mol
    
        # store density object
        if pbc:
            # generate density fitter
            self.df_obj = gs_calc.with_df
        else:
            self.df_obj = None

        # check if localization is needed and run it
        self._run_localization(self.gs_data, self.parameters["loc"])
        
        # write output fchk files if using mokit
        if self.oset["print_fchk"]:
            self._print_fchk_files()

        # mark that GS has been run
        self._ran_GS = True
        
        self.logger.info(u"GS finished in {:.1f} s.\n".format(time.time() - start_time))
        return
    
    # run localization procedure
    def _run_localization(self, dft_calc, loc_type):
        
        # define list of relevant atoms
        ato_idxs = [cc for cc, ato in enumerate(self.structure) if ato.symbol != "H"]
        
        # check for degenerate delocalized orbitals and if necessary, do Boys
        s1_orbitals = []
        for ii in [0,1]:
            s1orb = find_1s_orbitals_pyscf(dft_calc.mol, dft_calc.mo_coeff[ii],
                                         dft_calc.mo_energy[ii],
                                         dft_calc.mo_occ[ii], as_list(ato_idxs),
                                         check_deg=True)
            s1_orbitals.append(s1orb)
        
        # only one s orb, no loc needed
        if len(s1_orbitals[1]) <= 1:
            return dft_calc.mo_coeff, False
        
        # localize up to highest degenerate orbital #TEST
        if loc_type.endswith("m"):
            s1_orbitals = [list(range(np.max(orb))) for orb in s1_orbitals]
            
        
        mo_loc = do_localization_pyscf(dft_calc, s1_orbitals, loc_type)
        
        self.logger.info("{} localization : {}".format(loc_type.upper(), s1_orbitals[1]))
        
        self._used_loc = True
        
        # update MO coeff if localization was used
        if self._used_loc:
            self.gs_data.mo_coeff_del = self.gs_data.mo_coeff.copy()
            self.gs_data.mo_coeff     = mo_loc
        
        return
    

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
    

    def get_mbxas_spectra(self, ato_idx, axis=None, sigma=0.5, npoints=3001, tol=0.01,
                          erange=None):
        
        ato_idxs = atoms_to_indexes(self.structure, ato_idx)
        
        energies    = []
        intensities = []
        for exc in self.excitations:
            if exc.ato_idx not in ato_idxs:
                continue
            energies.append(exc.mbxas["energies"])
            intensities.append(exc.mbxas["absorption"])
            
        energies    = Ha*np.concatenate(energies)
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
        
        if oname is None:
            oname = self.oset["save_name"]

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

        return
    
    # restart object from pkl file previously saved
    def _restart_from_pickle(self, pkl_file):

        # open previously generated gpw file
        with open(pkl_file, "rb") as fin:
            restart = dill.load(fin)
        
        # convert to dict
        data = restart.__dict__.copy()
            
        # make compatible with older version of pymbxas (<= 0.5.0)
        for old_key in ["excitations"]:
            if old_key in data:
                new_key = "_" + old_key
                change_key(data, old_key, new_key)
        for del_key in ["excited_idxs"]:
            data.pop(del_key, None)

        self.__dict__ = data

        return
    
    @property
    def oset(self):
        return self._output_settings.copy()
    
    @property
    def parameters(self):
        return self._parameters.copy()
    
    @property
    def excitations(self):
        return self._excitations
    
    @property
    def excited_idxs(self):
        return [exc.ato_idx for exc in self.excitations]
    
    def to_spectra(self, excitation=None):
        
        if excitation is None:
            indexes = list(range(len(self.excitations)))
        else:
            indexes = as_list(excitation)
            
        spectras = [Spectra(self, excitation=cc) for cc in indexes]
        
        if len(spectras) == 1:
            return spectras[0]
        else:
            return Spectras(spectras)
