#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 18:13:29 2023

@author: roncoroni
"""

# os 'n similar
import os
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
from pymbxas.io.config import configure_logger
from pymbxas.io import h5
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
                 pbc          = None, # infer from structure, or override with explicit bool
                 solvent      = None,  # specify solvent epsilon
                 calc_type    = "UKS", # UKS or UHF
                 do_xch       = True,  # do XCH to align energy
                 loc_type     = "ibo", # localization routine

                 target_dir   = None, # run the calculation in a target dir

                 xas_verbose  = 3,    # verbose level of pymbxas
                 xas_logfile  = None, # file for mbxas log
                 dft_verbose  = 3,    # verbose level of pyscf
                 dft_logfile  = None, # file for pyscf log
                 dft_output   = True, # print pyscf output or not

                 print_fchk   = False, # print FCHK files as calculation goes

                 save         = True,  # save object as pkl file
                 save_chk     = False, # save calculation as chkfile
                 save_name    = "pymbxas_obj.h5", # name of saved file
                 save_path    = None, # path of saved object
                 gpu          = False,
                 ):

        # store directories and path
        self._cdir = os.getcwd() # current directory
        self._tdir = os.getcwd() if target_dir is None \
            else os.path.abspath(target_dir) # target directory

        if not os.path.exists(self._tdir):
            os.makedirs(self._tdir)

        self._initialize_from_scratch(structure, charge, spin,
                                      xc, basis, pbc, solvent, calc_type,
                                      do_xch, xas_verbose, xas_logfile,
                                      dft_verbose, dft_logfile, dft_output,
                                      print_fchk, save, loc_type,
                                      save_name, save_path, save_chk, gpu)

        return

    def _initialize_from_scratch(self, structure, charge, spin,
                                  xc, basis, pbc, solvent, calc_type,
                                  do_xch, xas_verbose, xas_logfile,
                                  dft_verbose, dft_logfile, dft_output,
                                  print_fchk, save, loc_type,
                                  save_name, save_path, save_chk, gpu):

        # output, verbose and printing
        self._output_settings = {
            "print_fchk"   : print_fchk,
            "xas_verbose"  : xas_verbose,
            "xas_logfile"  : xas_logfile,
            "dft_verbose"  : dft_verbose,
            "dft_logfile"  : dft_logfile,
            "dft_output"   : dft_output,
            "save"         : save,
            "save_chk"     : save_chk,
            "save_path"    : save_path,
            "save_name"    : save_name,
            "is_gpu"       : gpu,
            }
        
        # logger
        configure_logger(xas_verbose, log_file=xas_logfile)
        self.logger = logging.getLogger(__name__)  # Logger tied to this class

        # check (#TODO in future allow to restart from a GS calculation)
        self._ran_GS   = False
        self._used_loc = False # becomes true if localization was needed

        # store calculation details
        self.structure = structure

        # resolve and check PBC
        pbc_resolved = check.check_pbc(pbc, structure)
        if pbc_resolved:
            raise NotImplementedError("MBXAS is not supported under periodic boundary conditions: the position operator used for the transition dipoles (int1e_r) is not periodic, so the lattice-summed integrals are not physically meaningful. Use a molecular or cluster model instead.")

        self._parameters = {
            "charge"   : charge,
            "spin"     : spin,
            "xc"       : xc,
            "basis"    : basis,
            "solvent"  : solvent,
            "pbc"      : pbc_resolved,
            "loc"      : loc_type,
            "xch"      : do_xch,
            "calc_type": calc_type,
            # ... add more parameters as needed
            }

        # initialize empty stuff
        self.output        = {}
        self.data          = {}
        self._excitations  = []
        self._h5_path      = None

        return
    
    # run all pymbxas from scratch
    def kernel(self, excitation):
        """
        Run all pymbxas calculations from scratch.
        
        Parameters:
        excitation (list or int): Atom indices to excite.
        
        Returns:
        None
        """
        
        
        header = """
           |----------------------------------|
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

        # run all specified excitations
        self.excite(excitation)

        # save object if needed
        if self.oset["save"]:
            self.logger.info("Saved everything as {}".format(self.save_object()))
            
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

        try:
            excitation = Excitation(self.structure, self.gs_data, ato_idx,
                                    self.parameters, channel, self.df_obj,
                                    self.oset, self.logger)
            self._excitations.append(excitation)
        except (ValueError, RuntimeError) as e:
            self.logger.error(str(e))

        return
        

    # run the GS calculation
    def run_ground_state(self, force=False):

        # check if GS was already performed, if so: skip
        if self._ran_GS and not force:
            self.logger.warning("GS already ran with this configuration. Skipping.")
            return

        if force and self._excitations:
            raise RuntimeError(
                "Cannot re-run the ground state: {} excitations were computed against "
                "the current one. Start a new calculation instead.".format(
                    len(self._excitations)))

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
        gs_mol = self.make_mol(charge, spin, basis, pbc)
        
        # generate KS calculator
        gs_calc = make_pyscf_calculator(gs_mol, xc, pbc=pbc, solvent=solvent,
                                        calc_type=calc_type, dens_fit=None,
                                        calc_name="gs", save=self.oset["save_chk"],
                                        is_gpu=self.oset["is_gpu"])

        # run SCF #TODO: check how to change convergence parameters
        gs_calc.kernel()
        if not gs_calc.converged:
            raise RuntimeError("Ground state SCF did not converge")

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
        self.mol.stdout.close()
        
        self.logger.info(u"GS finished in {:.1f} s.\n".format(time.time() - start_time))
        return
    
    # run localization procedure
    def _run_localization(self, dft_calc, loc_type):
        """
        Run localization procedure if needed.
        
        Parameters:
        dft_calc: DFT calculation object.
        loc_type (str): Localization type.
        
        Returns:
        None
        """
        
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
            s1_orbitals = [list(range(np.max(orb) + 1)) for orb in s1_orbitals]
            
        
        mo_loc = do_localization_pyscf(dft_calc, s1_orbitals, loc_type)
        
        self.logger.info("{} localization : {}".format(loc_type.upper(), s1_orbitals[1]))
        
        self._used_loc = True
        
        # update MO coeff if localization was used
        self.gs_data.mo_coeff_del = self.gs_data.mo_coeff.copy()
        if self._used_loc:
            self.gs_data.mo_coeff = mo_loc
        
        return
    

    def _print_fchk_files(self):

        # write MOs
        if self._used_loc:
            write_data_to_fchk(self.mol,
                               mo_coeff = self.gs_data.mo_coeff_del,
                               oname    = self._tdir + "/output_gs_del.fchk",
                               )

            write_data_to_fchk(self.mol,
                               mo_coeff = self.gs_data.mo_coeff,
                               oname    = self._tdir + "/output_gs_loc.fchk",
                               )

        else:
            write_data_to_fchk(self.mol,
                               mo_coeff = self.gs_data.mo_coeff,
                               oname    = self._tdir + "/output_gs.fchk",
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
        intensities = np.concatenate(intensities, axis=1)**2  # |d|² per axis

        erange, spectras = get_mbxas_spectra(energies, intensities,
                                              sigma=sigma, npoints=npoints,
                                              tol=tol, erange=erange)

        if axis is None:
            spectras = np.mean(spectras, axis=0)
        else:
            spectras = spectras[axis]

        return erange, spectras

    def save_object(self, oname=None, save_path=None):
        """
        Write the calculation to an HDF5 file, appending any excitation that
        is not on disk yet.

        Returns:
        str: path of the file written.
        """

        if not self._ran_GS:
            raise RuntimeError("Cannot save before the ground state has been run.")

        path = self._resolve_save_path(oname, save_path)

        if not os.path.exists(path):
            self._write_header(path)

        self._append_excitations(path)
        self._h5_path = path

        return path

    def _resolve_save_path(self, oname, save_path):

        if oname is None:
            oname = self.oset["save_name"]

        root = self._tdir if save_path is None else save_path

        if not oname.endswith(".h5"):
            oname = os.path.splitext(oname)[0] + ".h5"

        return os.path.join(root, oname)

    def _write_header(self, path):

        with h5.create(path, h5.KIND_CALCULATION) as f:
            f.attrs["ran_GS"]   = bool(self._ran_GS)
            f.attrs["used_loc"] = bool(self._used_loc)

            h5.write_structure(f, "structure", self.structure)
            h5.write_json(f, "parameters", self._parameters)
            h5.write_json(f, "output_settings", self._output_settings)
            h5.write_text(f, "output", self.output if isinstance(self.output, str) else "")
            h5.write_snapshot(f, self.gs_data)

            f.create_group("excitations")

        return

    def _append_excitations(self, path):

        with h5.append(path) as f:
            root = f["excitations"]

            for idx, exc in enumerate(self._excitations):
                key = "{:03d}".format(idx)

                if key in root:
                    if root[key].attrs.get("complete", False):
                        continue
                    del root[key]

                group = root.create_group(key)
                group.attrs["ato_idx"] = int(exc.ato_idx)
                group.attrs["symbol"]  = exc.symbol
                group.attrs["channel"] = int(exc.channel)
                group.attrs["orb_idx"] = int(exc.orb_idx)

                for name in ("fch", "xch"):
                    if name not in exc.data:
                        continue
                    sub = group.create_group(name)
                    h5.write_snapshot(sub, exc.data[name])
                    h5.write_text(sub, "output", exc.output.get(name, ""))

                mbxas = group.create_group("mbxas")
                for name, value in exc.mbxas.items():
                    h5.write_array(mbxas, name, np.asarray(value))

                group.attrs["complete"] = True

        return
    
    # restart object from pkl file previously saved
    @classmethod
    def load(cls, filename):
        """Reopen a calculation previously written with save_object()."""

        obj = cls.__new__(cls)
        obj._load_h5(filename)
        return obj

    def _load_h5(self, filename):

        path = os.path.abspath(filename)

        with h5.open_read(path, h5.KIND_CALCULATION) as f:
            self.structure        = h5.read_structure(f, "structure")
            self._parameters      = h5.read_json(f, "parameters")
            self._output_settings = h5.read_json(f, "output_settings")
            self.output           = h5.read_text(f, "output")
            self._ran_GS          = bool(f.attrs["ran_GS"])
            self._used_loc        = bool(f.attrs["used_loc"])

            complete   = []
            incomplete = []
            for key in sorted(f["excitations"]):
                if f["excitations"][key].attrs.get("complete", False):
                    complete.append("excitations/" + key)
                else:
                    incomplete.append(key)

        configure_logger(self._output_settings["xas_verbose"],
                         log_file=self._output_settings["xas_logfile"])
        self.logger = logging.getLogger(__name__)

        for key in incomplete:
            self.logger.warning("Skipping incomplete excitation {} in {}".format(key, path))

        self.gs_data     = h5.read_snapshot(path, "/")
        self.mol         = self.gs_data.mol
        self.mol.verbose = self._output_settings["dft_verbose"]
        self.df_obj      = None

        self._cdir        = os.getcwd()
        self._tdir        = os.path.dirname(path) or os.getcwd()
        self._h5_path     = path
        self._excitations = [Excitation.from_h5(path, key) for key in complete]

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
    
    # convert object to Spectra object
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
        
    # internal function to generate a pyscf mol obj
    def make_mol(self, charge, spin, basis, pbc):
        
        mol = ase_to_mole(self.structure, charge, spin, basis=basis, pbc=pbc,
                             verbose=self.oset["dft_verbose"],
                             print_output=self.oset["dft_output"],
                             log_file=self.oset["dft_logfile"],
                             is_gpu=self.oset["is_gpu"])
        
        return mol
