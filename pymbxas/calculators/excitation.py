#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 13:17:41 2024

@author: roncoroni
"""

import time
import copy

# self module utilities
from pymbxas.io.data import pyscf_data
from pymbxas.build.structure import ase_to_mole
from pymbxas.build.input_pyscf import make_pyscf_calculator
from pymbxas.utils.orbitals import find_1s_orbitals_pyscf
from pymbxas.mbxas.mbxas import run_MBXAS_pyscf

# pyscf stuff
from pyscf.scf.addons import mom_occ

# MOKIT stuff
from pymbxas.io.write import write_data_to_fchk
    
#%%
    
class Excitation(object):
    
    def __init__(self, structure, gs_data, ato_idx, parameters, channel,
                 df_obj, oset, logger):
        
        # output stuff
        self.oset   = oset
        self.logger = logger
        
        # set up excitation info
        self.ato_idx   = ato_idx
        self.channel   = channel
        
        # store output
        self.output = {}
        self.data   = {}
        
        # find index of orbital to excite
        self.orb_idx = find_1s_orbitals_pyscf(gs_data.mol,
                                              gs_data.mo_coeff[channel],
                                              gs_data.mo_energy[channel],
                                              gs_data.mo_occ[channel],
                                              [ato_idx],
                                              check_deg=False)
        
        # make it work even if it uses GPU
        if self.oset["is_gpu"]:
            gs_data = gs_data.to_gpu()
        
        if len(self.orb_idx) > 1:
            self.logger.error("It seems that the atomic orbitals are still delocalized.")
            return
        
        # run excitation
        self._excite(structure, gs_data, parameters, df_obj)
        
        return
    
    
    # use it to run excitation of the selected atom
    def _excite(self,structure, gs_data, parameters, df_obj):
        
        self.logger.info("------- Exciting atom #{:>2} -------|".format(self.ato_idx))
        
        # run FCH
        self._run_fch(structure, gs_data, parameters, df_obj)
        
        # run XCH
        self._run_xch(structure, gs_data, parameters, df_obj)
        
        # run MBXAS
        self._run_mbxas(gs_data)
        
        return
    
    # run the FCH calculation
    def _run_fch(self, structure, gs_data, parameters, df_obj):
        
        self.logger.info(">>> Started FCH calculation.")
        
        # retrieve parameters
        pbc       = parameters["pbc"]
        charge    = parameters["charge"] + 1 if not pbc else 0 #TODO: check if works properly for PBC
        spin      = parameters["spin"] + self.channel*2 - 1 if not pbc else 0 #TODO: check if works properly for PBC
        basis     = parameters["basis"]
        xc        = parameters["xc"]
        solvent   = parameters["solvent"]

        start_time  = time.time()

        # Read MO coefficients and occupation number from GS
        scf_guess  = copy.deepcopy(gs_data.mo_coeff)
        occupation = copy.deepcopy(gs_data.mo_occ)

        # Assign initial occupation pattern --> kick orbital N
        occupation[self.channel][self.orb_idx] = 0

        # change charge
        fch_mol = ase_to_mole(structure, charge=charge, spin=spin, basis=basis,
                              pbc=pbc, verbose=self.oset["verbose"],
                              print_output=self.oset["print_output"])

        # Defnine new SCF calculator
        fch_calc = make_pyscf_calculator(fch_mol, xc, pbc=pbc, solvent=solvent,
                                         dens_fit=df_obj, calc_name="fch",
                                         save=self.oset["save_chk"],
                                         gpu=self.oset["is_gpu"],)

        # Construct new density matrix with new occupation pattern
        dm_u = fch_calc.make_rdm1(scf_guess, occupation)

        # Apply mom occupation principle
        fch_calc = mom_occ(fch_calc, scf_guess, occupation)

        # Start new SCF with new density matrix
        fch_calc.scf(dm_u)

        # store input/output
        self.output["fch"] = fch_calc.stdout.log.getvalue()
        self.data["fch"]   = pyscf_data(fch_calc)

        # if pobj._print_fchk:
        #     write_data_to_fchk(pobj.mol, fch_calc,
        #                        oname = pobj._tdir + "/output_fch_{}.fchk".format(
        #                            self.ato_idx),
        #                        )
            
        self.logger.info(">>> FCH finished in {:.1f} s.".format(time.time() - start_time))
        return

    # run the XCH calculation
    def _run_xch(self, structure, gs_data, parameters, df_obj):
        
        self.logger.info(">>> Started XCH calculation.")
        
        # retrieve parameters
        pbc       = parameters["pbc"]
        charge    = parameters["charge"] 
        spin      = parameters["spin"]
        basis     = parameters["basis"]
        xc        = parameters["xc"]
        solvent   = parameters["solvent"]
        
        if self.oset["is_gpu"]:
            data = self.data["fch"].to_gpu()
        else:
            data = self.data["fch"]
        
        start_time = time.time()

        # Read MO coefficients and occupation number from GS
        scf_guess  = copy.deepcopy(data.mo_coeff)
        occupation = copy.deepcopy(data.mo_occ)

        # Assign initial occupation pattern --> kick orbital N
        # occupation[channel][self.orb_idx] = 0
        occupation[self.channel][gs_data.nelec[self.channel]] = 1

        # make XCH molecule
        xch_mol = ase_to_mole(structure, charge=charge, spin=spin, basis=basis,
                              pbc=pbc, verbose=self.oset["verbose"],
                              print_output=self.oset["print_output"],)

        # define new SCF calculator
        xch_calc = make_pyscf_calculator(xch_mol, xc, pbc=pbc, solvent=solvent,
                                         dens_fit=df_obj, calc_name="xch",
                                         save=self.oset["save_chk"],
                                         gpu=self.oset["is_gpu"])

        # Construct new density matrix with new occupation pattern
        dm_u = xch_calc.make_rdm1(scf_guess, occupation)

        # Apply mom occupation principle
        xch_calc = mom_occ(xch_calc, scf_guess, occupation)

        # Start new SCF with new density matrix
        xch_calc.scf(dm_u)
        
        # reconvert to cpu cause otherwise stuff is iffy
        # if self.oset["is_gpu"]:
            # xch_calc = xch_calc.to_cpu()

        # store input/output
        self.output["xch"] = xch_calc.stdout.log.getvalue()
        self.data["xch"]   = pyscf_data(xch_calc)

        # if pobj._print_fchk:
        #     write_data_to_fchk(pobj.mol, xch_calc,
        #                        oname = pobj._tdir + "/output_xch_{}.fchk".format(
        #                            self.ato_idx),
        #                        )

        self.logger.info(">>> XCH finished in {:.1f} s.".format(time.time() - start_time))
        
        return
    
    # run MBXAS from a set of pySCF calculations
    def _run_mbxas(self, gs_data):
        
        start_time = time.time()

        energies, absorption, mb_ovlp, dip_KS, b_ovlp = run_MBXAS_pyscf(
            gs_data.mol, gs_data.to_cpu(), self.data["fch"].to_cpu(),
            self.orb_idx, channel=self.channel, xch_calc=self.data["xch"].to_cpu())

        self.mbxas = {
            "energies"   : energies,
            "absorption" : absorption,
            "mb_overlap" : mb_ovlp,
            "dipole_KS"  : dip_KS,
            "basis_ovlp" : b_ovlp
            }
        
        self.logger.info(">>> MBXAS finished in {:.1f} s [\u2713].".format(time.time() - start_time))
        self.logger.info("---------------------------------|\n")
        return