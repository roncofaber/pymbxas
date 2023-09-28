#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 14:49:52 2023

@author: roncoroni
"""

import time
import copy

# self module utilities
from pymbxas.io.pyscf import parse_pyscf_calculator
from pymbxas.build.structure import ase_to_mole
from pymbxas.build.input_pyscf import make_pyscf_calculator
from pymbxas.utils.orbitals import find_1s_orbitals_pyscf
from pymbxas.mbxas.mbxas import run_MBXAS_pyscf

# pyscf stuff
from pyscf.scf.addons import mom_occ

# MOKIT stuff
try:
    from mokit.lib.py2fch_direct import fchk as write_to_fchk
    is_mokit = True
except:
    is_mokit = False
    
#%%
    
class Excitation(object):
    
    def __init__(self, ato_idx, pobj, channel=1):
        
        assert pobj._ran_GS, "Please run a GS calculation first."
        
        self._print_all = pobj._print_all
        
        self.ato_idx   = ato_idx
        self.channel   = channel
        
        # store output
        self.output = {}
        self.data   = {}
        
        # find index of orbital to excite
        self.orb_idx = find_1s_orbitals_pyscf(pobj.data.mol,
                                              pobj.data.mo_coeff[channel],
                                              pobj.data.mo_energy[channel],
                                              pobj.data.mo_occ[channel],
                                              [ato_idx],
                                              check_deg=False)
        
        if len(self.orb_idx) > 1:
            self.print("It seems that the atomic orbitals are still delocalized.")
            return
        
        # run excitation
        self._excite(pobj)
        
        return
    
    # use it to run excitation of the selected atom
    def _excite(self, pobj):
        
        self.print(">>  Exciting atom #{}  <<".format(self.ato_idx), flush=True)
        
        self.print("> FCH:", end="")
        
        __, ft = self.run_fch(pobj)
        
        self.print(" {:.1f} s | XCH:".format(ft), end="")
        
        __, xt = self.run_xch(pobj)
        
        self.print(" {:.1f} s | MBXAS:".format(xt), end="")
        
        mt = self.run_mbxas(pobj)
        
        self.print(u" {:.1f} s [\u2713]".format(mt))
        
        return
    
    # run the FCH calculation
    def run_fch(self, pobj):
        
        structure = pobj.structure
        charge    = pobj.charge + 1 if not pobj.pbc else 0 #TODO: check if works properly for PBC
        basis     = pobj.basis
        xc        = pobj.xc
        pbc       = pobj.pbc
        solvent   = pobj.solvent
        spin      = pobj.spin + self.channel*2 - 1 if not pobj.pbc else 0 #TODO: check if works properly for PBC
        
        start_time  = time.time()

        # Read MO coefficients and occupation number from GS
        scf_guess  = copy.deepcopy(pobj.data.mo_coeff)
        occupation = copy.deepcopy(pobj.data.mo_occ)

        # Assign initial occupation pattern --> kick orbital N
        occupation[self.channel][self.orb_idx] = 0

        # change charge
        fch_mol = ase_to_mole(structure, charge=charge, spin=spin, basis=basis,
                              pbc=pbc, verbose=pobj._verbose,
                              print_output=pobj._print_output)

        # Defnine new SCF calculator
        fch_calc = make_pyscf_calculator(fch_mol, xc, pbc=pbc, solvent=solvent,
                                         dens_fit=pobj.df_obj, calc_name="fch",
                                         save=pobj._save_chk)

        # Construct new density matrix with new occupation pattern
        dm_u = fch_calc.make_rdm1(scf_guess, occupation)

        # Apply mom occupation principle
        fch_calc = mom_occ(fch_calc, scf_guess, occupation)

        # Start new SCF with new density matrix
        fch_calc.scf(dm_u)

        # store input/output
        self.output["fch"] = fch_calc.stdout.log.getvalue()
        self.data["fch"]   = parse_pyscf_calculator(fch_calc)

        if pobj._print_fchk and is_mokit:
            write_to_fchk(fch_calc, pobj._tdir + "/output_fch_{}.fchk".format(
                self.ato_idx))
            
        
        fch_time = time.time() - start_time
        
        return fch_calc, fch_time

    # run the XCH calculation
    def run_xch(self, pobj):
        
        structure = pobj.structure
        charge    = pobj.charge
        basis     = pobj.basis
        xc        = pobj.xc
        pbc       = pobj.pbc
        solvent   = pobj.solvent
        spin      = pobj.spin
        
        start_time = time.time()

        # Read MO coefficients and occupation number from GS
        scf_guess  = copy.deepcopy(self.data["fch"].mo_coeff)
        occupation = copy.deepcopy(self.data["fch"].mo_occ)

        # Assign initial occupation pattern --> kick orbital N
        # occupation[channel][self.orb_idx] = 0
        occupation[self.channel][pobj.n_electrons[self.channel]] = 1

        # make XCH molecule
        xch_mol = ase_to_mole(structure, charge=charge, spin=spin, basis=basis,
                              pbc=pbc, verbose=pobj._verbose,
                              print_output=pobj._print_output)

        # define new SCF calculator
        xch_calc = make_pyscf_calculator(xch_mol, xc, pbc=pbc, solvent=solvent,
                                         dens_fit=pobj.df_obj, calc_name="xch",
                                         save=pobj._save_chk)

        # Construct new density matrix with new occupation pattern
        dm_u = xch_calc.make_rdm1(scf_guess, occupation)

        # Apply mom occupation principle
        xch_calc = mom_occ(xch_calc, scf_guess, occupation)

        # Start new SCF with new density matrix
        xch_calc.scf(dm_u)

        # store input/output
        self.output["xch"] = xch_calc.stdout.log.getvalue()
        self.data["xch"]   = parse_pyscf_calculator(xch_calc)

        if pobj._print_fchk and is_mokit:
            write_to_fchk(xch_calc, pobj._tdir + "/output_xch_{}.fchk".format(
                self.ato_idx))

        xch_time = time.time() - start_time
        
        return xch_calc, xch_time
    
    # run MBXAS from a set of pySCF calculations
    def run_mbxas(self, pobj):
        
        start_time = time.time()

        energies, absorption, mb_ovlp, dip_KS, b_ovlp = run_MBXAS_pyscf(
            pobj.data, self.data["fch"], self.orb_idx, channel=self.channel,
            xch_calc=self.data["xch"])

        self.mbxas = {
            "energies"   : energies,
            "absorption" : absorption,
            "mb_overlap" : mb_ovlp,
            "dipole_KS"  : dip_KS,
            "basis_ovlp" : b_ovlp
            }
        
        mbxas_time = time.time() - start_time

        return mbxas_time
    
    
    def print(self, toprint, end="\n", flush=True):
        if self._print_all:
            print(toprint, end=end, flush=flush)