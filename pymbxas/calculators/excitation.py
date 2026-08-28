#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 13:17:41 2024

@author: roncoroni
"""

import time
import copy
import os

# self module utilities
from pymbxas.io.data import pyscf_data
from pymbxas.io import h5
from pymbxas.config import CalculationConfig, ExcitationConfig, RuntimeConfig
from pymbxas.io.config import (
    format_log_fields, log_scf_completion, with_log_context,
)
from pymbxas.build.structure import ase_to_mole
from pymbxas.build.input_pyscf import make_pyscf_calculator
from pymbxas.utils.orbitals import find_1s_orbitals_pyscf
from pymbxas.mbxas.mbxas import core_hole_index, run_MBXAS_pyscf

# pyscf stuff
from pymbxas.calculators.maxvol import apply_occupation_method
from pymbxas.calculators.scf import run_constrained_scf
import numpy as np
from pyscf.scf import uhf

#%%

CORE_HOLE_OVERLAP_TOL = 0.8
XCH_ORBITAL_OVERLAP_TOL = 0.8
FCH_SUBSPACE_SINGULAR_TOL = 0.8
FCH_SPIN_CONTAMINATION_TOL = 0.1


def _as_numpy(array):
    """Return a NumPy view of a NumPy/CuPy orbital array."""
    return array.get() if hasattr(array, "get") else np.asarray(array)


def _occupation_counts(mo_occ):
    return tuple(int(np.count_nonzero(_as_numpy(channel) == 1))
                 for channel in mo_occ)


def _fch_state_diagnostics(calculator, reference_coeff, target_occ, overlap):
    """Measure spin purity and connection to the target occupied subspace."""
    coefficients = _as_numpy(calculator.mo_coeff)
    occupations = _as_numpy(calculator.mo_occ)
    references = _as_numpy(reference_coeff)
    targets = _as_numpy(target_occ)
    occupied = tuple(
        coefficients[spin][:, occupations[spin] > 0.5]
        for spin in range(2))
    target = tuple(
        references[spin][:, targets[spin] > 0.5]
        for spin in range(2))

    spin_square, multiplicity = uhf.spin_square(occupied, overlap)
    intended_spin = abs(float(calculator.mol.spin)) / 2.0
    ideal_spin_square = intended_spin * (intended_spin + 1.0)

    singular_values = []
    for spin in range(2):
        matrix = target[spin].conj().T @ overlap @ occupied[spin]
        singular_values.append(np.linalg.svd(matrix, compute_uv=False))
    singular_values = np.concatenate(singular_values)
    clipped = np.clip(singular_values, np.finfo(float).tiny, None)
    return {
        "spin_square": float(spin_square),
        "multiplicity": float(multiplicity),
        "ideal_spin_square": ideal_spin_square,
        "spin_contamination": float(spin_square - ideal_spin_square),
        "occupied_determinant": float(np.exp(np.log(clipped).sum())),
        "minimum_singular_value": float(singular_values.min()),
    }


def _fch_core_hole_index(data, gs_data, channel, gs_core_idx, overlap):
    mb_overlap = (
        _as_numpy(data.mo_coeff[channel]).T
        @ overlap
        @ _as_numpy(gs_data.mo_coeff[channel]))
    return core_hole_index(
        mb_overlap, _as_numpy(data.mo_occ[channel]), gs_core_idx)


def _xch_target_index(data, core_hole_idx, channel):
    """Choose the lowest-energy ordinary FCH virtual for the XCH electron."""
    occupation = _as_numpy(data.mo_occ[channel])
    virtual = np.flatnonzero(occupation == 0)
    virtual = virtual[virtual != int(core_hole_idx)]
    if virtual.size == 0:
        raise RuntimeError("Cannot build XCH state: no virtual orbital remains after the core hole")
    energies = _as_numpy(data.mo_energy[channel])
    return int(virtual[np.argmin(energies[virtual])])


def _validate_electron_counts(label, calculator):
    actual = _occupation_counts(calculator.mo_occ)
    expected = tuple(int(value) for value in calculator.mol.nelec)
    if actual != expected:
        raise RuntimeError(
            f"{label} occupation has electron counts {actual}, but the "
            f"PySCF molecule requires {expected}")
    
class Excitation(object):
    
    def __init__(self, structure, gs_data, ato_idx, config):
        """Describe one excitation without executing electronic structure."""
        if not isinstance(config, ExcitationConfig):
            raise TypeError("config must be an ExcitationConfig")
        
        # set up excitation info
        self.ato_idx = ato_idx
        self.symbol  = structure.get_chemical_symbols()[ato_idx]
        self.config = config
        self.channel = config.channel_index
        
        # store output
        self.output = {}
        self.data   = {}
        
        # find index of orbital to excite
        orb_idx = find_1s_orbitals_pyscf(
                                         gs_data.mol,
                                         gs_data.mo_coeff[self.channel],
                                         gs_data.mo_energy[self.channel],
                                         gs_data.mo_occ[self.channel],
                                         [ato_idx],
                                         check_deg=False)
        
        # check that the orbitals are not still delocalized
        if not len(orb_idx) == 1:
            raise ValueError("Excitation of {:<2} atom #{:>2}: orbital is still delocalized (found {} 1s orbitals).".format(
                self.symbol, self.ato_idx, len(orb_idx)))
        
        # assign index of the orbital to excite
        self.orb_idx = orb_idx[0]
        
    def run(self, structure, gs_data, calculation, runtime, df_obj, logger):
        """Execute FCH, optional XCH, and MBXAS for this request."""
        if not isinstance(calculation, CalculationConfig):
            raise TypeError("calculation must be a CalculationConfig")
        if not isinstance(runtime, RuntimeConfig):
            raise TypeError("runtime must be a RuntimeConfig")
        if self.data:
            raise RuntimeError("Excitation has already been run")
        import pymbxas
        import pyscf
        self.provenance = {
            "device": runtime.device.value,
            "pymbxas_version": pymbxas.__version__,
            "pyscf_version": pyscf.__version__,
        }
        if runtime.is_gpu:
            gs_data = gs_data.to_gpu()
        self._excite(
            structure, gs_data, calculation, df_obj, logger, runtime)
        return self

    @classmethod
    def from_h5(cls, path, key):
        exc = cls.__new__(cls)

        with h5.open_plain(path) as f:
            group = f[key]
            exc.ato_idx = int(group.attrs["ato_idx"])
            exc.symbol  = h5.read_attr_str(group, "symbol")
            exc.channel = int(group.attrs["channel"])
            exc.orb_idx = int(group.attrs["orb_idx"])
            exc.config = ExcitationConfig.from_dict(
                h5.read_json(group, "config"))
            exc.provenance = h5.read_json(group, "provenance")

            names      = [name for name in ("fch", "xch") if name in group]
            exc.output = {name: h5.read_text(group[name], "output") for name in names}
            exc.mbxas  = {name: group["mbxas"][name][()] for name in group["mbxas"]}

        exc.data = {name: h5.read_snapshot(path, "{}/{}".format(key, name), lazy=True)
                    for name in names}

        return exc

    # use it to run excitation of the selected atom
    def _excite(self, structure, gs_data, calculation, df_obj, logger, runtime):
        logger.info("")
        site_logger = with_log_context(
            logger, site=f"{self.symbol}:{self.ato_idx}")
        site_logger.info("Excitation")
        
        # run FCH
        self._run_fch(
            structure, gs_data, calculation, df_obj,
            with_log_context(site_logger, stage="FCH"), runtime)

        # run XCH if enabled
        if self.config.xch:
            self._run_xch(
                structure, gs_data, calculation, df_obj,
                with_log_context(site_logger, stage="XCH"), runtime)
        else:
            with_log_context(site_logger, stage="XCH").info(
                "Alignment skipped; returned energies are raw FCH eigenvalues")

        # run MBXAS
        self._run_mbxas(
            gs_data, with_log_context(site_logger, stage="MBXAS"))
        
        site_logger.info("Excitation succeeded")
        return
    
    # run the FCH calculation
    def _run_fch(self, structure, gs_data, calculation, df_obj, logger, runtime):
        
        start_time = time.time()
        
        # retrieve parameters
        pbc       = calculation.pbc
        charge    = calculation.charge + 1
        spin      = calculation.spin + self.channel*2 - 1
        basis     = calculation.basis
        xc        = calculation.xc
        solvent   = calculation.solvent
        calc_type = calculation.method
        scf_config = self.config.resolved_fch_scf
        occupation_method = self.config.occupation.value
        maxvol_warmup_calls = self.config.mom_warmup_calls
        logging_config = runtime.logging
        logger.info("Starting calculation\n%s", format_log_fields({
            "occupation method": occupation_method,
            "MOM warm-up calls": (
                maxvol_warmup_calls if occupation_method == "mixed" else None),
        }))

        # Read MO coefficients and occupation number from GS
        scf_guess  = copy.deepcopy(gs_data.mo_coeff)
        occupation = copy.deepcopy(gs_data.mo_occ)

        # Assign initial occupation pattern --> kick orbital N
        occupation[self.channel][self.orb_idx] = 0
        
        # assign magmom
        magmom = len(structure)*[0]
        magmom[self.ato_idx] = 1-2*self.channel

        # change charge
        fch_mol = ase_to_mole(structure, charge=charge, spin=spin, basis=basis,
                              pbc=pbc, verbose=logging_config.pyscf_verbosity,
                              print_output=logging_config.pyscf_console,
                              log_file=logging_config.pyscf_logfile,
                              magmom=magmom, is_gpu=runtime.is_gpu, append=True,
                              log_context={
                                  "site": f"{self.symbol}:{self.ato_idx}",
                                  "stage": "FCH",
                                  "channel": ("alpha", "beta")[self.channel],
                                  "calculator": calc_type.upper(),
                                  "xc": (("LDA" if xc is None else xc)
                                         if "KS" in calc_type.upper() else None),
                                  "occupation": occupation_method,
                                  "mom_warmup_calls": (
                                      maxvol_warmup_calls
                                      if occupation_method == "mixed" else None),
                              })

        # Defnine new SCF calculator
        fch_calc = make_pyscf_calculator(fch_mol, xc=xc, calc_type=calc_type,
                                         pbc=pbc, solvent=solvent,
                                         dens_fit=df_obj,
                                         calc_name=os.path.join(
                                             runtime.work_directory,
                                             f"{self.symbol}_{self.ato_idx}_fch"),
                                         save=runtime.checkpoint.pyscf_chkfiles,
                                         is_gpu=runtime.is_gpu,
                                         max_cycle=scf_config.max_cycles,
                                         conv_tol=scf_config.convergence_tolerance,
                                         grid_level=scf_config.grid_level)

        # Construct new density matrix with new occupation pattern
        dm_u = fch_calc.make_rdm1(scf_guess, occupation)

        # Apply the requested reversible occupation constraint.
        fch_calc = apply_occupation_method(
            fch_calc, scf_guess, occupation, occupation_method,
            maxvol_warmup_calls=maxvol_warmup_calls)
        logger.debug("Attached %s occupation controller", occupation_method)

        try:
            # Start new SCF with new density matrix
            fch_calc = run_constrained_scf(
                fch_calc, dm_u, logger,
                max_cycle=scf_config.max_cycles,
                diis_cycles=scf_config.diis_cycles,
                mixing_cycles=scf_config.mixing_cycles,
                damping=scf_config.damping,
                level_shift=scf_config.level_shift,
                second_order=scf_config.second_order)
            if not fch_calc.converged:
                raise RuntimeError("FCH SCF did not converge for {} atom #{}".format(
                    self.symbol, self.ato_idx))

            _validate_electron_counts("FCH", fch_calc)

            # Check that the constrained core hole survived the SCF.
            # Select it by overlap, not by its position among occupation zeros.
            fch_overlap = (
                _as_numpy(fch_calc.mo_coeff[self.channel]).T
                @ fch_mol.intor("int1e_ovlp")
                @ _as_numpy(gs_data.mo_coeff[self.channel]))
            hole_idx = core_hole_index(
                fch_overlap, _as_numpy(fch_calc.mo_occ[self.channel]),
                self.orb_idx)
            c_hole = fch_calc.mo_coeff[self.channel][:, hole_idx]
            S = fch_mol.intor("int1e_ovlp")
            gs_coeff = gs_data.mo_coeff[self.channel][:, self.orb_idx]
            c_hole = _as_numpy(c_hole)
            gs_coeff = _as_numpy(gs_coeff)
            overlap = abs(c_hole @ S @ gs_coeff)
            logger.debug("Core hole overlap for {} atom #{}: {:.5f}".format(
                self.symbol, self.ato_idx, overlap))
            if overlap < CORE_HOLE_OVERLAP_TOL:
                raise RuntimeError("Core hole collapsed to ground state for {} atom #{} (overlap {:.5f} < {})".format(
                    self.symbol, self.ato_idx, overlap, CORE_HOLE_OVERLAP_TOL))
            state_diagnostics = _fch_state_diagnostics(
                fch_calc, scf_guess, occupation, S)
            if (state_diagnostics["spin_contamination"]
                    > FCH_SPIN_CONTAMINATION_TOL):
                logger.warning(
                    "FCH spin contamination exceeds tolerance\n%s",
                    format_log_fields({
                        "<S^2>": f"{state_diagnostics['spin_square']:.6f}",
                        "ideal <S^2>": (
                            f"{state_diagnostics['ideal_spin_square']:.6f}"),
                        "excess": (
                            f"{state_diagnostics['spin_contamination']:.6f}"),
                        "tolerance": FCH_SPIN_CONTAMINATION_TOL,
                    }))
            if (state_diagnostics["minimum_singular_value"]
                    < FCH_SUBSPACE_SINGULAR_TOL):
                logger.warning(
                    "FCH occupied subspace drift exceeds tolerance\n%s",
                    format_log_fields({
                        "minimum singular value": (
                            f"{state_diagnostics['minimum_singular_value']:.6f}"),
                        "tolerance": FCH_SUBSPACE_SINGULAR_TOL,
                        "occupied determinant": (
                            f"{state_diagnostics['occupied_determinant']:.6f}"),
                    }))
        finally:
            # store input/output
            self.output["fch"] = fch_calc.stdout.log.getvalue()
            self.data["fch"]   = pyscf_data(fch_calc)

            # close logfile of mol if exists
            fch_mol.stdout.close()
            
        log_scf_completion(
            logger, fch_calc, time.time() - start_time,
            occupation_method=occupation_method,
            core_hole_mo=hole_idx,
            core_hole_overlap=f"{overlap:.5f}",
            spin_square=f"{state_diagnostics['spin_square']:.6f}",
            ideal_spin_square=f"{state_diagnostics['ideal_spin_square']:.6f}",
            spin_contamination=(
                f"{state_diagnostics['spin_contamination']:.6f}"),
            occupied_determinant=(
                f"{state_diagnostics['occupied_determinant']:.6f}"),
            minimum_occupied_singular_value=(
                f"{state_diagnostics['minimum_singular_value']:.6f}"))
        return

    # run the XCH calculation
    def _run_xch(self, structure, gs_data, calculation, df_obj, logger, runtime):
        
        start_time = time.time()
        
        # retrieve parameters
        pbc       = calculation.pbc
        charge    = calculation.charge
        spin      = calculation.spin
        basis     = calculation.basis
        xc        = calculation.xc
        solvent   = calculation.solvent
        calc_type = calculation.method
        scf_config = self.config.resolved_xch_scf
        occupation_method = self.config.occupation.value
        maxvol_warmup_calls = self.config.mom_warmup_calls
        logging_config = runtime.logging
        logger.info("Starting calculation\n%s", format_log_fields({
            "occupation method": occupation_method,
            "MOM warm-up calls": (
                maxvol_warmup_calls if occupation_method == "mixed" else None),
        }))
        
        if runtime.is_gpu:
            data = self.data["fch"].to_gpu()
        else:
            data = self.data["fch"]

        # Read MO coefficients and occupation number from GS
        scf_guess  = copy.deepcopy(data.mo_coeff)
        occupation = copy.deepcopy(data.mo_occ)

        # Add the excited electron to the lowest-energy ordinary FCH virtual.
        # The MO whose numerical index equals the GS electron count may
        # already be occupied in a non-Aufbau MOM solution.
        S = data.mol.intor("int1e_ovlp")
        hole_idx = _fch_core_hole_index(
            data, gs_data, self.channel, self.orb_idx, S)
        target_idx = _xch_target_index(data, hole_idx, self.channel)
        if _as_numpy(occupation[self.channel])[target_idx] != 0:
            raise RuntimeError(
                f"Selected XCH target orbital {target_idx} is already occupied")
        occupation[self.channel][target_idx] = 1

        intended_counts = list(_occupation_counts(occupation))
        expected_counts = list(data.nelec)
        expected_counts[self.channel] += 1
        if intended_counts != expected_counts:
            raise RuntimeError(
                f"XCH target has electron counts {tuple(intended_counts)}, "
                f"expected {tuple(expected_counts)}")
        logger.debug(
            "XCH target for %s atom #%s: FCH core hole MO %d, excited MO %d",
            self.symbol, self.ato_idx, hole_idx, target_idx)

        # make XCH molecule
        xch_mol = ase_to_mole(structure, charge=charge, spin=spin, basis=basis,
                              pbc=pbc, verbose=logging_config.pyscf_verbosity,
                              print_output=logging_config.pyscf_console,
                              log_file=logging_config.pyscf_logfile,
                              is_gpu=runtime.is_gpu, append=True,
                              log_context={
                                  "site": f"{self.symbol}:{self.ato_idx}",
                                  "stage": "XCH",
                                  "channel": ("alpha", "beta")[self.channel],
                                  "calculator": calc_type.upper(),
                                  "xc": (("LDA" if xc is None else xc)
                                         if "KS" in calc_type.upper() else None),
                                  "occupation": occupation_method,
                                  "mom_warmup_calls": (
                                      maxvol_warmup_calls
                                      if occupation_method == "mixed" else None),
                              })

        # define new SCF calculator
        xch_calc = make_pyscf_calculator(xch_mol, xc=xc, calc_type=calc_type,
                                         pbc=pbc, solvent=solvent,
                                         dens_fit=df_obj,
                                         calc_name=os.path.join(
                                             runtime.work_directory,
                                             f"{self.symbol}_{self.ato_idx}_xch"),
                                         save=runtime.checkpoint.pyscf_chkfiles,
                                         is_gpu=runtime.is_gpu,
                                         max_cycle=scf_config.max_cycles,
                                         conv_tol=scf_config.convergence_tolerance,
                                         grid_level=scf_config.grid_level)

        # Construct new density matrix with new occupation pattern
        dm_u = xch_calc.make_rdm1(scf_guess, occupation)

        # Apply the same occupation constraint used for FCH.
        xch_calc = apply_occupation_method(
            xch_calc, scf_guess, occupation, occupation_method,
            maxvol_warmup_calls=maxvol_warmup_calls)
        logger.debug("Attached %s occupation controller", occupation_method)

        try:
            # Start new SCF with new density matrix
            xch_calc = run_constrained_scf(
                xch_calc, dm_u, logger,
                max_cycle=scf_config.max_cycles,
                diis_cycles=scf_config.diis_cycles,
                mixing_cycles=scf_config.mixing_cycles,
                damping=scf_config.damping,
                level_shift=scf_config.level_shift,
                second_order=scf_config.second_order)
            if not xch_calc.converged:
                raise RuntimeError("XCH SCF did not converge for {} atom #{}".format(
                    self.symbol, self.ato_idx))
            _validate_electron_counts("XCH", xch_calc)

            xch_coeff = _as_numpy(xch_calc.mo_coeff[self.channel])
            xch_occ = _as_numpy(xch_calc.mo_occ[self.channel])
            fch_coeff = _as_numpy(data.mo_coeff[self.channel])
            xch_overlap = xch_coeff.T @ S @ fch_coeff

            unoccupied = np.flatnonzero(xch_occ == 0)
            core_overlaps = np.abs(xch_overlap[unoccupied, hole_idx])
            core_overlap = float(core_overlaps.max()) if core_overlaps.size else 0.0
            if core_overlap < CORE_HOLE_OVERLAP_TOL:
                raise RuntimeError(
                    "XCH core hole collapsed for {} atom #{} "
                    "(overlap {:.5f} < {})".format(
                        self.symbol, self.ato_idx, core_overlap,
                        CORE_HOLE_OVERLAP_TOL))

            occupied = np.flatnonzero(xch_occ == 1)
            target_overlaps = np.abs(xch_overlap[occupied, target_idx])
            target_overlap = float(target_overlaps.max()) if target_overlaps.size else 0.0
            if target_overlap < XCH_ORBITAL_OVERLAP_TOL:
                raise RuntimeError(
                    "XCH excited electron collapsed for {} atom #{} "
                    "(target MO {} overlap {:.5f} < {})".format(
                        self.symbol, self.ato_idx, target_idx, target_overlap,
                        XCH_ORBITAL_OVERLAP_TOL))
            logger.debug(
                "XCH validation for %s atom #%s: core-hole overlap %.5f, "
                "excited-orbital overlap %.5f",
                self.symbol, self.ato_idx, core_overlap, target_overlap)
        finally:
            # store input/output
            self.output["xch"] = xch_calc.stdout.log.getvalue()
            self.data["xch"]   = pyscf_data(xch_calc)

            # close logfile of mol if exists
            xch_mol.stdout.close()

        log_scf_completion(
            logger, xch_calc, time.time() - start_time,
            occupation_method=occupation_method,
            core_hole_overlap=f"{core_overlap:.5f}",
            spectator_mo=target_idx,
            spectator_overlap=f"{target_overlap:.5f}")
        
        return
    
    # run MBXAS from a set of pySCF calculations
    def _run_mbxas(self, gs_data, logger):

        start_time = time.time()
        logger.info("Starting calculation")

        xch_data = self.data["xch"].to_cpu() if "xch" in self.data else None
        energies, absorption, mb_ovlp, dip_KS, b_ovlp = run_MBXAS_pyscf(
            gs_data.mol, gs_data.to_cpu(), self.data["fch"].to_cpu(),
            self.orb_idx, channel=self.channel, xch_calc=xch_data)

        self.mbxas = {
            "energies"   : energies,
            "absorption" : absorption,
            "mb_overlap" : mb_ovlp,
            "dipole_KS"  : dip_KS,
            "basis_ovlp" : b_ovlp
            }
        
        logger.info("Finished in %.1f s", time.time() - start_time)
        return
