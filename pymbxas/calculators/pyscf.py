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
from dataclasses import dataclass

# good ol' numpy
import numpy as np

# self module utilities
import pymbxas
import pyscf
from pymbxas.calculators.excitation import Excitation
from pymbxas.config import (
    CalculationConfig, CheckpointConfig, ExcitationConfig, LoggingConfig,
    RuntimeConfig, SCFConfig, snapshot_config,
)
import pymbxas.utils.check_keywords as check
from pymbxas.utils.auxiliary import as_list
from pymbxas.utils.indexing import atoms_to_indexes
from pymbxas.io.data import pyscf_data
from pymbxas.io.config import (
    configure_logger, format_log_fields, log_scf_completion,
    with_log_context,
)
from pymbxas.io import h5
from pymbxas.build.structure import ase_to_mole
from pymbxas.build.input_pyscf import make_pyscf_calculator
from pymbxas.utils.orbitals import find_1s_orbitals_pyscf
from pymbxas.utils.boys import do_localization_pyscf
# from pymbxas.mbxas.broaden import get_mbxas_spectra
from pymbxas.io.cleanup import remove_tmp_files
from pymbxas.io.write import write_data_to_fchk
from pymbxas.spectra import Spectra
from pymbxas.spectras import Spectras

#%%


@dataclass(frozen=True)
class ExcitationOutcome:
    """Result of one requested atom excitation."""

    atom_index: int
    symbol: str
    status: str
    message: str = ""


def _resolve_log_path(path, calculation_dir):
    """Resolve a configured logfile relative to the calculation directory."""
    if path is None:
        return None
    path = os.fspath(path)
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(calculation_dir, path))


class PySCFMBXAS:
    """One immutable ground-state definition with zero or more excitations."""

    def __init__(self, structure, *, calculation=None, gs_scf=None,
                 checkpoint="pymbxas.h5"):
        if structure is None or not hasattr(structure, "copy"):
            raise TypeError("structure must be an ASE Atoms object")

        self.structure = structure.copy()
        calculation = snapshot_config(
            calculation, CalculationConfig, "calculation")
        pbc_resolved = check.check_pbc(calculation.pbc, self.structure)
        if pbc_resolved:
            raise NotImplementedError(
                "MBXAS is not supported under periodic boundary conditions: "
                "the position operator is not periodic. Use a molecular or "
                "cluster model instead.")
        resolved = calculation.copy().set(pbc=pbc_resolved).snapshot()
        self._calculation = resolved
        self._gs_scf = snapshot_config(gs_scf, SCFConfig, "gs_scf")
        self._checkpoint = self._resolve_checkpoint(checkpoint)
        self._runtime = RuntimeConfig().snapshot()
        self._logging = LoggingConfig().snapshot()
        self.logger = logging.getLogger(__name__)

        self._ran_GS = False
        self._used_loc = False
        self.output = ""
        self._excitations = []
        self._last_excitation_outcomes = ()
        self._h5_path = None
        self._ground_state_provenance = {}
        self._execution_started = False

    @staticmethod
    def _resolve_checkpoint(checkpoint):
        if checkpoint is None:
            return None
        if isinstance(checkpoint, (str, os.PathLike)):
            checkpoint = CheckpointConfig(path=checkpoint)
        if not isinstance(checkpoint, CheckpointConfig):
            raise TypeError(
                "checkpoint must be a path, CheckpointConfig, or None")
        return checkpoint.copy(path=os.path.abspath(checkpoint.path)).snapshot()

    def _prepare_execution(self, runtime=None, logging_config=None):
        if runtime is None:
            runtime = self.runtime
        if logging_config is None:
            logging_config = self.logging
        runtime = snapshot_config(runtime, RuntimeConfig, "runtime")
        logging_config = snapshot_config(
            logging_config, LoggingConfig, "logging")
        os.makedirs(runtime.work_directory, exist_ok=True)
        logging_config = logging_config.copy().set(
            pymbxas_logfile=_resolve_log_path(
                logging_config.pymbxas_logfile, runtime.work_directory),
            pyscf_logfile=_resolve_log_path(
                logging_config.pyscf_logfile, runtime.work_directory),
        ).snapshot()
        configure_logger(
            logging_config.pymbxas_verbosity,
            log_file=logging_config.pymbxas_logfile,
            file_mode="a" if self._execution_started or self._h5_path else "w")
        self._runtime = runtime
        self._logging = logging_config
        self._execution_started = True

    def run(self, sites, *, excitation=None, fch_scf=None, xch_scf=None,
            runtime=None, logging=None):
        """Run the missing ground state and the requested excitations."""
        self._prepare_execution(runtime, logging)
        excitation_config = snapshot_config(
            excitation, ExcitationConfig, "excitation")
        fch_config = snapshot_config(fch_scf, SCFConfig, "fch_scf")
        xch_config = snapshot_config(xch_scf, SCFConfig, "xch_scf")
        if not excitation_config.xch and xch_scf is not None:
            raise ValueError("xch_scf was supplied while excitation.xch is False")
        self.logger.info(
            "Starting PyMBXAS\n%s",
            format_log_fields({
                "version": pymbxas.__version__,
                "release date": pymbxas.__date__,
                "formula": self.structure.get_chemical_formula(),
                "atoms": len(self.structure),
                "requested excitation": sites,
                "target directory": self.runtime.work_directory,
                "device": self.runtime.device.value.upper(),
                "calculator": self.calculation.method,
                "xc": self.calculation.xc,
                "basis": self.calculation.basis,
                "XCH alignment": excitation_config.xch,
                "occupation method": excitation_config.occupation.value,
                "MOM warm-up calls": (
                    excitation_config.mom_warmup_calls
                    if excitation_config.occupation.value == "mixed"
                    else None),
                "SCF DIIS / mixing cycles": (
                    f"{fch_config.diis_cycles} / {fch_config.mixing_cycles}"),
                "second-order recovery": fch_config.second_order,
            }))
        self.logger.info("")
        if not self._ran_GS:
            self._run_gs()
        outcomes = self._excite(
            sites, excitation_config, fch_config, xch_config)
        remove_tmp_files(self.runtime.work_directory)
        return outcomes

    def excite(self, sites, *, excitation=None, fch_scf=None, xch_scf=None,
               runtime=None, logging=None):
        """Run excitation-specific work against an existing ground state."""
        self._prepare_execution(runtime, logging)
        excitation_config = snapshot_config(
            excitation, ExcitationConfig, "excitation")
        fch_config = snapshot_config(fch_scf, SCFConfig, "fch_scf")
        xch_config = snapshot_config(xch_scf, SCFConfig, "xch_scf")
        if not excitation_config.xch and xch_scf is not None:
            raise ValueError("xch_scf was supplied while excitation.xch is False")
        return self._excite(
            sites, excitation_config, fch_config, xch_config)

    def _excite(self, sites, excitation_config, fch_scf, xch_scf):
        if not self._ran_GS:
            message = "Cannot excite atoms before the ground state has run"
            self.logger.error(message)
            raise RuntimeError(message)
            
        # convert into atom indexes
        to_excite = atoms_to_indexes(self.structure, sites)
        
        # iterate over the indexes 
        outcomes = []
        for ato_idx in to_excite:
            outcomes.append(self._single_excite(
                ato_idx, excitation_config, fch_scf, xch_scf))
        
        self._last_excitation_outcomes = tuple(outcomes)
        self._log_excitation_summary(self._last_excitation_outcomes)
        return self._last_excitation_outcomes

    # perform a single excitation
    def _single_excite(self, ato_idx, config, fch_scf, xch_scf):

        symbol = self.structure.get_chemical_symbols()[ato_idx]
        site_logger = with_log_context(
            self.logger, site=f"{symbol}:{ato_idx}")
        identity = (ato_idx, config.channel_index, config, fch_scf, xch_scf)
        if any((exc.ato_idx, exc.channel, exc.config,
                exc.fch_scf, exc.xch_scf) == identity
               for exc in self.excitations):
            site_logger.info("Equivalent excitation already exists; skipping")
            return ExcitationOutcome(ato_idx, symbol, "skipped")

        try:
            excitation = Excitation(
                self.structure, self.gs_data, ato_idx, config,
                fch_scf, xch_scf)
            excitation.run(
                self.structure, self.gs_data, self.calculation, self.runtime,
                self.logging, self.checkpoint, self.df_obj, self.logger)
            self._excitations.append(excitation)
            if self.checkpoint is not None:
                self.save()
        except (ValueError, RuntimeError) as e:
            site_logger.error("Excitation failed: %s", e)
            return ExcitationOutcome(ato_idx, symbol, "failed", str(e))

        return ExcitationOutcome(ato_idx, symbol, "succeeded")

    def _log_excitation_summary(self, outcomes):
        """Log a truthful aggregate result for the current request."""
        outcomes = tuple(outcomes)
        succeeded = [item for item in outcomes if item.status == "succeeded"]
        failed = [item for item in outcomes if item.status == "failed"]
        skipped = [item for item in outcomes if item.status == "skipped"]
        self.logger.info("")
        message = "Run completed" if not failed else "Run completed with failures"
        fields = format_log_fields({
            "succeeded": len(succeeded),
            "failed": len(failed),
            "skipped": len(skipped),
        })
        if failed:
            self.logger.error("%s\n%s", message, fields)
            for item in failed:
                self.logger.error(
                    "[%s:%d] %s", item.symbol, item.atom_index, item.message)
        else:
            self.logger.info("%s\n%s", message, fields)
        

    def run_gs(self, *, runtime=None, logging=None, force=False):
        """Run only the ground-state SCF calculation."""
        self._prepare_execution(runtime, logging)
        return self._run_gs(force=force)

    def _run_gs(self, force=False):

        # check if GS was already performed, if so: skip
        if self._ran_GS and not force:
            with_log_context(self.logger, stage="GS").warning(
                "Ground state already exists; skipping")
            return

        if force and self._excitations:
            raise RuntimeError(
                "Cannot re-run the ground state: {} excitations were computed against "
                "the current one. Start a new calculation instead.".format(
                    len(self._excitations)))

        if force:
            # A successful replacement GS must rewrite, not append to, the
            # previous checkpoint header.
            self._h5_path = None

        gs_logger = with_log_context(self.logger, stage="GS")
        gs_logger.info("Starting calculation")
        
        start_time  = time.time()

        xc        = self.calculation.xc
        pbc       = self.calculation.pbc
        solvent   = self.calculation.solvent
        calc_type = self.calculation.method
        scf_config = self.gs_scf
        
        # generate molecule
        gs_mol = self._make_ground_state_mol()
        
        # generate KS calculator
        gs_calc = make_pyscf_calculator(gs_mol, xc, pbc=pbc, solvent=solvent,
                                        calc_type=calc_type, dens_fit=None,
                                        calc_name=os.path.join(
                                            self.runtime.work_directory, "gs"),
                                        save=(self.checkpoint is not None and
                                              self.checkpoint.pyscf_chkfiles),
                                        is_gpu=self.runtime.is_gpu,
                                        max_cycle=scf_config.max_cycles,
                                        conv_tol=scf_config.convergence_tolerance,
                                        grid_level=scf_config.grid_level)

        # Run the ordinary ground-state SCF.
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
        self._run_localization(self.gs_data, self.calculation.localization)
        
        # write output fchk files if using mokit
        if self.checkpoint is not None and self.checkpoint.fchk_files:
            self._print_fchk_files()

        # mark that GS has been run
        self._ran_GS = True
        self._ground_state_provenance = {
            "device": self.runtime.device.value,
            "pymbxas_version": pymbxas.__version__,
            "pyscf_version": pyscf.__version__,
        }
        self.mol.stdout.close()
        
        log_scf_completion(
            gs_logger, gs_calc, time.time() - start_time)
        if self.checkpoint is not None:
            self.save()
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
        
        # channel is chosen per excitation, not here, so localization must be
        # decided for both spin channels: skip only if neither is delocalized
        if all(len(s1orb) <= 1 for s1orb in s1_orbitals):
            return dft_calc.mo_coeff, False

        # localize up to highest degenerate orbital #TEST
        if loc_type.endswith("m"):
            s1_orbitals = [list(range(np.max(orb) + 1)) if orb else orb
                          for orb in s1_orbitals]


        mo_loc = do_localization_pyscf(dft_calc, s1_orbitals, loc_type)

        localization_logger = with_log_context(
            self.logger, stage="GS localization")
        localization_logger.info(
            "%s completed\n%s", loc_type.upper(), format_log_fields({
                "alpha orbital count": len(s1_orbitals[0]),
                "beta orbital count": len(s1_orbitals[1]),
            }))
        localization_logger.debug(
            "Localized orbital indices\n%s", format_log_fields({
                "alpha": s1_orbitals[0],
                "beta": s1_orbitals[1],
            }))
        
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
                               oname=os.path.join(
                                   self.runtime.work_directory,
                                   "output_gs_del.fchk"),
                               )

            write_data_to_fchk(self.mol,
                               mo_coeff = self.gs_data.mo_coeff,
                               oname=os.path.join(
                                   self.runtime.work_directory,
                                   "output_gs_loc.fchk"),
                               )

        else:
            write_data_to_fchk(self.mol,
                               mo_coeff = self.gs_data.mo_coeff,
                               oname=os.path.join(
                                   self.runtime.work_directory,
                                   "output_gs.fchk"),
                               )

        return
    

    def get_mbxas_spectra(self, ato_idx, axis=None, sigma=0.5, npoints=3001, tol=0.01,
                          erange=None, f_order=1, spectator_order="auto",
                          max_total_order=None,
                          max_configurations=2_000_000):

        ato_idxs = atoms_to_indexes(self.structure, ato_idx)
        matched = [i for i, exc in enumerate(self.excitations) if exc.ato_idx in ato_idxs]
        if not matched:
            raise ValueError(f"No excitations found for atom index/label {ato_idx!r}")
        matched_sites = [self.excitations[i].ato_idx for i in matched]
        if len(matched_sites) != len(set(matched_sites)):
            raise ValueError(
                "Multiple excitation configurations exist for at least one "
                "requested site; select an explicit result with "
                "to_spectra(index=...)")

        spectras = [self.to_spectra(index=i) for i in matched]

        if erange is None:
            all_energies = np.concatenate([sp.energies for sp in spectras])
            erange = [all_energies.min(), all_energies.max()]

        energy = None
        intensity_sum = None
        for sp in spectras:
            energy, intensity = sp.get_mbxas_spectra(axis=axis, sigma=sigma,
                                                      npoints=npoints, tol=tol,
                                                      erange=erange,
                                                      f_order=f_order,
                                                      spectator_order=spectator_order,
                                                      max_total_order=max_total_order,
                                                      max_configurations=max_configurations)
            intensity_sum = intensity if intensity_sum is None else intensity_sum + intensity

        return energy, intensity_sum

    def save(self, path=None):
        """Checkpoint the ground state and every completed excitation."""

        if not self._ran_GS:
            raise RuntimeError("Cannot save before the ground state has been run.")

        if path is None:
            if self._h5_path is not None:
                path = self._h5_path
            elif self.checkpoint is not None:
                path = self.checkpoint.path
            else:
                raise RuntimeError(
                    "Checkpointing is disabled; pass an explicit path to save()")
        path = os.path.abspath(os.fspath(path))
        os.makedirs(os.path.dirname(path), exist_ok=True)

        same_backing_file = (
            self._h5_path is not None
            and os.path.abspath(self._h5_path) == os.path.abspath(path)
        )
        if not same_backing_file or not os.path.exists(path):
            self._write_header(path)

        self._append_excitations(path)
        self._h5_path = path

        return path

    def _write_header(self, path):

        with h5.create(path, h5.KIND_CALCULATION) as f:
            f.attrs["ran_GS"]   = bool(self._ran_GS)
            f.attrs["used_loc"] = bool(self._used_loc)

            h5.write_structure(f, "structure", self.structure)
            h5.write_json(
                f, "calculation_config", self.calculation.to_dict())
            h5.write_json(f, "gs_scf_config", self.gs_scf.to_dict())
            h5.write_json(
                f, "checkpoint_config",
                None if self.checkpoint is None else self.checkpoint.to_dict())
            h5.write_json(
                f, "ground_state_provenance",
                self._ground_state_provenance)
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
                h5.write_json(group, "config", exc.config.to_dict())
                h5.write_json(group, "fch_scf_config", exc.fch_scf.to_dict())
                h5.write_json(group, "xch_scf_config", exc.xch_scf.to_dict())
                h5.write_json(group, "provenance", exc.provenance)

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
    
    @classmethod
    def load(cls, filename):
        """Restore scientific state and use the file as its checkpoint."""

        obj = cls.__new__(cls)
        obj._load_h5(filename)
        return obj

    def _load_h5(self, filename):

        path = os.path.abspath(filename)
        self._runtime = RuntimeConfig(
            work_directory=os.path.dirname(path)).snapshot()
        self._logging = LoggingConfig().snapshot()

        with h5.open_read(path, h5.KIND_CALCULATION) as f:
            if "calculation_config" not in f or "gs_scf_config" not in f:
                raise ValueError(
                    "Checkpoint uses an older calculation configuration layout; "
                    "re-run the calculation with this PyMBXAS version")
            self.structure        = h5.read_structure(f, "structure")
            self._calculation = CalculationConfig.from_dict(
                h5.read_json(f, "calculation_config")).snapshot()
            self._gs_scf = SCFConfig.from_dict(
                h5.read_json(f, "gs_scf_config")).snapshot()
            self._ground_state_provenance = h5.read_json(
                f, "ground_state_provenance")
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

        self.logger = logging.getLogger(__name__)

        for key in incomplete:
            self.logger.warning("Skipping incomplete excitation {} in {}".format(key, path))

        self.gs_data     = h5.read_snapshot(path, "/")
        self.mol         = self.gs_data.mol
        self.mol.verbose = self.logging.pyscf_verbosity
        self.df_obj      = None

        self._h5_path     = path
        stored_checkpoint = None
        with h5.open_plain(path) as f:
            if "checkpoint_config" in f:
                stored_checkpoint = h5.read_json(f, "checkpoint_config")
        if stored_checkpoint is None:
            self._checkpoint = CheckpointConfig(path=path).snapshot()
        else:
            stored_checkpoint["path"] = path
            self._checkpoint = CheckpointConfig.from_dict(
                stored_checkpoint).snapshot()
        self._excitations = [Excitation.from_h5(path, key) for key in complete]
        self._last_excitation_outcomes = ()
        self._execution_started = False

        return
    
    @property
    def excitations(self):
        return self._excitations

    @property
    def calculation(self):
        return self._calculation

    @property
    def config(self):
        """Internal compatibility view used by the spectra layer."""
        return self._calculation

    @property
    def gs_scf(self):
        return self._gs_scf

    @property
    def checkpoint(self):
        return self._checkpoint

    @property
    def runtime(self):
        return self._runtime

    @property
    def logging(self):
        return self._logging

    @property
    def last_excitation_outcomes(self):
        """Outcomes from the most recent :meth:`excite` request."""
        return self._last_excitation_outcomes
    
    @property
    def excited_idxs(self):
        return [exc.ato_idx for exc in self.excitations]

    @property
    def excitation_keys(self):
        return [
            (exc.ato_idx, exc.channel, exc.config, exc.fch_scf, exc.xch_scf)
            for exc in self.excitations
        ]

    # convert object to Spectra object
    def to_spectra(self, *, index=None):

        if index is None:
            indexes = list(range(len(self.excitations)))
        else:
            indexes = as_list(index)

        spectras = [Spectra(self, excitation=cc) for cc in indexes]

        if len(spectras) == 1:
            return spectras[0]
        else:
            return Spectras(spectras)

    # internal function to generate a pyscf mol obj
    def _make_ground_state_mol(self):
        logging_config = self.logging

        mol = ase_to_mole(
            self.structure, self.calculation.charge, self.calculation.spin,
            basis=self.calculation.basis, pbc=self.calculation.pbc,
                             verbose=logging_config.pyscf_verbosity,
                             print_output=logging_config.pyscf_console,
                             log_file=logging_config.pyscf_logfile,
                             is_gpu=self.runtime.is_gpu,
                             log_context={
                                 "stage": "GS",
                                 "calculator": self.calculation.method,
                                 "xc": (("LDA" if self.calculation.xc is None
                                         else self.calculation.xc)
                                        if "KS" in self.calculation.method
                                        else None),
                             })
        
        return mol


__all__ = ["ExcitationOutcome", "PySCFMBXAS"]
