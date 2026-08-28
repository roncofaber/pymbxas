#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 15:50:13 2025

@author: roncofaber
"""

import logging

#%%

def pyscf_acquire(structure, sites, *, calculation=None, gs_scf=None,
                  checkpoint="pymbxas.h5", excitation=None, fch_scf=None,
                  xch_scf=None, runtime=None, logging=None):
    """Performs a PySCF calculation and returns a Spectra object.

    Args:
        structure: The ASE structure.
        sites: Atom index(es)/symbol(s) to excite.
        calculation: Optional ``CalculationConfig``.
        gs_scf: Optional ground-state ``SCFConfig``.
        checkpoint: Calculation checkpoint path, policy, or ``None``.
        excitation: Optional reusable ``ExcitationConfig``.
        fch_scf: Optional FCH ``SCFConfig``.
        xch_scf: Optional XCH ``SCFConfig``.
        runtime: Optional ``RuntimeConfig``.
        logging: Optional ``LoggingConfig``.

    Returns:
        Spectra object or None if calculation fails.
    """
    logger = logging.getLogger(__name__)
    from pymbxas.calculators.pyscf import PySCFMBXAS

    formula = structure.get_chemical_formula()
    logger.info("Starting PySCF calculation for %s (sites=%s)", formula, sites)

    try:
        obj = PySCFMBXAS(
            structure, calculation=calculation, gs_scf=gs_scf,
            checkpoint=checkpoint)
        obj.run(
            sites, excitation=excitation, fch_scf=fch_scf,
            xch_scf=xch_scf, runtime=runtime, logging=logging)
        logger.info("Calculation succeeded for %s (sites=%s)", formula, sites)
        return obj.to_spectra()
    except Exception:
        logger.exception(
            "PySCF calculation failed for %s (sites=%s)", formula, sites)
        return None
