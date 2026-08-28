#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 15:50:13 2025

@author: roncofaber
"""

import logging

#%%

def pyscf_acquire(structure, sites, *, calculation=None, runtime=None,
                  excitation=None, **excitation_settings):
    """Performs a PySCF calculation and returns a Spectra object.

    Args:
        structure: The ASE structure.
        sites: Atom index(es)/symbol(s) to excite.
        calculation: Optional ``CalculationConfig``.
        runtime: Optional ``RuntimeConfig``.
        excitation: Optional reusable ``ExcitationConfig``.
        **excitation_settings: Direct settings accepted by ``excite``.

    Returns:
        Spectra object or None if calculation fails.
    """
    logger = logging.getLogger(__name__)
    from pymbxas.calculators.pyscf import PySCFMBXAS

    formula = structure.get_chemical_formula()
    logger.info("Starting PySCF calculation for %s (sites=%s)", formula, sites)

    try:
        obj = PySCFMBXAS(structure, config=calculation, runtime=runtime)
        obj.run(sites, config=excitation, **excitation_settings)
        logger.info("Calculation succeeded for %s (sites=%s)", formula, sites)
        return obj.to_spectra()
    except Exception:
        logger.exception("PySCF calculation failed for %s (to_excite=%s)", formula, to_excite)
        return None
