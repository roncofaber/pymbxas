#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 15:29:31 2023

@author: roncoroni
"""

import logging
import inspect

import ase

from pyscf import gto
from pyscf.pbc import gto as pgto

from pymbxas.io.logger import Logger
from pymbxas.utils.check_keywords import check_pbc
from pymbxas.utils.auxiliary import get_available_memory

logger = logging.getLogger(__name__)


def _valid_build_keywords(pbc):
    """Keywords explicitly supported by the installed PySCF build APIs."""
    methods = [gto.Mole.build]
    if pbc:
        methods.append(pgto.Cell.build)
    valid = set()
    for method in methods:
        valid.update(
            name for name, parameter in inspect.signature(method).parameters.items()
            if name != "self" and parameter.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        )
    return valid

#%%

# convert an ase Atoms object to a mole or cell object for pyscf
def ase_to_mole(structure, charge=0, spin=0, basis='def2-svpd', pbc=None,
                verbose=4, print_output=True, log_file=None, symmetry=False,
                is_gpu=False, append=False, log_context=None, **kwargs):

    # generate atom list to feed to object
    atom_list = []
    for ii in range(len(structure)):
        atom_list.append([
            structure.get_chemical_symbols()[ii],
            tuple(structure.get_positions()[ii])
            ])
    
    if pbc is None:
        pbc = check_pbc(pbc, structure)

    unknown = sorted(set(kwargs) - _valid_build_keywords(pbc))
    if unknown:
        raise TypeError(
            "ase_to_mole received keyword(s) not supported by the installed "
            f"PySCF {'Cell' if pbc else 'Mole'}.build API: {unknown}"
        )
    
    # Create Logger instance (tees PySCF's own stdout, unrelated to the
    # module-level `logger` above)
    context = None
    if log_context is not None:
        context = dict(log_context)
        context.update({
            "charge": charge,
            "spin": spin,
            "basis": basis,
            "device": "GPU" if is_gpu else "CPU",
        })
    stdout_logger = Logger(
        print_to_terminal=print_output, log_file=log_file, append=append,
        section_context=context)

    # periodic system
    if pbc:
        mol = pgto.Cell(
            atom  = atom_list,
            basis = basis,
            charge = charge,
            spin = spin,
            verbose = verbose,
            stdout = stdout_logger,
            a = structure.get_cell().array,
            ke_cutoff = 100.0,
            symmetry = symmetry,
            **kwargs
            )

    # non periodic system
    else:
        mol = gto.Mole(
            atom  = atom_list,
            basis = basis,
            charge = charge,
            spin = spin,
            verbose = verbose,
            stdout = stdout_logger,
            max_memory = get_available_memory(is_gpu),
            unit = 'Angstrom',
            symmetry = symmetry,
            **kwargs
            )

    mol.build()
    
    # overwrite integrals if PBC with proper ones #TODO: check if needed and correct
    if pbc:
        mol.intor = mol.pbc_intor
    
    return mol

# convert a mol object to ase Atoms
def mole_to_ase(mol, units="Angstrom", **kwargs):
    
    structure = ase.Atoms(
        mol.elements,
        mol.atom_coords(unit=units),
        **kwargs
        )
    
    return structure

# rotate, translate, permute, inverse a structure
def rotate_structure(structure, rot, tr, perm, inv, rtype):
    try:
        import sea_urchin.alignement.align as ali
        has_SU = True
    except ImportError:
        has_SU = False
    assert has_SU, "Please install Sea Urchin to use this"
    return ali.align_structure(structure, rot, tr, perm, inv, rtype)
    
