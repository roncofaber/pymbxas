#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Set of functions to set up AIMD and relaxations

Created on Wed Mar 13 10:35:07 2024

@author: roncofaber
"""

import pyscf.geomopt as gopt

from pymbxas.build.structure import ase_to_mole
from pymbxas.build.input_pyscf import make_pyscf_calculator
from pymbxas.aimd.callback import OptimizerTraj
#%%
# perform geometry optimization of the provided structure
def geometry_optimization(structure, basis, xc, opt_type="geometric", charge=0,
                          spin=0, pbc=False, solvent=None, maxsteps=100,
                          write=True, oname="optimization.xyz",
                          append=False, verbose=0, print_output=False,
                          calc_name="gopt", save=False):
    
    # check solver type
    if opt_type == "geometric":
        from pyscf.geomopt.geometric_solver import optimize
    elif opt_type == "berny":
        from pyscf.geomopt.berny_solver import optimize
    else:
        raise "WRONG 'opt_type' selected. 'geometric' and 'berny' accepted."
        
    # generate molecule
    mol = ase_to_mole(structure, charge, spin, basis=basis, pbc=pbc,
                         verbose=verbose, print_output=print_output)
    
    # generate KS calculator
    calc = make_pyscf_calculator(mol, xc, pbc=pbc, solvent=solvent,
                                 dens_fit=None, calc_name=calc_name, save=save)
    
    # generate optimizer object
    opt = OptimizerTraj(write=write, oname=oname, append=append)

    # optimize mol geometry
    mol_opt = optimize(calc, maxsteps=maxsteps, callback=opt)
    
    return opt