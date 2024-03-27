#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Set of functions to set up AIMD and relaxations

Created on Wed Mar 13 10:35:07 2024

@author: roncofaber
"""

from pyscf.md.distributions import MaxwellBoltzmannVelocity
import pyscf.geomopt as gopt

from pymbxas.build.structure import ase_to_mole
from pymbxas.build.input_pyscf import make_pyscf_calculator
from pymbxas.md.callback import OptimizerTraj, AIMDTrajWriter
from pymbxas.md.integrator import NVTBerendson

#%%
# perform geometry optimization of the provided structure
def geometry_optimization(pyscf_calc, opt_type="geometric", maxsteps=100,
                          write=True, oname="optimization.xyz",
                          append=False, density_fit=True):
    
    # check solver type
    if opt_type == "geometric":
        from pyscf.geomopt.geometric_solver import optimize
    elif opt_type == "berny":
        from pyscf.geomopt.berny_solver import optimize
    else:
        raise "WRONG 'opt_type' selected. 'geometric' and 'berny' accepted."
        
    # make dens. fitting
    if density_fit:
        pyscf_calc.density_fit()
    
    # generate optimizer object
    opt = OptimizerTraj(write=write, oname=oname, append=append)

    # optimize mol geometry
    _ = optimize(pyscf_calc, maxsteps=maxsteps, callback=opt)
    
    return opt

# run aimd calculation
def aimd_run(pyscf_calc, T, dt, nsteps, ofreq=1, oname="aimd_traj.xyz",
             output="output.log"):
    
    # run GS calc
    if not pyscf_calc.converged:
        pyscf_calc.kernel()
    
    # initialize velocity
    init_veloc = MaxwellBoltzmannVelocity(pyscf_calc.mol, T=T)

    # make a logger
    logger = AIMDTrajWriter(
        oname        = oname, # name output traj
        nstep        = ofreq,    # frequency at which traj is written
        data_output  = output, # also write output
        append       = False # append to existing traj
        )

    aimd_calc = NVTBerendson(pyscf_calc,
                            dt    = dt,
                            steps = nsteps,
                            T     = T,
                            taut  = 50*dt,
                            veloc =  init_veloc,
                            callback = logger,
                            )

    aimd_calc.run()
    
    return aimd_calc