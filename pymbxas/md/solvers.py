#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Set of functions to set up AIMD and relaxations

Created on Wed Mar 13 10:35:07 2024

@author: roncofaber
"""

import ase
import ase.visualize

from pyscf.md.distributions import MaxwellBoltzmannVelocity
import pyscf.geomopt

from pymbxas.build.structure import ase_to_mole
from pymbxas.build.input_pyscf import make_pyscf_calculator
from pymbxas.md.callback import OptimizerTraj, AIMDTrajWriter
from pymbxas.md.integrator import NVTBerendson

#%%

class Geometry_optimizer():
    """Class to handle geometry optimization using PySCF.

    Args:
        pyscf_calc (pyscf object): A PySCF calculation object.
        opt_type (str, optional): Type of optimization algorithm ('geometric' or 'berny'). Defaults to 'geometric'.
        write (bool, optional): Whether to write trajectory to a file. Defaults to True.
        oname (str, optional): Output filename. Defaults to "optimization.xyz".
        append (bool, optional): Whether to append to an existing file. Defaults to False.
        density_fit (bool, optional): Whether to use density fitting for the calculation. Defaults to True.
    """
    
    def __init__(self, pyscf_calc, opt_type="geometric",
                 write=True, oname="optimization.xyz", append=False,
                 density_fit=True):
        
        self.opt_type = opt_type
        
        # make dens. fitting
        if density_fit:
            pyscf_calc.density_fit()
            
        # assign calculator
        self.calc = pyscf_calc
        
        # generate optimizer object
        self.opt = OptimizerTraj(write=write, oname=oname, append=append)

        return
    
    def optimize(self, maxsteps=100):
        
        optimize = self._get_opt_type()
        
        # optimize mol geometry
        self.mol_eq = optimize(self.calc, maxsteps=maxsteps, callback=self.opt)
        
        return
    
    def _get_opt_type(self):
        
        # check solver type
        if self.opt_type == "geometric":
            from pyscf.geomopt.geometric_solver import optimize
        elif self.opt_type == "berny":
            from pyscf.geomopt.berny_solver import optimize
        else:
            raise "WRONG 'opt_type' selected. 'geometric' and 'berny' accepted."
        
        return optimize
    
    def view(self):
        self.opt.view()
        return
    
    # make class iterable
    def __getitem__(self, index):
        return self.opt[index]
    
    def __iter__(self):
        return iter(self.opt._traj)
    
    # get elements
    @property
    def mol(self):
        return self.opt.mol
    @property
    def traj(self):
        return self.opt.traj
    @property
    def atoms(self):
        return self.opt.atoms
        
    

class AIMD_solver():
    """
    """
    def __init__(self, pyscf_calc, T, taut=100, dt=10, nsteps=1, ofreq=1,
                 oname="aimd_traj.xyz", veloc=None, output="output.log",
                 fixcom=True, fixrot=True):
        
        # run GS calc
        if not pyscf_calc.converged:
            pyscf_calc.kernel()
        
        # initialize velocity
        if veloc == "rand":
            veloc = MaxwellBoltzmannVelocity(pyscf_calc.mol, T=T)

        # make a logger
        self.logger = AIMDTrajWriter(
            oname        = oname, # name output traj
            nstep        = ofreq,    # frequency at which traj is written
            data_output  = output, # also write output
            append       = False # append to existing traj
            )
        
        # make scanner
        scanner = pyscf_calc.as_scanner()

        self.integrator = NVTBerendson(scanner,
                                  dt    = dt,
                                  steps = nsteps,
                                  T     = T,
                                  taut  = taut,
                                  veloc =  veloc,
                                  callback = self.logger,
                                  fixrot = fixrot,
                                  fixcom = fixcom
                                  )
        
        return
    
    # run AIMD
    def run(self, veloc=None, steps=None, dump_flags=True, verbose=None):
        self.integrator.kernel(veloc=veloc, steps=steps,
                               dump_flags=dump_flags, verbose=verbose)
        return