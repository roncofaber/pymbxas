#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:36:48 2024

@author: roncofaber
"""

# ase stuff
import ase
from ase import units
import ase.visualize
from ase.calculators.singlepoint import SinglePointCalculator

# system stuff
import os
# this repo
from pymbxas.build.structure import mole_to_ase

#%%

au_2_fs = 0.02418884254

# class to add to a AIMD callback to print traj every n steps as ase atoms xyz
class AIMDTrajWriter():
    
    def __init__(self,
                 oname  = "aimd_trajectory.xyz", # name output traj
                 nstep  = 1,    # frequency at which traj is written
                 data_output  = None, # also write output
                 append       = False # append to existing traj
                 ):
        
        self.nstep = nstep
        self.oname = oname
        
        if os.path.exists(oname) and not append:
            os.remove(oname)
            
        self._setup_data_output(data_output)
        
        return
    
    def _setup_data_output(self, data_output):
        
        # avoid opening data_output file twice
        if type(data_output) is str:
            if os.path.isfile(data_output):
                print('overwrite data output file: %s' %
                      data_output)
            else:
                print('data output file: %s' % data_output)

            if data_output == '/dev/null':
                self.data_output = open(os.devnull, 'w')

            else:
                self.data_output = open(data_output, 'w')
                self.data_output.write(
                    'time          Epot                 Ekin                 '
                    'Etot                 T\n'
                )
        
        return
    
    def _write_data(self, data):
        '''Writes out the potential, kinetic, and total energy, temperature to the
        self.data_output stream. '''

        output = '%8.2f  %.12E  %.12E  %.12E %3.4f' % (data.time,
                                                       data.epot,
                                                       data.ekin,
                                                       data.ekin + data.epot,
                                                       data.temperature())

        # We follow OM of writing all the states at the end of the line
        if getattr(data.scanner.base, 'e_states', None) is not None:
            if len(data.scanner.base.e_states) > 1:
                for e in data.scanner.base.e_states:
                    output += '  %.12E' % e

        self.data_output.write(output + '\n')

        # If we don't flush, there is a possibility of losing data
        self.data_output.flush()
        
        return
    
    def __call__(self, aimd):
        
        data = aimd["self"]
        
        # if not multiple, do nothing unless is last frame
        if data._step % self.nstep != 0:
            if data._step == data.steps:
                pass
            else:
                return
        
        # convert mol to ase
        atoms = mole_to_ase(data.mol)
    
        atoms.info["epot"]  = data.epot
        atoms.info["ekin"]  = data.ekin
        atoms.info["veloc"] = data.veloc
        atoms.info["etot"]  = data.epot + data.ekin
        atoms.info["time"]  = data.time
        
        # set velocity
        atoms.set_velocities(data.veloc*units.Bohr/(au_2_fs*units.fs))
        
        # assign energy as single point calc #TODO expand with forces and such
        calc = SinglePointCalculator(atoms, energy=data.epot*units.Ha)
        atoms.set_calculator(calc)
        
        # write traj
        ase.io.write(self.oname, atoms, append=True)
        
        if self.data_output is not None:
            self._write_data(data)
        return
    
# class to callback a geometry optimization
class OptimizerTraj():
    
    def __init__(self, write=True, oname="optimization.xyz", append=False):
        
        self._traj   = []
        self.oname  = oname
        self._write = write
        
        if write and os.path.exists(oname) and not append:
            os.remove(oname)
            
        return
    
    # function that is called by the optimizer
    def __call__(self, opt):

        atoms = mole_to_ase(opt["mol"])
        atoms.info["etot"] = opt["energy"]
        
        # assign energy as single point calc #TODO expand with forces and such
        calc = SinglePointCalculator(atoms, energy=opt["energy"]*units.Ha)
        atoms.set_calculator(calc)
        
        # update optimizer internal data
        self.append(atoms)
        self.set_mol(opt["mol"], atoms)
        
        # write output if specified
        if self._write:
            ase.io.write(self.oname, atoms, append=True)
        
        return
    
    def set_mol(self, mol, atoms=None):
        self._mol   = mol
        
        if atoms is None:
            atoms = mole_to_ase(mol)
        else:
            atoms = atoms
        self._atoms = atoms
        return
    
    def append(self, element):
        self._traj.append(element)
        return
    
    def view(self):
        ase.visualize.view(self)
        return
    
    # make class iterable
    def __getitem__(self, index):
        return self._traj[index]
    
    def __iter__(self):
        return iter(self._traj)
    
    # get elements
    @property
    def mol(self):
        return self._mol
    @property
    def traj(self):
        return self._traj
    @property
    def atoms(self):
        return self._atoms