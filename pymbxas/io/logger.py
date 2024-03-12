#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 17:41:10 2023

@author: roncoroni
"""

# ase stuff
import ase

# system stuff
import os
import sys
from io import StringIO

# this repo
from pymbxas.build.structure import mole_to_ase

#%%

# Logger to both print to terminal but store the output as string
class Logger(object):
    def __init__(self, print_to_terminal=True):
        self.print_to_terminal = print_to_terminal
        if print_to_terminal:
            self.terminal_write = sys.stdout.write
        else:
            self.terminal_write = lambda message: None  # A dummy function that does nothing
        self.log = StringIO()
        
    def write(self, message):
        self.terminal_write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass
    
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
        
        print(data._step)
        
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
        
        ase.io.write(self.oname, atoms, append=True)
        
        if self.data_output is not None:
            self._write_data(data)
        
        return