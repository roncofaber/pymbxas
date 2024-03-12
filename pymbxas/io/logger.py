#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 17:41:10 2023

@author: roncoroni
"""

import sys
from io import StringIO

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
    
