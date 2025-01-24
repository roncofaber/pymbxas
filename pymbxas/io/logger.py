#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 17:41:10 2023

@author: roncoroni
"""

import sys
from io import StringIO

#%%

class Logger(object):
    """
    Custom logger to print to terminal and store the output as a string.
    
    Args:
        print_to_terminal (bool): If True, prints to terminal.
        log_file (str): Optional path to a log file.
    """
    def __init__(self, print_to_terminal=True, log_file=None, append=False):
        self.print_to_terminal = print_to_terminal
        self.log = StringIO()
        self.log_file = log_file
        
        if print_to_terminal:
            self.terminal_write = sys.stdout.write
        else:
            self.terminal_write = lambda message: None  # A dummy function that does nothing
        
        if log_file:
            if append:
                self.file = open(log_file, 'a')
            else:
                self.file = open(log_file, 'w')
        else:
            self.file = None
    
    def write(self, message):
        self.terminal_write(message)
        self.log.write(message)
        if self.file:
            self.file.write(message)
    
    def flush(self):
        """
        This flush method is needed for Python 3 compatibility.
        This handles the flush command by doing nothing.
        """
        pass
    
    def close(self):
        """
        Close the file handler if it was opened.
        """
        if self.file:
            self.file.close()

    def get_log(self):
        """
        Retrieve the log messages stored in the StringIO object.
        """
        return self.log.getvalue()
