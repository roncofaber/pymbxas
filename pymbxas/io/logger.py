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
    Custom logger to print to terminal and store the output as a string. Tracks file closure.
    """
    def __init__(self, print_to_terminal=True, log_file=None, append=False):
        self.print_to_terminal = print_to_terminal
        self.log = StringIO()
        self.log_file = log_file
        self.file = None
        self._isclosed = False

        if print_to_terminal:
            self.terminal_write = sys.stdout.write
        else:
            self.terminal_write = lambda message: None

        if log_file:
            try:
                if append:
                    self.file = open(log_file, 'a')
                else:
                    self.file = open(log_file, 'w')
            except (FileNotFoundError, OSError) as e:
                print(f"Error opening log file: {e}")
                self._isclosed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def write(self, message):
        self.terminal_write(message)
        self.log.write(message)
        if self.file and not self._isclosed:
            try:
                self.file.write(message)
            except (ValueError, OSError) as e:
                print(f"Error writing to log file: {e}. Setting closed flag.")
                self._isclosed = True

    def flush(self):
        pass

    def close(self):
        if self.file and not self._isclosed:
            try:
                self.file.close()
                self._isclosed = True
            except Exception as e:
                print(f"Error closing log file: {e}")

    def get_log(self):
        return self.log.getvalue()

    def is_closed(self):
        return self._isclosed
