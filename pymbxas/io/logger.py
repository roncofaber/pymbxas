#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 17:41:10 2023

@author: roncoroni
"""

import sys
from io import StringIO

import logging
# set up logger object
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s |%(message)s",  # Modified format
    datefmt = "[%H:%M:%S]",
)

#%%

def configure_logger(level):
    
    # Define a mapping from user input to logging levels
    level_mapping = {
        1: logging.ERROR,
        2: logging.WARNING,
        3: logging.INFO,
        4: logging.DEBUG,
        5: logging.CRITICAL,
    }
    
    # level
    level = level_mapping[level]
    
    # Get the root logger
    root_logger = logging.getLogger()
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set the logging level
    root_logger.setLevel(level)
    
    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Create a formatter
    formatter = logging.Formatter("%(asctime)s |%(message)s", datefmt = "[%H:%M:%S]")
    
    # Set the formatter for the handler
    console_handler.setFormatter(formatter)
    
    # Add the handler to the root logger
    root_logger.addHandler(console_handler)
    
    return




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
    
