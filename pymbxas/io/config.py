#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 14:35:55 2025

@author: roncofaber
"""

import sys
import logging
# set up logger object
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s |%(message)s",  # Modified format
    datefmt = "[%H:%M:%S]",
)

#%%

class LevelFirstLetterFilter(logging.Filter):
    """
    Custom filter to add the first letter of the log level to the log record.
    """
    def filter(self, record):
        record.level_first_letter = record.levelname[0]
        return True

def configure_logger(level, log_file=None):
    """
    Configures the root logger with the specified logging level.
    
    Args:
        level (int): Logging level (1-5).
        log_file (str): Optional path to a log file.
    """
    # Define a mapping from user input to logging levels
    level_mapping = {
        1: logging.ERROR,
        2: logging.WARNING,
        3: logging.INFO,
        4: logging.DEBUG,
        5: logging.CRITICAL,
    }
    
    if level not in level_mapping:
        raise ValueError("Invalid logging level: {}. Choose a level between 1 and 5.".format(level))
    
    # Map the user input level to logging level
    level = level_mapping[level]
    
    # Get the root logger
    root_logger = logging.getLogger()
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Set the logging level
    root_logger.setLevel(level)
    
    # Create a console handler using stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Create a formatter
    formatter = logging.Formatter("%(asctime)s |(%(level_first_letter)s) %(message)s", datefmt="[%H:%M:%S]")
    
    # Set the formatter for the handler
    console_handler.setFormatter(formatter)
    
    # Add the custom filter to the handler
    console_handler.addFilter(LevelFirstLetterFilter())
    
    # Add the handler to the root logger
    root_logger.addHandler(console_handler)
    
    # If log_file is specified, create a file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(LevelFirstLetterFilter())
        root_logger.addHandler(file_handler)
        
    return
