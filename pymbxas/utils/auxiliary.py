#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 11:23:47 2023

@author: roncoroni
"""

import subprocess
import logging
import psutil
import numpy as np
import collections.abc

from ase import Atoms


logger = logging.getLogger(__name__)

#%%
# get available memory either for CPU or GPU
def get_available_memory(is_gpu=False):
    if is_gpu:
        try:
            command = "nvidia-smi --query-gpu=memory.free --format=csv"
            memory_free_info = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
            memory_free_values = [int(x.split()[0]) for x in memory_free_info]
            return memory_free_values[0]  # Return all GPU memory values
        except FileNotFoundError:
            logger.warning("Cannot query GPU memory: nvidia-smi was not found")
            return None
        except subprocess.CalledProcessError as e:
            logger.warning("Cannot query GPU memory: nvidia-smi failed: %s", e)
            return None
        except Exception as e:
            logger.warning("Cannot query GPU memory: %s", e)
            return None
    else:
        try:
            return int(psutil.virtual_memory().available / 1e6)
        except Exception as e:
            logger.warning("Cannot query available system memory: %s", e)
            return None

    
    
# return copy of input as list if not one
def as_list(inp):
    if inp is None:
        return None
    elif isinstance(inp, (int, np.integer)):
        return [inp]
    elif isinstance(inp, list):
        return inp
    elif isinstance(inp, Atoms):
        return [inp]
    elif isinstance(inp, collections.abc.Iterable) and not isinstance(inp, str):
        return list(inp)
    else:
        raise TypeError(f"Cannot convert type {type(inp)} to list")
