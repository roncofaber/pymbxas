#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 11:23:47 2023

@author: roncoroni
"""

import subprocess as sp
import os
import psutil

def get_available_memory(is_gpu=False):
    if is_gpu:
        command = "nvidia-smi --query-gpu=memory.free --format=csv"
        memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
        return int(memory_free_values[0])
    
    else:
        return int(psutil.virtual_memory().available / 1e6)
    
    
# return copy of input as list if not one
def as_list(inp):
    return [inp] if not isinstance(inp, list) else inp.copy()

def s2i(string):
    if string == "beta":
        return 1
    elif string == "alpha":
        return 0
    else:
        raise "ERROR CHANNEL"