#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 11:23:47 2023

@author: roncoroni
"""

import subprocess
import psutil
import numpy as np
import collections.abc

#%%
# get available memory either for CPU or GPU
def get_available_memory(is_gpu=False):
    if is_gpu:
        command = "nvidia-smi --query-gpu=memory.free --format=csv"
        memory_free_info = subprocess.check_output(command.split()).decode(
            'ascii').split('\n')[:-1][1:]
        memory_free_values = [int(x.split()[0]) for i, x in enumerate(
            memory_free_info)]
        return int(memory_free_values[0])
    
    else:
        return int(psutil.virtual_memory().available / 1e6)
    
    
# return copy of input as list if not one
def as_list(inp):
    if inp is None:
        return None
    elif isinstance(inp, int) or isinstance(inp, np.int64):
        return [inp]
    elif isinstance(inp, collections.abc.Iterable) and not isinstance(inp, str): 
        # Handles lists, tuples, NumPy arrays, etc. (Excludes strings)
        return list(inp)  
    else:
        raise TypeError(f"Cannot convert type {type(inp)} to list")
    

def s2i(string):
    if string.lower() == "beta":
        return 1
    elif string.lower() == "alpha":
        return 0
    else:
        raise TypeError
     
# see the docstring
def standardCount(label_arr, labels = None):
    '''
    Most Common Number of occurances of each label in each row of label_arr

    Parameters
    ----------
    labels : array (n_labels)
        List of target labels. If None assumes all unique elements of label_arr
    label_arr : array (n_rows, n_col)
        reference array to find most common elements.

    Returns
    -------
    List of most common frequency of each label (n_labels).

    '''
    labels = as_list(labels)
    
    label_arr = np.array(label_arr)
    
    if labels is None:
        labels = set(label_arr.reshape(-1))
        labels.discard(-1)
    ## How many occurances of a label in each structure
    label_count = np.array([np.sum(label_arr == label, axis= 1) for label in labels]).T
    
    ## The standard number of occurances of each cluster
    unique, counts = np.unique(label_count, axis= 0, return_counts= True)
    standard_count = unique[counts.argmax()]
    
    return np.squeeze(standard_count)
