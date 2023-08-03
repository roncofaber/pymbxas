#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 18:12:42 2023

@author: roncoroni
"""

import os
import glob

#%%

def remove_tmp_files(current_directory):
    
    file_list = glob.glob(current_directory + "/tmp*")
    
    for file in file_list:
        if os.path.isfile(file):
            os.remove(file)
            
    return