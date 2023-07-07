#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 13:09:15 2023

@author: roncofaber
"""

import os
import glob
import shutil

#%%

def copy_output_files(source_dir, target_dir):
    try:
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
            
        file_list = glob.glob(source_dir + "/*.txt")
        for file in file_list:
            shutil.copy(file, target_dir)
    except OSError:
        print("Copy failed!")
        pass
    
    return