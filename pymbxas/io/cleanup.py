#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 18:12:42 2023

@author: roncoroni
"""

import os
import glob
import logging

logger = logging.getLogger(__name__)

#%%

def remove_tmp_files(current_directory):

    file_list = glob.glob(os.path.join(current_directory, "tmp*.h5"))
    file_list.extend(glob.glob(os.path.join(current_directory, "tmp*.chk")))

    for file in file_list:
        if os.path.isfile(file):
            try:
                os.remove(file)
                logger.debug("Removed temporary file: %s", file)
            except OSError as e:
                logger.warning("Failed to remove temporary file %s: %s", file, e)

    return