#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 15:32:40 2023

@author: roncofaber
"""

import os

import pymbxas as pym
import pymbxas.io

#%%

# initialize variables
run_path = os.getcwd()

mbxas_parameters = {
    'gridP'    : 1000,
    'highE'    : 0,#1127.5,
    'lowE'     : 0,#20,
    'sigma'    : 0.3,
    'do_align' : False,
    'DoRIXS'   : False 
    }
                       
pym.io.write.write_mbxas_input(mbxas_parameters, run_path)

mbxas_dir = '/global/home/groups/nano/share/software/electrolyte_machine/gitlab_repo/CleaRIXS/'

pym.io.run.submit_mbxas_job(run_path, mbxas_dir)
