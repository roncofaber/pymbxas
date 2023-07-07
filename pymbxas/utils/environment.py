#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:42:44 2023

@author: roncofaber
"""

import os
from pymbxas import __qcauxdir__, __qcdir__#, __scratchdir__
#%%

def set_qchem_environment(scratchdir=os.getcwd()):

    # Clearixs qchem env. variables
    os.environ["QC"]         =  __qcdir__
    os.environ["QCAUX"]      = __qcauxdir__
    os.environ["QCTHREADS"]  = '1'
    os.environ["QCPARALLEL"] = 'True'
    os.environ["QCSCRATCH"]  = scratchdir
    
    # Path variables
    os.environ["PATH"] += "{}/bin:{}/bin/perl".format(__qcdir__, __qcdir__)

    return