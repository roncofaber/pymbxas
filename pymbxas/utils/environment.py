#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:42:44 2023

@author: roncofaber
"""

import os
from pymbxas import __qcauxdir__, __qcdir__
#%%

def get_qchem_version_from_output(output):
    for line in output.splitlines():
        if line.strip().startswith("Q-Chem"):
            version = float(line.strip().split()[1])
            return version

def set_qchem_environment(scratchdir=None):

    # scratch in current directory if not specified
    if scratchdir is None:
        scratchdir = os.getcwd()

    # Clearixs qchem env. variables
    os.environ["QC"]         =  __qcdir__
    os.environ["QCAUX"]      = __qcauxdir__
    os.environ["QCTHREADS"]  = '1'
    os.environ["QCPARALLEL"] = 'True'
    os.environ["QCSCRATCH"]  = scratchdir

    # Path variables
    path_string = "{}/bin:{}/bin/perl".format(__qcdir__, __qcdir__)
    if path_string not in os.environ["PATH"]:
        os.environ["PATH"] += "{}/bin:{}/bin/perl".format(__qcdir__, __qcdir__)

    return