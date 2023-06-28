#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 10:33:37 2023

@author: roncofaber
"""

import numpy as np
import os

from .io.read import read_qchem_job

#%%
class XAS_spectra():

    def __init__(self,
                 mbxas_spectra, #file with the mbxas spectra
                 qchem_out = "qchem.output"
                 ):

        # define path of data
        self.__path = os.path.dirname(mbxas_spectra)

        # read spectra
        self.x, self.y = np.loadtxt(mbxas_spectra).T


        self.po = read_qchem_job(qchem_out, self.__path, "sp")

        return

    # def