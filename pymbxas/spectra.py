#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 10:33:37 2023

@author: roncofaber
"""

# data manipulation
import scipy as sp
import numpy as np

# os and stuff
import os

# save 'n load
import dill

# internal imports
from .io.read import read_qchem_job

#%%
class Spectra():

    def __init__(self,
                 mbxas_spectra,                   # file with the mbxas spectra
                 qchem_out  = "qchem.output",     # name of qchem output file
                 read_qchem = False,              # read or not qchem out file
                 save       = False,              # save or not
                 save_name  = "mbxas_spectra.pkl" # name of saved spectra (if save)
                 ):
        
        if mbxas_spectra.endswith(".pkl"): #restart from pickle
            self.__restart_from_pickle(mbxas_spectra)
        else:
            # read spectra
            self.__initialize_spectra(mbxas_spectra, qchem_out, read_qchem,
                                      save_name)

        return
    
    # function that reads and initialize the spectra object
    def __initialize_spectra(self, mbxas_spectra, qchem_out, read_qchem,
                             save_name):
        
        # define path of data and other internal variables
        self.__path     = os.path.dirname(mbxas_spectra)
        self.__savename = save_name
        
        # read x,y values (ideally save sticks and not broad. spectra)
        x, y = np.loadtxt(mbxas_spectra).T
        
        # interpolate spectra
        intf = sp.interpolate.interp1d(x, y,
                                       fill_value   = 0,
                                       bounds_error = False)
        
        # parse other qchem information #TODO WIP
        if read_qchem:
            qchem = read_qchem_job(qchem_out, self.__path, "sp")
        else:
            qchem = None
        
        # assign variables
        self.x = x
        self.y = y
        self.fspectra = intf
        self.qchem = qchem
        
        return
    
    # save object as pkl file
    def save(self, oname=None, save_path=None):

        if oname is None:
            oname = self.__savename

        if save_path is None:
            path = self.__path
        else:
            path = save_path

        if oname.endswith(".pkl"):
            oname =  path + "/" + oname
        else:
            oname = path + "/" + oname.split(".")[-1] + ".pkl"

        with open(oname, 'wb') as fout:
            dill.dump(self, fout)

        print("Saved everything in:\n{}".format(oname))
        return

    # restart object from pkl file previously saved
    def __restart_from_pickle(self, mbxas_file):

        # open previously generated mbxas file
        with open(mbxas_file, "rb") as fin:
            restart = dill.load(fin)

        self.__dict__ = restart.__dict__.copy()

        # assert hasattr(self, "sea_urchin_object"), "Cannot load non compatible object!"

        return