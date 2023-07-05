#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 12:08:17 2023

@author: roncofaber
"""

import numpy as np
import scipy as sp

import os

import dill

from .spectra import Spectra

#%%

class Spectras():
    
    def __init__(self,
                 spectras,
                 path       = os.getcwd(),
                 npoints    = 3001,
                 save       = False,               # save or not
                 save_name  = "mbxas_spectras.pkl", # name of saved spectra (if save)
                 ):
        
        if isinstance(spectras, list):
            self.__initialize_spectras(spectras, path,  npoints,
                                       save, save_name)
        
        elif isinstance(spectras, str) and spectras.endswith(".pkl"): #restart from pickle
            self.__restart_from_pickle(spectras)
        else:
            raise "Wrong datatype"
        
        return
    
    def __initialize_spectras(self, spectras, path, npoints, save, save_name):
        
        
        # make spectra objects if not already
        if isinstance(spectras[0], str):
            spectras = [Spectra(file) for file in spectras]
            
            
        # get all signals to determine energy range
        all_x = np.array([sp.x for sp in spectras])
        
        # get emin, emax, erange
        emin   = np.min(all_x)
        emax   = np.max(all_x)
        erange = np.linspace(emin, emax, npoints)

        mean_signal = np.mean([sp.fspectra(erange) for sp in spectras], axis=0)
        
        # interpolate spectra
        intf = sp.interpolate.interp1d(erange, mean_signal,
                                       fill_value   = 0,
                                       bounds_error = False)
        
        # assign variables
        self.spectras = spectras
        self.x = erange
        self.y = mean_signal
        self.fspectra = intf
        
        # set standard variables
        self.__path = path
        self.__npoints = npoints
        
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