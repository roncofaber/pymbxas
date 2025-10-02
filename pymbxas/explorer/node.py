#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 15:33:50 2024

@author: roncofaber
"""

import gpflow
import numpy as np

from pymbxas.explorer.gpmodels import GPflow_model, MTGPyTorch_model
import pymbxas.utils.metrics as met

#%%
# Base class to perform spectra fitting.
class SpectralNode(object):
    
    def __init__(self, peak_label, broaden, isotropic, yscaler):
        
        # assign local variables
        self._label     = peak_label
        self._broaden   = broaden
        self._isotropic = isotropic
        self._eps       = 1e-8
        
        # initialize empty models
        self.kr_a, self.kr_e = None, None
        self.lgtshist = []
        
        # generate scalers
        self.yscaler = met.generate_scaler(yscaler)
        self.escaler = met.generate_scaler(yscaler)
        return
    
    @property
    def label(self):
        return self._label
    
    def _fit_amplitudes(self, Xs, Ys, parameters=None):
        assert self._isotropic, "So far only isotropic calculated"

        model = GPflow_model(Xs, Ys, parameters=parameters)
        model.train()
        
        # model = GPyTorch_model(Xs, Ys, parameters=parameters)
        # model = MTGPyTorch_model(Xs, Ys, parameters=parameters)
        # model.to_cuda()
        # model.train()
        
        return model
    
    def _fit_energies(self, Xs, Es, parameters=None):

        model = GPflow_model(Xs, Es, parameters=parameters)
        model.train()
        
        # model = GPyTorch_model(Xs, Es, parameters=parameters)
        # model = MTGPyTorch_model(Xs, Es, parameters=parameters)
        # model.to_cuda()
        # model.train()
        
        return model

    def predict(self, Xscaled):
        e_pre, e_std = self._predict_energy(Xscaled)
        Y_pre, Y_var = self._predict_amplitude(Xscaled)
        return e_pre, e_std, Y_pre, Y_var
    
    def _predict_amplitude(self, Xtest):
        
        # predict values
        Y_pre, Y_var = self.kr_a.predict(Xtest)
        
        # reshape and make it numpy
        Y_pre = Y_pre.reshape(-1, self._npoints)
        Y_var = Y_var.reshape(-1, self._npoints)
        
        Y_pre_uns, Y_var_uns = self.inverse_transform(Y_pre, Y_var)
        
        return np.squeeze(Y_pre_uns), np.squeeze(Y_var_uns)
    
    @property
    def n_targets(self):
        return self._npoints
    
    def fit_transform(self, Y):
        Ylog = np.log(Y + self._eps)
        return self.yscaler.fit_transform(Ylog)
    
    def transform(self, Y):
        Ylog = np.log(Y + self._eps)
        return self.yscaler.transform(Ylog)
    
    def inverse_transform(self, Ys, Yvar=None):
            # Inverse standard scaling
            Ylog = self.yscaler.inverse_transform(Ys)
            # Exponentiation
            Y = np.exp(Ylog)# - self._eps
    
            if Yvar is not None:
                # Inverse standard scaling of uncertainties
                Ylog_std = np.sqrt(Yvar)  # Yvar is the variance, so we take the sqrt to get the standard deviation
                Ylog_std_transformed = Ylog_std #* self.yscaler.scale_
    
                # Exponentiation of uncertainties
                Y_std = Y * Ylog_std_transformed
    
                # Convert standard deviation back to variance
                Yvar_transformed = Y_std ** 2
    
                return Y, Yvar_transformed
            else:
                return Y

#%%
# class of a single electronic cluster - broadened

class BroadenedNode(SpectralNode):
    
    def __init__(self,  spectras, Xdata, yscaler="standard", broaden=None,
                 peak_label=None, isotropic=True, ykernel=None):
        
        # run super
        super().__init__(peak_label, broaden, isotropic, yscaler)

        # assign local variables
        self._npoints = broaden["npoints"]
        
        # train model
        self.train(spectras, Xdata)
        
        return
    
    # read data from spectra and return them
    def _read_data(self, spectras, Xdata):
        
        assert isinstance(self._broaden, dict) #just make sure we are working
        
        npoints = self._broaden["npoints"]
        erange  = self._broaden["erange"]
        sigma   = self._broaden["sigma"]
          
        # read spectral data
        Y    = []
        Xout = []
        for cc, spectra in enumerate(spectras):
            
            energy, amplitude = spectra.get_mbxas_spectra(npoints  = npoints,
                                                          erange   = erange,
                                                          sigma    = sigma,
                                                          el_label = self._label)
            if amplitude is None:
                continue # skip this spectra and forget about it
            else: # add data
                Y.append(amplitude)
                Xout.append(Xdata[cc])
                E    = np.array(energy)
            
        # define values for fitting (convert to eV)
        Xout = np.array(Xout)
        Y    = np.array(Y).reshape(-1, npoints)
        
        return Xout, E, Y
    
    # dummy replace for en(200ish)ergy prediction (not necessary)
    def _predict_energy(self, Xtest):
        e_pre = np.squeeze(np.tile(self._E, (len(Xtest), 1)))
        e_std = np.zeros(e_pre.shape)
        return e_pre, e_std
    
    # do a training cycle
    def train(self, spectras, Xdata, retrain=False, premodel=None):
        
        # read data to use for fitting
        Xs, E, Y  = self._read_data(spectras, Xdata)
        
        # scale data accordingly
        Ys = self.fit_transform(Y)
        
        if retrain: # reuse parameters
            if premodel is None:
                if self.kr_a is None: raise ValueError("Model has not been initialized.")
                parameters = gpflow.utilities.parameter_dict(self.kr_a)
            else:
                parameters = gpflow.utilities.parameter_dict(premodel)
        else:
            parameters = None
        
        # do fitting for energies and amplitudes
        self.kr_a = self._fit_amplitudes(Xs, Ys, parameters=parameters)
        
        # store data to check
        self._Xs, self._E, self._Y, self._Ys = Xs, E, Y, Ys
        
        return



#%%
# class of a single electronic cluster - discrete
class DiscreteNode(SpectralNode):
    
    def __init__(self,  spectras, Xdata, yscaler="standard", broaden=None,
                 peak_label=None, isotropic=True, ykernel=None):
        
        # run super
        super().__init__(peak_label, broaden, isotropic, yscaler)

        # assign local variables
        self._npoints = broaden["npeaks"]
        
        # train model
        self.train(spectras, Xdata)
        
        return
    
    # read data from spectra and return them
    def _read_data(self, spectras, Xdata):
        
        assert isinstance(self._broaden, dict) #just make sure we are working
        
        npeaks = self._broaden["npeaks"]
        sigma  = self._broaden["sigma"]
          
        # read spectral data
        E    = []
        Y    = []
        Xout = []
        for cc, spectra in enumerate(spectras):
            
            erange = spectra.energies[:npeaks]
            
            energy, amplitude = spectra.get_mbxas_spectra(erange   = erange,
                                                          sigma    = sigma,
                                                          el_label = self._label)
            if amplitude is None:
                continue # skip this spectra and forget about it
            else: # add data
                Y.append(amplitude)
                Xout.append(Xdata[cc])
                E.append(energy)
            
        # define values for fitting (convert to eV)
        Xout = np.array(Xout)
        Y    = np.array(Y).reshape(-1, npeaks)
        E    = np.array(E)
        
        return Xout, E, Y
    
    # predict energy
    def _predict_energy(self, Xtest):
        
        # predict values
        E_pre, E_var = self.kr_e.predict(Xtest)
        
        # reshape and make it numpy
        E_pre = E_pre.numpy().reshape(-1, self._npoints)
        E_var = E_var.numpy().reshape(-1, self._npoints)
        
        E_pre_uns = self.escaler.inverse_transform(E_pre)#, Y_var)
        
        return np.squeeze(E_pre_uns), np.squeeze(E_var)
    
    # do a training cycle
    def train(self, spectras, Xdata, retrain=False, premodel=None):
        
        # read data to use for fitting
        Xs, E, Y  = self._read_data(spectras, Xdata)
        
        # scale data accordingly
        Ys = self.fit_transform(Y)
        Es = self.escaler.fit_transform(E)
        
        if retrain: # reuse parameters
            if premodel is None:
                if self.kr_a is None: raise ValueError("Model has not been initialized.")
                parameters = gpflow.utilities.parameter_dict(self.kr_a)
            else:
                parameters = gpflow.utilities.parameter_dict(premodel)
        else:
            parameters = None
        
        # do fitting for energies and amplitudes
        self.kr_a = self._fit_amplitudes(Xs, Ys, parameters=parameters)
        self.kr_e = self._fit_energies(Xs, Es, parameters=None)
        
        # store data to check
        self._Xs, self._E, self._Es, self._Y, self._Ys = Xs, E, Es, Y, Ys
        
        return
