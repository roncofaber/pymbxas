#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 14:48:34 2024

@author: roncofaber
"""

import numpy as np

import pymbxas.utils.metrics as met
import pymbxas.utils.auxiliary as aux

import gpflow
import tensorflow as tf
#%%

# class to learn broadened XAS spectra
class BroadXAS():
    
    def __init__(self, spectras, xscaler="standard", yscaler="standard", metric=None,
                 isotropic=True, verbose=0, broaden=None, ykernel=None):
        
        # define metric (must work on ASE atoms object)
        if metric is None:
            metric = met.get_distances
        self._metric = metric
        
        # generate scaler for feature vector
        self.xscale = met.generate_scaler(xscaler)
        
        # calculate feature vector for all spectra
        structures = [sp.structure for sp in spectras]
        X  = self.metric(structures)
        Xs = self.xscale.fit_transform(X)
        
        # read data to use for fitting
        y_E, y_A = self._read_data(spectras, isotropic, broaden)
        
        self._y_A = y_A
        self._y_E = y_E
        
        self._npoints = broaden["npoints"]
        
        # scale data accordingly
        ys_A = self._scale_data(y_E, y_A, yscaler)
        
        # do fitting for energies and amplitudes
        self.kr_a = self._fit_amplitudes(Xs, ys_A, isotropic, ykernel)
        
        return
    
    # read data from spectra and return them
    @staticmethod
    def _read_data(spectras, isotropic, broaden):
        
        assert isinstance(broaden, dict)
        
        npoints = broaden["npoints"]
        erange  = broaden["erange"]
        sigma   = broaden["sigma"]
          
        # read spectral data
        y_A  = []
        for cc, spectra in enumerate(spectras):
            
            e, i = spectra.get_mbxas_spectra(npoints=npoints, erange=erange, sigma=sigma)
    
            y_A.append(i)
            
        # define values for fitting (convert to eV)
        y_E  = np.array(e)
        y_A  = np.array(y_A).reshape(-1, npoints)
        
        return y_E, y_A
    
    # take read data and return scaled data while generating the scalers
    def _scale_data(self, y_E, y_A, yscaler):
        
        self.yscale_A = met.generate_scaler(yscaler)
      
        # scale 'em
        ys_A = self.yscale_A.fit_transform(y_A)
        
        self._std_A = np.sqrt(self.yscale_A.var_)
        
        return ys_A
    
    @staticmethod
    def _fit_amplitudes(Xs, ys_A, isotropic, ykernel):
        
        assert isotropic, "Only isotropic implemented"
        
        model_A = gpflow.models.GPR(
            (Xs, ys_A),
            kernel=gpflow.kernels.SquaredExponential(),
        )
        
        # opt_A = tf.keras.optimizers.Adam()

        # @tf.function
        # def step() -> tf.Tensor:
        #     opt_A.minimize(model_A.training_loss, model_A.trainable_variables)
        
        # maxiter = 2000
        # for i in range(maxiter):
        #     step()
        #     if i % 100 == 0:
        #         print(i, model_A.training_loss().numpy())
        
        opt_A = gpflow.optimizers.Scipy()
        opt_A.minimize(model_A.training_loss, model_A.trainable_variables)
        
        return model_A
    
    def _predict_energy(self, Xtest):
        
        return self._y_E, np.zeros(len(self._y_E))
    
    def _predict_amplitude(self, Xtest):
        
        a_pre, a_std = self.kr_a.predict_y(Xtest)
        
        a_pre = a_pre.numpy().reshape(-1, self._npoints)
        a_std = self._std_A*a_std.numpy()
        
        return np.squeeze(self.yscale_A.inverse_transform(a_pre)), np.squeeze(a_std)
    
    
    def predict(self, structures):
        
        if not isinstance(structures, list): # it's a single structure
            Xtest = self.metric([structures])
        else:
            Xtest = self.metric(structures)
        
        # scale new input
        Xtest_scaled = self.xscale.transform(Xtest)
        
        e_pre, e_std = self._predict_energy(Xtest_scaled)
        a_pre, a_std = self._predict_amplitude(Xtest_scaled)
        
        return e_pre, e_std, a_pre, a_std
    
    def metric(self, X):
        return self._metric(X)
