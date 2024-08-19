#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 15:33:50 2024

@author: roncofaber
"""

import numpy as np

import sea_urchin.clustering.clusterize as clf

from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as sker

import pymbxas.utils.auxiliary as aux

import copy

import gpflow

# from ase import units
# Ha = units.Ha

#%%

# class of a single electronic cluster
class spectralNode():
    
    def __init__(self, spectras, label, Xdata, yscaler="standard",
                 isotropic=False, ykernel=None):
        
        # read data to use for fitting
        Xs, y_E, y_A, n_targets = self._read_data(spectras, label, Xdata, isotropic)
        
        self.y_A       = y_A
        self.n_targets = n_targets
        
        # scale data accordinglspectray
        ys_E, ys_A = self._scale_data(y_E, y_A, yscaler)
        
        # do fitting for energies and amplitudes
        self.kr_e = self._fit_energy(Xs, ys_E, n_targets, ykernel)
        self.kr_a = self._fit_amplitudes(Xs, ys_A, n_targets, isotropic, ykernel)
        
        # assign label variable
        self.label = label
            
        return
    
    # read data from spectra and return them
    @staticmethod
    def _read_data(spectras, label, Xdata, isotropic):
        
        # obtain number of targets
        n_targets = int(aux.standardCount([sp._el_labels for sp in spectras], label))
        
        # read spectral data
        y_E  = []
        y_A  = []
        Xout = []
        for cc, spectra in enumerate(spectras):
            
            idxs = np.where(spectra._el_labels == label)[0]
            
            # ignore if wrong number of targets
            if len(idxs) != n_targets:
                continue
    
            # append energies
            y_E.append(spectra.energies[idxs])
            
            # append values to fit amplitude
            amp = spectra.amplitude[:,idxs]
            y_A.append(amp**2) # append square value of the amplitude
            
            # store training coordinates
            Xout.append(Xdata[cc])
        
        # define values for fitting (convert to eV)
        y_E  = np.array(y_E).reshape(-1, n_targets)
        Xout = np.array(Xout) 
        y_A  = np.array(y_A)
        
        if isotropic:
            # y_A = np.mean(y_A, axis=1).reshape(-1, 1, n_targets)
            y_A = np.mean(y_A, axis=1).reshape(-1, n_targets)
        
        return Xout, y_E, y_A, n_targets
    
    # take read data and return scaled data while generating the scalers
    def _scale_data(self, y_E, y_A, yscaler):
        
        # generate data scalers
        self.yscale_E = clf.generate_scaler(yscaler)
        self.yscale_A = clf.generate_scaler(yscaler)
      
        # scale 'em
        ys_E = self.yscale_E.fit_transform(y_E)
        ys_A = self.yscale_A.fit_transform(y_A)
        
        return ys_E, ys_A
    
    @staticmethod
    def _fit_energy(Xs, ys_E, n_targets, ykernel):
        
        # if ykernel is None:
        #     kernel = 5.0 * sker.RBF(
        #         length_scale        = 15.0,
        #         length_scale_bounds = (1e-3, 1e3)
        #         )
        # else:
        #     kernel = copy.deepcopy(ykernel)
        
        # k_e = GaussianProcessRegressor(
        #     kernel,
        #     n_restarts_optimizer = 25,
        #     n_targets            = n_targets
        #     ).fit(Xs, ys_E)
        
        model_E = gpflow.models.GPR(
            (Xs, ys_E),
            kernel=gpflow.kernels.SquaredExponential(),
        )
        
        opt_E = gpflow.optimizers.Scipy()
        opt_E.minimize(model_E.training_loss, model_E.trainable_variables)
        
        return model_E
    
    @staticmethod
    def _fit_amplitudes(Xs, ys_A, n_targets, isotropic, ykernel):
        
        assert isotropic, "Only isotropic implemented"
        
        
        # if ykernel is None:
        #     kernel = sker.RBF() + sker.Matern()
        # else:
        #     kernel = copy.deepcopy(ykernel)

        # k_a = GaussianProcessRegressor(kernel,
        #                                n_restarts_optimizer = 10,
        #                                n_targets            = n_targets
        #                                ).fit(Xs, ys_A)
        
        model_A = gpflow.models.GPR(
            (Xs, ys_A),
            kernel=gpflow.kernels.SquaredExponential(),
        )
        
        opt_A = gpflow.optimizers.Scipy()
        opt_A.minimize(model_A.training_loss, model_A.trainable_variables)
        
        return model_A
    
    def _predict_energy(self, Xtest):
        xas_energy, _ = self.kr_e.predict_f(Xtest)
        xas_energy = np.array(xas_energy).reshape(-1, self.n_targets)
        return self.yscale_E.inverse_transform(xas_energy)
    
    def _predict_amplitude(self, Xtest):
        
        # xas_amplitudes = []
        # for kr_a in self.kr_a:
        #     xas_amplitude = kr_a.predict(Xtest)
        #     xas_amplitudes.append(xas_amplitude)
        
        xas_amplitudes, _ = self.kr_a.predict_f(Xtest)
        
        # xas_amplitudes = np.concatenate(xas_amplitudes).reshape(-1,self.n_targets)
        xas_amplitudes = np.array(xas_amplitudes).reshape(-1, self.n_targets)
        
        return self.yscale_A.inverse_transform(xas_amplitudes)
    
    
    def predict(self, Xtest_scaled):
        
        y_e_pred = self._predict_energy(Xtest_scaled)
        y_a_pred = self._predict_amplitude(Xtest_scaled)
        
        return np.squeeze(y_e_pred), np.squeeze(y_a_pred)
