#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 15:33:50 2024

@author: roncofaber
"""

import numpy as np

import pymbxas.utils.metrics as met
import pymbxas.utils.auxiliary as aux

import gpflow

#%%
# Base class to perform spectra fitting.
class SpectralNode(object):
    
    def __init__(self, peak_label):
        
        # assign label variable
        self._label = peak_label
        
        return
    
    @property
    def label(self):
        return self._label
    
    def predict(self, Xscaled):
        e_pre, e_std = self._predict_energy(Xscaled)
        a_pre, a_std = self._predict_amplitude(Xscaled)
        return e_pre, e_std, a_pre, a_std
    
    def _predict_amplitude(self, Xtest):
        
        a_pre, a_std = self.kr_a.predict_f(Xtest)
        
        a_pre = a_pre.numpy().reshape(-1, self._npoints)
        a_std = self._std_A*a_std.numpy()
        
        return np.squeeze(self.yscale_A.inverse_transform(a_pre)), np.squeeze(a_std)
    
    @staticmethod
    def _fit_amplitudes(Xs, ys_A, npoints, isotropic, ykernel):
        
        assert isotropic, "So far only isotropic calculated"

        ## SIMPLE CASE
        lgts = np.ones(Xs.shape[1])
        vras = 1.6
        my_kernel = gpflow.kernels.Matern32(variance=vras, lengthscales=lgts)
        
        model_A = gpflow.models.GPR(
            (Xs, ys_A),
            kernel         = my_kernel,
            # num_latent_gps = self._npoints,
            # noise_variance = 1e-6,
            )
        
        opt_A = gpflow.optimizers.Scipy()
        opt_A.minimize(model_A.training_loss, model_A.trainable_variables)
        
        return model_A
    
    @property
    def n_targets(self):
        return self._npoints
    
#%%
# class of a single electronic cluster - discrete

class DiscreteNode(SpectralNode):
    
    def __init__(self, spectras, Xdata, yscaler="standard",
                 isotropic=False, ykernel=None, peak_label=None):
        
        super().__init__(peak_label)
        
        # read data to use for fitting
        Xs, y_E, y_A, n_targets = self._read_data(spectras, Xdata, peak_label, isotropic)
        
        self.y_A      = y_A
        self._npoints = n_targets
        
        # scale data accordinglspectray
        ys_E, ys_A = self._scale_data(y_E, y_A, yscaler)
        
        # do fitting for energies and amplitudes
        self.kr_e = self._fit_energy(Xs, ys_E, n_targets, ykernel)
        self.kr_a = self._fit_amplitudes(Xs, ys_A, n_targets, isotropic, ykernel)
        
        return
    
    # read data from spectra and return them
    @staticmethod
    def _read_data(spectras, Xdata, peak_label, isotropic):
        
        # obtain number of targets
        n_targets = int(aux.standardCount([sp._el_labels for sp in spectras], peak_label))
        
        if n_targets == 0:
            print("WARNING: 0")
            n_targets = 1
        
        # read spectral data
        y_E  = []
        y_A  = []
        Xout = []
        for cc, spectra in enumerate(spectras):
            
            idxs = np.where(spectra._el_labels == peak_label)[0]
            
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
        self.yscale_E = met.generate_scaler(yscaler)
        self.yscale_A = met.generate_scaler(yscaler)
      
        # scale 'em
        ys_E = self.yscale_E.fit_transform(y_E)
        ys_A = self.yscale_A.fit_transform(y_A)
        
        self._std_E = np.sqrt(self.yscale_E.var_)
        self._std_A = np.sqrt(self.yscale_A.var_)
        
        return ys_E, ys_A
    
    @staticmethod
    def _fit_energy(Xs, ys_E, n_targets, ykernel):
        
        model_E = gpflow.models.GPR(
            (Xs, ys_E),
            kernel=gpflow.kernels.SquaredExponential(),
        )
        
        opt_E = gpflow.optimizers.Scipy()
        opt_E.minimize(model_E.training_loss, model_E.trainable_variables)
        
        return model_E
    
    def _predict_energy(self, Xtest):
        
        e_pre, e_std = self.kr_e.predict_f(Xtest)
        
        e_pre = e_pre.numpy().reshape(-1, self.n_targets)
        e_std = self._std_E*e_std.numpy()
        
        return np.squeeze(self.yscale_E.inverse_transform(e_pre)), np.squeeze(e_std)

#%%
# class of a single electronic cluster - broadened

class BroadenedNode(SpectralNode):
    
    def __init__(self,  spectras, Xdata, yscaler="standard", broaden=None,
                 peak_label=None, isotropic=True, ykernel=None):
        
        super().__init__(peak_label)
        
        # read data to use for fitting
        Xs, y_E, y_A  = self._read_data(spectras, Xdata, broaden, peak_label)
        self._y_E = y_E
        self._y_A = y_A
        
        self._npoints = broaden["npoints"]
        
        # scale data accordingly
        ys_A = self._scale_data(y_E, y_A, yscaler)
        
        # do fitting for energies and amplitudes
        self.kr_a = self._fit_amplitudes(Xs, ys_A, broaden["npoints"], isotropic, ykernel)
        
        return
    
    # read data from spectra and return them
    @staticmethod
    def _read_data(spectras, Xdata, broaden, el_label):
        
        assert isinstance(broaden, dict)
        
        npoints = broaden["npoints"]
        erange  = broaden["erange"]
        sigma   = broaden["sigma"]
          
        # read spectral data
        y_A  = []
        Xout = []
        for cc, spectra in enumerate(spectras):
            
            e, i = spectra.get_mbxas_spectra(npoints=npoints, erange=erange,
                                             sigma=sigma, el_label=el_label)
            if i is None:
                continue
            else:
                y_A.append(i)
                Xout.append(Xdata[cc])
            
        # define values for fitting (convert to eV)
        Xout = np.array(Xout)
        y_E  = np.array(e)
        y_A  = np.array(y_A).reshape(-1, npoints)
        
        return Xout, y_E, y_A
    
    # take read data and return scaled data while generating the scalers
    def _scale_data(self, y_E, y_A, yscaler):
        
        self.yscale_A = met.generate_scaler(yscaler)
      
        # scale 'em
        ys_A = self.yscale_A.fit_transform(y_A)
        
        self._std_A = np.sqrt(self.yscale_A.var_)
        
        return ys_A
    
    # dummy replace for energy prediction (not necessary)
    def _predict_energy(self, Xtest):
        e_pre = np.squeeze(np.tile(self._y_E, (len(Xtest), 1)))
        e_std = np.zeros(e_pre.shape)
        return e_pre, e_std
    

    
