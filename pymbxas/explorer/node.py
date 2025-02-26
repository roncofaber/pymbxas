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

from scipy.spatial.distance import pdist, squareform

#%%
# Base class to perform spectra fitting.
class SpectralNode(object):
    
    def __init__(self, peak_label, broaden, isotropic, yscaler):
        
        # assign local variables
        self._label     = peak_label
        self._broaden   = broaden
        self._isotropic = isotropic
        
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

        # Visual inspection: Estimate variance from the range of y
        vras = np.var(Ys) # Or a visual estimate
        
        # Heuristic for lengthscales:
        distances = pdist(Xs.T)
        distances_matrix = squareform(distances)
        lgts = np.median(distances_matrix, axis=0)

        my_kernel = gpflow.kernels.Matern32(variance     = vras,
                                            lengthscales = lgts,
                                            )
        
        # create GP model
        # mean_function = gpflow.mean_functions.Constant() <<- assume data is scaled to zero
        model = gpflow.models.GPR(
            (Xs, Ys),
            kernel         = my_kernel,
            # mean_function  = mean_function,
            noise_variance = 5e-6 # smaller than this gives error... annoying...
        )
        
        # reassign parameters
        if parameters is not None:
            gpflow.utilities.multiple_assign(model, parameters)
            
        # set variance as NOT trainable --> check this
        gpflow.utilities.set_trainable(model.likelihood, False)
        
        # run optimizer
        opt = gpflow.optimizers.Scipy()
        opt.minimize(
            model.training_loss,
            model.trainable_variables,
            options = dict(maxiter=2500),
            method  = "l-bfgs-b",
        )
        
        # add parameters to history
        self.lgtshist.append(model.parameters[0].numpy())
        
        return model, opt
    
    def predict(self, Xscaled):
        e_pre, e_std = self._predict_energy(Xscaled)
        Y_pre, Y_var = self._predict_amplitude(Xscaled)
        return e_pre, e_std, Y_pre, Y_var
    
    def _predict_amplitude(self, Xtest):
        
        # predict values
        Y_pre, Y_var = self.kr_a.predict_f(Xtest)
        
        # reshape and make it numpy
        Y_pre = Y_pre.numpy().reshape(-1, self._npoints)
        Y_var = Y_var.numpy().reshape(-1, self._npoints)
        
        Y_pre_uns, Y_var_uns = self.inverse_transform(Y_pre, Y_var)
        
        return np.squeeze(Y_pre_uns), np.squeeze(Y_var_uns)
    
    @property
    def n_targets(self):
        return self._npoints
    
    @property
    def _std_A(self):
        try:
            variance = self.yscaler.var_
        except:
            return 0
        if variance is None:
            return 1
        else:
            return np.sqrt(variance)
    
    @property
    def _std_E(self):
        variance = self.escaler.var_
        if variance is None:
            return 1
        else:
            return np.sqrt(variance)
        
    def fit_transform(self, Y):
        Ylog = np.log(Y + 1e-12)
        return self.yscaler.fit_transform(Ylog)
    
    def transform(self, Y):
        Ylog = np.log(Y + 1e-12)
        return self.yscaler.transform(Ylog)
    
    def inverse_transform(self, Ys, Yvar=None):
        
        Ylog = self.yscaler.inverse_transform(Ys)
        Y = np.exp(Ylog)

        if Yvar is not None:
            
            # Yvar_log = self.yscaler.var_*Yvar
            # Yvar = (np.exp(Yvar_log) - 1) * np.exp(2*self.yscaler.mean_ + Yvar_log)
            
            return Y, Yvar
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
            
        # define values for fitting (convert to eV)
        Xout = np.array(Xout)
        E    = np.array(energy)
        Y    = np.array(Y).reshape(-1, npoints)
        
        return Xout, E, Y
    
    # dummy replace for en(200ish)ergy prediction (not necessary)
    def _predict_energy(self, Xtest):
        e_pre = np.squeeze(np.tile(self._E, (len(Xtest), 1)))
        e_std = np.zeros(e_pre.shape)
        return e_pre, e_std
    
    # do a training cycle
    def train(self, spectras, Xdata, retrain=False):
        
        # read data to use for fitting
        Xs, E, Y  = self._read_data(spectras, Xdata)
        
        # scale data accordingly
        Ys = self.fit_transform(Y)
        
        if retrain: # reuse parameters
            if self.kr_a is None: raise ValueError("Model has not been initialized.")
            parameters = gpflow.utilities.parameter_dict(self.kr_a)
        else:
            parameters = None
            
        # do fitting for energies and amplitudes
        self.kr_a, self.opt_a = self._fit_amplitudes(Xs, Ys, parameters=parameters)
        
        # store data to check
        self._Xs, self._E, self._Y, self._Ys = Xs, E, Y, Ys
        
        return



#%%
# class of a single electronic cluster - discrete

class DiscreteNode(SpectralNode):
    
    def __init__(self, spectras, Xdata, yscaler="standard",
                 isotropic=False, ykernel=None, peak_label=None):
        
        super().__init__(peak_label, None, isotropic)
        
        # read data to use for fitting
        self._Xs, E, A, n_targets = self._read_data(spectras, Xdata)
        
        self._E = E
        self._A = A
        self._npoints = n_targets
        
        # scale data accordinglspectray
        self._Es, self._Ys = self._generate_scaler_and_scale(E, A, yscaler)
        
        
        # do fitting for energies and amplitudes
        self.kr_e = self._fit_energy(self._Xsyscaler, self._Es, n_targets, ykernel)
        self.kr_a = self._fit_amplitudes(self._Xs, self._Ys)
        
        return
    
    # read data from spectra and return them
    def _read_data(self, spectras, Xdata):
        
        # obtain number of targets
        n_targets = int(aux.standardCount([sp._el_labels for sp in spectras],
                                          self._label))
        
        if n_targets == 0:
            print("WARNING: 0")
            n_targets = 1
        
        # read spectral data
        E    = []
        Y    = []
        Xout = []
        for cc, spectra in enumerate(spectras):
            
            idxs = np.where(spectra._el_labels == self._label)[0]
            
            # ignore if wrong number of targets
            if len(idxs) != n_targets:
                continue
    
            # append energies
            E.append(spectra.energies[idxs])
            
            # append values to fit amplitude
            amp = spectra.amplitude[:,idxs]
            Y.append(amp**2) # append square value of the amplitude
            
            # store training coordinates
            Xout.append(Xdata[cc])
        
        # define values for fitting (convert to eV)
        E    = np.array(E).reshape(-1, n_targets)
        Xout = np.array(Xout) 
        Y    = np.array(Y)
        
        if self._isotropic:
            Y = np.mean(Y, axis=1).reshape(-1, n_targets)
        
        return Xout, E, Y, n_targets
    
    # take read data and return scaled data while generating the scalers
    def _generate_scaler_and_scale(self, E, Y, yscaler):
        
        # generate data scalers
        self.escaler = met.generate_scaler(yscaler)
        self.yscaler = met.generate_scaler(yscaler)
      
        # scale 'em
        Es = self.escaler.fit_transform(E)
        Ys = self.yscaler.fit_transform(Y)

        return Es, Ys
    
    @staticmethod
    def _fit_energy(Xs, Es, n_targets, ykernel):
        
        model_E = gpflow.models.GPR(
            (Xs, Es),
            kernel=gpflow.kernels.SquaredExponential(),
        )
        
        opt_E = gpflow.optimizers.Scipy()
        opt_E.minimize(model_E.training_loss, model_E.trainable_variables)
        
        return model_E
    
    def _predict_energy(self, Xtest):
        
        e_pre, e_std = self.kr_e.predict_y(Xtest)
        
        e_pre = e_pre.numpy().reshape(-1, self.n_targets)
        e_std = self._std_E*e_std.numpy()
        
        return np.squeeze(self.escaler.inverse_transform(e_pre)), np.squeeze(e_std)
