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

Ha = 27.2113862161
#%%

def principal_invariants(A):
  """
  Calculates the principal invariants of a 3x3 tensor.

  Args:
    A: A 3x3 numpy array representing the tensor.

  Returns:
    A tuple containing the three principal invariants (I1, I2, I3).
  """

  # Calculate the trace of the tensor
  tr = np.trace(A)

  # Calculate the principal invariants
  I1 = tr
  I2 = 0.5*(tr**2 - np.trace(np.dot(A,A)))
  I3 = np.linalg.det(A)

  return I1, I2, I3

# class of a single electronic cluster
class spectralNode():
    
    def __init__(self, spectra_list, label, Xdata, yscaler="standard"):
        
        # read data to use for fitting
        Xs, y_E, y_A = self.__read_data(spectra_list, label, Xdata)
        
        # scale data accordingly
        ys_E, ys_A = self.__scale_data(y_E, y_A, yscaler)
        
        # do fitting for energies and amplitudes
        self.kr_e = self.__fit_energy(Xs, ys_E)
        self.kr_a = self.__fit_amplitudes(Xs, ys_A)
        
        # assign label variable
        self.label = label
            
        return
    
    # read data from spectra and return them
    @staticmethod
    def __read_data(spectra_list, label, Xdata):
        
        # read spectral data
        y_E  = []
        y_A  = []
        Xout = []
        for cc, spectra in enumerate(spectra_list):
            
            idxs = np.where(spectra._el_labels == label)[0]
            
            if len(idxs) != 1: #TODO fix for idxs > 1
                continue
            
            tensor = spectra.get_amplitude_tensor()
            
            idx = idxs[0]
            
            # I1, I2, I3 = principal_invariants(tensor[:,:,idx])
            
            y_E.append(spectra.energies[idx])
            y_A.append(spectra.amplitude[:,idx])
            # y_A.append([I1, I2, I3])
            Xout.append(Xdata[cc])
        
        # define values for fitting (convert to eV)
        y_E = Ha*np.array(y_E)
        y_A = abs(np.array(y_A)) #TODO ABS
        y_A = np.vstack([y_A.T, np.sum(y_A**2, axis=1)]).T
        Xout  = np.array(Xout) 
        
        return Xout, y_E, y_A
    
    # take read data and return scaled data while generating the scalers
    def __scale_data(self, y_E, y_A, yscaler):
        
        # generate data scalers
        self.yscale_E = clf.generate_scaler(yscaler)
        self.yscale_A = clf.generate_scaler(yscaler)
      
        # scale 'em
        ys_E = self.yscale_E.fit_transform(y_E.reshape(-1, 1))
        ys_A = self.yscale_A.fit_transform(y_A)
        
        return ys_E, ys_A
    
    @staticmethod
    def __fit_energy(Xs, ys_E):
        
        kernel = 5.0 * sker.RBF(
            length_scale        = 15.0,
            length_scale_bounds = (1e-3, 1e3)
            )
        
        k_e = GaussianProcessRegressor(
            kernel,
            n_restarts_optimizer = 25
            ).fit(Xs, ys_E)
        
        return k_e
    
    @staticmethod
    def __fit_amplitude(Xs, ys_A_axis):
        
        # kernel = 1*sker.RBF(
        #     length_scale        = 15.0,
        #     length_scale_bounds = (1e-4, 1e3)
        #     )# +1* sker.WhiteKernel()
            
        kernel = 1*sker.Matern(
                length_scale        = 15.0,
                length_scale_bounds = (1e-3, 1e3),
                nu = 2.5
            )
        
        k_a = GaussianProcessRegressor(
            kernel,
            n_restarts_optimizer = 10,
            # n_targets            = 1
            ).fit(Xs, ys_A_axis)
        
        return k_a
    
    def __fit_amplitudes(self, Xs, ys_A):
        
        k_amplitudes = []
        for ys_A_axis in ys_A.T:
            k_a = self.__fit_amplitude(Xs, ys_A_axis.reshape(-1, 1))
            k_amplitudes.append(k_a)
            
        return k_amplitudes
    
    def __predict_energy(self, Xtest):
        xas_energy = self.kr_e.predict(Xtest).reshape(-1, 1)
        return self.yscale_E.inverse_transform(xas_energy)
    
    def __predict_amplitude(self, Xtest):
        
        xas_amplitudes = []
        for kr_a in self.kr_a:
            xas_amplitude = kr_a.predict(Xtest)
            xas_amplitudes.append(xas_amplitude)
        
        xas_amplitudes = np.concatenate(xas_amplitudes).reshape(-1,4)
        
        return self.yscale_A.inverse_transform(xas_amplitudes)
    
    
    def predict(self, Xtest_scaled):
        
        y_e_pred = self.__predict_energy(Xtest_scaled)
        y_a_pred = self.__predict_amplitude(Xtest_scaled)
        
        return np.concatenate(y_e_pred), y_a_pred