#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 15:32:48 2024

@author: roncofaber
"""

# numpy is my rock and I am ready to roll
import numpy as np

# sea urchin, maybe not needed in the future #TODO
import sea_urchin.clustering.metrics as met
import sea_urchin.clustering.clusterize as clf

# pymbxas stuff
from pymbxas.explorer.node import spectralNode
from pymbxas.mbxas.broaden import get_mbxas_spectra

# learning stuff
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as sker

# instead of random floats, let's make some order
from ase import units
Ha = units.Ha

#%%

# wrapper class to predict a spectra
class MBXASplorer():
    
    def __init__(self, spectras, fit_labels=None, xscaler="standard",
                 yscaler="standard", metric=None, isotropic=True, verbose=0,
                 ykernel=None):
        
        # set up internal variables
        self._verbose = verbose
        self._is_isotropic = isotropic
        self._ykernel = ykernel
        
        self._nodes  = []
        self._labels = []
        
        # define labels to fit
        if fit_labels is None:
            fit_labels = np.unique([sp._el_labels for sp in spectras])
        else:
            fit_labels = [ii for ii in np.unique(fit_labels)
                          if ii in np.unique([sp._el_labels for sp in spectras])]
            
        # define metric (must work on ASE atoms object)
        if metric is None:
            metric = met.get_distances
        self._metric = metric
        
        # generate scaler for feature vector
        self.xscale = clf.generate_scaler(xscaler)
        
        # calculate feature vector for all spectra
        structures = [sp.structure for sp in spectras]
        X  = self.metric(structures)
        Xs = self.xscale.fit_transform(X)
        
        # read GS data
        ys_G = self._read_scale_GS_energy(spectras, yscaler)
        
        # first, predict GS energy
        self.kr_g = self._fit_gs_energy(Xs, ys_G)

        # fit each peak
        for peak_label in fit_labels:
            self._make_node(spectras, peak_label, Xs, yscaler, isotropic, ykernel)
            
        return
    
    def _make_node(self, spectras, peak_label, Xs, yscaler, isotropic, ykernel):
        
        if peak_label == -1 or peak_label in self._labels: # ignore noise
            return
        
        # fit a single excitation
        sn = spectralNode(spectras, peak_label, Xs, yscaler=yscaler,
                          isotropic=isotropic, ykernel=ykernel)
          
        # add it to the list #TODO: add later as well
        self._nodes.append(sn)
        self._labels.append(peak_label)
        
        # if self._verbose:
            # print(sn.kr_a.kernel_)
                
        return
    
    def predict(self, structures, node=None):
        
        if not isinstance(structures, list): # it's a single structure
            Xtest = self.metric([structures])
        else:
            Xtest = self.metric(structures)
        
        # scale new input
        Xtest_scaled = self.xscale.transform(Xtest)
        
        y_G = self._predict_gs_energy(Xtest_scaled)
        
        if node is not None:
            return *self[node].predict(Xtest_scaled), np.squeeze(y_G)
        
        else:
            
            y_E, y_A = [], []
            
            for node in self:
                e, a = node.predict(Xtest_scaled)
                y_E.append(e.reshape(-1, node.n_targets))
                y_A.append(a.reshape(-1, node.n_targets))
                
            # y_E = np.squeeze(y_E) #TODO check this with isotropic and not
            # y_A = np.squeeze(y_A)
        
        y_G = np.squeeze(y_G)
                
        return np.hstack(y_E), np.hstack(y_A), y_G
    
    # generate a MBXAS spectra predicted from a structure
    def get_mbxas_spectra(self, structure, axis=None, sigma=0.02, npoints=3001,
                          tol=0.01, erange=None):
        
        # predict E and A
        energies, amplitude, __ = self.predict(structure)
        
        # get broadened spectra
        erange, spectras = get_mbxas_spectra(energies, amplitude,
                                             sigma=sigma, npoints=npoints,
                                             tol=tol, erange=erange,
                                             isotropic=self._is_isotropic)
        
        if isinstance(structure, list):
            norm = len(structure)
        else:
            norm = 1
        
        #return
        if axis is None:
            spectras = np.mean(spectras, axis=0)/norm
        else:
            spectras = spectras[axis]/norm
        
        return erange, spectras
    
    def _read_scale_GS_energy(self, spectras, yscaler):
        
        y_G = np.array([sp.gs_energy for sp in spectras])
  
        self.yscale_G = clf.generate_scaler("none")
        
        ys_G = self.yscale_G.fit_transform(y_G.reshape(-1, 1))
        
        return ys_G
    
    @staticmethod
    def _fit_gs_energy(Xs, ys_G):
        
        kernel = 3.0 * sker.RBF(
            length_scale        = 15.0,
            length_scale_bounds = (1e-3, 1e3)
            )
        
        kr_g = GaussianProcessRegressor(
            kernel,
            n_restarts_optimizer = 30
            ).fit(Xs, ys_G)

        return kr_g
    
    def _predict_gs_energy(self, Xtest):
        gs_energy = self.kr_g.predict(Xtest).reshape(-1, 1)
        return Ha*self.yscale_G.inverse_transform(gs_energy)
    
    # make class iterable
    def __getitem__(self, index):
        return self._nodes[index]
    
    def __iter__(self):
        return iter(self._nodes)
    
    def __len__(self):
        return len(self.nodes)
    
    @property
    def nodes(self):
        return self._nodes

    def metric(self, X):
        return self._metric(X)
