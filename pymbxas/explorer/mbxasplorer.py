#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 15:32:48 2024

@author: roncofaber
"""

# numpy is my rock and I am ready to roll
import numpy as np

# pymbxas stuff
import pymbxas.utils.metrics as met
import pymbxas.utils.auxiliary as aux
from pymbxas.mbxas.broaden import get_mbxas_spectra
from pymbxas.explorer.node import DiscreteNode, BroadenedNode

# learning stuff
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as sker

#%%

# wrapper class to predict a spectra
class MBXASplorer(object):
    
    def __init__(self, spectras, fit_labels=None, xscaler="standard",
                 yscaler="standard", metric=None, isotropic=True, verbose=0,
                 ykernel=None, broaden=None):
        
        # set up internal variables
        self._verbose      = verbose
        self._is_isotropic = isotropic
        self._ykernel      = ykernel
        self._broaden      = broaden
        
        # initialize nodes
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
        self.xscale = met.generate_scaler(xscaler)
        
        # calculate feature vector for all spectra
        structures = [sp.structure for sp in spectras]
        X  = self.metric(structures)
        Xs = self.xscale.fit_transform(X)
        self._Xs = Xs

        # fit each peak
        for peak_label in fit_labels:
            self._make_node(spectras, peak_label, Xs, yscaler, isotropic, ykernel,
                            broaden)
            
        return
    
    def _make_node(self, spectras, peak_label, Xs, yscaler, isotropic, ykernel,
                   broaden):
        
        if peak_label in self._labels: # ignore noise
            return
        
        if broaden is None:
            # fit a single excitation
            sn = DiscreteNode(spectras, Xs, yscaler=yscaler, isotropic=isotropic,
                              ykernel=ykernel, peak_label=peak_label)
        else:
            sn = BroadenedNode(spectras, Xs, yscaler=yscaler, broaden=broaden,
                         peak_label=peak_label)
          
        # add it to the list #TODO: add later as well
        self._nodes.append(sn)
        self._labels.append(peak_label)
                
        return
    
    def predict(self, structures, node=None, return_std=True):
        
        structures = aux.as_list(structures)
        Xtest      = self.metric(structures)
        
        # scale new input
        Xtest_scaled = self.xscale.transform(Xtest)
        
        if node is None:
            nodelist = self._labels
        else:
            nodelist = aux.as_list(node)
        
        E_pre, E_std, A_pre, A_std = [], [], [], []
        for label in nodelist:
            
            node = self._get_node(label)
            
            e_pre, e_std, a_pre, a_std = node.predict(Xtest_scaled)
            
            E_pre.append(e_pre.reshape(-1, node.n_targets))
            E_std.append(e_std.reshape(-1, node.n_targets))
            A_pre.append(a_pre.reshape(-1, node.n_targets))
            A_std.append(a_std.reshape(-1, node.n_targets))
        
        E_pre = np.squeeze(np.hstack(E_pre))
        E_std = np.squeeze(np.hstack(E_std))
        A_pre = np.squeeze(np.hstack(A_pre))
        A_std = np.squeeze(np.hstack(A_std))
                
        if return_std:
            return E_pre, E_std, A_pre, A_std
        else:
            return E_pre, A_pre
            
    
    # generate a MBXAS spectra predicted from a structure
    def get_mbxas_spectra(self, structures, axis=None, sigma=0.5, npoints=3001,
                          tol=0.01, erange=None, node=None):
        
        # make strucute a list
        structures = aux.as_list(structures)
        
        # predict E and A
        E_pre, E_std, A_pre, A_std = self.predict(structures, node=node)

        # get broadened spectra - if not already
        if not self._broaden:
            erange, spectras = get_mbxas_spectra(E_pre, A_pre, sigma=sigma,
                                                 npoints=npoints, tol=tol,
                                                 erange=erange, isotropic=self._is_isotropic)
            if axis is None:
                spectras = np.mean(spectras, axis=0)
            
        else:
            npts   = self._broaden['npoints']
            E_pre = E_pre.reshape((len(structures), -1))
            A_pre = A_pre.reshape((len(structures), -1))
            
            erange = E_pre[0, :npts]
            spectras = A_pre.sum(axis=0).reshape((-1, npts)).sum(axis=0)

        return erange, spectras/len(structures)
    
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

    def _get_node(self, label):
        idx = np.where(np.array(self._labels) == label)[0]
        assert len(idx) == 1, "TOO MANY"
        return self[idx[0]]