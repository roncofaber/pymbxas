#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 15:32:48 2024

@author: roncofaber
"""

# numpy is my rock and I am ready to roll
import numpy as np
import copy

# pymbxas stuff
import pymbxas.utils.metrics as met
from pymbxas.utils.auxiliary import as_list
from pymbxas.mbxas.broaden import get_mbxas_spectra
from pymbxas.explorer.node import DiscreteNode, BroadenedNode
from pymbxas.drivers.acquisitor import pyscf_acquire

#%%

# wrapper class to predict a spectra
class MBXASplorer(object):
    
    def __init__(self, spectras=None, fit_labels=None, xscaler="standard",
                 yscaler="standard", metric=None, isotropic=True, verbose=0,
                 ykernel=None, broaden=None, train=False):
        
        # set up internal variables
        self._verbose      = verbose
        self._is_isotropic = isotropic
        self._ykernel      = ykernel
        self._broaden      = broaden
        self._trained      = False
        self._scaler_type  = {
            "xdata": xscaler,
            "ydata": yscaler
            }
        
        # initialize nodes
        self._nodes  = []
        self._labels = []
            
        # define metric (must work on ASE atoms object)
        if metric is None:
            metric = met.get_zmatlike_distances #met.get_distances
        self._metric = copy.deepcopy(metric)
        
        # generate scaler for feature vector
        self.xscaler = met.generate_scaler(xscaler)
        
        if train:
            self.train(spectras, fit_labels=fit_labels, ykernel=ykernel,
                       broaden=broaden, yscaler=yscaler)
        
        return
    
    # train MBXASplorer from scratch
    def train(self, spectras, fit_labels=None, ykernel=None, broaden=None,
              yscaler=None):
        
        # get class values
        if broaden is None:
            broaden = self._broaden
        if ykernel is None:
            ykernel = self._ykernel
        if yscaler is None:
            yscaler = self._scaler_type["ydata"]
        
        # define labels to fit
        if fit_labels is None:
            fit_labels = np.unique([sp._el_labels for sp in spectras])
        else:
            fit_labels = [ii for ii in np.unique(fit_labels)
                          if ii in np.unique([sp._el_labels for sp in spectras])]
                
        # calculate feature vector for all spectra
        X  = self.metric([sp.structure for sp in spectras])
        Xs = self.xscaler.fit_transform(X)

        # fit each peak
        for peak_label in fit_labels:
            self._make_node(spectras, peak_label, Xs, yscaler, self._is_isotropic,
                            ykernel, broaden)
            
        # assign some last variables
        self._X  = X
        self._Xs = Xs
        self._trained = True
        
        return
    
    def retrain(self, spectras):
        
        if not self._trained: raise ValueError("Train the model first")
        
        # calculate feature vector for all spectra and refit xscaler
        X  = self.metric([sp.structure for sp in spectras])
        Xs = self.xscaler.fit_transform(X)
        
        for node in self.nodes:
            node.train(spectras, Xs, retrain=True)
        
        self._X  = X
        self._Xs = Xs
        
        return
    
    def explore(self, structure_pool, niter, nmin=20, batch_size=1,
                initial_spectras=None, acquire=None):
        
        # store acquisition function internally
        if acquire is None:
            self._acquire = pyscf_acquire
        else:
            self._acquire = acquire
        
        # setup initial model as a starting point
        self._setup_explorer(structure_pool, initial_spectras, nmin)
            
        # now iterate and train better
        for ii in range(niter):
            
            # get new structures and new spectra
            new_structures = self._find_where_to_look(structure_pool, nsamples=batch_size)
            new_spectras   = [self._acquire(structure) for structure in new_structures]
            self._spectras.extend(new_spectras)
            
            self.retrain(self._spectras)
            
            # initialize performance test
            m, s = self._assert_performance(structure_pool)
            self._mean_var.append(m)
            self._std_var.append(s)
            
        return
    
    def _setup_explorer(self, structure_pool, initial_spectras, nmin):
        
        if initial_spectras is None:
            # empty list
            self._spectras = []
            
            # indexing
            iidxs = np.random.choice(list(range(len(structure_pool))), nmin)
            # do initial training
            initial_structures = [structure_pool[ii] for ii in iidxs]
            for structure in initial_structures:
                spectra = self._acquire(structure)
                self._spectras.append(spectra)
        else:
            if not isinstance(initial_spectras, list):
                raise TypeError("List needed for initial spectra")
            self._spectras = initial_spectras
        
        # train it
        self.train(self._spectras)
        
        # initialize performance test
        m, s = self._assert_performance(structure_pool)
        self._mean_var = [m]
        self._std_var  = [s]
        
        return
    
    def _find_where_to_look(self, structure_pool, beta=5.0, nsamples=1):
        
        _, _, mean, var = self.predict(structure_pool)
        
        #Upper Confidence Bound (UCB)
        ucb = mean + beta * np.sqrt(var)
        
        tidxs = np.argsort(ucb.max(axis=1))[-nsamples:]
        
        return [structure_pool[ii] for ii in tidxs]
    
    def _assert_performance(self, structures):
        
        _, _, _, var = self.predict(structures)
        
        mean_var = np.mean(var.sum(axis=1))
        std_var  = np.std(var.sum(axis=1))
        
        return mean_var, std_var
    
    
    def _make_node(self, spectras, peak_label, Xs, yscaler, isotropic, ykernel,
                   broaden):
        
        if peak_label in self._labels: # ignore noise
            print(f"Node {peak_label} already exists.")
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
        
        if not self._trained: raise ValueError("Train the model first")
        
        Xtest_s = self.scaled_metric(structures)
        
        if node is None:
            nodelist = self._labels
        else:
            nodelist = as_list(node)
        
        E_pre, E_std, A_pre, A_std = [], [], [], []
        for label in nodelist:
            
            node = self._get_node(label)
            
            e_pre, e_std, a_pre, a_std = node.predict(Xtest_s)
            
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
        structures = as_list(structures)
        
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
    
    # return metric of input
    def metric(self, X):
        return self._metric(X)
    
    # return scaled metric of input
    def scaled_metric(self, X):
        Xtest   = self.metric(X)
        Xtest_s = self.xscaler.transform(Xtest)
        return np.asarray(Xtest_s)

    def _get_node(self, label):
        assert label in self._labels, "This node does not exists."
        
        idx = np.where(np.array(self._labels) == label)[0]
        
        if not len(idx) == 1: raise ValueError
        
        return self[idx[0]]
