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
from pymbxas import Spectras
import pymbxas.utils.metrics as met
from pymbxas.utils.auxiliary import as_list
from pymbxas.mbxas.broaden import get_mbxas_spectra
from pymbxas.explorer.node import DiscreteNode, BroadenedNode
from pymbxas.drivers.acquisitor import pyscf_acquire

# let's add more packages
from sklearn.model_selection import train_test_split

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
            metric = met.get_distances
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
    
    def explore(self, structure_pool, niter, nmin=25, batch_size=1,
                initial_spectras=None, acquire=None, test_spectras=None,
                set_aside=None, hard_test=False, next_guess="ucb",
                rseed=None):
        
        # store acquisition function internally
        if acquire is None:
            self._acquire = pyscf_acquire
        else:
            self._acquire = acquire
        
        # setup initial model as a starting point
        self._setup_explorer(structure_pool, initial_spectras, nmin, set_aside,
                             hard_test, rseed)
        
        # train it for the first time
        self._initialize_training()
            
        # now iterate and train better
        for ii in range(niter):
            self._training_step(batch_size=batch_size, next_guess=next_guess)
            
        return
    
    def _setup_explorer(self, structure_pool, initial_spectras, nmin, set_aside,
                        hard_test, rseed):
        
        # initialize variables
        self._benchmark   = None
        self._spe2test    = None
        self._uncertainty = []
        self._error       = []
        
        # split training and testing set
        if set_aside is not None:
            str2train, str2test = train_test_split(structure_pool,
                                                   test_size=set_aside,
                                                   random_state=rseed)
            
            # calculate also the spectra on the testing set
            # (slow but necessary for testing)
            if hard_test:
                test_spectras = [self._acquire(structure) for structure in str2test]
                self._spe2test = Spectras(test_spectras)
                
                self._benchmark = self._spe2test.get_mbxas_spectras(
                    sigma=self._broaden["sigma"], npoints=self._broaden["npoints"],
                    erange=self._broaden["erange"])[1]
            
        else:
            str2train = structure_pool
            str2test  = None
        
        # store structures internally
        self._str2train = str2train
        self._str2test  = str2test
        
        self._idx2train = [] # keep in memory indexes used for training
        if initial_spectras is None:
            # empty list
            self._spectras = []
            
            # indexing to choose where to start: TODO better sampling!
            iidxs = np.random.choice(list(range(len(str2train))), nmin, replace=False)
            
            # do initial training
            for idx in iidxs:
                structure = str2train[idx]
                spectra = self._acquire(structure)
                
                self._spectras.append(spectra)
                self._idx2train.append(idx)
        else:
            if not isinstance(initial_spectras, list):
                raise TypeError("List needed for initial spectra")
            self._spectras = initial_spectras
        
        return
    
    def _initialize_training(self):
        
        # train it
        self.train(self._spectras)
        
        # initialize performance test
        unc, err = self._assert_performance(self._str2test, spe2test=self._benchmark)
        self._uncertainty.append(unc)
        self._error.append(err)
        
        return
    
    def _training_step(self, batch_size=1, beta=10.0, next_guess="ucb"):
        
        # get new structures
        new_structures, new_idxs = self._find_where_to_look(
            self._str2train, beta=beta, nsamples=batch_size,
            ignore_idxs=self._idx2train, next_guess=next_guess
            )
        
        # calculate new spectra
        new_spectras = [self._acquire(structure) for structure in new_structures]
        
        # add to trained pool
        self._spectras.extend(new_spectras)
        self._idx2train.extend(new_idxs)
        
        # retrain
        self.retrain(self._spectras)
        
        # run performance test on retrained data
        unc, err = self._assert_performance(self._str2test, spe2test=self._benchmark)
        self._uncertainty.append(unc)
        self._error.append(err)
        
        return
    
    def _find_where_to_look(self, str2train, beta=10.0, nsamples=1, ignore_idxs=[],
                            next_guess="ucb"):
        
        if next_guess == "ucb":
            _, _, mean, var = self.predict(str2train)
            
            #Upper Confidence Bound (UCB)
            # ucb = mean + beta * np.sqrt(var)
            ucb = beta * np.sqrt(var) # TODO for the moment only variance
            
            sorted_idxs = np.argsort(ucb.max(axis=1))[::-1]
        
        elif next_guess == "random":
            sorted_idxs = list(range(len(str2train)))
            np.random.shuffle(sorted_idxs)
            
        
        new_structures = []
        used_idxs      = []
        for idx in sorted_idxs:
            if idx not in ignore_idxs:
                new_structures.append(str2train[idx])
                used_idxs.append(idx)
                
            if len(new_structures) == nsamples:
                break
                
        return new_structures, used_idxs
    
    def _assert_performance(self, str2test, spe2test=None):
        
        _, _, mean, uncertainty = self.predict(str2test)
        
        error = None
        if spe2test is not None:
            error = np.sum((spe2test-mean)**2, axis=1)
            
        return uncertainty, error
    
    
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

    def _save_self(self, oname):
        
        import dill
        with open(oname, "wb") as fout:
            dill.dump(self, fout)
