#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 15:32:48 2024

@author: roncofaber
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 13:07:36 2024

@author: roncofaber
"""

import numpy as np

import sea_urchin.clustering.metrics as met
import sea_urchin.clustering.clusterize as clf

from pymbxas.explorer.node import spectralNode

Ha = 27.2113862161

# wrapper class to predict a spectra
class MBXASplorer():
    
    def __init__(self, spectra_list, fit_labels=None, xscaler="standard",
                 yscaler="standard", metric=None, verbose=0):
        
        # define labels to fit
        if fit_labels is None:
            fit_labels = np.unique([sp._el_labels for sp in spectra_list])
            
        # define metric (must work on ASE atoms object)
        if metric is None:
            metric = met.get_distances
        self.metric = metric
        
        # generate scaler for feature vector
        self.xscale = clf.generate_scaler(xscaler)
        
        # calculate feature vector for all spectra
        structures = [sp.structure for sp in spectra_list]
        X  = self.metric(structures)
        Xs = self.xscale.fit_transform(X)

        # fit each peak
        nodes = []
        for peak_label in fit_labels:
            
            if peak_label == -1: # ignore noise
                continue
            
            # fit a single excitation
            sn = spectralNode(spectra_list, peak_label, Xs, yscaler=yscaler)
              
            # add it to the list #TODO: add later as well
            nodes.append(sn)
            
            if verbose:
                for kr_a in sn.kr_a:
                    print(kr_a.kernel_)
        
        # save internal variables
        self.nodes = nodes
        
        return
    
    def predict(self, structures, node=None):
        
        if not isinstance(structures, list): # it's a single structure
            Xtest = self.metric([structures])
        else:
            Xtest = self.metric(structures)
        
        # scale new input
        Xtest_scaled = self.xscale.transform(Xtest)
        
        if node is not None:
            return self.nodes[node].predict(Xtest_scaled)
        
        else:
            
            y_E, y_A, y_G = [], [], []
            
            for node in self.nodes:
                e, a, g = node.predict(Xtest_scaled)
                y_E.append(e)
                y_A.append(a)
                y_G.append(g)
                
        return np.concatenate(y_E), np.concatenate(y_A), np.concatenate(y_G)
    
    # make class iterable
    def __getitem__(self, index):
        return self.nodes[index]
    
    def __iter__(self):
        return iter(self.nodes)