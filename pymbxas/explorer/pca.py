#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 15:35:55 2024

@author: roncofaber
"""

import numpy as np
from sklearn.decomposition import PCA
import ase
import ase.visualize

#%%
class PCAmodel():
    
    def __init__(self, clusters):
        
        # get positions
        X = np.array([clu.get_positions().flatten() for clu in clusters])

        # get mean (meaningful because of SU)
        mean_pos = np.mean(X, axis=0).reshape(-1,3)
        
        # save mean structure
        self.mean_structure = clusters[0].copy()
        self.mean_structure.set_positions(mean_pos)
        
        # perform mass weighting
        self.mass_weight = np.sqrt(np.repeat(clusters[0].get_masses(),3))
        X_w = self.mass_weight*X

        # FIT PCA MODEL
        pca_model = PCA()
        pca_model.fit(X_w)

        # save model
        self.pca_model = pca_model
        self.pca_modes = self.generate_pca_modes(pca_model)

        return
    
    def generate_pca_modes(self, pca_model):
        
        # x_pca = np.dot(X_w, evecs) #TODO understand if this is needed
        # mode_scale = np.sqrt(np.sum(np.var(np.dot(x_pca, evecs[:, 0]))))
        
        evals = pca_model.singular_values_
        evecs = pca_model.components_
        
        pca_modes = []
        for pca_comp in range(pca_model.n_components_):

            displacement = np.dot(evecs[pca_comp], np.sqrt(evals[pca_comp]))
            displacement = displacement/self.mass_weight
            displacement = displacement.reshape(-1, 3)

            fac = np.linspace(0,1,50)
            tclus = []
            for ii in fac:
                tclu = self.mean_structure.copy()
                tclu.set_pbc(False)
                tclu.set_positions(self.mean_structure.get_positions() + ii*displacement)
                tclus.append(tclu)

            pca_modes.append(tclus)
        
        return pca_modes
    
    def transform(self, cluster):
        
        if not isinstance(cluster, list):
            cluster = [cluster]
            
        X = np.array([clu.get_positions().flatten() for clu in cluster])
        X_w = self.mass_weight*X
        
        return self.pca_model.transform(X_w)
    
    def view(self, pca_idx):
        
        ase.visualize.view(self.pca_modes[pca_idx])
        
        return