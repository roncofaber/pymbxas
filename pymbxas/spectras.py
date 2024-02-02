#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 12:08:17 2023

@author: roncofaber
"""

import numpy as np


import copy

# can we use the sea urchin here?
try:
    # import sea_urchin.alignement.align as ali
    import sea_urchin.clustering.metrics as met
    SeaUrchin_exists = True
except:
    SeaUrchin_exists = False
    
#%%

"""
Class for a collection of spectras - WIP

Input: list of pyscf objects

"""

class Spectras():
    
    def __init__(self, spectra_list,
                 labels     = None,
                 comp       = None,
                 post_align = False,
                 alignment  = None
                 ):
        
        self.__initialize_collection(spectra_list, labels, comp, post_align,
                                     alignment)
        
        return
    
    # start from list of pyscf objects
    def __initialize_collection(self, spectra_list, labels, comp, post_align,
                                alignment):
        
        self.spectras = copy.deepcopy(spectra_list)
        
        if comp is None:       
            self.labels = len(spectra_list)*[-1]
            self.ref_structures = None
            
        else:
            self.labels = copy.deepcopy(comp.labels)
            self.ref_structures = comp.get_representatives()
        
        # assign labels
        for cc, spectra in enumerate(self.spectras):
            spectra.label = self.labels[cc]
        
        if post_align:
            if alignment is None:
                alignment = comp.alignment
        
            self.ref_spectras = self.align_to_references(self.ref_structures,
                                                         alignment)
            
        return
    
    # return all spectra with given atomic cluster label
    def get_spectra_with_label(self, label):
        return [sp for sp in self.spectras if sp.label == label]
    
    # get all spectras with a specific label
    def get_mbxas_spectras(self, axis=None, sigma=0.02, npoints=1001, tol=0.01,
                          erange=None, label=None, el_label=None):
        if label is None:
            spectras = self.spectras
        else:
            spectras = self.get_spectra_with_label(label)
        
        I_list = []
        for spectra in spectras:
            E, I = spectra.get_mbxas_spectra(axis=axis, sigma=sigma,
                                             npoints=npoints, tol=tol,
                                             erange=erange, el_label=el_label)
            I_list.append(I)
        
        return E, np.array(I_list)
   
    # get the average spectra
    def get_mbxas_spectra(self, axis=None, sigma=0.02, npoints=1001, tol=0.01,
                          erange=None, label=None, el_label=None):
        
        E, I_list = self.get_mbxas_spectras(axis=axis, sigma=sigma,
                                            npoints=npoints, tol=tol,
                                            erange=erange, label=label,
                                            el_label=el_label)
        
        return E, np.mean(I_list, axis=0)
    
  
    # align all spectra to their reference structure
    def align_to_references(self, ref_structures, alignment, labels=None):
        
        if labels is None:
            labels = self.labels
        elif isinstance(labels, int):
            labels = [labels]

        # iterate over labels and return the structure closest to mean structure
        for lab in set(labels):
            
            if lab == -1: # ignore noise
                continue
            
            # get relevant spectra
            spectras = self.get_spectra_with_label(lab)
            
            # get relevant reference structure
            ref_structure = ref_structures[lab]
            
            for spectra in spectras:
                spectra.align_to_reference(ref_structure, alignment)
            
        return
    
 
    # generate a set of IAOS basis 
    def generate_iaos_basis(self, minao="minao"):
        
        assert self.comp is not None, "Can only do this if comp is provided"
        
        mean_structures = self.comp.get_mean_structures()
        
        iaos_list    = []
        ref_spectras = []
        for lab in set(self.labels):
            
            if lab == -1: # ignore noise
                continue
            
            # get indexes where labels
            tidx = np.where(self.labels == lab)[0]
            
            # get them clusters
            clusters = self.comp.get_clusters_with_label(lab)
            
            # get distance matrix
            dists = met.get_simple_distances([mean_structures[lab]], clusters)
            
            # find actual structure closest to mean structure and spectra
            ref_idx = tidx[np.argmin(dists)]
            spectra = self.spectras[ref_idx]
            
            # make the iaos
            iaos = self.make_iaos(spectra.mol, spectra._mo_coeff, spectra._mo_occ, minao)
            iaos_list.append(iaos)
            ref_spectras.append(spectra)

        return np.array(iaos_list), copy.deepcopy(ref_spectras)
    
    
    def get_feature_vector(self, label=None):
        
        assert label in self.labels and label != -1, "Invalid label provided"
        
        energies   = []
        amplitudes = []
        overlaps   = []
        # CMOs       = []
        for spectra in self.get_spectra_with_label(label):
                  
            # calculate overlap
            ovlp = spectra.get_CMO_orth_proj()
            
            # append data
            overlaps.append(ovlp**2)
            energies.append(spectra.energies)
            amplitudes.append(spectra.amplitude)

        return np.array(overlaps), np.array(energies), np.array(amplitudes)
    
    # make class iterable
    def __getitem__(self, index):
        return self.spectras[index]
    
    def __iter__(self):
        return iter(self.spectras)