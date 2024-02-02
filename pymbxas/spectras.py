#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 12:08:17 2023

@author: roncofaber
"""

import numpy as np
from functools import reduce

from pyscf.lo import iao, orth

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

"""

class Spectras():
    
    def __init__(self, spectra_list, labels=None, comp=None):
        
        self.__initialize_collection(spectra_list, labels, comp)
        
        return
    
    
    def __initialize_collection(self, spectra_list, labels, comp):
        
        self.spectras = copy.deepcopy(spectra_list)
        
        if comp is None:
            self.comp   = None
            self.labels = copy.deepcopy(labels)
            self.ref_spectras = None
        else:
            self.comp   = copy.deepcopy(comp)
            self.labels = copy.deepcopy(comp.labels)
            self.ref_spectras = self.get_reference_spectra()
        
        return
    
    
    def get_spectra_with_label(self, label):
        
        if self.labels is None:
            return None
        
        output = [self.spectras[cc] for cc, lab in enumerate(self.labels)
                  if lab == label]
        
        return output
    
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
    
    # generate iaos given a structure and a basis (assumes FCH)
    @staticmethod
    def make_iaos(mol, mo_coeff, mo_occ, minao):
        
        maxidx = np.where(mo_occ == 1)[0].max()
        
        b_ovlp = mol.intor_symmetric('int1e_ovlp')
        
        iaos = iao.iao(mol, mo_coeff[:,:maxidx], minao=minao)
        
        return np.dot(iaos, orth.lowdin(reduce(np.dot, (iaos.T,b_ovlp,iaos))))
    
    def get_reference_spectra(self):
        
        assert self.comp is not None, "Can only do this if comp is provided"
        
        # get mean structures
        mean_structures = self.comp.get_mean_structures()
        
        # iterate over labels and return the structure closest to mean structure
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
            
            ref_spectras.append(spectra)
        
        return copy.deepcopy(ref_spectras)
    
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
    
    
    def get_feature_vector(self, label=None, alignment=None):
        
        assert label in self.labels and label != -1, "Invalid label provided"
        
        # get reference structure
        ref_structure = self.ref_spectras[label].structure.copy()
        
        energies   = []
        amplitudes = []
        overlaps   = []
        # CMOs       = []
        for spectra in self.get_spectra_with_label(label):
            
            # if needed, align to reference
            if alignment is not None:
                spectra.align_to_reference(ref_structure, alignment)
            
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