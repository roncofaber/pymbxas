#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 12:08:17 2023

@author: roncofaber
"""

import numpy as np
import copy
import dill, lzma

from pymbxas import Spectra

# can we use the sea urchin here?
try:
    # import sea_urchin.alignement.align as ali
    import sea_urchin.clustering.metrics as met
    import sea_urchin.alignement.align as ali
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
        
        if isinstance(spectra_list, list):
            self.__initialize_collection(spectra_list, labels, comp, post_align,
                                     alignment)
        else:
            self.__restart(spectra_list)
        
        return
    
    # start from list of pyscf objects
    def __initialize_collection(self, spectra_list, labels, comp, post_align,
                                alignment):
        
        self.spectras = copy.deepcopy(spectra_list)
        
        if comp is None:       
            self.labels = len(spectra_list)*[-1]
        else:
            self.labels = copy.deepcopy(comp.labels)
        
        # assign labels
        for cc, spectra in enumerate(self.spectras):
            spectra.label = self.labels[cc]
        
        if post_align:
            if alignment is None:
                alignment = comp.alignment
        
            self.align_labels_to_mean_structures(alignment)
            
        return
    
    def __restart(self, spectra_obj):
        
        #load a pkl or dict
        if isinstance(spectra_obj, dict):
            self.__dict__ = spectra_obj.copy()
        elif isinstance(spectra_obj, str):
            self.__dict__ = self.__pkl_to_dict(spectra_obj)
            
        spectras = [Spectra(ii) for ii in self.spectras]
        
        self.spectras = spectras
            
        return
    
    def __pkl_to_dict(self, filename):
        
        with open(filename, 'rb') as fin:
            data = fin.read()
            
        compressed_data = lzma.decompress(data)
        data = dill.loads(compressed_data)
  
        return data.copy()
    
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
    
    def align_labels_to_mean_structures(self, alignment, labels=None):
        
        if labels is None: # get comp labels
            labels = self.labels
            
        for lab in set(labels):
            
            if lab == -1:
                continue
            
            self.align_label_to_mean_structure(lab, alignment)
        
        return
    
    def align_label_to_mean_structure(self, label, alignment):
        
        # assert isinstance(label, int)
        
        # get spectras and structures
        spectras   = self.get_spectra_with_label(label)
        structures = [sp.structure for sp in spectras]
        
        # calculate mean structure
        
        __, mstrus = ali.align_to_mean_structure(structures, alignment,
                                                 start_structure = structures[0])
        
        mean_structure = mstrus[-1]
        
        # get alignments to mean structure
        rot, tr, perm, inv, dh = ali.get_RTPI(structures, mean_structure, alignment)
        
        for cc, spectra in enumerate(spectras):
            spectra.transform(rot=rot[cc], tr=tr[cc], perm=perm[cc],
                              inv=inv[cc], atype=alignment["type"])
        
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
    
    def _prepare_for_save(self):
        
        data = self.__dict__.copy()
        
        spectras = [sp._prepare_for_save() for sp in self.spectras]
        
        data["spectras"] = spectras
        
        return data
    
    def save(self, filename="spectras.pkl"):
        """Saves the object to a file."""
        
        data = self._prepare_for_save()
        serialized_data = dill.dumps(data)

        # Compress using lzma for space efficiency
        compressed_data = lzma.compress(serialized_data)

        # Save to file
        with open(filename, 'wb') as fout:
            fout.write(compressed_data)
            
        return
    
    # use it to return a copy of the spectra
    def copy(self):
        data = self._prepare_for_save()
        return Spectras(data)