#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 12:08:17 2023

@author: roncofaber
"""

import numpy as np
import copy
import dill

from pymbxas import Spectra

#%%

"""
Class for a collection of spectras - WIP

Input: list of pyscf objects

"""

class Spectras():
    
    def __init__(self, spectra_list,
                 labels     = None,
                 post_align = False,
                 alignment  = None
                 ):
        
        if isinstance(spectra_list, Spectra):
            spectra_list = [spectra_list]
        
        if isinstance(spectra_list, list):
            self.__initialize_collection(spectra_list, labels, post_align,
                                     alignment)
        else:
            self.__restart(spectra_list)
        
        # store internal variables for later
        self._update_erange()
        
        return
    
    # start from list of pyscf objects
    def __initialize_collection(self, spectra_list, labels, post_align,
                                alignment):
        
        # copy input spectras
        self.spectras = copy.deepcopy(spectra_list)
        
        # assign labels, if existend
        self.assign_atomic_labels(labels)
        
        # align
        if post_align:
            assert alignment is not None, "Provide an alignment"
            self.align_labels_to_mean_structures(alignment)
            
        return
    
    # restart from pkl file or another spectras object
    def __restart(self, spectra_obj):
        
        #load a pkl or dict
        if isinstance(spectra_obj, dict):
            self.__dict__ = spectra_obj.copy()
        elif isinstance(spectra_obj, str):
            self.__dict__ = self.__pkl_to_dict(spectra_obj)
            
        spectras = [Spectra(ii) for ii in self.spectras]
        
        self.spectras = spectras
            
        return
    
    @staticmethod
    def __pkl_to_dict(filename):
        with open(filename, 'rb') as fin:
            data = dill.load(fin)
        return data
    
    def assign_atomic_labels(self, labels):
        
        # check if labels are provided
        if labels is None:
            self.labels = len(self.spectras)*[-1]
        else:
            self.labels = labels.copy()
                
        # check that dims match
        assert len(self.labels) == len(self.spectras), "Wrong labels"
        
        # assign labels
        for cc, spectra in enumerate(self.spectras):
            spectra._label = self.labels[cc]

        # reset aligned keyword        
        self._aligned = False
        
        return
    
    def assign_electronic_labels(self, labels=None, label=None, reset=False):
        
        if reset:
            for sp in self:
                sp._el_labels = np.array([-1]*sp.CMO.shape[1])
            return
        
        if label is None:
            spectras = self.spectras
        else:
            spectras = self.__get_atomic_label(label)
        
        # check is all good
        assert len(labels) == len(spectras)
        
        imax = labels.shape[1]
        
        # assign electronic labels
        for cc, sp in enumerate(spectras):
            tlab = -np.ones(len(sp.energies), dtype=int)
            tlab[:imax] = labels[cc]
            
            sp._el_labels = tlab
        
        return
    
    # return sliced object with specific label
    def get_spectra_with_label(self, label):
        
        sp_list = self.__get_atomic_label(label)

        return Spectras(sp_list, labels=len(sp_list)*[label])
    
    # get all spectras with a specific label
    def get_mbxas_spectras(self, axis=None, sigma=0.5, npoints=3001, tol=0.01,
                          erange=None, label=None, el_label=None):
        if label is None:
            spectras = self.spectras
        else:
            spectras = self.__get_atomic_label(label)
            
        if erange is None:
            erange = self._erange
        
        E = None
        I_list = []
        for spectra in spectras:
            Et, I = spectra.get_mbxas_spectra(axis=axis, sigma=sigma,
                                             npoints=npoints, tol=tol,
                                             erange=erange, el_label=el_label)

            E = Et
            I_list.append(I)
        
        return E, np.array(I_list)
   
    # get the average spectra
    def get_mbxas_spectra(self, axis=None, sigma=0.5, npoints=3001, tol=0.01,
                          erange=None, label=None, el_label=None, average=True):
        
        if erange is None:
            erange=self._erange
        
        E, I_list = self.get_mbxas_spectras(axis=axis, sigma=sigma,
                                            npoints=npoints, tol=tol,
                                            erange=erange, label=label,
                                            el_label=el_label)
        
        if average:
            I_list = np.mean(I_list, axis=0)
        else:
            I_list = np.sum(I_list, axis=0)
        
        return E, I_list
    
    def align_labels_to_mean_structures(self, alignment):
        

        for lab in set(self.labels):
            
            if lab == -1: #ignore noise
                continue
            
            self._align_label_to_mean_structure(lab, alignment)
            
        self._aligned = True
        
        return
    
    def align_spectras_to_structure(self, ref_structure, alignment):
        
        # can we use the sea urchin here?
        try:
            import sea_urchin.alignement.align as ali
        except:
            raise ImportError("You need SeaUrchin compiled for this to work.")
        
        # get structures
        structures = [sp.structure for sp in self]
        
        # get alignments to mean structure
        rot, tr, perm, inv, dh = ali.get_RTPI(structures, ref_structure, alignment)
        
        for cc, spectra in enumerate(self):
            spectra.transform(rot=rot[cc], tr=tr[cc], perm=perm[cc],
                              inv=inv[cc], atype=alignment["type"])
        
        return
    
    def _align_label_to_mean_structure(self, label, alignment):
        
        # can we use the sea urchin here?
        try:
            import sea_urchin.alignement.align as ali
        except:
            raise ImportError("You need SeaUrchin compiled for this to work.")
        
        # get spectras and structures
        spectras   = self.__get_atomic_label(label)
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
    
    def get_mean_structure(self, label):
        
        # check alignment was done
        assert self._aligned, "You might want to align the structures before..."
            
        # get spectras and structures
        spectras   = self.__get_atomic_label(label)
        structures = [sp.structure for sp in spectras]

        positions = [cc.get_positions() for cc in structures]

        mean_structure = structures[0].copy()
        mean_structure.set_positions(np.mean(positions, axis=0))

        return mean_structure
    
    def get_feature_vector(self, label=None):
        
        if label is None:
            sp_list = self.spectras
        else:
            assert label in self.labels and label != -1, "Invalid label provided"
            sp_list = self.__get_atomic_label(label)
        
        energies   = []
        amplitudes = []
        overlaps   = []
        # CMOs       = []
        for spectra in sp_list:
                  
            # calculate overlap
            ovlp = spectra.get_CMO_orth_proj()
            
            # append data
            overlaps.append(ovlp**2)
            energies.append(spectra.energies)
            amplitudes.append(spectra.amplitude)

        return np.array(overlaps), np.array(energies), np.array(amplitudes)
    
    # make class iterable
    def __getitem__(self, index):
        if isinstance(index, (int, np.integer)):  # Include np.integer to handle NumPy integers
            return self.spectras[index]
        elif isinstance(index, slice):
            subset_spectras = self.spectras[index]
            return Spectras(subset_spectras)
        elif isinstance(index, (list, np.ndarray)):
            if all(isinstance(i, (bool, np.bool_)) for i in index):  # Check if all elements are booleans
                if len(index) != len(self.spectras):
                    raise IndexError("Boolean index list must have the same length as the spectras list")
                subset_spectras = [s for s, flag in zip(self.spectras, index) if flag]
            else:
                subset_spectras = [self.spectras[i] for i in index]
            return Spectras(subset_spectras)
        else:
            raise TypeError("Invalid index type")
    
    def __iter__(self):
        return iter(self.spectras)
    
    def __len__(self):
        return len(self.spectras)
    
    def __get_atomic_label(self, label):
        return [sp for sp in self if sp.label == label]
    
    def __add__(self, spectras):
        return Spectras(self.spectras + spectras.spectras)
    
    def _prepare_for_save(self):
        
        data = self.__dict__.copy()
        
        spectras = [sp._prepare_for_save() for sp in self.spectras]
        
        data["spectras"] = spectras
        
        return data
    
    def save(self, filename="spe            has_SU = Truectras.pkl"):
        """Saves the object to a file."""
        
        data = self._prepare_for_save()
      
        # Save to file
        with open(filename, 'wb') as fout:
            dill.dump(data, fout)
            
        return
    
    # use it to return a copy of the spectra collection object
    def copy(self):
        data = self._prepare_for_save()
        return Spectras(data)
    
    def append(self, spectra):
        """Appends a single Spectra object to the collection.

        Args:
            spectra: A Spectra object to append.  Raises a TypeError if not a Spectra object.
        """
        if not isinstance(spectra, Spectra):
            raise TypeError("Only Spectra objects can be appended.")
        self.spectras.append(copy.deepcopy(spectra))  # Deepcopy to avoid modification of the original
        self._update_erange() # Update energy range after appending
        self._aligned = False # Reset alignment flag

    def _update_erange(self):
        """Updates the _erange attribute after modifications to the spectras list."""
        energies = np.concatenate([sp.energies for sp in self])
        self._erange = [np.min(energies), np.max(energies)]
