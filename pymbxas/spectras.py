#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 12:08:17 2023

@author: roncofaber
"""

import numpy as np
import copy

from pymbxas import Spectra
from pymbxas.io import h5

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

        if not isinstance(spectra_list, list):
            raise TypeError("Spectras() takes a list of Spectra; use Spectras.load(path) for files.")

        self.__initialize_collection(spectra_list, labels, post_align, alignment)
        
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
    
    @classmethod
    def load(cls, filename):
        with h5.open_read(filename, h5.KIND_SPECTRAS) as f:
            spectras = [Spectra._from_group(f["spectras"][key])
                        for key in sorted(f["spectras"])]
            labels = [int(x) for x in f["labels"][()]]
            aligned = bool(f.attrs["aligned"])

        obj = cls(spectras, labels=labels)
        obj._aligned = aligned
        return obj

    def materialize(self):
        """Force every member's deferred coefficients to be read from disk."""
        for spectra in self.spectras:
            spectra.materialize()
        return

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
                          erange=None, label=None, el_label=None, f_order=1,
                          spectator_order="auto", max_total_order=None,
                          max_configurations=2_000_000):
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
                                             erange=erange, el_label=el_label,
                                             f_order=f_order,
                                             spectator_order=spectator_order,
                                             max_total_order=max_total_order,
                                             max_configurations=max_configurations)

            E = Et
            I_list.append(I)
        
        return E, np.array(I_list)
   
    # get the average spectra
    def get_mbxas_spectra(self, axis=None, sigma=0.5, npoints=3001, tol=0.01,
                          erange=None, label=None, el_label=None, average=True,
                          f_order=1, spectator_order="auto",
                          max_total_order=None,
                          max_configurations=2_000_000):
        
        if erange is None:
            erange=self._erange
        
        E, I_list = self.get_mbxas_spectras(axis=axis, sigma=sigma,
                                            npoints=npoints, tol=tol,
                                            erange=erange, label=label,
                                            el_label=el_label, f_order=f_order,
                                            spectator_order=spectator_order,
                                            max_total_order=max_total_order,
                                            max_configurations=max_configurations)
        
        if average:
            I_list = np.mean(I_list, axis=0)
        else:
            I_list = np.sum(I_list, axis=0)
        
        return E, I_list

    def get_mbxas_decomposition(
            self, f_order=2, sigma=0.5, npoints=3001, erange=None, tol=0.01,
            label=None, average=True, spectator_order="auto",
            max_total_order=None, max_configurations=2_000_000):
        """Return an aggregate many-body decomposition for this collection.

        The returned mapping has the same numerical fields as
        :meth:`Spectra.get_mbxas_decomposition`. Arrays and overlap masses are
        averaged by default, matching :meth:`get_mbxas_spectra`; pass
        ``average=False`` to sum independent absorbing sites. Site-resolved
        overlap reports remain available under ``overlap["per_site"]``.
        """
        spectras = (self.spectras if label is None
                    else self.__get_atomic_label(label))
        if not spectras:
            raise ValueError("No spectra match the requested collection")
        if erange is None:
            erange = self._erange

        site_data = [
            spectra.get_mbxas_decomposition(
                f_order=f_order, sigma=sigma, npoints=npoints, erange=erange,
                tol=tol, spectator_order=spectator_order,
                max_total_order=max_total_order,
                max_configurations=max_configurations)
            for spectra in spectras
        ]
        energy = np.asarray(site_data[0]["energy"])
        if any(not np.array_equal(data["energy"], energy)
               for data in site_data[1:]):
            raise ValueError("Site decompositions do not share an energy grid")

        scale = 1.0 / len(site_data) if average else 1.0

        def combine_arrays(values):
            return scale * np.sum(values, axis=0)

        orders = sorted(site_data[0]["contributions"])
        contributions = {
            order: combine_arrays([
                data["contributions"][order] for data in site_data])
            for order in orders
        }
        cumulative = {
            order: combine_arrays([
                data["cumulative"][order] for data in site_data])
            for order in orders
        }
        resolved_orders = sorted(site_data[0]["decomposition"])
        decomposition = {
            order: {
                component: combine_arrays([
                    data["decomposition"][order][component]
                    for data in site_data])
                for component in ("shakeup", "shakedown")
            }
            for order in resolved_orders
        }
        total = combine_arrays([data["total"] for data in site_data])
        integrated = {
            "contributions": {
                order: np.trapezoid(value, energy)
                for order, value in contributions.items()},
            "cumulative": {
                order: np.trapezoid(value, energy)
                for order, value in cumulative.items()},
            "decomposition": {
                order: {
                    component: np.trapezoid(value, energy)
                    for component, value in parts.items()}
                for order, parts in decomposition.items()},
            "total": np.trapezoid(total, energy),
        }

        probability_starts = [data["probability"][0][0]
                              for data in site_data]
        probability_stops = [data["probability"][0][-1]
                             for data in site_data]
        probability_energy = np.linspace(
            min(probability_starts), max(probability_stops), npoints)
        probability = combine_arrays([
            np.interp(
                probability_energy, data["probability"][0],
                data["probability"][1], left=0.0, right=0.0)
            for data in site_data
        ])
        probability_orders = site_data[0]["probability"][2]
        if any(data["probability"][2] != probability_orders
               for data in site_data[1:]):
            probability_orders = None

        site_overlap = tuple(data["overlap"] for data in site_data)
        overlap_orders = sorted(set().union(*(
            report["by_total_order"] for report in site_overlap)))
        by_total_order = {
            order: scale * sum(
                report["by_total_order"].get(order, 0.0)
                for report in site_overlap)
            for order in overlap_orders
        }
        captured = scale * sum(report["captured"] for report in site_overlap)
        available = scale * sum(report["available"] for report in site_overlap)
        overlap = {
            "by_total_order": by_total_order,
            "captured": captured,
            "available": available,
            "fraction": captured / available if available > 0 else 0.0,
            "per_site": site_overlap,
        }
        captured_sum = sum(report["captured"] for report in site_overlap)
        shakedown_fraction = (
            sum(data["shakedown_fraction"] * data["overlap"]["captured"]
                for data in site_data) / captured_sum
            if captured_sum > 0 else 0.0)

        return {
            "energy": energy,
            "contributions": contributions,
            "decomposition": decomposition,
            "cumulative": cumulative,
            "total": total,
            "integrated": integrated,
            "probability": (
                probability_energy, probability, probability_orders),
            "overlap": overlap,
            "shakedown_fraction": shakedown_fraction,
            "site_count": len(site_data),
            "aggregation": "mean" if average else "sum",
        }

    def plot_mbxas_decomposition(
            self, f_order=2, sigma=0.5, npoints=3001, erange=None, tol=0.01,
            label=None, average=True, spectator_order="auto",
            max_total_order=None, max_configurations=2_000_000,
            show_probability=True, show_resolved=False,
            show_cumulative=False, figsize=None):
        """Calculate and plot an aggregate collection decomposition."""
        from pymbxas.plotting import plot_mbxas_decomposition

        decomposition = self.get_mbxas_decomposition(
            f_order=f_order, sigma=sigma, npoints=npoints, erange=erange,
            tol=tol, label=label, average=average,
            spectator_order=spectator_order,
            max_total_order=max_total_order,
            max_configurations=max_configurations)
        return plot_mbxas_decomposition(
            decomposition, show_probability=show_probability,
            show_resolved=show_resolved,
            show_cumulative=show_cumulative, figsize=figsize)
    
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
        except ImportError:
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
        except ImportError:
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
    
    def save(self, filename="spectras.h5"):
        """Saves the collection to an HDF5 file."""
        with h5.create(filename, h5.KIND_SPECTRAS) as fout:
            fout.attrs["aligned"] = bool(self._aligned)
            h5.write_array(fout, "labels", np.asarray(self.labels, dtype=np.int64))
            root = fout.create_group("spectras")
            for cc, spectra in enumerate(self.spectras):
                spectra._write_into(root.create_group("{:03d}".format(cc)))
        return
    
    def copy(self):
        self.materialize()
        return copy.deepcopy(self)
    
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
