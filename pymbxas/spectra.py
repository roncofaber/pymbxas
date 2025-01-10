#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 10:33:37 2023

@author: roncofaber
"""

# data manipulation
import numpy as np
from functools import reduce
import dill

# pymbxas utils
from pymbxas.build.structure import rotate_structure, ase_to_mole
from pymbxas.utils.basis import get_AO_permutation, get_l_val
from pymbxas.mbxas.broaden import get_mbxas_spectra
from pymbxas.io.write import write_data_to_fchk
from pymbxas.utils.auxiliary import change_key

# can we use the sea urchin here?
try:
    import sea_urchin.alignement.align as ali
    has_SU = True
except:
    has_SU = False

# pyscf stuff
from pyscf import lo
from pyscf.lo import iao, orth

from ase import units
Ha = units.Ha

#%%

class Spectra():

    def __init__(self,
                 pyscf_obj,
                 excitation=None,
                 ):
        
        # read spectra from pyscf object
        if "calculators.pyscf" in str(type(pyscf_obj)):
            self.__initialize_spectra(pyscf_obj, excitation)
        else:
            self.__restart(pyscf_obj)
        
        return

    # function that reads and initialize the spectra object #TODO this is mostly to be updated
    def __initialize_spectra(self, pyscf_obj, excitation):
        
        if excitation is None:
            assert len(pyscf_obj.excitations) == 1, "Please specify one excitation"
            excitation = 0
        else:
            assert isinstance(excitation, int)
        
        # retrieve excitation
        excitation = pyscf_obj.excitations[excitation]
        
        # retreive calculation details
        self.mol       = pyscf_obj.mol
        self.structure = pyscf_obj.structure
        self._exc_idx  = excitation.ato_idx
        
        self.calc_settings = pyscf_obj.parameters
        
        # get excitation data
        data    = excitation.data["fch"]
        mbxas   = excitation.mbxas
        channel = excitation.channel
        
        # store XAS data
        self._gs_energy = pyscf_obj.gs_data.e_tot
        self._energies  = mbxas["energies"]
        self._amplitude = mbxas["absorption"]
        
        # store MO data
        self._mo_coeff = data.mo_coeff[channel]
        self._mo_occ   = data.mo_occ[channel]
        
        # metadata for clustering and such
        self._el_labels = np.array([-1]*self.CMO.shape[1])
        self._label     = -1
        
        return
    
    @property
    def energies(self):
        return Ha*self._energies
    
    @property
    def amplitude(self):
        return self._amplitude
    
    @property
    def gs_energy(self):
        return Ha*self._gs_energy
    
    def __restart(self, pyscf_obj):
        
        #load a pkl or dict
        if isinstance(pyscf_obj, dict):
            data = pyscf_obj.copy()
        elif isinstance(pyscf_obj, str):
            data = self.__pkl_to_dict(pyscf_obj)
        else:
            raise TypeError("pyscf_obj must be a dictionary or a string path to a pickle file.")
            
        # make compatible with older version of pymbxas (<= 0.4.1)
        for old_key in ["energies", "gs_energy", "amplitude"]:
            if old_key in data:
                new_key = "_" + old_key
                change_key(data, old_key, new_key)
        
        # add new keys
        for new_key in ["_exc_idx"]:
            if new_key not in data:
                data[new_key] = None
                
        # fix _el_labels
        if data["_el_labels"] is None:
            data["_el_labels"] = np.array([-1]*len(np.where(data["_mo_occ"] == 0)[0][1:]))
        
        # assign values
        self.__dict__ = data
        
        # make the mol
        self.make_mol()
            
        return
    
    def __pkl_to_dict(self, filename):
        with open(filename, 'rb') as fin:
            data = dill.load(fin)
        return data
    
    def transform(self, rot=None, tr=None, perm=None,
                       inv=None, atype=None):
        
        if rot is None:
            rot = np.eye(3)
        if tr is None:
            tr = np.zeros(3)
        # no inversion? Use det of rot #TODO
        if inv is None:
            inv = np.round(np.linalg.det(rot))
        if perm is None:
            perm = list(range(len(self.structure)))
        if atype is None:
            atype = "fastoverlap"
            print("Assuming FO as type")
        
        # generate rotated structure
        structure = rotate_structure(self.structure, rot, tr, perm, inv, atype)
        
        # convert to mole
        mol = ase_to_mole(structure, verbose=0, **self.calc_settings)
                        
        # generate rotation matrix from rotM
        U = mol.ao_rotation_matrix(rot)
        
        # get permutation of AOs to match structure perm
        AO_permutation = get_AO_permutation(mol, perm)
        
        # calculate inversion contribution
        inv_A = inv**get_l_val(mol)
        
        # calculate rotated MOs
        ali_MOs = (inv_A*U).T.dot(self._mo_coeff[AO_permutation])
        
        # reassign variables
        self.structure = structure
        self._mo_coeff = ali_MOs
        self.mol       = mol
        self._amplitude = inv*rot@(self.amplitude)
        
        return
    
    def align_to_reference(self, reference, alignment, subset=None):
        
        assert has_SU, "Please, install also Sea Urchin to do this"
        
        if subset is None:
            structure = self.structure
        else:
            structure = self.structure[subset]
        
        rot, tr, perm, inv, _ = ali.get_RTPI(structure, reference, alignment)
        
        assert inv in [-1, 1]
        
        self.transform(rot=rot, tr=tr, perm=perm, inv=inv, atype=alignment["type"])
        
        return
    
    def get_CMO_projection(self, AO_to_proj, mol=None):
        
        if mol is None:
            mol = self.mol
        
        basis_ovlp = mol.intor_symmetric("int1e_ovlp")
        
        return (self.CMO.T@basis_ovlp@AO_to_proj).T
    
    def get_CMO_orth_proj(self, orth_method="meta-lowdin"):
        
        # calculate basis overlap
        basis_ovlp = self.mol.intor_symmetric('int1e_ovlp')
        
        # get orth AOs
        lao = lo.orth.orth_ao(self.mol, orth_method, s=basis_ovlp)
        
        # return CMO proj on LAOs
        return (self.CMO.T@basis_ovlp@lao).T
    
    
    # generate iaos given a structure and a basis (assumes FCH)
    def make_iaos(self, minao="minao"):
        
        maxidx = np.where(self._mo_occ == 1)[0].max()
        
        b_ovlp = self.mol.intor_symmetric('int1e_ovlp')
        
        iaos = iao.iao(self.mol, self._mo_coeff[:,:maxidx], minao=minao)
        
        return np.dot(iaos, orth.lowdin(reduce(np.dot, (iaos.T,b_ovlp,iaos))))
    
    def get_mbxas_spectra(self, axis=None, sigma=0.5, npoints=3001, tol=0.01,
                          erange=None, el_label=None):
        
        if el_label is not None:
            if el_label not in self._el_labels:
                return None, None
            
            idxs = self._el_labels == el_label
            
            amplitude = self.amplitude[:,idxs]
            energies  = self.energies[idxs] 
        else:
            amplitude = self.amplitude
            energies  = self.energies
        
        erange, spectras = get_mbxas_spectra(energies, amplitude,
                                              sigma=sigma, npoints=npoints,
                                              tol=tol, erange=erange)

        if axis is None:
            spectras = np.mean(spectras, axis=0)
        else:
            spectras = spectras[axis]
        
        return erange, spectras
        
    def get_amplitude_tensor(self):
        return np.einsum("ij,jk->ikj", self.amplitude, self.amplitude.T)
    
    # get CMOs
    @property
    def CMO(self): 
        uno_idxs = np.where(self._mo_occ == 0)[0][1:]
        return self._mo_coeff[:, uno_idxs]
    
    @property
    def label(self):
        return self._label
    
    
    def write_CMO2fchk(self, center=True, oname="spectra_CMO.fchk",
                       mo_coeff=None):
        
        if mo_coeff is None:
            mo_coeff = self.CMO
        
        write_data_to_fchk(self.mol,
                           mo_coeff  = mo_coeff,
                           mo_energy = self.energies/Ha,
                           mo_occ    = np.zeros((2,len(self.energies))),
                           center    = center,
                           oname     = oname)
        
        return
    
    # use this to remake mol obj (useful for reloading with dill)
    def make_mol(self):
        self.mol = ase_to_mole(self.structure, verbose=0, **self.calc_settings)
        return
    
    def _prepare_for_save(self):
        """Subclasses can override this to customize saving behavior.

        Returns:
            dict: A dictionary containing data to be serialized.
        """
        data = self.__dict__.copy()
        del data["mol"]
        return data
    
    def save(self, filename="spectra.pkl"):
        """Saves the object to a file."""
        
        data = self._prepare_for_save()
      
        # Save to file
        with open(filename, 'wb') as fout:
            dill.dump(data, fout)
            
        return
    
    @property
    def exc_idx(self):
        return self._exc_idx
    
    # use it to return a copy of the spectra
    def copy(self):
        data = self._prepare_for_save()
        return Spectra(data)

    def __repr__(self):
        chemfor = self.structure.get_chemical_formula()
        ato_idx = self.exc_idx
        
        if ato_idx is None:
            return f"Spectra({chemfor}|??)"
        
        ato_sym = self.structure.get_chemical_symbols()[ato_idx]
        return f"Spectra({chemfor}|{ato_sym}#{ato_idx})"
    
    def get_orbitals_with_label(self, label):
        
        idxs = np.argwhere(self._el_labels == label)[:,0]
        
        return self.CMO[:,idxs]
