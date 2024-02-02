#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 10:33:37 2023

@author: roncofaber
"""

# data manipulation
import numpy as np
from functools import reduce

from pyscf.lo import iao, orth

# pymbxas utils
from pymbxas.build.structure import rotate_structure, ase_to_mole
from pymbxas.utils.basis import get_AO_permutation, get_l_val
from pymbxas.mbxas.broaden import get_mbxas_spectra
from pymbxas.io.write import write_data_to_fchk

# can we use the sea urchin here?
try:
    import sea_urchin.alignement.align as ali
    SeaUrchin_exists = True
except:
    SeaUrchin_exists = False

# pyscf stuff
from pyscf import lo

#%%

class Spectra():

    def __init__(self,
                 pyscf_obj,
                 excitation=0,
                 ):
        
        # read spectra
        self.__initialize_spectra(pyscf_obj, excitation)

    
    # function that reads and initialize the spectra object
    def __initialize_spectra(self, pyscf_obj, excitation):
        
        # retreive calculation details
        self.mol       = pyscf_obj.mol
        self.structure = pyscf_obj.structure
        
        self.calc_settings = {
            "charge"  : pyscf_obj.charge,
            "spin"    : pyscf_obj.spin,
            "xc"      : pyscf_obj.xc,
            "basis"   : pyscf_obj.basis,
            "solvent" : pyscf_obj.solvent,
            "pbc"     : pyscf_obj.pbc,
            }
        
        # get excitation data
        data  = pyscf_obj.excitations[excitation].data["fch"]
        mbxas = pyscf_obj.excitations[excitation].mbxas
        channel = pyscf_obj.excitations[excitation].channel
        
        self.gs_energy = pyscf_obj.data.e_tot
        self.energies  = mbxas["energies"]
        self.amplitude = mbxas["absorption"]
        
        self._mo_coeff = data.mo_coeff[channel]
        self._mo_occ   = data.mo_occ[channel]
        
        self._el_labels = None
        
        return
    
    def rotate_spectra(self, rot, perm, inv, reference=None):
        
        # generate rotated structure
        structure = rotate_structure(self.structure, rot, reference=reference,
                                     P=perm, inv=inv)
        
        mol = ase_to_mole(structure, verbose=0, **self.calc_settings)
        
        # generate rotation matrix from rotM
        U = mol.ao_rotation_matrix(rot)
        
        # get permutation of AOs to match structure perm
        AO_permutation = get_AO_permutation(mol, perm)
        
        # calculate inversion contribution
        inv_A = inv**get_l_val(mol)
        
        # calculate rotated MOs
        ali_MOs = (inv_A*U).T.dot(self._mo_coeff[AO_permutation])
        
        self.structure = structure
        self._mo_coeff = ali_MOs
        self.mol       = mol
        self.amplitude = rot@(self.amplitude)
        
        return
    
    def align_to_reference(self, reference, alignment, subset=None):
        
        assert SeaUrchin_exists, "Please, install also Sea Urchin to do this"
        
        if subset is None:
            structure = self.structure
        else:
            structure = self.structure[subset]
        
        __, __, R, P, I = ali.align_to_reference(structure, reference, alignment)
        
        self.rotate_spectra(R, P, I, reference=reference)
        
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
    
    def get_mbxas_spectra(self, axis=None, sigma=0.02, npoints=3001, tol=0.01,
                          erange=None, el_label=None):
        
        if el_label is not None:
            assert self._el_labels is not None, "first run el. clustering"
            
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
        
    
    # get CMOs
    @property
    def CMO(self): 
        uno_idxs = np.where(self._mo_occ == 0)[0][1:]
        return self._mo_coeff[:, uno_idxs]
    
    
    def write_CMO2fchk(self, center=True, oname="spectra_CMO.fchk"):
        
        write_data_to_fchk(self.mol,
                           mo_coeff  = self.CMO,
                           mo_energy = self.energies,
                           mo_occ    = np.zeros((2,len(self.energies))),
                           center    = center,
                           oname     = oname)
        
        return
    