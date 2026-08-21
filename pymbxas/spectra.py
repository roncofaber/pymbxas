#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 10:33:37 2023

@author: roncofaber
"""

# data manipulation
import numpy as np
from functools import reduce
import copy

# pymbxas utils
from pymbxas.build.structure import rotate_structure, ase_to_mole
from pymbxas.utils.basis import get_AO_permutation, get_l_val
from pymbxas.mbxas.broaden import get_mbxas_spectra
from pymbxas.io.write import write_data_to_fchk
from pymbxas.io import h5

# pyscf stuff
from pyscf import gto, lo
from pyscf.lo import iao, orth

from ase import units
Ha = units.Ha

#%%

class Spectra():
    """Represents a single spectrum, including molecular
    structure and electronic data."""

    def __init__(self, pyscf_obj, excitation=None):
        if isinstance(pyscf_obj, (str, dict)):
            raise TypeError("Spectra() no longer loads files; use Spectra.load(path).")
        if not hasattr(pyscf_obj, "excitations"):
            raise TypeError("Invalid pyscf_obj type. Must be a pyscf object with excitations.")
        self.__initialize_spectra(pyscf_obj, excitation)

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

        # store MO data for both spin channels from the FCH wavefunction.
        # _channel identifies which spin was excited.
        self._mo_coeff = data.mo_coeff   # shape: (2, nbasis, norb)
        self._mo_occ   = data.mo_occ     # shape: (2, norb)
        self._channel  = channel

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
    
    @classmethod
    def load(cls, filename):
        with h5.open_read(filename, h5.KIND_SPECTRA) as f:
            return cls._from_group(f)

    @classmethod
    def _from_group(cls, group):
        obj = cls.__new__(cls)
        obj._read_from(group)
        return obj

    def _write_into(self, group):
        h5.write_str(group, "mol", self.mol.dumps())
        h5.write_structure(group, "structure", self.structure)
        h5.write_json(group, "calc_settings", self.calc_settings)

        scf = group.create_group("scf")
        h5.write_array(scf, "mo_coeff", np.asarray(self._mo_coeff))
        h5.write_array(scf, "mo_occ", np.asarray(self._mo_occ))

        xas = group.create_group("xas")
        h5.write_array(xas, "energies", np.asarray(self._energies))
        h5.write_array(xas, "amplitude", np.asarray(self._amplitude))
        h5.write_array(xas, "el_labels", np.asarray(self._el_labels))

        group.attrs["channel"]   = int(self._channel)
        group.attrs["exc_idx"]   = -1 if self._exc_idx is None else int(self._exc_idx)
        group.attrs["label"]     = int(self._label)
        group.attrs["gs_energy"] = float(self._gs_energy)
        return

    def _read_from(self, group):
        self.structure     = h5.read_structure(group, "structure")
        self.calc_settings = h5.read_json(group, "calc_settings")

        xas = group["xas"]
        self._energies  = xas["energies"][()]
        self._amplitude = xas["amplitude"][()]
        self._el_labels = xas["el_labels"][()]

        exc_idx = int(group.attrs["exc_idx"])
        self._channel   = int(group.attrs["channel"])
        self._exc_idx   = None if exc_idx < 0 else exc_idx
        self._label     = int(group.attrs["label"])
        self._gs_energy = float(group.attrs["gs_energy"])

        self._h5_source = (group.file.filename, group.name)

        if "mol" in group:
            self.mol = gto.loads(h5.read_str(group, "mol"))
            self.mol.verbose = 0
        else:
            self.make_mol()
        return

    def __getattr__(self, name):
        if name in ("_mo_coeff", "_mo_occ"):
            source = self.__dict__.get("_h5_source")
            if source is not None:
                value = h5.read_lazy_field(source, name[1:])
                self.__dict__[name] = value
                return value
        raise AttributeError(name)

    def materialize(self):
        """Force deferred orbital coefficients to be read from disk."""
        if self.__dict__.get("_h5_source") is None:
            return
        self._mo_coeff
        self._mo_occ
        self.__dict__.pop("_h5_source", None)
        return
    
    def align_to(self, structure, alignment):
        
        # can we use the sea urchin here?
        try:
            import sea_urchin.alignement.align as ali
        except ImportError:
            raise ImportError("You need SeaUrchin compiled for this to work.")

        # get alignments to mean structure
        rot, tr, perm, inv, dh = ali.get_RTPI(self.structure, structure, alignment)
        
        # transform self
        self.transform(rot, tr, perm, inv, atype=alignment["type"])
        
        return
    
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
        
        # calculate rotated MOs - handle both old (2D) and new (3D) formats
        if np.asarray(self._mo_coeff).ndim == 3:
            ali_MOs = np.array([(inv_A*U).T.dot(self._mo_coeff[ch][AO_permutation])
                                for ch in range(self._mo_coeff.shape[0])])
        else:
            ali_MOs = (inv_A*U).T.dot(self._mo_coeff[AO_permutation])
        
        # reassign variables
        self.structure = structure
        self._mo_coeff = ali_MOs
        self.mol       = mol
        self._amplitude = inv*rot@(self.amplitude)
        
        # change excited index to follow permutation
        if perm is not None:
            self._exc_idx = np.argwhere(perm==self.exc_idx)[0][0]
        
        return
    
    def align_to_reference(self, reference, alignment, subset=None):
        
        # can we use the sea urchin here?
        try:
            import sea_urchin.alignement.align as ali
        except ImportError:
            raise ImportError("You need SeaUrchin compiled for this to work.")

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

        occ_idxs = np.where(self._active_mo_occ == 1)[0]

        b_ovlp = self.mol.intor_symmetric('int1e_ovlp')

        iaos = iao.iao(self.mol, self._active_mo_coeff[:, occ_idxs], minao=minao)

        return np.dot(iaos, orth.lowdin(reduce(np.dot, (iaos.T, b_ovlp, iaos))))
    
    def get_mbxas_spectra(self, axis=None, sigma=0.5, npoints=3001, tol=0.01,
                          erange=None, el_label=None):
        
        if el_label is not None:
            idxs      = self._el_labels == el_label
            amplitude = self.amplitude[:,idxs]
            energies  = self.energies[idxs] 
            
        else:
            amplitude = self.amplitude
            energies  = self.energies
            
        if erange is None:
            erange = [self.energies.min(), self.energies.max()]
            
        # convert amplitude to intensity
        if axis is None:
            intensities = self.amp2int(amplitude)
        else:
            intensities = amplitude[axis]**2
        
        erange, spectra = get_mbxas_spectra(energies, intensities,
                                              sigma=sigma, npoints=npoints,
                                              tol=tol, erange=erange)
        
        return erange, spectra
        
    def get_amplitude_tensor(self):
        return np.einsum("ij,jk->ikj", self.amplitude, self.amplitude.T)
    
    def amp2int(self, amplitude=None):
        if amplitude is None:
            amplitude = self.amplitude
        return np.sum(amplitude**2, axis=0) / amplitude.shape[0]
    
    # channel-aware accessors - handle both old (1D occ / 2D coeff)
    # and new (2D occ / 3D coeff) formats transparently
    @property
    def _active_mo_occ(self):
        mo_occ = self._mo_occ
        return mo_occ[self._channel] if np.asarray(mo_occ).ndim == 2 else mo_occ

    @property
    def _active_mo_coeff(self):
        mo_coeff = self._mo_coeff
        return mo_coeff[self._channel] if np.asarray(mo_coeff).ndim == 3 else mo_coeff

    # get CMOs
    @property
    def CMO(self):
        uno_idxs = np.where(self._active_mo_occ == 0)[0][1:]
        return self._active_mo_coeff[:, uno_idxs]

    @property
    def channel(self):
        return self._channel
    
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
    
    def make_mol(self):
        self.mol = ase_to_mole(self.structure, verbose=0, **self.calc_settings)
        return
    
    def save(self, filename="spectra.h5"):
        """Saves the object to an HDF5 file."""
        with h5.create(filename, h5.KIND_SPECTRA) as fout:
            self._write_into(fout)
        return
    
    @property
    def exc_idx(self):
        return self._exc_idx
    
    def copy(self):
        self.materialize()
        return copy.deepcopy(self)

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
