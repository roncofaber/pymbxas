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
import logging

# pymbxas utils
from pymbxas.build.structure import rotate_structure, ase_to_mole
from pymbxas.utils.basis import get_AO_permutation, get_l_val
from pymbxas.mbxas.broaden import get_mbxas_spectra
from pymbxas.io.write import write_data_to_fchk
from pymbxas.io import h5
from pymbxas.io.config import format_log_fields, with_log_context

# pyscf stuff
from pyscf import gto, lo
from pyscf.lo import iao, orth

from ase import units
Ha = units.Ha
logger = logging.getLogger(__name__)

_GAUSSIAN_SUPPORT = 6.0
_MAX_JOINT_ELEMENTS = 500_000


def _combine_spin_stick_block(xas_energy, oscillator, spectator_shift,
                              spectator_probability):
    """Combine spin-factor sticks using the complete final photon energy."""
    joint_energy = (
        np.asarray(xas_energy)[:, None]
        + np.asarray(spectator_shift)[None, :])
    joint_intensity = (
        joint_energy * np.asarray(oscillator)[:, None]
        * np.asarray(spectator_probability)[None, :])
    return joint_energy, joint_intensity


def _molecule_settings(calc_settings):
    """Select only settings that belong to PySCF's Mole/Cell build API."""
    return {key: calc_settings[key]
            for key in ("charge", "spin", "basis", "pbc")
            if key in calc_settings}

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
        
        self.calc_settings = pyscf_obj.config.to_dict()
        
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

        # data needed to rebuild A/K for shake-up satellites (both channels,
        # so the not-yet-implemented cross-spin extension needs no schema
        # change): shape (2, norb_fch, norb_gs), (2, norb_fch), (2, norb_gs)
        self._mb_overlap    = mbxas["mb_overlap"]
        self._fch_mo_energy = data.mo_energy
        self._gs_mo_energy  = pyscf_obj.gs_data.mo_energy
        self._gs_mo_occ     = pyscf_obj.gs_data.mo_occ
        self._core_orb_idx  = excitation.orb_idx

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

        shakeup = group.create_group("shakeup")
        h5.write_array(shakeup, "mb_overlap", np.asarray(self._mb_overlap))
        h5.write_array(shakeup, "fch_mo_energy", np.asarray(self._fch_mo_energy))
        h5.write_array(shakeup, "gs_mo_energy", np.asarray(self._gs_mo_energy))
        h5.write_array(shakeup, "gs_mo_occ", np.asarray(self._gs_mo_occ))
        shakeup.attrs["core_orb_idx"] = int(self._core_orb_idx)

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

        shakeup = group["shakeup"]
        self._mb_overlap    = shakeup["mb_overlap"][()]
        self._fch_mo_energy = shakeup["fch_mo_energy"][()]
        self._gs_mo_energy  = (
            shakeup["gs_mo_energy"][()]
            if "gs_mo_energy" in shakeup else None)
        self._gs_mo_occ     = shakeup["gs_mo_occ"][()]
        self._core_orb_idx  = int(shakeup.attrs["core_orb_idx"])

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
            logger.debug("No alignment type supplied; using fastoverlap")
        
        # generate rotated structure
        structure = rotate_structure(self.structure, rot, tr, perm, inv, atype)
        
        # convert to mole
        mol = ase_to_mole(
            structure, verbose=0, **_molecule_settings(self.calc_settings))
                        
        # generate rotation matrix from rotM
        U = mol.ao_rotation_matrix(rot)
        
        # get permutation of AOs to match structure perm
        AO_permutation = get_AO_permutation(mol, perm)
        
        # calculate inversion contribution
        inv_A = inv**get_l_val(mol)
        
        ali_MOs = np.array([(inv_A*U).T.dot(self._mo_coeff[ch][AO_permutation])
                            for ch in range(self._mo_coeff.shape[0])])
        
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
                          erange=None, el_label=None, f_order=1,
                          spectator_order="auto", max_total_order=None,
                          max_configurations=2_000_000):
        """Return the spin-complete MBXAS spectrum through a total order.

        ``f_order=1`` is f1=10, ``f_order=2`` is cumulative f1+f2 with
        f2=20+11, and ``f_order=3`` also adds f3=30+21+12. By default the
        spectator channel follows the requested f order. Pass
        ``spectator_order=None`` only for an excited-channel diagnostic.
        """

        if (not isinstance(f_order, (int, np.integer))
                or isinstance(f_order, (bool, np.bool_)) or f_order < 1):
            raise ValueError("f_order must be a positive integer")
        max_extra_order = int(f_order) - 1

        spectator_auto = spectator_order == "auto"
        if isinstance(spectator_order, str) and not spectator_auto:
            raise ValueError("spectator_order must be 'auto', None, or an integer")
        if (max_extra_order > 0 or
                spectator_order not in (None, "auto")) and el_label is not None:
            raise ValueError("el_label is not defined for many-body final determinants")

        excited_order = max_extra_order
        if el_label is None and spectator_auto:
            spectator_order = max_extra_order

        if el_label is None:
            return self._get_many_body_mbxas_spectra(
                axis=axis, sigma=sigma, npoints=npoints, tol=tol,
                erange=erange, max_extra_order=excited_order,
                spectator_order=spectator_order,
                max_total_order=max_total_order,
                shakedown_only=False,
                max_configurations=max_configurations)

        if el_label is not None:
            idxs        = self._el_labels == el_label
            amplitude   = self.amplitude[:,idxs]
            energies    = self.energies[idxs]
            energies_ha = self._energies[idxs]

        else:
            amplitude   = self.amplitude
            energies    = self.energies
            energies_ha = self._energies

        if erange is None:
            erange = [self.energies.min(), self.energies.max()]

        # convert amplitude to intensity: sigma(omega) ~ omega * |M|^2
        # (Eq. 4, PRB 107, 035146), weighted in the same atomic units (Ha)
        # as the amplitude
        if axis is None:
            intensities = self.amp2int(amplitude, energies_ha)
        else:
            intensities = energies_ha * amplitude[axis]**2

        erange, spectra = get_mbxas_spectra(energies, intensities,
                                              sigma=sigma, npoints=npoints,
                                              tol=tol, erange=erange)

        # Orbital-label projections remain meaningful at f1. Apply the
        # opposite-spin reference determinant without constructing higher
        # final determinants.
        if spectator_auto:
            spectra *= self._spectator_reference_weight()

        return erange, spectra

    def _spectator_reference_weight(self):
        """Return |det(A)|^2 for the non-excited spin channel."""
        from pymbxas.mbxas.mbxas import (
            build_A_K, spectator_occ_unocc_indices)

        channel = 1 - self._channel
        occ_gs, occ_fch, virt = spectator_occ_unocc_indices(
            self._gs_mo_occ[channel], self._mo_occ[channel])
        _, determinant, _, _ = build_A_K(
            self._mb_overlap[channel], occ_fch, occ_gs, virt)
        return float(abs(determinant) ** 2)

    def _fch_core_hole_index(self):
        """Return the excited-channel FCH core-hole MO by GS overlap."""
        from pymbxas.mbxas.mbxas import core_hole_index
        ch = self._channel
        return core_hole_index(
            self._mb_overlap[ch], self._mo_occ[ch], self._core_orb_idx)

    def _get_many_body_mbxas_spectra(self, axis, sigma, npoints, tol, erange,
                                     max_extra_order, spectator_order,
                                     max_total_order, shakedown_only,
                                     max_configurations,
                                     return_components=False):
        """QE-compatible explicit f1/f2/f3-style spectral assembly.

        ``max_extra_order`` counts extra particle-hole pairs in the excited
        channel. Spectator overlap orders are convolved afterwards, with
        determinant weights retained and the total order capped before the
        final sticks are broadened once. ``return_components`` additionally
        accumulates the shake-down subset from those same sticks.
        """
        from pymbxas.mbxas.mbxas import (
            build_A_K, occ_unocc_indices, spectator_occ_unocc_indices)
        from pymbxas.mbxas.shakeup import (
            mbxas_sticks_by_order, overlap_sticks_from_K,
            screened_overlap_doubles_from_K)

        excited_order = int(max_extra_order)
        spectator_max = 0 if spectator_order is None else int(spectator_order)
        if excited_order < 0 or spectator_max < 0:
            raise ValueError("internal extra-pair orders must be non-negative")
        if max_total_order is None:
            # QE groups terms by total extra-pair order. This default avoids
            # silently adding orders beyond either requested truncation.
            max_total_order = max(excited_order, spectator_max)
        max_total_order = int(max_total_order)
        if max_total_order < 0:
            raise ValueError("max_total_order must be non-negative")

        if erange is None:
            erange = [self.energies.min(), self.energies.max()]
        if len(erange) == 2:
            energy_grid = np.linspace(erange[0], erange[1], npoints)
        else:
            energy_grid = np.asarray(erange)
        output_min = float(np.min(energy_grid))
        output_max = float(np.max(energy_grid))
        # QE admits individual K(c,v) promotions across the requested
        # spectral span plus the six-sigma Gaussian support.  This selection
        # is independent of the dipole-final orbital f.
        pair_energy_max = (
            output_max - output_min + _GAUSSIAN_SUPPORT * sigma) / Ha
        final_energy_range = (
            (output_min - _GAUSSIAN_SUPPORT * sigma) / Ha,
            (output_max + _GAUSSIAN_SUPPORT * sigma) / Ha)

        selection = "shake-down" if shakedown_only else None
        screening_log = with_log_context(
            logging.getLogger("pymbxas.mbxas.shakeup"),
            site=self._site_label(),
            stage=" ".join(filter(None, (
                f"f{excited_order + 1}", selection))))

        ch = self._channel
        occ_gs, occ_fch, virt_fch = occ_unocc_indices(
            self._gs_mo_occ[ch], self._mo_occ[ch], self._core_orb_idx,
            core_hole_idx=self._fch_core_hole_index())
        _, det_A, K, _ = build_A_K(
            self._mb_overlap[ch], occ_fch, occ_gs, virt_fch)
        eps_occ = self._fch_mo_energy[ch][occ_fch]
        eps_unocc = self._fch_mo_energy[ch][virt_fch]
        xas = mbxas_sticks_by_order(
            self._energies, self._amplitude, K, eps_occ, eps_unocc,
            excited_order, shakedown_only=False,
            max_configurations=max_configurations, screen_tol=tol,
            pair_energy_max=pair_energy_max,
            final_energy_range=final_energy_range,
            determinant=det_A, log=screening_log)

        if spectator_order is None:
            spectator = {0: (np.array([0.0]), np.array([1.0]),
                             np.array([False]))}
        else:
            sp = 1 - ch
            sp_occ_gs, sp_occ_fch, sp_virt = spectator_occ_unocc_indices(
                self._gs_mo_occ[sp], self._mo_occ[sp])
            sp_A, _, sp_K, _ = build_A_K(
                self._mb_overlap[sp], sp_occ_fch, sp_occ_gs, sp_virt)
            sp_eps_occ = self._fch_mo_energy[sp][sp_occ_fch]
            sp_eps_unocc = self._fch_mo_energy[sp][sp_virt]
            spectator = {}
            for order in range(spectator_max + 1):
                if order == 2:
                    sticks = screened_overlap_doubles_from_K(
                        abs(np.linalg.det(sp_A)) ** 2,
                        sp_K, sp_eps_occ, sp_eps_unocc, tol=tol,
                        max_configurations=max_configurations,
                        pair_energy_max=pair_energy_max,
                        log=screening_log)
                else:
                    sticks = overlap_sticks_from_K(
                        abs(np.linalg.det(sp_A)) ** 2,
                        sp_K, sp_eps_occ, sp_eps_unocc, order,
                        max_configurations=max_configurations,
                        pair_energy_max=pair_energy_max)
                spectator[order] = (sticks.energy, sticks.weight, sticks.shakedown)

        from pymbxas.mbxas.broaden import broadened_spectrum
        want_full = not shakedown_only or return_components
        want_shakedown = shakedown_only or return_components
        spectrum = (np.zeros_like(energy_grid, dtype=float)
                    if want_full else None)
        shakedown_spectrum = (np.zeros_like(energy_grid, dtype=float)
                              if want_shakedown else None)
        for xas_order, (energy, amplitude, xas_down) in xas.items():
            for sp_order, (shift, probability, sp_down) in spectator.items():
                total_order = xas_order + sp_order
                if total_order > max_total_order:
                    continue
                # Shake-down is a decomposition of higher-order terms, not
                # an alternative physical spectrum.  In particular, f1 has
                # no shake-down component even if its reference overlap is
                # combined with a flagged zero-energy constituent.
                if axis is None:
                    oscillator = (np.sum(np.abs(amplitude) ** 2, axis=0)
                                  / amplitude.shape[0])
                else:
                    oscillator = np.abs(amplitude[axis]) ** 2
                # Stream the Cartesian product in bounded blocks. This is
                # algebraically identical to assembling all final sticks and
                # broadening once, but its peak memory is independent of the
                # full f x spectator-single count.
                shift_chunk = max(
                    1, _MAX_JOINT_ELEMENTS // max(1, energy.size))
                for start in range(0, shift.size, shift_chunk):
                    stop = min(start + shift_chunk, shift.size)
                    shift_block = shift[start:stop]
                    probability_block = probability[start:stop]
                    joint_energy, joint_intensity = _combine_spin_stick_block(
                        energy, oscillator, shift_block, probability_block)
                    window = (
                        (joint_energy >= final_energy_range[0])
                        & (joint_energy <= final_energy_range[1]))
                    if want_full and np.any(window):
                        spectrum += broadened_spectrum(
                            energy_grid, Ha * joint_energy[window],
                            joint_intensity[window], sigma)
                    if want_shakedown and total_order > 0:
                        down = window & (
                            xas_down[:, None]
                            | sp_down[None, start:stop])
                        if np.any(down):
                            shakedown_spectrum += broadened_spectrum(
                                energy_grid, Ha * joint_energy[down],
                                joint_intensity[down], sigma)

        if return_components:
            return energy_grid, spectrum, shakedown_spectrum
        if shakedown_only:
            return energy_grid, shakedown_spectrum
        return energy_grid, spectrum
        
    def get_amplitude_tensor(self):
        return np.einsum("ij,jk->ikj", self.amplitude, self.amplitude.T)
    
    def amp2int(self, amplitude=None, energies=None):
        """Amplitude -> isotropic intensity, sigma(omega) ~ omega * |M|^2
        (Eq. 4, PRB 107, 035146). `energies` must be in Hartree, matching
        the atomic-unit amplitude; defaults to this spectrum's own."""
        if amplitude is None:
            amplitude = self.amplitude
        if energies is None:
            energies = self._energies
        return energies * np.sum(amplitude**2, axis=0) / amplitude.shape[0]

    def _shakeup_sticks_by_order(self, order, channel, tol):
        """Cached {order: (delta_e_ev, weight)} for one spin channel -- the
        per-order form _shakeup_sticks concatenates, and the form
        combine_cross_channel_sticks (mbxas.shakeup) needs directly.

        Safe across transform(): see _shakeup_sticks."""
        from pymbxas.mbxas.mbxas import build_A_K, occ_unocc_indices
        from pymbxas.mbxas.shakeup import shakeup_sticks_by_order

        if channel is None:
            channel = self._channel

        if not hasattr(self, "_shakeup_cache_by_order"):
            self._shakeup_cache_by_order = {}

        key = (channel, order, tol)
        if key not in self._shakeup_cache_by_order:
            occ_idxs_gs, occ_idxs_fch, uno_idxs_fch = occ_unocc_indices(
                self._gs_mo_occ[channel], self._mo_occ[channel],
                self._core_orb_idx,
                core_hole_idx=self._fch_core_hole_index())

            AMat, _, _, APrimeMat = build_A_K(self._mb_overlap[channel], occ_idxs_fch,
                                              occ_idxs_gs, uno_idxs_fch)

            eps_occ   = self._fch_mo_energy[channel][occ_idxs_fch]
            eps_unocc = self._fch_mo_energy[channel][uno_idxs_fch]

            sticks_by_order, _ = shakeup_sticks_by_order(AMat, APrimeMat, eps_occ, eps_unocc, order=order, tol=tol)
            self._shakeup_cache_by_order[key] = {
                k: (Ha * e, w) for k, (e, w) in sticks_by_order.items()
            }

        return self._shakeup_cache_by_order[key]

    def _shakeup_sticks(self, order, channel, tol):
        """Cached (delta_e_ev, weight, orders_included) for one spin channel.
        `channel=None` defaults to the excited channel. Cross-spin processing
        uses the structurally distinct spectator helper below.

        Safe across transform(): that only rotates/permutes mo_coeff and
        amplitude, never mb_overlap/mo_occ/mo_energy/core_orb_idx, so a
        cache built before a transform() call stays valid after it."""
        if channel is None:
            channel = self._channel

        if not hasattr(self, "_shakeup_cache"):
            self._shakeup_cache = {}

        key = (channel, order, tol)
        if key not in self._shakeup_cache:
            sticks_by_order = self._shakeup_sticks_by_order(order, channel, tol)
            orders = sorted(sticks_by_order)
            all_e = [sticks_by_order[k][0] for k in orders]
            all_w = [sticks_by_order[k][1] for k in orders]
            self._shakeup_cache[key] = (np.concatenate(all_e), np.concatenate(all_w), orders)

        return self._shakeup_cache[key]

    def _spectator_shakeup_sticks(self, order, tol):
        """Cached {order: (delta_e_ev, weight)} for the spectator
        (non-excited) spin channel's own valence relaxation. It supplies the
        spectator factor in PyMBXAS's cross-spin overlap convolution. The
        production spectrum combines these with explicit order-resolved XAS
        inputs; see dev/shakeup.md. Same
        underlying persisted data as _shakeup_sticks
        (mb_overlap/mo_occ/mo_energy for both channels), but built from
        spectator_occ_unocc_indices since this channel keeps its full
        ground-state occupation in the FCH step."""
        from pymbxas.mbxas.mbxas import build_A_K, spectator_occ_unocc_indices
        from pymbxas.mbxas.shakeup import shakeup_sticks_by_order

        channel = 1 - self._channel

        if not hasattr(self, "_spectator_shakeup_cache"):
            self._spectator_shakeup_cache = {}

        key = (order, tol)
        if key not in self._spectator_shakeup_cache:
            occ_idxs_gs, occ_idxs_fch, uno_idxs_fch = spectator_occ_unocc_indices(
                self._gs_mo_occ[channel], self._mo_occ[channel])

            AMat, _, _, APrimeMat = build_A_K(self._mb_overlap[channel], occ_idxs_fch,
                                              occ_idxs_gs, uno_idxs_fch)

            eps_occ   = self._fch_mo_energy[channel][occ_idxs_fch]
            eps_unocc = self._fch_mo_energy[channel][uno_idxs_fch]

            sticks_by_order, _ = shakeup_sticks_by_order(AMat, APrimeMat, eps_occ, eps_unocc, order=order, tol=tol)
            self._spectator_shakeup_cache[key] = {
                k: (Ha * e, w) for k, (e, w) in sticks_by_order.items()
            }

        return self._spectator_shakeup_cache[key]

    def _combined_shakeup_sticks(self, order, spectator_order, max_total_order, tol, shakedown_only):
        """Compatibility view of the screened overlap distribution without order 0."""
        energy, weight, flags, orders = self._overlap_distribution(
            order, spectator_order, max_total_order, tol=tol)
        nonzero_order = orders > 0
        if shakedown_only:
            nonzero_order &= flags
        return energy[nonzero_order], weight[nonzero_order]

    def _overlap_factors(self, order, spectator_order="auto",
                         max_total_order=None, channel=None,
                         max_configurations=2_000_000, tol=0.01,
                         pair_energy_max=None):
        """Build screened order-resolved overlap factors for both spins."""
        from pymbxas.mbxas.mbxas import (
            build_A_K, occ_unocc_indices, spectator_occ_unocc_indices)
        from pymbxas.mbxas.shakeup import (
            available_overlap_weight, overlap_sticks_from_K,
            screened_overlap_doubles_from_K)

        if order is None:
            order = 0
        if order == "auto":
            raise ValueError("order='auto' is no longer supported; choose an explicit order")
        order = int(order)
        if spectator_order == "auto":
            spectator_order = order if channel is None else None
        elif isinstance(spectator_order, str):
            raise ValueError("spectator_order must be 'auto', None, or an integer")
        if channel is None:
            channel = self._channel
        if channel == self._channel:
            occ_gs, occ_fch, virt = occ_unocc_indices(
                self._gs_mo_occ[channel], self._mo_occ[channel],
                self._core_orb_idx,
                core_hole_idx=self._fch_core_hole_index())
        else:
            occ_gs, occ_fch, virt = spectator_occ_unocc_indices(
                self._gs_mo_occ[channel], self._mo_occ[channel])
        AMat, det_A, K, APrimeMat = build_A_K(
            self._mb_overlap[channel], occ_fch, occ_gs, virt)
        first_available = available_overlap_weight(AMat, APrimeMat)
        eps_occ = self._fch_mo_energy[channel][occ_fch]
        eps_virt = self._fch_mo_energy[channel][virt]
        screening_log = with_log_context(
            logging.getLogger("pymbxas.mbxas.shakeup"),
            site=self._site_label(), stage="overlap diagnostic")

        def build_orders(det_weight, matrix, occupied_energy,
                         virtual_energy, max_order):
            result = {}
            for current_order in range(max_order + 1):
                if current_order == 2:
                    result[current_order] = screened_overlap_doubles_from_K(
                        det_weight, matrix, occupied_energy, virtual_energy,
                        tol=tol, max_configurations=max_configurations,
                        pair_energy_max=pair_energy_max, log=screening_log)
                else:
                    result[current_order] = overlap_sticks_from_K(
                        det_weight, matrix, occupied_energy, virtual_energy,
                        current_order,
                        max_configurations=max_configurations,
                        pair_energy_max=pair_energy_max)
            return result

        first = build_orders(
            abs(det_A) ** 2, K, eps_occ, eps_virt, order)

        if spectator_order is None:
            second = build_orders(
                1.0, np.empty((0, 0)), np.empty(0), np.empty(0), 0)
            second_available = 1.0
        else:
            spectator_order = int(spectator_order)
            sp = 1 - self._channel
            sp_gs, sp_fch, sp_virt = spectator_occ_unocc_indices(
                self._gs_mo_occ[sp], self._mo_occ[sp])
            sp_A, sp_det, sp_K, sp_APrime = build_A_K(
                self._mb_overlap[sp], sp_fch, sp_gs, sp_virt)
            second_available = available_overlap_weight(sp_A, sp_APrime)
            sp_eps_occ = self._fch_mo_energy[sp][sp_fch]
            sp_eps_virt = self._fch_mo_energy[sp][sp_virt]
            second = build_orders(
                abs(sp_det) ** 2, sp_K, sp_eps_occ, sp_eps_virt,
                spectator_order)

        if max_total_order is None:
            max_total_order = max(order, 0 if spectator_order is None else spectator_order)
        max_total_order = int(max_total_order)
        if max_total_order < 0:
            raise ValueError("max_total_order must be non-negative")
        return (first, second, max_total_order, spectator_order,
                (first_available, second_available))

    def _overlap_distribution(self, order, spectator_order="auto",
                              max_total_order=None, channel=None,
                              max_configurations=2_000_000, tol=0.01,
                              pair_energy_max=None):
        """Materialize the screened determinant-overlap distribution.

        Production diagnostics use ``_overlap_diagnostic`` to stream the
        cross-spin products. This compatibility helper is intended for small
        validation systems and constituent inspection.
        """
        first, second, max_total_order, _, _ = self._overlap_factors(
            order, spectator_order, max_total_order, channel,
            max_configurations, tol, pair_energy_max)
        energies, weights, flags, orders = [], [], [], []
        for i, left in first.items():
            for j, right in second.items():
                if i + j > max_total_order:
                    continue
                energies.append((left.energy[:, None] + right.energy[None, :]).ravel())
                weights.append((left.weight[:, None] * right.weight[None, :]).ravel())
                flags.append((left.shakedown[:, None] | right.shakedown[None, :]).ravel())
                orders.append(np.full(left.energy.size * right.energy.size, i + j, dtype=int))
        return (Ha * np.concatenate(energies), np.concatenate(weights),
                np.concatenate(flags), np.concatenate(orders))

    def _overlap_diagnostic(self, order, sigma, npoints, erange,
                            spectator_order, max_total_order, channel,
                            max_configurations, tol,
                            pair_energy_max=None):
        """Stream the broadened two-spin overlap and its flagged mass."""
        from pymbxas.mbxas.broaden import broadened_spectrum

        (first, second, max_total_order, resolved_spectator,
         available_by_channel) = (
            self._overlap_factors(
                order, spectator_order, max_total_order, channel,
                max_configurations, tol, pair_energy_max))
        allowed = [
            (i, left, j, right)
            for i, left in first.items()
            for j, right in second.items()
            if i + j <= max_total_order
            and left.energy.size and right.energy.size]

        if erange is None:
            minima = [Ha * (left.energy.min() + right.energy.min())
                      for _, left, _, right in allowed]
            maxima = [Ha * (left.energy.max() + right.energy.max())
                      for _, left, _, right in allowed]
            lo = min([-5 * sigma] + [value - 5 * sigma for value in minima])
            hi = max([5 * sigma] + [value + 5 * sigma for value in maxima])
            erange = [lo, hi]
        egrid = np.linspace(erange[0], erange[1], npoints)
        curve = np.zeros_like(egrid, dtype=float)
        total_mass = 0.0
        shakedown_mass = 0.0
        mass_by_total_order = {}

        for left_order, left, right_order, right in allowed:
            left_mass = float(left.weight.sum())
            right_mass = float(right.weight.sum())
            block_mass = left_mass * right_mass
            total_mass += block_mass
            total_order = left_order + right_order
            mass_by_total_order[total_order] = (
                mass_by_total_order.get(total_order, 0.0) + block_mass)
            left_up_mass = float(left.weight[~left.shakedown].sum())
            right_up_mass = float(right.weight[~right.shakedown].sum())
            left_down_mass = float(left.weight[left.shakedown].sum())
            right_down_mass = float(right.weight[right.shakedown].sum())
            shakedown_mass += (
                left_down_mass * right_mass
                + left_up_mass * right_down_mass)

            right_chunk = max(
                1, _MAX_JOINT_ELEMENTS // max(1, left.energy.size))
            for start in range(0, right.energy.size, right_chunk):
                stop = min(start + right_chunk, right.energy.size)
                joint_energy = (
                    left.energy[:, None]
                    + right.energy[None, start:stop]).ravel()
                joint_weight = (
                    left.weight[:, None]
                    * right.weight[None, start:stop]).ravel()
                curve += broadened_spectrum(
                    egrid, Ha * joint_energy, joint_weight, sigma)

        orders = None
        if resolved_spectator is None:
            orders = sorted({
                i for i, sticks in first.items()
                if i > 0 and sticks.energy.size})
        available_mass = float(np.prod(available_by_channel))
        captured_fraction = (
            total_mass / available_mass if available_mass > 0 else 0.0)
        overlap_report = {
            "by_total_order": dict(sorted(mass_by_total_order.items())),
            "captured": total_mass,
            "available": available_mass,
            "fraction": captured_fraction,
        }
        return (egrid, curve, orders, total_mass, shakedown_mass,
                overlap_report)

    def get_shakeup_spectrum(self, order=1, channel=None, sigma=0.5,
                              npoints=3001, erange=None, tol=0.01,
                              spectator_order="auto", max_total_order=None,
                              max_configurations=2_000_000):
        """Broadened determinant-overlap distribution P(dE).

        This diagnostic is distinct from the explicit higher-order XAS
        amplitudes used by ``get_mbxas_spectra(f_order=...)``. Its
        weights include the reference determinant and are not unit-normalized.

        spectator_order and max_total_order combine with the
        spectator (non-excited) channel's own overlap kernel -- only valid with
        the default channel=None (the excited channel), since the
        combination fixes both channels' identity itself. For a cross-channel
        result, the third return value (orders_included) is None instead of a
        list because there is no single per-channel order list to report.
        """
        if spectator_order == "auto" and channel is not None:
            spectator_order = None
        if spectator_order is not None and channel is not None:
            raise ValueError(
                "spectator_order combines the excited channel with the "
                "spectator channel; pass channel=None (the default) "
                "rather than an explicit channel."
            )

        egrid, curve, orders, _, _, _ = self._overlap_diagnostic(
            order, sigma, npoints, erange, spectator_order,
            max_total_order, channel, max_configurations, tol)
        return egrid, curve, orders

    def get_mbxas_decomposition(self, f_order=2, sigma=0.5, npoints=3001,
                                erange=None, tol=0.01,
                                spectator_order="auto", max_total_order=None,
                                max_configurations=2_000_000):
        """Return resolved and cumulative spectra through ``f_order``.

            "energy"      : (npoints,) shared energy grid, eV
            "contributions": {1: f1, 2: f2, ..., f_order: fN}
            "decomposition": {2: {"shakeup": ..., "shakedown": ...}, ...}
            "cumulative"  : {1: f1, 2: f1+f2, ..., f_order: total}
            "total"       : cumulative[f_order]
            "integrated"  : matching contribution/cumulative/total integrals
            "probability" : (delta_e, curve, orders_included) from the
                             screened determinant-overlap diagnostic
            "overlap"     : captured mass by total order, total captured
                             and available mass, and captured fraction
            "shakedown_fraction" : fraction of determinant-overlap mass
                             carrying an any-negative-constituent flag, for
                             the spin-combined stick set backing the correction.
                             A warning is logged if this exceeds tol.

        Data only. Use ``print_mbxas_summary`` for reporting and
        ``plot_mbxas_decomposition`` for plotting.
        """
        if (not isinstance(f_order, (int, np.integer))
                or isinstance(f_order, (bool, np.bool_)) or f_order < 1):
            raise ValueError("f_order must be a positive integer")
        f_order = int(f_order)

        cumulative = {}
        shakedown_cumulative = {1: None}
        energy = None
        for current_f in range(1, f_order + 1):
            current_range = erange if energy is None else [energy[0], energy[-1]]
            current_total_order = current_f - 1
            if max_total_order is not None:
                current_total_order = min(
                    current_total_order, int(max_total_order))
            if current_f == 1:
                energy, intensity = self.get_mbxas_spectra(
                    sigma=sigma, npoints=npoints, erange=current_range,
                    tol=tol, f_order=current_f,
                    spectator_order=spectator_order,
                    max_total_order=current_total_order,
                    max_configurations=max_configurations)
                cumulative[current_f] = intensity
                shakedown_cumulative[1] = np.zeros_like(intensity)
                continue
            current_spectator_order = (
                current_f - 1 if spectator_order == "auto"
                else spectator_order)
            energy, intensity, down_intensity = self._get_many_body_mbxas_spectra(
                axis=None, sigma=sigma, npoints=npoints, tol=tol,
                erange=current_range,
                max_extra_order=current_f - 1,
                spectator_order=current_spectator_order,
                max_total_order=current_total_order,
                shakedown_only=False,
                max_configurations=max_configurations,
                return_components=True)
            cumulative[current_f] = intensity
            shakedown_cumulative[current_f] = down_intensity

        contributions = {1: cumulative[1].copy()}
        for current_f in range(2, f_order + 1):
            contributions[current_f] = (
                cumulative[current_f] - cumulative[current_f - 1])

        decomposition = {}
        for current_f in range(2, f_order + 1):
            shakedown = (
                shakedown_cumulative[current_f]
                - shakedown_cumulative[current_f - 1])
            decomposition[current_f] = {
                "shakeup": contributions[current_f] - shakedown,
                "shakedown": shakedown,
            }

        integrated = {
            "contributions": {
                key: np.trapezoid(value, energy)
                for key, value in contributions.items()},
            "cumulative": {
                key: np.trapezoid(value, energy)
                for key, value in cumulative.items()},
            "decomposition": {
                key: {
                    name: np.trapezoid(value, energy)
                    for name, value in parts.items()}
                for key, parts in decomposition.items()},
            "total": np.trapezoid(cumulative[f_order], energy),
        }

        max_extra_order = f_order - 1
        probability_total_order = max_extra_order
        if max_total_order is not None:
            probability_total_order = min(
                probability_total_order, int(max_total_order))

        pair_energy_max = (
            energy[-1] - energy[0] + _GAUSSIAN_SUPPORT * sigma) / Ha
        (prob_e, prob_curve, prob_orders, total_mass,
         shakedown_mass, overlap_report) = self._overlap_diagnostic(
            max_extra_order, sigma, npoints, None, spectator_order,
            probability_total_order, None, max_configurations, tol,
            pair_energy_max=pair_energy_max)
        shakedown_fraction = shakedown_mass / total_mass if total_mass > 0 else 0.0
        decomposition_log = with_log_context(
            logger, site=self._site_label(), stage="decomposition")
        if shakedown_fraction > tol:
            decomposition_log.warning(
                "shake-down fraction %.4f exceeds tol=%.3g: a non-negligible "
                "share of shake-up probability mass has delta_e < 0",
                shakedown_fraction, tol)

        summary_fields = {
            "highest order": f"f{f_order}",
            "energy window": f"{energy[0]:.3f} to {energy[-1]:.3f} eV",
            "screening tolerance": f"{tol:.3g}",
        }
        for current_f in sorted(integrated["contributions"]):
            summary_fields[f"integrated f{current_f}"] = (
                f"{integrated['contributions'][current_f]:.8e}")
            if current_f in integrated["decomposition"]:
                parts = integrated["decomposition"][current_f]
                summary_fields[f"f{current_f} shake-up"] = (
                    f"{parts['shakeup']:.8e}")
                summary_fields[f"f{current_f} shake-down"] = (
                    f"{parts['shakedown']:.8e}")
        summary_fields["integrated total"] = f"{integrated['total']:.8e}"
        summary_fields["captured overlap"] = (
            f"{overlap_report['captured']:.8e} / "
            f"{overlap_report['available']:.8e} "
            f"({100 * overlap_report['fraction']:.4f}%)")
        summary_fields["overlap shake-down"] = (
            f"{100 * shakedown_fraction:.4f}%")
        decomposition_log.info(
            "Completed\n%s", format_log_fields(summary_fields))

        return {
            "energy": energy,
            "contributions": contributions,
            "decomposition": decomposition,
            "cumulative": cumulative,
            "total": cumulative[f_order],
            "integrated": integrated,
            "probability": (prob_e, prob_curve, prob_orders),
            "overlap": overlap_report,
            "shakedown_fraction": shakedown_fraction,
        }

    def print_mbxas_summary(self, decomposition, file=None):
        """Print integrated MBXAS contributions from decomposition data."""
        import sys

        if file is None:
            file = sys.stdout
        contributions = decomposition["integrated"]["contributions"]
        parts = decomposition["integrated"]["decomposition"]
        highest_order = max(contributions)
        lines = [f"MBXAS decomposition through f{highest_order}"]
        for f_order in sorted(contributions):
            lines.append(
                f"  f{f_order} contribution : {contributions[f_order]:.8e}")
            if f_order in parts:
                lines.append(
                    f"    shake-up            : {parts[f_order]['shakeup']:.8e}")
                lines.append(
                    f"    shake-down          : {parts[f_order]['shakedown']:.8e}")
        lines.append(
            f"  total                  : "
            f"{decomposition['integrated']['total']:.8e}")
        lines.append(
            f"  overlap shake-down     : "
            f"{100 * decomposition['shakedown_fraction']:.4f}%")
        overlap = decomposition["overlap"]
        lines.append("Overlap convergence")
        for order, mass in overlap["by_total_order"].items():
            lines.append(f"  order {order:<2d}              : {mass:.8e}")
        lines.append(f"  captured               : {overlap['captured']:.8e}")
        lines.append(f"  available              : {overlap['available']:.8e}")
        lines.append(
            f"  captured fraction      : {100 * overlap['fraction']:.4f}%")
        print("\n".join(lines), file=file)

    def plot_mbxas_decomposition(
            self, f_order=2, sigma=0.5, npoints=3001, erange=None, tol=0.01,
            spectator_order="auto", max_total_order=None,
            max_configurations=2_000_000, show_probability=True,
            show_resolved=False, show_cumulative=False, figsize=None):
        """Calculate and plot this site's MBXAS decomposition.

        Matplotlib remains optional and is imported only by the plotting
        helper. Use :meth:`get_mbxas_decomposition` directly when the data
        need to be retained or combined before plotting.
        """
        from pymbxas.plotting import plot_mbxas_decomposition

        decomposition = self.get_mbxas_decomposition(
            f_order=f_order, sigma=sigma, npoints=npoints, erange=erange,
            tol=tol, spectator_order=spectator_order,
            max_total_order=max_total_order,
            max_configurations=max_configurations)
        return plot_mbxas_decomposition(
            decomposition, show_probability=show_probability,
            show_resolved=show_resolved,
            show_cumulative=show_cumulative, figsize=figsize)

    def get_orbital_rearrangement(
            self, energy_window=(-15.0, 15.0),
            energy_reference="gs_homo", min_overlap=0.05,
            include_core=False):
        """Return GS/FCH orbital levels and a one-to-one overlap assignment.

        Energies are returned in eV. By default the global GS HOMO is zero and
        only a frontier window is selected for display. ``include_core=True``
        also selects the excited GS core and its FCH partner; this can greatly
        expand the vertical range and compress the frontier levels.

        The assignment is inferred from the final squared MO overlap and is
        not a record of individual MOM/maxvol SCF steps. Near-degenerate
        subspaces can therefore have non-unique individual orbital labels.
        """
        from pymbxas.utils.orbitals import match_orbitals_by_overlap

        if self._gs_mo_energy is None:
            raise RuntimeError(
                "Ground-state orbital energies are absent from this historical "
                "Spectra file. Load the calculation checkpoint and call "
                "to_spectra() to rebuild the post-processing object without "
                "rerunning SCF.")
        if min_overlap < 0 or min_overlap > 1:
            raise ValueError("min_overlap must lie between zero and one")
        if energy_window is not None:
            if len(energy_window) != 2 or energy_window[0] >= energy_window[1]:
                raise ValueError("energy_window must be (minimum, maximum)")

        gs_energy = np.asarray(self._gs_mo_energy)
        fch_energy = np.asarray(self._fch_mo_energy)
        gs_occ = np.asarray(self._gs_mo_occ)
        fch_occ = np.asarray(self._mo_occ)
        overlap = np.asarray(self._mb_overlap)
        if any(array.ndim != 2 for array in (
                gs_energy, fch_energy, gs_occ, fch_occ)):
            raise ValueError("Orbital energies and occupations require spin axes")
        if overlap.ndim != 3 or overlap.shape[0] != 2:
            raise ValueError("MB overlap must have shape (2, n_fch, n_gs)")

        def frontier(energies, occupations, excluded_unoccupied=()):
            occupied = np.flatnonzero(occupations > 0.5)
            unoccupied = np.flatnonzero(occupations <= 0.5)
            if excluded_unoccupied:
                unoccupied = unoccupied[
                    ~np.isin(unoccupied, tuple(excluded_unoccupied))]
            homo = (int(occupied[np.argmax(energies[occupied])])
                    if occupied.size else None)
            lumo = (int(unoccupied[np.argmin(energies[unoccupied])])
                    if unoccupied.size else None)
            return homo, lumo

        gs_frontiers = [frontier(gs_energy[spin], gs_occ[spin])
                        for spin in range(2)]
        occupied_homo_energies = [
            gs_energy[spin, homo] for spin, (homo, _) in
            enumerate(gs_frontiers) if homo is not None]
        if energy_reference == "gs_homo":
            reference_ha = max(occupied_homo_energies)
            reference_label = "GS HOMO"
        elif energy_reference in ("absolute", "vacuum"):
            reference_ha = 0.0
            reference_label = "absolute"
        else:
            raise ValueError(
                "energy_reference must be 'gs_homo' or 'absolute'")

        channels = []
        for spin in range(2):
            gs_ev = (gs_energy[spin] - reference_ha) * Ha
            fch_ev = (fch_energy[spin] - reference_ha) * Ha
            gs_indices, fch_indices, weights = match_orbitals_by_overlap(
                overlap[spin])
            gs_homo, gs_lumo = gs_frontiers[spin]
            core_gs = self._core_orb_idx if spin == self._channel else None
            core_fch = (self._fch_core_hole_index()
                        if spin == self._channel else None)
            fch_homo, fch_lumo = frontier(
                fch_energy[spin], fch_occ[spin],
                excluded_unoccupied=(() if core_fch is None else (core_fch,)))

            if energy_window is None:
                selected_gs = np.ones(gs_ev.size, dtype=bool)
                selected_fch = np.ones(fch_ev.size, dtype=bool)
            else:
                selected_gs = ((gs_ev >= energy_window[0])
                               & (gs_ev <= energy_window[1]))
                selected_fch = ((fch_ev >= energy_window[0])
                                & (fch_ev <= energy_window[1]))
            if include_core and core_gs is not None:
                selected_gs[core_gs] = True
                selected_fch[core_fch] = True

            visible = (
                (weights >= min_overlap)
                & selected_gs[gs_indices]
                & selected_fch[fch_indices])
            matches = {
                "gs_index": gs_indices[visible],
                "fch_index": fch_indices[visible],
                "overlap": weights[visible],
            }
            all_matches = {
                "gs_index": gs_indices,
                "fch_index": fch_indices,
                "overlap": weights,
            }
            channels.append({
                "spin": spin,
                "role": "excited" if spin == self._channel else "spectator",
                "gs_energy": gs_ev,
                "fch_energy": fch_ev,
                "gs_occupation": gs_occ[spin],
                "fch_occupation": fch_occ[spin],
                "gs_homo": gs_homo,
                "gs_lumo": gs_lumo,
                "fch_homo": fch_homo,
                "fch_lumo": fch_lumo,
                "core_gs": core_gs,
                "core_fch": core_fch,
                "selected_gs": selected_gs,
                "selected_fch": selected_fch,
                "matches": matches,
                "all_matches": all_matches,
            })

        return {
            "channels": tuple(channels),
            "excited_channel": self._channel,
            "energy_reference": energy_reference,
            "reference_energy_ev": reference_ha * Ha,
            "reference_label": reference_label,
            "energy_window": energy_window,
            "min_overlap": min_overlap,
            "site": self._site_label(),
        }

    def plot_orbital_rearrangement(
            self, energy_window=(-15.0, 15.0),
            energy_reference="gs_homo", min_overlap=0.05,
            include_core=False, *, show_indices=False, show_dos=False,
            dos_sigma=0.25, figsize=(10.0, 6.0)):
        """Calculate and plot the two-spin GS-to-FCH orbital rearrangement."""
        from pymbxas.plotting import plot_orbital_rearrangement

        data = self.get_orbital_rearrangement(
            energy_window=energy_window, energy_reference=energy_reference,
            min_overlap=min_overlap, include_core=include_core)
        return plot_orbital_rearrangement(
            data, show_indices=show_indices, show_dos=show_dos,
            dos_sigma=dos_sigma, figsize=figsize)

    @property
    def _active_mo_occ(self):
        return self._mo_occ[self._channel]

    @property
    def _active_mo_coeff(self):
        return self._mo_coeff[self._channel]

    # get CMOs
    @property
    def CMO(self):
        uno_idxs = np.flatnonzero(self._active_mo_occ == 0)
        uno_idxs = uno_idxs[uno_idxs != self._fch_core_hole_index()]
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
        self.mol = ase_to_mole(
            self.structure, verbose=0,
            **_molecule_settings(self.calc_settings))
        return
    
    def save(self, filename="spectra.h5"):
        """Saves the object to an HDF5 file."""
        with h5.create(filename, h5.KIND_SPECTRA) as fout:
            self._write_into(fout)
        return
    
    @property
    def exc_idx(self):
        return self._exc_idx

    def _site_label(self):
        """Return the calculation-style atom label used in log context."""
        if self.exc_idx is None:
            return None
        symbol = self.structure.get_chemical_symbols()[self.exc_idx]
        return f"{symbol}:{self.exc_idx}"
    
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
