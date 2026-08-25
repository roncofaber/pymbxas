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

# pyscf stuff
from pyscf import gto, lo
from pyscf.lo import iao, orth

from ase import units
Ha = units.Ha
logger = logging.getLogger(__name__)

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

        # data needed to rebuild A/K for shake-up satellites (both channels,
        # so the not-yet-implemented cross-spin extension needs no schema
        # change): shape (2, norb_fch, norb_gs), (2, norb_fch), (2, norb_gs)
        self._mb_overlap    = mbxas["mb_overlap"]
        self._fch_mo_energy = data.mo_energy
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
                          erange=None, el_label=None, shakeup_order=None,
                          spectator_order=None, max_total_order=None,
                          shakedown_only=False):

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

        if shakeup_order is not None or spectator_order is not None:
            from pymbxas.mbxas.shakeup import convolve_shakeup
            delta_e_ev, weight = self._combined_shakeup_sticks(
                shakeup_order, spectator_order, max_total_order, tol, shakedown_only)
            spectra = convolve_shakeup(erange, spectra, delta_e_ev, weight, sigma)

        return erange, spectra
        
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
                self._gs_mo_occ[channel], self._mo_occ[channel], self._core_orb_idx)

            _, _, K, _ = build_A_K(self._mb_overlap[channel], occ_idxs_fch,
                                occ_idxs_gs, uno_idxs_fch)

            eps_occ   = self._fch_mo_energy[channel][occ_idxs_fch]
            eps_unocc = self._fch_mo_energy[channel][uno_idxs_fch]

            sticks_by_order, _ = shakeup_sticks_by_order(K, eps_occ, eps_unocc, order=order, tol=tol)
            self._shakeup_cache_by_order[key] = {
                k: (Ha * e, w) for k, (e, w) in sticks_by_order.items()
            }

        return self._shakeup_cache_by_order[key]

    def _shakeup_sticks(self, order, channel, tol):
        """Cached (delta_e_ev, weight, orders_included) for one spin channel.
        `channel=None` defaults to the excited channel; an explicit channel
        is accepted so a future cross-spin feature can call this on the
        other channel without a signature change.

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
        (non-excited) spin channel's own valence relaxation -- the
        cross-spin contribution of mbxas-qe's spin_convolve_spectrum
        (spec.f90). Same underlying persisted data as _shakeup_sticks
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

            _, _, K, _ = build_A_K(self._mb_overlap[channel], occ_idxs_fch,
                                occ_idxs_gs, uno_idxs_fch)

            eps_occ   = self._fch_mo_energy[channel][occ_idxs_fch]
            eps_unocc = self._fch_mo_energy[channel][uno_idxs_fch]

            sticks_by_order, _ = shakeup_sticks_by_order(K, eps_occ, eps_unocc, order=order, tol=tol)
            self._spectator_shakeup_cache[key] = {
                k: (Ha * e, w) for k, (e, w) in sticks_by_order.items()
            }

        return self._spectator_shakeup_cache[key]

    def _combined_shakeup_sticks(self, shakeup_order, spectator_order, max_total_order, tol, shakedown_only):
        """Resolve the (possibly cross-channel-combined, possibly
        shakedown-filtered) shake-up sticks for this spectrum, in eV.

        spectator_order=None takes the exact pre-cross-spin code path
        (self._shakeup_sticks), so shakeup_order alone stays byte-identical
        to before this feature existed. Otherwise, both channels' per-order
        sticks are combined via mbxas.shakeup.combine_cross_channel_sticks,
        physically treating the two spin channels' relaxations as
        independent processes.
        """
        if spectator_order is None:
            if shakeup_order is None:
                return np.empty(0), np.empty(0)
            delta_e, weight, _ = self._shakeup_sticks(shakeup_order, None, tol)
        else:
            from pymbxas.mbxas.shakeup import combine_cross_channel_sticks

            sticks_a = {} if shakeup_order is None else self._shakeup_sticks_by_order(shakeup_order, None, tol)
            sticks_b = self._spectator_shakeup_sticks(spectator_order, tol)

            if max_total_order is None:
                max_total_order = (max(sticks_a) if sticks_a else 0) + max(sticks_b)

            delta_e, weight = combine_cross_channel_sticks(sticks_a, sticks_b, max_total_order, tol=tol)

        if shakedown_only:
            mask = delta_e < 0
            delta_e, weight = delta_e[mask], weight[mask]

        return delta_e, weight

    def get_shakeup_spectrum(self, order="auto", channel=None, sigma=0.5,
                              npoints=3001, erange=None, tol=0.01,
                              spectator_order=None, max_total_order=None,
                              shakedown_only=False):
        """Broadened valence shake-up probability spectrum P(dE), the
        f^(n) terms beyond the one-body truncation (see dev/method.md).
        Convolve this onto a main spectrum's own grid with
        pymbxas.mbxas.shakeup.convolve_shakeup, or use
        get_mbxas_spectra(shakeup_order=...) to do that automatically.

        spectator_order, max_total_order, shakedown_only: combine with the
        spectator (non-excited) channel's own shake-up -- only valid with
        the default channel=None (the excited channel), since the
        combination fixes both channels' identity itself. When any of
        these three is used, the third return value (orders_included) is
        None instead of a list: a cross-channel or shakedown-filtered
        result has no single per-channel order list to report.
        """
        from pymbxas.mbxas.shakeup import broaden_shakeup

        if spectator_order is not None and channel is not None:
            raise ValueError(
                "spectator_order combines the excited channel with the "
                "spectator channel; pass channel=None (the default) "
                "rather than an explicit channel."
            )

        if spectator_order is None and max_total_order is None and not shakedown_only:
            delta_e_ev, weight, orders = self._shakeup_sticks(order, channel, tol)
        else:
            delta_e_ev, weight = self._combined_shakeup_sticks(
                order, spectator_order, max_total_order, tol, shakedown_only)
            orders = None

        if erange is None:
            # widen on both sides, never narrower than +-5*sigma around the
            # n=0 term -- delta_e_ev can be negative for a non-aufbau
            # MOM-converged state, where a formally-unoccupied orbital sits
            # below a formally-occupied one
            if len(delta_e_ev):
                lo = min(-5 * sigma, delta_e_ev.min() - 5 * sigma)
                hi = max(5 * sigma, delta_e_ev.max() + 5 * sigma)
            else:
                lo, hi = -5 * sigma, 5 * sigma
            erange = [lo, hi]
        egrid = np.linspace(erange[0], erange[1], npoints)

        return egrid, broaden_shakeup(delta_e_ev, weight, egrid, sigma), orders

    def get_shakeup_summary(self, order=2, sigma=0.5, npoints=3001, erange=None,
                              tol=0.01, spectator_order=None, max_total_order=None,
                              shakedown_only=False):
        """Compare a spectrum with and without the shake-up correction, up
        to and including the given order (both bare and every intermediate
        order 1..order are included, plus the shake-up probability curve
        itself). Returns a dict:

            "energy"      : (npoints,) shared energy grid, eV
            "spectra"     : {0: bare, 1: order-1, ..., order: order-1..order}
                            plus "cross" (the fully combined excited +
                            spectator correction) when spectator_order is given
            "integrated"  : {same keys} -> trapezoidal integral of each spectrum
            "probability" : (delta_e, curve, orders_included) from
                             get_shakeup_spectrum(order=order, sigma=sigma, ...)
            "shakedown_fraction" : fraction of shake-up probability mass with
                             delta_e < 0 ("shake-down", mbxas-qe's
                             kpoint_spectral_details.f90 convention), for
                             whichever stick set (cross-combined if
                             spectator_order is given, else the plain
                             excited-channel one) backs the correction above.
                             A warning is logged if this exceeds tol.

        Data only, no plotting -- see dev/method.md for what the numbers mean.
        """
        spectra = {0: self.get_mbxas_spectra(sigma=sigma, npoints=npoints,
                                             erange=erange, tol=tol)}
        energy = spectra[0][0]
        spectra[0] = spectra[0][1]

        for k in range(1, order + 1):
            _, intensity = self.get_mbxas_spectra(sigma=sigma, npoints=npoints,
                                                   erange=[energy[0], energy[-1]],
                                                   tol=tol, shakeup_order=k)
            spectra[k] = intensity

        if spectator_order is not None:
            _, intensity_cross = self.get_mbxas_spectra(
                sigma=sigma, npoints=npoints, erange=[energy[0], energy[-1]],
                tol=tol, shakeup_order=order, spectator_order=spectator_order,
                max_total_order=max_total_order, shakedown_only=shakedown_only)
            spectra["cross"] = intensity_cross

        integrated = {k: np.trapezoid(I, energy) for k, I in spectra.items()}

        prob_e, prob_curve, prob_orders = self.get_shakeup_spectrum(
            order=order, sigma=sigma, tol=tol, spectator_order=spectator_order,
            max_total_order=max_total_order)

        delta_e_frac, weight_frac = self._combined_shakeup_sticks(
            order, spectator_order, max_total_order, tol, False)
        total_mass = weight_frac.sum() + 1.0  # +1 for the implicit n=0 "no shake-up" term
        shakedown_mass = weight_frac[delta_e_frac < 0].sum() if len(delta_e_frac) else 0.0
        shakedown_fraction = shakedown_mass / total_mass if total_mass > 0 else 0.0
        if shakedown_fraction > tol:
            logger.warning(
                "shake-down fraction %.4f exceeds tol=%.3g: a non-negligible "
                "share of shake-up probability mass has delta_e < 0",
                shakedown_fraction, tol)

        return {
            "energy": energy,
            "spectra": spectra,
            "integrated": integrated,
            "probability": (prob_e, prob_curve, prob_orders),
            "shakedown_fraction": shakedown_fraction,
        }

    @property
    def _active_mo_occ(self):
        return self._mo_occ[self._channel]

    @property
    def _active_mo_coeff(self):
        return self._mo_coeff[self._channel]

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
