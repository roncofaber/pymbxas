#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Order-resolved many-body XAS and spectator-overlap stick spectra.

The formulas follow ``mbxas-qe``'s ``mbxas_spectra.f90``. PyMBXAS uses
finite all-electron molecular orbitals rather than plane waves, but the
determinant algebra is unchanged.
"""

from dataclasses import dataclass
from itertools import combinations
from math import comb
import logging

import numpy as np

from pymbxas.io.config import format_log_fields


logger = logging.getLogger(__name__)

_QE_INITIAL_ELEMENT_TOL = 0.1
_QE_ELEMENT_REDUCTION = 0.7
_QE_SPECTRUM_BATCH = 1024 * 1024 // 10
_QE_BUILD_BATCH = _QE_SPECTRUM_BATCH
_QE_MAX_SCREEN_ITERATIONS = 200


@dataclass(frozen=True)
class StickSet:
    """Energy shifts, weights, and QE-compatible shake-down flags."""

    energy: np.ndarray
    weight: np.ndarray
    shakedown: np.ndarray

    @classmethod
    def empty(cls):
        return cls(np.empty(0), np.empty(0), np.empty(0, dtype=bool))

    def selected(self, shakedown_only=False):
        if not shakedown_only:
            return self
        mask = self.shakedown
        return StickSet(self.energy[mask], self.weight[mask], self.shakedown[mask])


def available_overlap_weight(AMat, APrimeMat):
    """Return the Cauchy--Binet overlap mass available in a row manifold.

    Stacking the occupied-reference rows ``A`` and eligible replacement rows
    ``A'`` gives every FCH row allowed in the determinant expansion. The Gram
    determinant equals the sum of squared determinants over every occupied-size
    row subset, independent of how those subsets are grouped by order.
    """
    AMat = np.asarray(AMat)
    APrimeMat = np.asarray(APrimeMat)
    if AMat.ndim != 2 or AMat.shape[0] != AMat.shape[1]:
        raise ValueError("AMat must be square")
    if APrimeMat.ndim != 2 or APrimeMat.shape[1] != AMat.shape[1]:
        raise ValueError("APrimeMat must have the same column count as AMat")
    rows = np.vstack((AMat, APrimeMat))
    available = float(np.real(np.linalg.det(rows.conj().T @ rows)))
    # A Gram determinant is non-negative. Permit only roundoff-sized negative
    # values rather than reporting an unphysical captured-overlap denominator.
    scale = max(1.0, float(np.linalg.norm(rows) ** (2 * AMat.shape[1])))
    if available < -100 * np.finfo(float).eps * scale:
        raise ValueError(
            f"overlap Gram determinant is unexpectedly negative: {available}")
    return max(0.0, available)


def _validate_order(order, n_occ, n_unocc):
    if not isinstance(order, (int, np.integer)) or order < 0:
        raise ValueError(f"order must be a non-negative integer, got {order!r}")
    return order <= n_occ and order <= n_unocc


def _check_configuration_count(count, max_configurations, kind, order):
    if max_configurations is not None and count > max_configurations:
        raise ValueError(
            f"{kind} order {order} requires {count:,} configurations, exceeding "
            f"max_configurations={max_configurations:,}. Lower the order or "
            "explicitly raise the limit."
        )


def _qe_tolerance_count(sorted_score, threshold):
    """Number of ranked K elements admitted by QE at ``threshold``.

    ``energy_sequence_tolerance_index`` returns the index of the first score
    strictly below the squared threshold. The following inclusive Fortran
    loop admits that boundary element as well. Reproducing this conservative
    endpoint is part of matching the reference implementation.
    """
    sorted_score = np.asarray(sorted_score)
    below = np.flatnonzero(sorted_score < threshold ** 2)
    return int(below[0] + 1) if below.size else len(sorted_score)


def _next_qe_element_threshold(threshold, configuration_count,
                               spectrum_batch_size=_QE_SPECTRUM_BATCH):
    """Return QE's next MB2/MB3 K-element threshold in serial execution."""
    if configuration_count >= spectrum_batch_size:
        return threshold / (
            1.0 + _QE_ELEMENT_REDUCTION * spectrum_batch_size
            / configuration_count)
    return threshold * _QE_ELEMENT_REDUCTION


def _next_qe_overlap_doubles_threshold(
        threshold, configuration_count,
        spectrum_batch_size=1024 ** 2):
    """Return QE's next spectator-double element threshold.

    ``doubles_overlap`` uses the same geometric reduction as MB2 until its
    first overlap buffer fills. Afterwards it reduces the threshold by the
    square root of the count-dependent factor because a double candidate is
    selected from a product of two ranked K elements.
    """
    if configuration_count >= spectrum_batch_size:
        return threshold / np.sqrt(
            1.0 + _QE_ELEMENT_REDUCTION * spectrum_batch_size
            / configuration_count)
    return threshold * _QE_ELEMENT_REDUCTION


def _qe_screening_converged(series, tol):
    """QE's accumulated-weight delta/derivative/curvature test."""
    if len(series) < 4 or series[-1] == 0:
        return False
    delta = series[-1] - series[-2]
    deriv = series[-1] - 2 * series[-2] + series[-3]
    curv = series[-1] - 3 * series[-2] + 3 * series[-3] - series[-4]
    return (abs(delta) < tol and abs(deriv) < 0.5 * tol
            and deriv * curv < tol ** 2)


def overlap_sticks(AMat, APrimeMat, eps_occ, eps_unocc, order,
                   max_configurations=2_000_000, pair_energy_max=None):
    """Exact determinant-overlap sticks of one particle-hole order.

    Weights include ``|det(A)|**2``. A configuration is marked as
    shake-down when any constituent promotion has negative energy, matching
    mbxas-qe. Occupied combinations are paired in ascending order with
    virtual combinations in descending order, QE's doubles/triples convention.
    """
    det_weight = float(abs(np.linalg.det(AMat)) ** 2)
    K = np.linalg.solve(np.asarray(AMat).T, np.asarray(APrimeMat).T).T
    return overlap_sticks_from_K(
        det_weight, K, eps_occ, eps_unocc, order, max_configurations,
        pair_energy_max=pair_energy_max)


def overlap_sticks_from_K(det_weight, K, eps_occ, eps_unocc, order,
                          max_configurations=2_000_000,
                          pair_energy_max=None):
    """Exact overlap sticks from a precomputed K matrix.

    ``pair_energy_max`` is an optional upper bound on each constituent
    ``eps_unocc[c] - eps_occ[v]`` promotion. Negative-energy shake-down pairs
    remain eligible. The bound reduces the energy-relevant configuration
    manifold without selecting orbitals by their array position.
    """
    eps_occ = np.asarray(eps_occ)
    eps_unocc = np.asarray(eps_unocc)
    K = np.asarray(K)
    n_occ, n_unocc = len(eps_occ), len(eps_unocc)
    if K.shape != (n_unocc, n_occ):
        raise ValueError("K shape must match eps_unocc x eps_occ")
    if not _validate_order(order, n_occ, n_unocc):
        return StickSet.empty()
    if order == 0:
        return StickSet(np.array([0.0]), np.array([float(det_weight)]), np.array([False]))
    if pair_energy_max is None:
        _check_configuration_count(
            comb(n_occ, order) * comb(n_unocc, order), max_configurations,
            "overlap", order)

    energies, weights, flags = [], [], []
    for occ in combinations(range(n_occ), order):
        for virt_asc in combinations(range(n_unocc), order):
            virt = tuple(reversed(virt_asc))
            constituent = eps_unocc[list(virt)] - eps_occ[list(occ)]
            if (pair_energy_max is not None
                    and np.any(constituent > pair_energy_max)):
                continue
            _check_configuration_count(
                len(energies) + 1, max_configurations, "overlap", order)
            minor = np.linalg.det(K[np.ix_(virt_asc, occ)])
            energies.append(constituent.sum())
            weights.append(float(det_weight) * abs(minor) ** 2)
            flags.append(np.any(constituent < 0))
    return StickSet(np.asarray(energies), np.asarray(weights), np.asarray(flags, dtype=bool))


def screened_overlap_doubles_from_K(
        det_weight, K, eps_occ, eps_unocc, tol=0.01,
        max_configurations=2_000_000, pair_energy_max=None, *, log=None):
    """QE-style adaptively screened spectator order-2 overlap sticks.

    This is the molecular equivalent of ``mbxas-qe``'s
    ``doubles_overlap`` loop. Elementary ``det(A) K(c,v)`` entries are ranked
    by squared magnitude. Candidate products are admitted through QE's
    decreasing element threshold, but every retained weight is still the
    exact squared 2x2 K minor multiplied by ``|det(A)|**2``.

    ``pair_energy_max`` applies both to each elementary promotion and to the
    sum of the two promotions, matching QE's ``Kener`` and final ``e_range``
    checks. Negative-energy constituents remain eligible.
    """
    if tol <= 0:
        raise ValueError("tol must be positive")
    log = logger if log is None else log
    eps_occ = np.asarray(eps_occ)
    eps_unocc = np.asarray(eps_unocc)
    K = np.asarray(K)
    n_occ, n_unocc = len(eps_occ), len(eps_unocc)
    if K.shape != (n_unocc, n_occ):
        raise ValueError("K shape must match eps_unocc x eps_occ")
    if not _validate_order(2, n_occ, n_unocc):
        return StickSet.empty()

    promotion = eps_unocc[:, None] - eps_occ[None, :]
    candidate = np.ones(K.shape, dtype=bool)
    if pair_energy_max is not None:
        candidate &= promotion <= pair_energy_max
    flat = np.flatnonzero(candidate.ravel())
    if not flat.size:
        return StickSet.empty()

    # Kener stores det(A)*K and ranks |Kener%a|^2. Stable sorting gives a
    # deterministic equivalent of its descending index array for ties.
    score = float(det_weight) * np.abs(K) ** 2
    rank = flat[np.argsort(-score.ravel()[flat], kind="stable")]
    ranked_score = score.ravel()[rank]
    if ranked_score[0] == 0:
        return StickSet.empty()

    next_inner = np.zeros(len(rank), dtype=int)
    energy_chunks, weight_chunks, flag_chunks = [], [], []
    energy_buffer, weight_buffer, flag_buffer = [], [], []

    def flush_buffer():
        if not energy_buffer:
            return
        energy_chunks.append(np.asarray(energy_buffer))
        weight_chunks.append(np.asarray(weight_buffer))
        flag_chunks.append(np.asarray(flag_buffer, dtype=bool))
        energy_buffer.clear()
        weight_buffer.clear()
        flag_buffer.clear()

    series = []
    threshold = tol * _QE_ELEMENT_REDUCTION
    configuration_count = 0
    accumulated = 0.0
    stop_reason = "maximum iterations"

    for iteration in range(1, _QE_MAX_SCREEN_ITERATIONS + 1):
        # energy_sequence_tolerance_index receives squared product bounds and
        # includes the first element below each bound in the Fortran loop.
        outer_count = _qe_tolerance_count(
            ranked_score, threshold / np.sqrt(ranked_score[0]))
        added = 0
        for outer_position in range(outer_count):
            outer_flat = rank[outer_position]
            c, v = np.unravel_index(outer_flat, K.shape)
            outer_score = ranked_score[outer_position]
            if outer_score == 0:
                continue
            inner_count = _qe_tolerance_count(
                ranked_score, threshold / np.sqrt(outer_score))
            start = next_inner[outer_position]
            for inner_position in range(start, inner_count):
                inner_flat = rank[inner_position]
                cp, vp = np.unravel_index(inner_flat, K.shape)
                # QE's unique canonical pairing: v < vp and cp < c.
                if vp <= v or cp >= c:
                    continue
                de_outer = promotion[c, v]
                de_inner = promotion[cp, vp]
                energy = de_outer + de_inner
                if (pair_energy_max is not None
                        and energy > pair_energy_max):
                    continue
                _check_configuration_count(
                    configuration_count + 1, max_configurations,
                    "screened overlap", 2)
                minor = (K[c, v] * K[cp, vp]
                         - K[cp, v] * K[c, vp])
                energy_buffer.append(energy)
                weight = float(det_weight) * abs(minor) ** 2
                weight_buffer.append(weight)
                flag_buffer.append(de_outer < 0 or de_inner < 0)
                configuration_count += 1
                added += 1
                accumulated += weight
                if len(energy_buffer) == _QE_BUILD_BATCH:
                    flush_buffer()
            next_inner[outer_position] = max(
                next_inner[outer_position], inner_count)

        if accumulated > 0:
            series.append(accumulated)
        delta = series[-1] - series[-2] if len(series) > 1 else 0.0
        deriv = (series[-1] - 2 * series[-2] + series[-3]
                 if len(series) > 2 else 0.0)
        curv = (series[-1] - 3 * series[-2] + 3 * series[-3] - series[-4]
                if len(series) > 3 else 0.0)
        log.debug(
            "Spectator doubles screening iteration %d\n%s", iteration,
            format_log_fields({
                "threshold": f"{threshold:.12g}",
                "new configurations": added,
                "configurations": configuration_count,
                "series": f"{series[-1] if series else 0.0:.12g}",
                "delta": f"{delta:.12g}",
                "derivative": f"{deriv:.12g}",
                "derivative x curvature": f"{deriv * curv:.12g}",
            }))
        if _qe_screening_converged(series, tol):
            stop_reason = "converged"
            break
        if (outer_count == len(rank)
                and np.all(next_inner[:outer_count] == len(rank))):
            stop_reason = "all energy-relevant pair products exhausted"
            break
        threshold = _next_qe_overlap_doubles_threshold(
            threshold, configuration_count)
    else:
        log.warning(
            "Spectator doubles screening did not converge\n%s",
            format_log_fields({
                "iterations": _QE_MAX_SCREEN_ITERATIONS,
                "retained configurations": configuration_count,
            }))

    log.info("Spectator doubles screening complete\n%s", format_log_fields({
        "stop reason": stop_reason,
        "iterations": iteration,
        "final threshold": f"{threshold:.12g}",
        "final series": f"{series[-1] if series else 0.0:.12g}",
        "final delta": f"{delta:.12g}",
        "final derivative": f"{deriv:.12g}",
        "final derivative x curvature": f"{deriv * curv:.12g}",
        "retained configurations": configuration_count,
        "energy-relevant K elements": len(rank),
    }))
    flush_buffer()
    return StickSet(
        np.concatenate(energy_chunks) if energy_chunks else np.empty(0),
        np.concatenate(weight_chunks) if weight_chunks else np.empty(0),
        np.concatenate(flag_chunks) if flag_chunks else np.empty(0, dtype=bool))


def shakeup_sticks(AMat, APrimeMat, eps_occ, eps_unocc, order,
                   shakedown_only=False, tol=0.01,
                   max_configurations=2_000_000):
    """Compatibility wrapper returning exact order-k overlap sticks.

    ``tol`` is accepted for API compatibility but exact enumeration does not
    use it. Returned weights now include ``|det(A)|**2``.
    """
    del tol
    sticks = overlap_sticks(
        AMat, APrimeMat, eps_occ, eps_unocc, order,
        max_configurations=max_configurations).selected(shakedown_only)
    return sticks.energy, sticks.weight


def shakeup_sticks_by_order(AMat, APrimeMat, eps_occ, eps_unocc, order=1,
                            tol=0.01, shakedown_only=False,
                            max_configurations=2_000_000):
    """Return exact overlap sticks for orders 1 through ``order``.

    Automatic selection belonged to the removed maxvol-style heuristic, so
    callers must now choose an explicit truncation.
    """
    del tol
    if order == "auto":
        raise ValueError("order='auto' is no longer supported; choose an explicit order")
    max_order = int(order)
    if max_order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    result = {}
    for k in range(1, max_order + 1):
        sticks = overlap_sticks(
            AMat, APrimeMat, eps_occ, eps_unocc, k,
            max_configurations=max_configurations).selected(shakedown_only)
        result[k] = (sticks.energy, sticks.weight)
    return result, list(result)


def shakeup_spectrum(AMat, APrimeMat, eps_occ, eps_unocc, order=1,
                     tol=0.01, shakedown_only=False,
                     max_configurations=2_000_000):
    """Concatenate exact overlap sticks from order 1 through ``order``."""
    per_order, orders = shakeup_sticks_by_order(
        AMat, APrimeMat, eps_occ, eps_unocc, order, tol, shakedown_only,
        max_configurations)
    return (np.concatenate([per_order[k][0] for k in orders]),
            np.concatenate([per_order[k][1] for k in orders]), orders)


def mbxas_sticks_by_order(base_energy, base_amplitude, K, eps_occ, eps_unocc,
                          max_extra_order, shakedown_only=False,
                          max_configurations=2_000_000, screen_tol=None,
                          pair_energy_max=None, final_energy_range=None,
                          determinant=1.0, *, log=None):
    """Construct explicit order-resolved MBXAS transition sticks.

    Dictionary key ``k`` counts extra valence particle-hole pairs: key 0 is
    f1, key 1 is QE's f2/``mb2_spectrum``, and key 2 is f3/``mb3_spectrum``.
    Values are ``(energy, amplitude, shakedown)``. ``base_amplitude`` is the
    determinant-weighted f1 amplitude with shape ``(npol, nvirt)``.
    ``pair_energy_max`` filters each K(c,v) promotion. Independently,
    ``final_energy_range`` filters the resulting photon energy, so a strong
    dipole-final orbital is never removed merely because it lies beyond a
    positional virtual-orbital cutoff.
    """
    base_energy = np.asarray(base_energy)
    base_amplitude = np.asarray(base_amplitude)
    K = np.asarray(K)
    eps_occ = np.asarray(eps_occ)
    eps_unocc = np.asarray(eps_unocc)
    log = logger if log is None else log
    if base_amplitude.ndim != 2 or base_amplitude.shape[1] != len(eps_unocc):
        raise ValueError("base_amplitude must have shape (npol, nvirt)")
    if len(base_energy) != len(eps_unocc) or K.shape != (len(eps_unocc), len(eps_occ)):
        raise ValueError("base energies, K, and orbital-energy dimensions disagree")
    if max_extra_order < 0:
        raise ValueError("max_extra_order must be non-negative")

    result = {0: (base_energy.copy(), base_amplitude.copy(),
                  np.zeros(len(base_energy), dtype=bool))}
    max_possible = min(len(eps_occ), len(eps_unocc) - 1)
    for order in range(1, min(max_extra_order, max_possible) + 1):
        if order == 1 and screen_tol is not None:
            result[order] = _screened_mb2_sticks(
                base_energy, base_amplitude, K, eps_occ, eps_unocc,
                screen_tol, pair_energy_max, final_energy_range,
                determinant, max_configurations, shakedown_only, log)
            continue
        if order == 2 and screen_tol is not None:
            result[order] = _screened_mb3_sticks(
                base_energy, base_amplitude, K, eps_occ, eps_unocc,
                screen_tol, pair_energy_max, final_energy_range,
                determinant, max_configurations, shakedown_only, log)
            continue
        if pair_energy_max is None and final_energy_range is None:
            _check_configuration_count(
                comb(len(eps_occ), order) * comb(len(eps_unocc), order + 1),
                max_configurations, "MBXAS", order)
        energies, amplitudes, flags = [], [], []
        for occ in combinations(range(len(eps_occ)), order):
            for final_virtuals in combinations(range(len(eps_unocc)), order + 1):
                shake_virtuals = tuple(reversed(final_virtuals[:-1]))
                final = final_virtuals[-1]
                constituent = eps_unocc[list(shake_virtuals)] - eps_occ[list(occ)]
                if (pair_energy_max is not None
                        and np.any(constituent > pair_energy_max)):
                    continue
                final_energy = base_energy[final] + constituent.sum()
                if (final_energy_range is not None
                        and not final_energy_range[0] <= final_energy <= final_energy_range[1]):
                    continue
                _check_configuration_count(
                    len(energies) + 1, max_configurations, "MBXAS", order)
                amp = np.zeros(base_amplitude.shape[0], dtype=np.result_type(base_amplitude, K))
                for position, virtual in enumerate(final_virtuals):
                    remaining = final_virtuals[:position] + final_virtuals[position + 1:]
                    minor = np.linalg.det(K[np.ix_(remaining, occ)])
                    amp += (-1) ** (order - position) * base_amplitude[:, virtual] * minor
                # QE writes the shake virtuals in descending band order.
                # Reversing ``order`` minor rows contributes this parity.
                amp *= (-1) ** (order * (order - 1) // 2)
                energies.append(final_energy)
                amplitudes.append(amp)
                flags.append(np.any(constituent < 0))
        energy = np.asarray(energies)
        amplitude = (np.asarray(amplitudes).T if amplitudes
                     else np.empty((base_amplitude.shape[0], 0)))
        flag = np.asarray(flags, dtype=bool)
        if shakedown_only:
            energy, amplitude, flag = energy[flag], amplitude[:, flag], flag[flag]
        result[order] = (energy, amplitude, flag)
    return result


def _screened_mb2_sticks(base_energy, base_amplitude, K, eps_occ,
                         eps_unocc, tol, pair_energy_max,
                         final_energy_range, determinant, max_configurations,
                         shakedown_only, log):
    """QE-style adaptive element screening for one-extra-pair MB2 sticks.

    ``mbxas-qe`` sorts ``det(A) K[c,v]`` by magnitude and begins at an
    element threshold of 0.07. It lowers that threshold geometrically until
    its first spectral buffer fills, then switches to a slower
    configuration-count-dependent reduction. Successive accumulated spectral
    weights are tested by QE's delta/derivative/curvature criteria.
    Energy-window rejection happens before amplitudes are retained.
    """
    if tol <= 0:
        raise ValueError("screen_tol must be positive")
    promotion = eps_unocc[:, None] - eps_occ[None, :]
    score = np.abs(determinant * K) ** 2
    candidate = np.ones(score.shape, dtype=bool)
    if pair_energy_max is not None:
        candidate &= promotion <= pair_energy_max
    flat = np.flatnonzero(candidate.ravel())
    rank = flat[np.argsort(-score.ravel()[flat], kind="stable")]

    energies, amplitudes, flags = [], [], []
    series = []
    threshold = _QE_INITIAL_ELEMENT_TOL * _QE_ELEMENT_REDUCTION
    previous = 0
    configuration_count = 0
    det_weight = float(abs(determinant) ** 2)
    norm = float(np.sum(np.abs(base_amplitude) ** 2))
    if det_weight:
        norm /= det_weight  # QE's A1tot excludes the determinant factor
    accumulated = 0.0
    ranked_score = score.ravel()[rank]
    stop_reason = "maximum iterations"
    for iteration in range(1, _QE_MAX_SCREEN_ITERATIONS + 1):
        count = _qe_tolerance_count(ranked_score, threshold)
        admitted_pairs = count - previous
        for flat_index in rank[previous:count]:
            c, v = np.unravel_index(flat_index, K.shape)
            de = promotion[c, v]
            final = np.arange(c + 1, len(eps_unocc))
            if final_energy_range is not None:
                final_energy = base_energy[final] + de
                final = final[
                    (final_energy >= final_energy_range[0])
                    & (final_energy <= final_energy_range[1])]
            if shakedown_only and de >= 0:
                continue
            if (max_configurations is not None
                    and configuration_count + len(final) > max_configurations):
                _check_configuration_count(
                    configuration_count + len(final), max_configurations,
                    "screened MBXAS", 1)
            if not len(final):
                continue
            configuration_count += len(final)
            for start in range(0, len(final), _QE_BUILD_BATCH):
                final_block = final[start:start + _QE_BUILD_BATCH]
                amp = (base_amplitude[:, final_block] * K[c, v]
                       - base_amplitude[:, c, None]
                       * K[final_block, v][None, :])
                energies.append(base_energy[final_block] + de)
                amplitudes.append(amp)
                flags.append(np.full(len(final_block), de < 0, dtype=bool))
                accumulated += float(np.sum(np.abs(amp) ** 2))
        previous = count
        series.append(accumulated / norm if norm else accumulated)
        delta = series[-1] - series[-2] if len(series) > 1 else 0.0
        deriv = (series[-1] - 2 * series[-2] + series[-3]
                 if len(series) > 2 else 0.0)
        curv = (series[-1] - 3 * series[-2] + 3 * series[-3] - series[-4]
                if len(series) > 3 else 0.0)
        log.debug("MB2 screening iteration %d\n%s", iteration,
                  format_log_fields({
                         "threshold": f"{threshold:.12g}",
                         "new pairs": admitted_pairs,
                         "configurations": configuration_count,
                         "series": f"{series[-1]:.12g}",
                         "delta": f"{delta:.12g}",
                         "derivative": f"{deriv:.12g}",
                         "derivative x curvature": f"{deriv * curv:.12g}",
                     }))
        if _qe_screening_converged(series, tol):
            stop_reason = "converged"
            break
        if count == len(rank):
            stop_reason = "all energy-relevant pairs exhausted"
            break
        threshold = _next_qe_element_threshold(
            threshold, configuration_count)
    else:
        log.warning("MB2 screening did not converge\n%s", format_log_fields({
            "iterations": _QE_MAX_SCREEN_ITERATIONS,
            "examined energy-relevant pairs": f"{previous}/{len(rank)}",
        }))

    log.info("MB2 screening complete\n%s", format_log_fields({
        "stop reason": stop_reason,
        "iterations": iteration,
        "final threshold": f"{threshold:.12g}",
        "final series": f"{series[-1]:.12g}",
        "final delta": f"{delta:.12g}",
        "final derivative": f"{deriv:.12g}",
        "final derivative x curvature": f"{deriv * curv:.12g}",
        "retained configurations": configuration_count,
        "examined energy-relevant pairs": f"{previous}/{len(rank)}",
    }))

    energy = (np.concatenate(energies) if energies
              else np.empty(0, dtype=base_energy.dtype))
    amplitude = (np.concatenate(amplitudes, axis=1) if amplitudes
                 else np.empty((base_amplitude.shape[0], 0)))
    flag = (np.concatenate(flags) if flags else np.empty(0, dtype=bool))
    return energy, amplitude, flag


def _screened_mb3_sticks(base_energy, base_amplitude, K, eps_occ,
                         eps_unocc, tol, pair_energy_max,
                         final_energy_range, determinant, max_configurations,
                         shakedown_only, log):
    """QE-style adaptive product screening for two-extra-pair MB3 sticks.

    This mirrors ``mbxas-qe``'s ``mb3_spectrum`` loop. Candidate elementary
    promotions are ranked by ``|det(A) K(c,v)|**2``. Products are admitted
    through QE's decreasing threshold, while every retained transition uses
    the exact three-term MB3 amplitude.
    """
    if tol <= 0:
        raise ValueError("screen_tol must be positive")
    promotion = eps_unocc[:, None] - eps_occ[None, :]
    det_weight = float(abs(determinant) ** 2)
    score = det_weight * np.abs(K) ** 2
    candidate = np.ones(score.shape, dtype=bool)
    if pair_energy_max is not None:
        candidate &= promotion <= pair_energy_max
    flat = np.flatnonzero(candidate.ravel())
    if not flat.size:
        return (np.empty(0, dtype=base_energy.dtype),
                np.empty((base_amplitude.shape[0], 0)),
                np.empty(0, dtype=bool))
    rank = flat[np.argsort(-score.ravel()[flat], kind="stable")]
    ranked_score = score.ravel()[rank]
    if ranked_score[0] == 0:
        return (np.empty(0, dtype=base_energy.dtype),
                np.empty((base_amplitude.shape[0], 0)),
                np.empty(0, dtype=bool))

    next_inner = np.zeros(len(rank), dtype=int)
    energies, amplitudes, flags = [], [], []
    series = []
    threshold = _QE_INITIAL_ELEMENT_TOL * _QE_ELEMENT_REDUCTION
    configuration_count = 0
    accumulated = 0.0
    norm = float(np.sum(np.abs(base_amplitude) ** 2))
    if det_weight:
        norm /= det_weight
    stop_reason = "maximum iterations"

    for iteration in range(1, _QE_MAX_SCREEN_ITERATIONS + 1):
        outer_count = _qe_tolerance_count(
            ranked_score, threshold / np.sqrt(ranked_score[0]))
        admitted_pairs = 0
        added_configurations = 0
        for outer_position in range(outer_count):
            outer_flat = rank[outer_position]
            c, v = np.unravel_index(outer_flat, K.shape)
            outer_score = ranked_score[outer_position]
            if outer_score == 0:
                continue
            inner_count = _qe_tolerance_count(
                ranked_score, threshold / np.sqrt(outer_score))
            start = next_inner[outer_position]
            for inner_position in range(start, inner_count):
                inner_flat = rank[inner_position]
                cp, vp = np.unravel_index(inner_flat, K.shape)
                # QE's unique MB3 ordering: v < vp and cp < c < f.
                if vp <= v or cp >= c:
                    continue
                de_outer = promotion[c, v]
                de_inner = promotion[cp, vp]
                if shakedown_only and de_outer >= 0 and de_inner >= 0:
                    continue
                final = np.arange(c + 1, len(eps_unocc))
                final_energy = (
                    base_energy[final] + de_outer + de_inner)
                if final_energy_range is not None:
                    final = final[
                        (final_energy >= final_energy_range[0])
                        & (final_energy <= final_energy_range[1])]
                    final_energy = (
                        base_energy[final] + de_outer + de_inner)
                if not len(final):
                    continue
                if (max_configurations is not None
                        and configuration_count + len(final)
                        > max_configurations):
                    _check_configuration_count(
                        configuration_count + len(final),
                        max_configurations, "screened MBXAS", 2)
                configuration_count += len(final)
                added_configurations += len(final)
                admitted_pairs += 1

                pair_minor = (
                    K[c, v] * K[cp, vp]
                    - K[cp, v] * K[c, vp])
                for block_start in range(0, len(final), _QE_BUILD_BATCH):
                    f = final[block_start:block_start + _QE_BUILD_BATCH]
                    energy_block = (
                        base_energy[f] + de_outer + de_inner)
                    c_minor = (
                        K[f, v] * K[cp, vp]
                        - K[cp, v] * K[f, vp])
                    cp_minor = (
                        K[f, v] * K[c, vp]
                        - K[c, v] * K[f, vp])
                    amplitude_block = (
                        base_amplitude[:, f] * pair_minor
                        - base_amplitude[:, c, None] * c_minor[None, :]
                        + base_amplitude[:, cp, None] * cp_minor[None, :])
                    energies.append(energy_block)
                    amplitudes.append(amplitude_block)
                    flags.append(np.full(
                        len(f), de_outer < 0 or de_inner < 0,
                        dtype=bool))
                    accumulated += float(
                        np.sum(np.abs(amplitude_block) ** 2))
            next_inner[outer_position] = max(
                next_inner[outer_position], inner_count)

        series.append(accumulated / norm if norm else accumulated)
        delta = series[-1] - series[-2] if len(series) > 1 else 0.0
        deriv = (series[-1] - 2 * series[-2] + series[-3]
                 if len(series) > 2 else 0.0)
        curv = (series[-1] - 3 * series[-2] + 3 * series[-3] - series[-4]
                if len(series) > 3 else 0.0)
        log.debug("MB3 screening iteration %d\n%s", iteration,
                  format_log_fields({
                         "threshold": f"{threshold:.12g}",
                         "new pairs": admitted_pairs,
                         "new configurations": added_configurations,
                         "configurations": configuration_count,
                         "series": f"{series[-1]:.12g}",
                         "delta": f"{delta:.12g}",
                         "derivative": f"{deriv:.12g}",
                         "derivative x curvature": f"{deriv * curv:.12g}",
                     }))
        if _qe_screening_converged(series, tol):
            stop_reason = "converged"
            break
        if (outer_count == len(rank)
                and np.all(next_inner[:outer_count] == len(rank))):
            stop_reason = "all energy-relevant pair products exhausted"
            break
        threshold = _next_qe_element_threshold(
            threshold, configuration_count)
    else:
        log.warning("MB3 screening did not converge\n%s", format_log_fields({
            "iterations": _QE_MAX_SCREEN_ITERATIONS,
            "retained configurations": configuration_count,
        }))

    log.info("MB3 screening complete\n%s", format_log_fields({
        "stop reason": stop_reason,
        "iterations": iteration,
        "final threshold": f"{threshold:.12g}",
        "final series": f"{series[-1]:.12g}",
        "final delta": f"{delta:.12g}",
        "final derivative": f"{deriv:.12g}",
        "final derivative x curvature": f"{deriv * curv:.12g}",
        "retained configurations": configuration_count,
        "energy-relevant K elements": len(rank),
    }))
    energy = (np.concatenate(energies) if energies
              else np.empty(0, dtype=base_energy.dtype))
    amplitude = (np.concatenate(amplitudes, axis=1) if amplitudes
                 else np.empty((base_amplitude.shape[0], 0)))
    flag = (np.concatenate(flags) if flags else np.empty(0, dtype=bool))
    return energy, amplitude, flag


def _prune_outer_product(e_i, w_i, e_j, w_j, tol):
    """Outer sum/product retaining at least ``1-tol`` of exact block mass."""
    if len(w_i) == 0 or len(w_j) == 0:
        return np.empty(0), np.empty(0)
    total_mass = w_i.sum() * w_j.sum()
    if total_mass == 0:
        return np.empty(0), np.empty(0)
    ri, rj = np.argsort(-w_i), np.argsort(-w_j)
    e_i, w_i, e_j, w_j = e_i[ri], w_i[ri], e_j[rj], w_j[rj]
    ni, nj = len(w_i), len(w_j)
    mi, mj = min(64, ni), min(64, nj)
    while True:
        e = (e_i[:mi, None] + e_j[None, :mj]).ravel()
        w = (w_i[:mi, None] * w_j[None, :mj]).ravel()
        if (mi == ni and mj == nj) or w.sum() >= (1 - tol) * total_mass:
            return e, w
        mi, mj = min(4 * mi, ni), min(4 * mj, nj)


def combine_cross_channel_sticks(sticks_a_by_order, sticks_b_by_order,
                                 max_total_order, tol=0.01):
    """Compatibility helper combining two relative overlap-stick mappings."""
    trivial = (np.array([0.0]), np.array([1.0]))
    all_e, all_w = [], []
    for i in [0] + sorted(sticks_a_by_order):
        for j in [0] + sorted(sticks_b_by_order):
            if (i == 0 and j == 0) or i + j > max_total_order:
                continue
            e_i, w_i = trivial if i == 0 else sticks_a_by_order[i]
            e_j, w_j = trivial if j == 0 else sticks_b_by_order[j]
            e, w = _prune_outer_product(e_i, w_i, e_j, w_j, tol)
            all_e.append(e)
            all_w.append(w)
    if not all_e:
        return np.empty(0), np.empty(0)
    return np.concatenate(all_e), np.concatenate(all_w)


def broaden_shakeup(delta_e, weight, egrid, sigma, reference_weight=1.0):
    """Broaden overlap sticks, including a zero-order reference stick."""
    from pymbxas.mbxas.broaden import broadened_spectrum
    return broadened_spectrum(egrid, np.concatenate([[0.0], delta_e]),
                              np.concatenate([[reference_weight], weight]), sigma)


def convolve_shakeup(egrid, main_intensity, delta_e, weight, sigma,
                     reference_weight=1.0):
    """Convolve with an unnormalized determinant-overlap kernel.

    The production path uses explicit MBXAS amplitudes and broadens final
    sticks once; this low-level helper remains for compatibility.
    """
    from pymbxas.mbxas.broaden import broadened_spectrum
    de = egrid[1] - egrid[0]
    extent = np.abs(delta_e).max() if len(delta_e) else 0.0
    n_half = int(np.ceil((extent + 5 * sigma) / de))
    kgrid = np.arange(-n_half, n_half + 1) * de
    kernel = np.zeros_like(kgrid)
    kernel[n_half] = reference_weight / de
    if len(delta_e):
        kernel += broadened_spectrum(kgrid, delta_e, weight, sigma)

    def apply(y):
        full = np.convolve(y, kernel, mode="full") * de
        return full[n_half:n_half + len(y)]
    main_intensity = np.asarray(main_intensity)
    return apply(main_intensity) if main_intensity.ndim == 1 else np.array([apply(y) for y in main_intensity])
