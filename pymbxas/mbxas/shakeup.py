#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Order-k valence shake-up satellite spectra: the f^(n) terms of Eq. 32-35,
PRB 107, 035146, beyond the one-body truncation `run_MBXAS_pyscf` keeps.

The weight for a k-fold simultaneous valence-to-conduction excitation is
the k x k minor determinant of the same K = A' @ inv(A) matrix used for the
ordinary n=1 amplitude (mbxas.mbxas.build_A_K) -- this is the exact
generalization of the matrix-determinant-lemma identity K already relies
on, not a separate approximation. See dev/method.md and
docs/superpowers/specs/2026-08-21-shakeup-satellites-design.md.
"""

import itertools
import logging

import numpy as np

from pymbxas.io.config import TRACE

MAX_IMPLEMENTED_ORDER = 2

logger = logging.getLogger(__name__)


def shakeup_sticks(K, eps_occ, eps_unocc, order, shakedown_only=False):
    """Order-k valence shake-up stick spectrum.

    K: (n_unocc, n_occ) matrix for one spin channel (mbxas.mbxas.build_A_K).
    eps_occ: (n_occ,) orbital energies of the valence manifold indexing K's columns.
    eps_unocc: (n_unocc,) orbital energies of the conduction manifold indexing K's rows.
    order: number of simultaneous valence -> conduction excitations.
    shakedown_only: if True, keep only combinations whose electron-hole
        energy delta_e is negative -- mbxas-qe's "shakedown" case
        (kpoint_spectral_details.f90: shakedown = any(de < 0)). A
        diagnostic isolation of the sign-anomalous combinations, not a
        different formula.

    Returns (delta_e, weight): flat 1D arrays, one entry per combination of
    `order` valence orbitals promoted to `order` conduction orbitals.
    weight = |det(K[c_combo, v_combo])|**2.
    """
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    if order > MAX_IMPLEMENTED_ORDER:
        raise NotImplementedError(
            f"shake-up order {order} is not implemented: the number of "
            f"{order}-fold valence/conduction combinations grows as "
            f"O(n_occ**{order} * n_unocc**{order}), and nothing here prunes "
            "that combinatorics the way mbxas-qe's adaptive-tolerance "
            "doubles_overlap/triples_overlap do. Implemented orders: 1-"
            f"{MAX_IMPLEMENTED_ORDER}."
        )

    n_occ = len(eps_occ)
    n_unocc = len(eps_unocc)
    if order > n_occ or order > n_unocc:
        return np.empty(0), np.empty(0)

    v_combos = np.array(list(itertools.combinations(range(n_occ), order)))
    c_combos = np.array(list(itertools.combinations(range(n_unocc), order)))

    # K from build_A_K has shape (n_unocc, n_occ); index with c_combos on rows, v_combos on columns
    # sub[i, j] is the (order, order) submatrix of K for valence combo i,
    # conduction combo j; numpy.linalg.det batches over leading dimensions
    sub = K[c_combos[None, :, :, None], v_combos[:, None, None, :]]
    weight = np.abs(np.linalg.det(sub)) ** 2  # (n_v_combos, n_c_combos)

    delta_e = (eps_unocc[c_combos].sum(axis=1)[None, :]
               - eps_occ[v_combos].sum(axis=1)[:, None])  # (n_v_combos, n_c_combos)

    delta_e = delta_e.ravel()
    weight = weight.ravel()
    if shakedown_only:
        mask = delta_e < 0
        delta_e, weight = delta_e[mask], weight[mask]
    return delta_e, weight


def shakeup_sticks_by_order(K, eps_occ, eps_unocc, order="auto", tol=0.01, shakedown_only=False):
    """Per-order valence shake-up sticks, order 1 up to the requested order.

    Same order/tol/shakedown_only semantics as shakeup_spectrum. Returns
    (sticks_by_order, orders_included): sticks_by_order is
    {order: (delta_e, weight)} for each order actually included -- the
    per-order breakdown mbxas.shakeup.combine_cross_channel_sticks needs;
    shakeup_spectrum concatenates this into its flat (delta_e, weight)
    contract for callers that don't need the breakdown.
    """
    e1, w1 = shakeup_sticks(K, eps_occ, eps_unocc, 1, shakedown_only=shakedown_only)
    mass1 = w1.sum()
    sticks_by_order = {1: (e1, w1)}
    orders_included = [1]

    if order == "auto":
        for k in range(2, MAX_IMPLEMENTED_ORDER + 1):
            ek, wk = shakeup_sticks(K, eps_occ, eps_unocc, k, shakedown_only=shakedown_only)
            massk = wk.sum()
            include = mass1 > 0 and massk > tol * mass1
            logger.log(TRACE,
                "shake-up auto-order: order %d mass=%.6e (order-1 mass=%.6e, "
                "tol=%.3g) -> %s", k, massk, mass1, tol,
                "included" if include else "stopped")
            if not include:
                break
            sticks_by_order[k] = (ek, wk)
            orders_included.append(k)
        return sticks_by_order, orders_included

    order = int(order)
    if order < 1:
        raise ValueError(f"order must be >= 1 or 'auto', got {order}")
    for k in range(2, order + 1):
        ek, wk = shakeup_sticks(K, eps_occ, eps_unocc, k, shakedown_only=shakedown_only)
        sticks_by_order[k] = (ek, wk)
        orders_included.append(k)
    return sticks_by_order, orders_included


def shakeup_spectrum(K, eps_occ, eps_unocc, order="auto", tol=0.01, shakedown_only=False):
    """Combined shake-up stick spectrum up to the requested order.

    order: "auto" adds each order from 2 up to MAX_IMPLEMENTED_ORDER only
        while its total weight exceeds tol * (order-1 total weight),
        stopping at the first order that fails -- so adding a higher
        MAX_IMPLEMENTED_ORDER later extends "auto" automatically, no
        separate rewrite needed here. An explicit int always includes
        every order from 1 up to and including that int, no tolerance check.
    shakedown_only: see shakeup_sticks.

    Returns (delta_e, weight, orders_included): concatenated sticks across
    all included orders, plus the sorted list of orders actually included.
    Delegates the per-order construction to shakeup_sticks_by_order.
    """
    sticks_by_order, orders_included = shakeup_sticks_by_order(
        K, eps_occ, eps_unocc, order=order, tol=tol, shakedown_only=shakedown_only)
    all_e = [sticks_by_order[k][0] for k in orders_included]
    all_w = [sticks_by_order[k][1] for k in orders_included]
    return np.concatenate(all_e), np.concatenate(all_w), orders_included


def broaden_shakeup(delta_e, weight, egrid, sigma):
    """Broaden shake-up sticks onto egrid, including the implicit n=0
    (no extra shake-up) term at delta_e=0 with unit weight -- this is what
    makes the result usable directly as a convolution kernel."""
    from pymbxas.mbxas.broaden import broadened_spectrum

    all_e = np.concatenate([[0.0], delta_e])
    all_w = np.concatenate([[1.0], weight])
    return broadened_spectrum(egrid, all_e, all_w, sigma)


def convolve_shakeup(egrid, main_intensity, delta_e, weight, sigma):
    """Convolve a broadened main spectrum with the shake-up probability
    kernel: an exact delta at delta_e=0 for the n=0 ("no extra shake-up")
    term, plus the supplied (delta_e, weight) sticks Gaussian-broadened by
    sigma. The n=0 term is deliberately NOT broadened here -- the main
    spectrum this convolves against has already been broadened by sigma
    once upstream, so re-broadening it here would double the effective
    width of every peak that has no shake-up satellites.

    egrid: (npoints,) uniform grid main_intensity is defined on.
    main_intensity: (npoints,) or (naxes, npoints).
    Returns an array the same shape as main_intensity, on the same egrid.
    """
    from pymbxas.mbxas.broaden import broadened_spectrum

    de = egrid[1] - egrid[0]

    stick_extent = np.abs(delta_e).max() if len(delta_e) else 0.0
    half_width = stick_extent + 5 * sigma
    n_half = int(np.ceil(half_width / de))
    kgrid = np.arange(-n_half, n_half + 1) * de  # guaranteed symmetric, kgrid[n_half] == 0.0 exactly

    kernel = np.zeros_like(kgrid)
    kernel[n_half] = 1.0 / de  # exact delta at delta_e=0, unbroadened
    if len(delta_e):
        kernel = kernel + broadened_spectrum(kgrid, delta_e, weight, sigma)

    kernel = kernel / (kernel.sum() * de)  # normalize to unit probability

    def _convolve_1d(y):
        full = np.convolve(y, kernel, mode="full") * de
        return full[n_half:n_half + len(y)]

    main_intensity = np.asarray(main_intensity)
    if main_intensity.ndim == 1:
        return _convolve_1d(main_intensity)
    return np.array([_convolve_1d(row) for row in main_intensity])
