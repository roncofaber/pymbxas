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


def shakeup_sticks(K, eps_occ, eps_unocc, order, shakedown_only=False, tol=0.01):
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
    tol: only used for order == 2, see _shakeup_sticks_order2.

    Returns (delta_e, weight): flat 1D arrays, one entry per combination of
    `order` valence orbitals promoted to `order` conduction orbitals.
    weight = |det(K[c_combo, v_combo])|**2. For order == 2 this is not
    every combination -- see _shakeup_sticks_order2.
    """
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    if order > MAX_IMPLEMENTED_ORDER:
        raise NotImplementedError(
            f"shake-up order {order} is not implemented: the number of "
            f"{order}-fold valence/conduction combinations grows as "
            f"O(n_occ**{order} * n_unocc**{order}), and nothing here prunes "
            "that combinatorics the way mbxas-qe's adaptive-tolerance "
            "triples_overlap does. Implemented orders: 1-"
            f"{MAX_IMPLEMENTED_ORDER}."
        )

    n_occ = len(eps_occ)
    n_unocc = len(eps_unocc)
    if order > n_occ or order > n_unocc:
        return np.empty(0), np.empty(0)

    if order == 2:
        delta_e, weight = _shakeup_sticks_order2(K, eps_occ, eps_unocc, tol)
        if shakedown_only:
            mask = delta_e < 0
            delta_e, weight = delta_e[mask], weight[mask]
        return delta_e, weight

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


def _shakeup_sticks_order2(K, eps_occ, eps_unocc, tol):
    """Order-2 valence shake-up sticks, pruned by |K|**2 magnitude.

    The brute-force route (all C(n_occ, 2) valence pairs times all
    C(n_unocc, 2) conduction pairs) enumerates every 2x2 minor of K
    regardless of size, which is what actually explodes for real systems
    (147M sticks for a modest active space) -- most valence electrons
    barely relax when a core hole forms, so almost all of those minors are
    negligible. This mirrors what mbxas-qe's doubles_overlap
    (SHIRLEY/src/mbxas_spectra.f90) does instead: treat every single
    valence->conduction transition (v, c) as one candidate ranked by
    |K(c, v)|**2, and only form 2x2 minors between candidates that are
    both large. Concretely: sort all n_occ*n_unocc singles by |K|**2
    descending, then iteratively grow an "active" prefix of that list
    (geometric growth stands in for QE's shrinking-tolerance schedule --
    both just mean "look at progressively smaller matrix elements"),
    forming all valid pairs (distinct v, distinct c) within the active set
    each round, until the accumulated order-2 weight stops changing by
    more than tol relative to the order-1 mass -- the same mass-based
    convergence convention shakeup_sticks_by_order's "auto" order already
    uses. If K has no sparsity at all, the active set grows to cover every
    singles pair and this costs the same as brute force; that is a
    property of K, not something pruning can fix.
    """
    n_occ = len(eps_occ)
    n_unocc = len(eps_unocc)
    n_singles = n_occ * n_unocc
    if n_occ < 2 or n_unocc < 2:
        return np.empty(0), np.empty(0)

    c_idx = np.repeat(np.arange(n_unocc), n_occ)
    v_idx = np.tile(np.arange(n_occ), n_unocc)
    k1 = K.ravel()
    importance = np.abs(k1) ** 2
    mass1 = importance.sum()
    if mass1 == 0:
        return np.empty(0), np.empty(0)
    delta_e1 = eps_unocc[c_idx] - eps_occ[v_idx]

    rank = np.argsort(-importance)
    v_s, c_s, k_s, de_s = v_idx[rank], c_idx[rank], k1[rank], delta_e1[rank]

    delta_e = weight = np.empty(0)
    prev_mass = 0.0
    m = min(64, n_singles)
    while True:
        v_a, c_a, k_a, de_a = v_s[:m], c_s[:m], k_s[:m], de_s[:m]

        # Each {v,v'}x{c,c'} quartet appears twice among distinct-v,
        # distinct-c singles pairs -- once as (v,c)&(v',c'), once as
        # (v,c')&(v',c) -- and both give the same |2x2 minor|**2. Keep
        # only the "concordant" matching (smaller v paired with smaller c)
        # to count each combo once.
        i, j = np.triu_indices(m, k=1)
        keep = (v_a[i] != v_a[j]) & (c_a[i] != c_a[j]) \
            & ((v_a[i].astype(np.int64) - v_a[j]) * (c_a[i].astype(np.int64) - c_a[j]) > 0)
        i, j = i[keep], j[keep]

        diag = k_a[i] * k_a[j]
        cross = K[c_a[j], v_a[i]] * K[c_a[i], v_a[j]]
        weight = np.abs(diag - cross) ** 2
        delta_e = de_a[i] + de_a[j]

        mass = weight.sum()
        converged = m >= n_singles or (mass > 0 and abs(mass - prev_mass) < tol * max(mass1, mass))
        logger.log(TRACE,
            "shake-up order-2 pruning: active singles=%d/%d pairs=%d "
            "mass=%.6e (order-1 mass=%.6e, tol=%.3g) -> %s",
            m, n_singles, len(weight), mass, mass1, tol,
            "converged" if converged else "widening")
        if converged:
            break
        prev_mass = mass
        m = min(m * 4, n_singles)

    if m > 256:
        logger.warning(
            "shake-up order-2 pruning needed an active set of %d/%d "
            "singles to converge (tol=%.3g) -- K is not very sparse here, "
            "so this is close to the brute-force cost.", m, n_singles, tol)

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
    e1, w1 = shakeup_sticks(K, eps_occ, eps_unocc, 1, shakedown_only=shakedown_only, tol=tol)
    mass1 = w1.sum()
    sticks_by_order = {1: (e1, w1)}
    orders_included = [1]

    if order == "auto":
        for k in range(2, MAX_IMPLEMENTED_ORDER + 1):
            ek, wk = shakeup_sticks(K, eps_occ, eps_unocc, k, shakedown_only=shakedown_only, tol=tol)
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
        ek, wk = shakeup_sticks(K, eps_occ, eps_unocc, k, shakedown_only=shakedown_only, tol=tol)
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


def _prune_outer_product(e_i, w_i, e_j, w_j, tol):
    """Weight-pruned outer sum/product of two independent stick lists.

    A plain outer join (e_i[:, None] + e_j[None, :], w_i[:, None] * w_j[None, :])
    is what actually overflows memory once both channels' order-1 stick
    counts (each just n_occ*n_unocc, not itself a combinatorial blow-up)
    reach the tens of thousands -- their cross product alone lands in the
    hundreds of millions. Unlike _shakeup_sticks_order2, here the two
    lists are independent (no shared-index double counting), and the true
    total mass sum(w_i)*sum(w_j) is known exactly without doing the join.
    So: sort each list by weight descending, grow active prefixes of both
    geometrically, and stop as soon as the join restricted to those
    prefixes has captured a (1 - tol) fraction of the exact total -- every
    excluded product is <= the smallest included one, so this can't
    over-converge.
    """
    if len(w_i) == 0 or len(w_j) == 0:
        return np.empty(0), np.empty(0)
    total_mass = w_i.sum() * w_j.sum()
    if total_mass == 0:
        return np.empty(0), np.empty(0)

    ri, rj = np.argsort(-w_i), np.argsort(-w_j)
    e_i_s, w_i_s = e_i[ri], w_i[ri]
    e_j_s, w_j_s = e_j[rj], w_j[rj]
    ni, nj = len(w_i), len(w_j)

    mi, mj = min(64, ni), min(64, nj)
    while True:
        e_sub = (e_i_s[:mi, None] + e_j_s[None, :mj]).ravel()
        w_sub = (w_i_s[:mi, None] * w_j_s[None, :mj]).ravel()
        captured = w_sub.sum()
        if (mi >= ni and mj >= nj) or captured >= (1 - tol) * total_mass:
            break
        mi, mj = min(mi * 4, ni), min(mj * 4, nj)
    return e_sub, w_sub


def combine_cross_channel_sticks(sticks_a_by_order, sticks_b_by_order, max_total_order, tol=0.01):
    """Cross-channel shake-up combination.

    Physically, the excited channel's own valence relaxation and the
    spectator channel's own valence relaxation are treated as independent
    processes (sudden-approximation factorization, mbxas-qe's
    spin_convolve_spectrum in spec.f90): the joint probability of an
    order-i excited-channel combination together with an order-j
    spectator-channel combination is the product of the two probabilities,
    and the joint electron-hole energy cost is their sum. In stick form
    that is exactly the outer product of weights and outer sum of energies:

        E_ij = e_i[:, None] + e_j[None, :]
        W_ij = w_i[:, None] * w_j[None, :]

    sticks_a_by_order, sticks_b_by_order: {order: (delta_e, weight)} as
    returned by shakeup_sticks_by_order. Order 0 ("no extra excitation in
    this channel") is implicit -- a trivial (delta_e=0, weight=1) stick --
    and does not need to be a key in either dict.

    max_total_order: only (i, j) pairs with i + j <= max_total_order
    contribute. Includes the "pure" terms (i, 0) and (0, j), which reduce
    to that channel's own sticks unchanged (outer sum/product with the
    trivial stick is the identity) -- so if sticks_b_by_order is empty,
    the result is exactly the concatenation of sticks_a_by_order's entries,
    i.e. plain single-channel shake-up.

    tol: each (i, j) block's outer join is pruned by weight magnitude
    (_prune_outer_product) rather than formed in full -- a single order-1
    x order-1 join between two channels is already n_occ*n_unocc large on
    each side, so a plain outer product is the thing that actually
    exhausts memory. Kept to a (1 - tol) fraction of that block's exact
    mass, same convention as shakeup_sticks_by_order's tol.

    Returns (delta_e, weight): concatenated sticks for every included
    (i, j) pair except the trivial (0, 0) "no shake-up anywhere" term --
    broaden_shakeup/convolve_shakeup already add that term themselves.
    """
    trivial_e, trivial_w = np.array([0.0]), np.array([1.0])

    def _get(sticks_by_order, k):
        return sticks_by_order[k] if k else (trivial_e, trivial_w)

    orders_a = [0] + sorted(sticks_a_by_order)
    orders_b = [0] + sorted(sticks_b_by_order)

    all_e, all_w = [], []
    for i in orders_a:
        for j in orders_b:
            if i == 0 and j == 0:
                continue
            if i + j > max_total_order:
                continue
            e_i, w_i = _get(sticks_a_by_order, i)
            e_j, w_j = _get(sticks_b_by_order, j)
            e_ij, w_ij = _prune_outer_product(e_i, w_i, e_j, w_j, tol)
            all_e.append(e_ij)
            all_w.append(w_ij)

    if not all_e:
        return np.empty(0), np.empty(0)
    return np.concatenate(all_e), np.concatenate(all_w)


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

    # A stick whose |delta_e| exceeds the main spectrum's own span plus a
    # broadening margin shifts the entire main spectrum off this egrid, so
    # it cannot contribute anything above Gaussian-tail precision to the
    # sliced output below -- drop it before sizing the kernel. Without this,
    # a spectator-channel or cross-channel combination reaching into the
    # diffuse, no-core-hole virtual manifold (delta_e up to hundreds of eV)
    # blows up n_half and the dense (n_kgrid, n_sticks) broadcast inside
    # broadened_spectrum to tens of GB for a system with a handful of orbitals.
    relevant_bound = (egrid[-1] - egrid[0]) + 5 * sigma
    if len(delta_e):
        relevant = np.abs(delta_e) <= relevant_bound
        delta_e, weight = delta_e[relevant], weight[relevant]

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
