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

import logging

import numpy as np

from pymbxas.io.config import TRACE
from pymbxas.mbxas.maxvol import sherman_morrison_row_update

logger = logging.getLogger(__name__)


def shakeup_sticks(AMat, APrimeMat, eps_occ, eps_unocc, order, shakedown_only=False, tol=0.01):
    """Order-k valence shake-up stick spectrum.

    AMat: (n_occ, n_occ) valence overlap matrix (mbxas.mbxas.build_A_K).
    APrimeMat: (n_unocc, n_occ) unoccupied-valence overlap matrix
        (mbxas.mbxas.build_A_K) -- rows are candidate virtual orbitals,
        columns match AMat's columns.
    eps_occ: (n_occ,) orbital energies of the valence manifold.
    eps_unocc: (n_unocc,) orbital energies of the conduction manifold.
    order: number of simultaneous valence -> conduction excitations.
    shakedown_only: if True, keep only combinations whose electron-hole
        energy delta_e is negative -- mbxas-qe's "shakedown" case
        (kpoint_spectral_details.f90: shakedown = any(de < 0)). A
        diagnostic isolation of the sign-anomalous combinations, not a
        different formula.
    tol: passed to the order>=2 maxvol search (_maxvol_shakeup_configs);
        unused for order == 1, which is always exact.

    Returns (delta_e, weight): flat 1D arrays. weight =
    |det(K[c_combo, v_combo])|**2 for whatever combinations the order==1
    exact enumeration, or the order>=2 maxvol search, actually finds.
    """
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")

    n_occ = len(eps_occ)
    n_unocc = len(eps_unocc)
    if order > n_occ or order > n_unocc:
        return np.empty(0), np.empty(0)

    K = APrimeMat @ np.linalg.inv(AMat)

    if order == 1:
        delta_e, weight = _order1_sticks(K, eps_occ, eps_unocc)
    else:
        configs = _maxvol_shakeup_configs(AMat, APrimeMat, K, eps_occ, eps_unocc, tol, min_order=order)
        if order not in configs:
            return np.empty(0), np.empty(0)
        delta_e, weight = configs[order]

    if shakedown_only:
        mask = delta_e < 0
        delta_e, weight = delta_e[mask], weight[mask]
    return delta_e, weight


def _order1_sticks(K, eps_occ, eps_unocc):
    """Exact order-1 shake-up sticks: every valence->conduction single,
    no pruning (n_occ*n_unocc is always cheap). Shared by shakeup_sticks
    and shakeup_sticks_by_order so the formula lives in one place."""
    delta_e = (eps_unocc[:, None] - eps_occ[None, :]).ravel()
    weight = (np.abs(K) ** 2).ravel()
    return delta_e, weight


def _maxvol_shakeup_configs(AMat, APrimeMat, K, eps_occ, eps_unocc, tol, min_order=None):
    """Order>=2 shake-up configurations via a maxvol-style swap search.

    Order 1 is exact and handled elsewhere (shakeup_sticks_by_order):
    a plain enumeration of all n_occ*n_unocc singles is already cheap and
    complete, so there is nothing to search for at order 1. This searches
    order 2 and beyond, where brute-force enumeration is what actually
    explodes -- see docs/superpowers/specs/2026-08-24-maxvol-shakeup-search-design.md.

    A configuration is a set of k occupied valence orbitals ("cols_out")
    simultaneously swapped for k virtual orbitals ("rows_in"). Its weight
    is |det(K[rows_in, cols_out])|**2 -- the Jacobi complementary-minor
    identity that makes any k-swap weight a plain minor of K (the same
    formula shakeup_sticks already used for order 2, no longer hardcoded
    to k=2).

    Search strategy, mirroring mbxas-qe's maxvol_multi_mod.f90: seed from
    the top-ranked order-1 singles (already known via |K|**2, no cost to
    reuse); from each seed, use the Sherman-Morrison-updated pivot inverse
    to find the best next swap (the B-matrix candidate step); extend
    breadth-first order by order; stop when a new order's total captured
    weight drops below tol * order-1 mass (mass1) -- the same convergence
    convention shakeup_sticks_by_order's "auto" mode already uses.

    min_order: if given, keep searching past the tol-based stop until at
    least this order has been attempted (or the search runs out of
    candidates), matching shakeup_sticks_by_order's "explicit order never
    silently downgrades" contract. Ignored (tol-based stopping always
    applies) when None.

    Returns {order: (delta_e, weight)} for order >= 2, empty if nothing
    found or nothing exceeds tol.
    """
    n_occ = len(eps_occ)
    n_unocc = len(eps_unocc)
    if n_occ < 2 or n_unocc < 2:
        return {}

    importance = np.abs(K) ** 2  # (n_unocc, n_occ), same as order-1 weights
    mass1 = importance.sum()
    if mass1 == 0:
        return {}

    n_seeds = min(64, n_unocc * n_occ)
    flat_rank = np.argsort(-importance, axis=None)[:n_seeds]
    seed_c, seed_v = np.unravel_index(flat_rank, importance.shape)

    A_inv0 = np.linalg.inv(AMat)

    # Each active branch: (cols_out, rows_in, A_pivot, A_inv). cols_out are
    # valence-slot indices (K's column axis) already swapped out, rows_in
    # are candidate unocc indices (K's row axis) already swapped in --
    # both tuples, kept sorted, order == len(cols_out) == len(rows_in).
    active = []
    for c0, v0 in zip(seed_c.tolist(), seed_v.tolist()):
        try:
            A_pivot1, A_inv1 = sherman_morrison_row_update(AMat, A_inv0, v0, APrimeMat[c0])
            active.append(((v0,), (c0,), A_pivot1, A_inv1))
        except np.linalg.LinAlgError:
            # Candidate rows are linearly dependent; heuristic search continues with the next seed.
            logger.log(TRACE, "seed candidate (c=%d, v=%d) singular, skipping", c0, v0)
            continue

    result = {}
    seen = set()
    order = 2
    max_order = min(n_occ, n_unocc)
    while active and order <= max_order:
        found = {}  # (cols_out, rows_in) -> (A_pivot, A_inv)
        for cols_out, rows_in, A_pivot, A_inv in active:
            avail_slots = [q for q in range(n_occ) if q not in cols_out]
            avail_c = [c for c in range(n_unocc) if c not in rows_in]
            if not avail_slots or not avail_c:
                continue
            B = APrimeMat[avail_c] @ A_inv  # (n_avail_c, n_occ)
            B_masked = B[:, avail_slots]
            flat = np.argmax(np.abs(B_masked))
            cand_pos, slot_pos = np.unravel_index(flat, B_masked.shape)
            b_val = B_masked[cand_pos, slot_pos]
            if abs(b_val) <= 1.0 + tol:
                continue
            new_c = avail_c[cand_pos]
            new_v = avail_slots[slot_pos]
            key = (tuple(sorted(cols_out + (new_v,))), tuple(sorted(rows_in + (new_c,))))
            if key in seen or key in found:
                continue
            try:
                A_pivot_new, A_inv_new = sherman_morrison_row_update(A_pivot, A_inv, new_v, APrimeMat[new_c])
                found[key] = (A_pivot_new, A_inv_new)
            except np.linalg.LinAlgError:
                # Candidate rows are linearly dependent; heuristic search continues with the next candidate.
                logger.log(TRACE, "extension candidate (new_c=%d, new_v=%d) singular, skipping", new_c, new_v)
                continue

        if not found:
            break

        cols_list = [np.array(k[0]) for k in found]
        rows_list = [np.array(k[1]) for k in found]
        weight = np.array([
            np.abs(np.linalg.det(K[np.ix_(r, c)])) ** 2
            for c, r in zip(cols_list, rows_list)
        ])
        delta_e = np.array([
            eps_unocc[r].sum() - eps_occ[c].sum()
            for c, r in zip(cols_list, rows_list)
        ])

        captured = weight.sum()
        force_keep = min_order is not None and order <= min_order
        if not force_keep and captured < tol * mass1:
            break

        result[order] = (delta_e, weight)
        seen.update(found.keys())
        active = [(k[0], k[1], A_pivot, A_inv) for k, (A_pivot, A_inv) in found.items()]
        order += 1

    return result


def shakeup_sticks_by_order(AMat, APrimeMat, eps_occ, eps_unocc, order="auto", tol=0.01, shakedown_only=False):
    """Per-order valence shake-up sticks, order 1 up to whatever the
    maxvol search (order>=2) finds.

    Same order/tol/shakedown_only semantics as shakeup_spectrum. Returns
    (sticks_by_order, orders_included): sticks_by_order is
    {order: (delta_e, weight)} for each order actually included -- the
    per-order breakdown mbxas.shakeup.combine_cross_channel_sticks needs;
    shakeup_spectrum concatenates this into its flat (delta_e, weight)
    contract for callers that don't need the breakdown.

    order="auto": order 1 (always) plus every order>=2 the maxvol search's
    own tol-based stopping includes.
    order=N (int): same, but forces the search to keep going past its
    natural tol-based stop until order N has been attempted (or the search
    runs out of candidates) -- "explicit order never silently downgrades",
    matching the pre-existing contract.
    """
    K = APrimeMat @ np.linalg.inv(AMat)
    e1, w1 = _order1_sticks(K, eps_occ, eps_unocc)
    if shakedown_only:
        mask = e1 < 0
        e1, w1 = e1[mask], w1[mask]
    sticks_by_order = {1: (e1, w1)}
    orders_included = [1]

    min_order = None if order == "auto" else int(order)
    if min_order is not None and min_order < 1:
        raise ValueError(f"order must be >= 1 or 'auto', got {order}")

    configs = _maxvol_shakeup_configs(AMat, APrimeMat, K, eps_occ, eps_unocc, tol, min_order=min_order)
    for k in sorted(configs):
        if min_order is not None and k > min_order:
            break
        ek, wk = configs[k]
        if shakedown_only:
            mask = ek < 0
            ek, wk = ek[mask], wk[mask]
        sticks_by_order[k] = (ek, wk)
        orders_included.append(k)
    return sticks_by_order, orders_included


def shakeup_spectrum(AMat, APrimeMat, eps_occ, eps_unocc, order="auto", tol=0.01, shakedown_only=False):
    """Combined shake-up stick spectrum up to the requested order.

    order: "auto" includes order 1 plus every order the maxvol search's
        own tol-based convergence includes (see shakeup_sticks_by_order).
        An explicit int forces inclusion up to that order (no silent
        downgrade below it, though the search may still find nothing
        beyond what actually exists for the system).
    shakedown_only: see shakeup_sticks.

    Returns (delta_e, weight, orders_included): concatenated sticks across
    all included orders, plus the sorted list of orders actually included.
    Delegates the per-order construction to shakeup_sticks_by_order.
    """
    sticks_by_order, orders_included = shakeup_sticks_by_order(
        AMat, APrimeMat, eps_occ, eps_unocc, order=order, tol=tol, shakedown_only=shakedown_only)
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
