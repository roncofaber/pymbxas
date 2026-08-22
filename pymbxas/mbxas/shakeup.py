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

import numpy as np

MAX_IMPLEMENTED_ORDER = 2


def shakeup_sticks(K, eps_occ, eps_unocc, order):
    """Order-k valence shake-up stick spectrum.

    K: (n_unocc, n_occ) matrix for one spin channel (mbxas.mbxas.build_A_K).
    eps_occ: (n_occ,) orbital energies of the valence manifold indexing K's columns.
    eps_unocc: (n_unocc,) orbital energies of the conduction manifold indexing K's rows.
    order: number of simultaneous valence -> conduction excitations.

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

    return delta_e.ravel(), weight.ravel()


def shakeup_spectrum(K, eps_occ, eps_unocc, order="auto", tol=0.01):
    """Combined shake-up stick spectrum up to the requested order.

    order: "auto" includes order 2 only if its total weight exceeds
        tol * (order-1 total weight); an explicit int always includes every
        order from 1 up to and including that int, no tolerance check.

    Returns (delta_e, weight, orders_included): concatenated sticks across
    all included orders, plus the sorted list of orders actually included.
    """
    e1, w1 = shakeup_sticks(K, eps_occ, eps_unocc, 1)
    mass1 = w1.sum()

    if order == "auto":
        e2, w2 = shakeup_sticks(K, eps_occ, eps_unocc, 2)
        mass2 = w2.sum()
        if mass1 > 0 and mass2 > tol * mass1:
            return np.concatenate([e1, e2]), np.concatenate([w1, w2]), [1, 2]
        return e1, w1, [1]

    order = int(order)
    if order < 1:
        raise ValueError(f"order must be >= 1 or 'auto', got {order}")
    if order == 1:
        return e1, w1, [1]

    all_e = [e1]
    all_w = [w1]
    for k in range(2, order + 1):
        ek, wk = shakeup_sticks(K, eps_occ, eps_unocc, k)
        all_e.append(ek)
        all_w.append(wk)
    return np.concatenate(all_e), np.concatenate(all_w), list(range(1, order + 1))
