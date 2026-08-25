#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sherman-Morrison rank-1 row updates for maxvol-style pivot swap search.

Replacing one row of a square pivot matrix A with a new row is a rank-1
update of A: A_new = A + e_r (new_row - old_row)^T. Recomputing inv(A) from
scratch after every swap -- what mbxas-qe's maxvol_multi_mod.f90 does via
zgetrf/zgetri each iteration -- is O(n^3) per swap; Sherman-Morrison updates
the existing inverse in O(n^2), the standard approach for this (e.g. what
maxvolpy's own core loop does), not a bespoke optimization here.
"""

import numpy as np

_SINGULAR_TOL = 1e-12


def sherman_morrison_row_update(A, A_inv, row_idx, new_row):
    """Update A and inv(A) after replacing row `row_idx` with `new_row`.

    A: (n, n) current pivot matrix. A_inv: (n, n) its inverse.
    row_idx: int, row being replaced. new_row: (n,) replacement row values.

    Returns (A_new, A_inv_new). Raises np.linalg.LinAlgError if the update
    would make the new matrix singular (denominator ~0), rather than
    silently returning a garbage inverse.
    """
    v = new_row - A[row_idx]
    col = A_inv[:, row_idx]
    denom = 1.0 + v @ col
    if abs(denom) < _SINGULAR_TOL:
        raise np.linalg.LinAlgError(
            f"Sherman-Morrison row update is singular (row {row_idx}, "
            f"denom={denom:.3e})")
    A_new = A.copy()
    A_new[row_idx] = new_row
    A_inv_new = A_inv - np.outer(col, v @ A_inv) / denom
    return A_new, A_inv_new
