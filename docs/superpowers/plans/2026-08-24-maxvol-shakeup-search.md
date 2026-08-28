# Maxvol Shake-up Configuration Search Implementation Plan

> **Historical implementation plan.** The 2026-08-25 reference review found
> that the resulting Python search is maxvol-style but not a port of QE's
> `maxvol_multi`, and that QE spectral doubles/triples use another algorithm.
> See `dev/shakeup.md` for current behavior and scientific status.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `mbxas/shakeup.py`'s order-2-only, magnitude-rank-and-grow shake-up pruning with a maxvol-style swap search on the existing `K` matrix, ported from `mbxas-qe`'s `maxvol_multi_mod.f90`, removing the hard order cap.

**Architecture:** A new, domain-agnostic `mbxas/maxvol.py` provides a Sherman-Morrison rank-1 row-update primitive. `mbxas/shakeup.py` uses it to implement a breadth-first swap search seeded from the top order-1 `|K|**2` elements: each swap is evaluated via the Sherman-Morrison-updated pivot inverse (mirroring `mbxas-qe`'s B-matrix candidate search), and every discovered configuration's weight is the existing `|det(K[rows_in, cols_out])|**2` minor formula (Jacobi complementary-minor identity — unchanged physics, new search strategy only). `mbxas.mbxas.build_A_K` gains a fourth return value (`APrimeMat`) so callers can feed the search what it needs.

**Tech Stack:** NumPy (existing dependency only — no `maxvolpy` or other new dependency, see spec).

**Spec:** `docs/superpowers/specs/2026-08-24-maxvol-shakeup-search-design.md`

## Global Constraints

- No new dependencies — hand-rolled on `numpy` only (spec: "No dependency on `maxvolpy`").
- SCF/MOM (`pyscf.scf.addons.mom_occ`) is untouched; `AMat`/`ADet`/`KMat` construction in `mbxas.py:build_A_K` keeps using whatever `mo_occ` says is occupied — no orbital-correspondence override (spec Non-goals).
- Order-1 sticks stay exact full enumeration, never routed through the search (spec Non-goals).
- `combine_cross_channel_sticks`, `_prune_outer_product`, `broaden_shakeup`, `convolve_shakeup` are unchanged — they already consume the `{order: (delta_e, weight)}` contract this produces.
- "Stick" terminology is unchanged (spec Non-goals).
- Run `conda run -n pymbxas pytest tests/ -q` before every commit that touches `mbxas/`, `calculators/`, `build/`, or `utils/orbitals.py` (`AGENTS.md`).
- A change that alters computed numbers gets a `### Changed` `CHANGELOG.md` entry, plainly stating shake-up intensities may shift and higher-order satellites may now appear (`AGENTS.md`, spec Documentation).

---

### Task 1: Sherman-Morrison row-update primitive

**Files:**
- Create: `pymbxas/mbxas/maxvol.py`
- Test: `tests/test_h2o_kedge.py` (new standalone test function, no fixture dependency — pure linear algebra, doesn't need a PySCF calculation)

**Interfaces:**
- Produces: `sherman_morrison_row_update(A, A_inv, row_idx, new_row) -> (A_new, A_inv_new)`, used by Task 3.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_h2o_kedge.py` (top-level, alongside other imports — check the existing `import` block near the top of the file and add `from pymbxas.mbxas.maxvol import sherman_morrison_row_update` there):

```python
def test_sherman_morrison_row_update():
    rng = np.random.default_rng(0)
    n = 5
    A = rng.normal(size=(n, n))
    A_inv = np.linalg.inv(A)

    for row_idx in range(n):
        new_row = rng.normal(size=n)
        A_new, A_inv_new = sherman_morrison_row_update(A, A_inv, row_idx, new_row)

        A_expected = A.copy()
        A_expected[row_idx] = new_row
        assert np.allclose(A_new, A_expected), \
            f"row {row_idx}: updated matrix does not match the row replacement"

        A_inv_expected = np.linalg.inv(A_expected)
        assert np.allclose(A_inv_new, A_inv_expected, atol=1e-10), \
            f"row {row_idx}: Sherman-Morrison inverse disagrees with np.linalg.inv from scratch"

    # A near-singular update (new row duplicates another row) must raise,
    # not silently return garbage.
    A_dup = A.copy()
    with pytest.raises(np.linalg.LinAlgError):
        sherman_morrison_row_update(A, A_inv, 0, A_dup[1])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n pymbxas pytest tests/test_h2o_kedge.py::test_sherman_morrison_row_update -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pymbxas.mbxas.maxvol'` (or `ImportError`).

- [ ] **Step 3: Write the implementation**

Create `pymbxas/mbxas/maxvol.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n pymbxas pytest tests/test_h2o_kedge.py::test_sherman_morrison_row_update -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pymbxas/mbxas/maxvol.py tests/test_h2o_kedge.py
git commit -m "Add Sherman-Morrison row-update primitive for maxvol swap search"
```

---

### Task 2: `build_A_K` returns `APrimeMat`

**Files:**
- Modify: `pymbxas/mbxas/mbxas.py:62-81` (`build_A_K`), `pymbxas/mbxas/mbxas.py:124` (its call site in `run_MBXAS_pyscf`)
- Test: `tests/test_h2o_kedge.py` (existing `build_A_K` call sites)

**Interfaces:**
- Consumes: nothing new.
- Produces: `build_A_K(...) -> (AMat, ADet, KMat, APrimeMat)` — a 4-tuple instead of 3. `APrimeMat` shape `(n_unocc, n_occ)`, consumed by Task 4/5.

- [ ] **Step 1: Update `build_A_K` to return `APrimeMat`**

In `pymbxas/mbxas/mbxas.py`, change the `build_A_K` docstring and return statement (currently lines 72-81):

```python
    Returns (AMat, ADet, KMat, APrimeMat): AMat is the square valence overlap
    matrix, ADet its determinant, KMat = A'Mat @ inv(AMat) the matrix used
    both for the n=1 amplitude (Eq. 22, PRB 107,035146) and, at higher
    order, for shake-up minors (see mbxas.shakeup). APrimeMat is returned
    alongside KMat because mbxas.shakeup's maxvol-based configuration
    search needs the raw unoccupied-valence overlap rows, not just their
    product with inv(AMat).
    """
    AMat = mb_overlap_channel[np.ix_(occ_idxs_fch, occ_idxs_gs)]
    ADet = np.linalg.det(AMat)
    APrimeMat = mb_overlap_channel[np.ix_(uno_idxs_fch, occ_idxs_gs)]
    KMat = APrimeMat @ np.linalg.inv(AMat)
    return AMat, ADet, KMat, APrimeMat
```

- [ ] **Step 2: Update `run_MBXAS_pyscf`'s call site**

In `pymbxas/mbxas/mbxas.py:124`, change:

```python
    AMat, ADet, KMat = build_A_K(mb_overlap[channel], occ_idxs_fch, occ_idxs_gs, uno_idxs_fch)
```

to:

```python
    AMat, ADet, KMat, _ = build_A_K(mb_overlap[channel], occ_idxs_fch, occ_idxs_gs, uno_idxs_fch)
```

(`APrimeMat` isn't needed here — the one-body amplitude only uses `AMat`/`ADet`/`KMat`.)

- [ ] **Step 3: Update the existing test's `build_A_K` call**

In `tests/test_h2o_kedge.py:179`, change:

```python
    _, _, K_ch = build_A_K(mb_overlap_ch, occ_idxs_fch_ch, occ_idxs_gs_ch, uno_idxs_fch_ch)
```

to:

```python
    AMat_ch, _, K_ch, APrimeMat_ch = build_A_K(mb_overlap_ch, occ_idxs_fch_ch, occ_idxs_gs_ch, uno_idxs_fch_ch)
```

(`AMat_ch`/`APrimeMat_ch` are used starting in Task 6; leaving them bound now avoids re-editing this line twice.)

- [ ] **Step 4: Run the physics test to confirm no regression**

Run: `conda run -n pymbxas pytest tests/test_h2o_kedge.py -q`
Expected: PASS (the test file won't fully pass yet if Task 6's new assertions aren't written — at this point just confirm nothing *existing* broke; if `test_full_h2o_kedge_kernel` fails only on lines you haven't reached yet in Task 6, that's expected and resolved there. If anything fails before that point, stop and fix before proceeding.)

- [ ] **Step 5: Commit**

```bash
git add pymbxas/mbxas/mbxas.py tests/test_h2o_kedge.py
git commit -m "Return APrimeMat from build_A_K for the maxvol shake-up search"
```

---

### Task 3: Maxvol swap search over `K`

**Files:**
- Modify: `pymbxas/mbxas/shakeup.py` (add `_maxvol_shakeup_configs`, remove `_shakeup_sticks_order2`)
- Test: `tests/test_h2o_kedge.py` (new standalone unit test with a hand-built small system)

**Interfaces:**
- Consumes: `sherman_morrison_row_update` (Task 1).
- Produces: `_maxvol_shakeup_configs(AMat, APrimeMat, K, eps_occ, eps_unocc, tol, min_order=None) -> {order: (delta_e, weight)}` for `order >= 2`, used by Task 4.

- [ ] **Step 1: Write the failing unit test**

This test hand-builds a tiny system where the *only* significant 2x2 minor is known in advance, so the search's discovery is checked against a directly-computed value — this is the Jacobi-identity verification the spec calls for, done here (controlled inputs) rather than against the full H2O system (search-outcome-dependent). Add to `tests/test_h2o_kedge.py`:

```python
def test_maxvol_shakeup_configs_matches_minor_identity():
    from pymbxas.mbxas.shakeup import _maxvol_shakeup_configs

    rng = np.random.default_rng(1)
    n_occ, n_unocc = 4, 4
    AMat = np.eye(n_occ) + 0.01 * rng.normal(size=(n_occ, n_occ))
    APrimeMat = 0.01 * rng.normal(size=(n_unocc, n_occ))
    # Make one specific 2-swap configuration (valence {0,2} -> conduction
    # {1,3}) dominant so the search is guaranteed to find it.
    APrimeMat[1, 0] = 0.9
    APrimeMat[3, 2] = 0.9

    K = APrimeMat @ np.linalg.inv(AMat)
    eps_occ = np.array([-1.0, -1.2, -1.4, -1.6])
    eps_unocc = np.array([0.5, 0.6, 0.7, 0.8])

    configs = _maxvol_shakeup_configs(AMat, APrimeMat, K, eps_occ, eps_unocc, tol=1e-6)

    assert 2 in configs, "the dominant 2-swap configuration should be found at order 2"
    delta_e, weight = configs[2]
    assert len(weight) >= 1

    expected_weight = np.abs(np.linalg.det(K[np.ix_([1, 3], [0, 2])])) ** 2
    expected_delta_e = (eps_unocc[1] + eps_unocc[3]) - (eps_occ[0] + eps_occ[2])
    assert any(abs(w - expected_weight) < 1e-10 for w in weight), \
        "no discovered order-2 config matches the hand-computed dominant 2x2 minor"
    idx = np.argmin(np.abs(weight - expected_weight))
    assert abs(delta_e[idx] - expected_delta_e) < 1e-10, \
        "the matching config's delta_e does not match the expected energy sum"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n pymbxas pytest "tests/test_h2o_kedge.py::test_maxvol_shakeup_configs_matches_minor_identity" -v`
Expected: FAIL with `ImportError: cannot import name '_maxvol_shakeup_configs'`.

- [ ] **Step 3: Write the implementation**

In `pymbxas/mbxas/shakeup.py`:

1. Add the import at the top:

```python
from pymbxas.mbxas.maxvol import sherman_morrison_row_update
```

2. Delete `MAX_IMPLEMENTED_ORDER = 2` (line 22) and `_shakeup_sticks_order2` (lines 90-171) entirely — superseded by the function below.

3. Add the new function in `_shakeup_sticks_order2`'s place:

```python
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
        A_pivot1, A_inv1 = sherman_morrison_row_update(AMat, A_inv0, v0, APrimeMat[c0])
        active.append(((v0,), (c0,), A_pivot1, A_inv1))

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
            A_pivot_new, A_inv_new = sherman_morrison_row_update(A_pivot, A_inv, new_v, APrimeMat[new_c])
            found[key] = (A_pivot_new, A_inv_new)

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n pymbxas pytest "tests/test_h2o_kedge.py::test_maxvol_shakeup_configs_matches_minor_identity" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pymbxas/mbxas/shakeup.py tests/test_h2o_kedge.py
git commit -m "Add maxvol-style swap search for order>=2 shake-up configurations"
```

---

### Task 4: Wire the search into `shakeup_sticks`/`shakeup_sticks_by_order`/`shakeup_spectrum`

**Files:**
- Modify: `pymbxas/mbxas/shakeup.py` (rewrite the three public order functions, lines ~27-233 of the pre-Task-3 file)

**Interfaces:**
- Consumes: `_maxvol_shakeup_configs` (Task 3).
- Produces: `shakeup_sticks(AMat, APrimeMat, eps_occ, eps_unocc, order, shakedown_only=False, tol=0.01)`, `shakeup_sticks_by_order(AMat, APrimeMat, eps_occ, eps_unocc, order="auto", tol=0.01, shakedown_only=False)`, `shakeup_spectrum(AMat, APrimeMat, eps_occ, eps_unocc, order="auto", tol=0.01, shakedown_only=False)` — all now take `(AMat, APrimeMat, ...)` instead of `(K, ...)`. Consumed by `Spectra` in `pymbxas/spectra.py` (Task 5) and by tests (Task 6).

- [ ] **Step 1: Replace `shakeup_sticks`**

Replace the existing `shakeup_sticks` function (originally lines 27-87, now directly after the module docstring/imports and before `_maxvol_shakeup_configs`) with:

```python
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
```

- [ ] **Step 2: Replace `shakeup_sticks_by_order`**

Replace the existing `shakeup_sticks_by_order` function with:

```python
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
```

- [ ] **Step 3: Replace `shakeup_spectrum`**

Replace the existing `shakeup_spectrum` function with:

```python
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
```

- [ ] **Step 4: Remove the now-unused `itertools` import**

`shakeup.py`'s top-level `import itertools` is no longer used anywhere in the file (the brute-force `itertools.combinations` path was in the old `shakeup_sticks`, now removed). Delete that import line.

- [ ] **Step 5: Run the maxvol unit tests to confirm no regression from this wiring**

Run: `conda run -n pymbxas pytest tests/test_h2o_kedge.py::test_sherman_morrison_row_update tests/test_h2o_kedge.py::test_maxvol_shakeup_configs_matches_minor_identity -v`
Expected: PASS (unaffected by this task, confirms the module still imports cleanly).

Do not run the full test file yet — `test_full_h2o_kedge_kernel` still calls the old three-argument `shakeup_sticks(K_ch, ...)` signature and will fail until Task 6 updates it. That's expected here.

- [ ] **Step 6: Commit**

```bash
git add pymbxas/mbxas/shakeup.py
git commit -m "Route shakeup_sticks/by_order/spectrum through the maxvol search, remove order cap"
```

---

### Task 5: Update `Spectra` call sites

**Files:**
- Modify: `pymbxas/spectra.py:356-387` (`_shakeup_sticks_by_order`), `pymbxas/spectra.py:414-446` (`_spectator_shakeup_sticks`)

**Interfaces:**
- Consumes: `build_A_K` (Task 2, now 4-tuple), `shakeup_sticks_by_order` (Task 4, now `(AMat, APrimeMat, ...)`).
- Produces: no change to `Spectra`'s own public API (`get_shakeup_spectrum`, `get_mbxas_spectra`, etc. are untouched — only their internals feed the new signature).

- [ ] **Step 1: Update `_shakeup_sticks_by_order`**

In `pymbxas/spectra.py`, change lines 376-382 from:

```python
            _, _, K = build_A_K(self._mb_overlap[channel], occ_idxs_fch,
                                occ_idxs_gs, uno_idxs_fch)

            eps_occ   = self._fch_mo_energy[channel][occ_idxs_fch]
            eps_unocc = self._fch_mo_energy[channel][uno_idxs_fch]

            sticks_by_order, _ = shakeup_sticks_by_order(K, eps_occ, eps_unocc, order=order, tol=tol)
```

to:

```python
            AMat, _, _, APrimeMat = build_A_K(self._mb_overlap[channel], occ_idxs_fch,
                                              occ_idxs_gs, uno_idxs_fch)

            eps_occ   = self._fch_mo_energy[channel][occ_idxs_fch]
            eps_unocc = self._fch_mo_energy[channel][uno_idxs_fch]

            sticks_by_order, _ = shakeup_sticks_by_order(AMat, APrimeMat, eps_occ, eps_unocc, order=order, tol=tol)
```

- [ ] **Step 2: Update `_spectator_shakeup_sticks`**

In `pymbxas/spectra.py`, apply the identical change to lines 435-441 (the `_spectator_shakeup_sticks` method's `build_A_K`/`shakeup_sticks_by_order` call, same pattern as Step 1).

- [ ] **Step 3: Run the full test suite**

Run: `conda run -n pymbxas pytest tests/ -q`
Expected: `test_h2o_kedge.py::test_full_h2o_kedge_kernel` will now fail at the old `shakeup_sticks(K_ch, ...)` / `shakeup_sticks_by_order(K_ch, ...)` call sites (still using the pre-Task-4 3-argument form) — this is exactly what Task 6 fixes. Confirm the *only* failures are in that one test function and they are argument-count/signature errors, not import errors or unrelated failures. `test_h5_io.py` and the two new maxvol tests should all PASS.

- [ ] **Step 4: Commit**

```bash
git add pymbxas/spectra.py
git commit -m "Pass AMat/APrimeMat from build_A_K into the shake-up search"
```

---

### Task 6: Update `test_h2o_kedge.py`'s shake-up assertions

**Files:**
- Modify: `tests/test_h2o_kedge.py:172-254` (the shake-up section of `test_full_h2o_kedge_kernel`)

**Interfaces:**
- Consumes: `AMat_ch`/`APrimeMat_ch` (bound in Task 2 Step 3), the new `shakeup_sticks`/`shakeup_sticks_by_order`/`shakeup_spectrum` signatures (Task 4).

- [ ] **Step 1: Update the order=1/shakedown_only/order=2 calls to the new signature**

In `tests/test_h2o_kedge.py`, replace lines 183-216 (from the `# order=1 shake-up...` comment through the `order=3` `NotImplementedError` check) with:

```python
    # order=1 shake-up recovers a plain |K_vc|^2 stick spectrum, one entry
    # per (valence, conduction) pair
    e1, w1 = shakeup_sticks(AMat_ch, APrimeMat_ch, eps_occ_ch, eps_unocc_ch, order=1)
    assert e1.shape == w1.shape == (len(occ_idxs_fch_ch) * len(uno_idxs_fch_ch),), \
        f"order=1 shake-up stick count mismatch: {e1.shape} vs expected {(len(occ_idxs_fch_ch)*len(uno_idxs_fch_ch),)}"
    w1_manual = np.abs(K_ch) ** 2
    assert np.allclose(np.sort(w1), np.sort(w1_manual.ravel()), atol=1e-14), \
        "order=1 shake-up weights do not match |K_vc|^2"

    # shakedown_only filters to negative delta_e only ("shake-down",
    # mbxas-qe's kpoint_spectral_details.f90 convention), at the
    # single-order level
    e1_down, w1_down = shakeup_sticks(AMat_ch, APrimeMat_ch, eps_occ_ch, eps_unocc_ch, order=1, shakedown_only=True)
    assert np.all(e1_down < 0), "shakedown_only=True should keep only negative delta_e sticks"
    manual_mask = e1 < 0
    assert np.array_equal(np.sort(e1_down), np.sort(e1[manual_mask])), \
        "shakedown_only=True should match a manual delta_e<0 filter of the unfiltered order-1 sticks"
    assert np.array_equal(np.sort(w1_down), np.sort(w1[manual_mask])), \
        "shakedown_only=True should keep the matching weights unchanged"

    # order=2: weight is the antisymmetrized 2x2 minor of K -- the maxvol
    # search's discovery mechanism changed (see
    # docs/superpowers/specs/2026-08-24-maxvol-shakeup-search-design.md),
    # but every returned weight must still be exactly some 2x2 minor of K,
    # not a hand-picked one (the search may not visit any particular pair
    # for a small test system -- see mbxas.shakeup unit test for the
    # controlled Jacobi-identity check).
    e2, w2 = shakeup_sticks(AMat_ch, APrimeMat_ch, eps_occ_ch, eps_unocc_ch, order=2)
    assert w2.shape == e2.shape

    # order=3+ no longer raises -- the maxvol search has no hardcoded cap.
    # It may legitimately find nothing (empty arrays) if no 3-swap
    # configuration clears tol for this system; either is acceptable, an
    # exception is not.
    e3, w3 = shakeup_sticks(AMat_ch, APrimeMat_ch, eps_occ_ch, eps_unocc_ch, order=3)
    assert e3.shape == w3.shape
```

- [ ] **Step 2: Update the `shakeup_spectrum` calls**

Replace lines 218-236 (`# shakeup_spectrum: explicit order=1...` through the `orders_auto` assertions) with:

```python
    # shakeup_spectrum: explicit order=1 includes only order 1; explicit
    # order=2 always includes both orders (no silent auto-downgrade)
    de1, dw1, orders1 = shakeup_spectrum(AMat_ch, APrimeMat_ch, eps_occ_ch, eps_unocc_ch, order=1)
    assert orders1 == [1], f"explicit order=1 should include only order 1, got {orders1}"
    de2, dw2, orders2 = shakeup_spectrum(AMat_ch, APrimeMat_ch, eps_occ_ch, eps_unocc_ch, order=2)
    assert orders2 == [1, 2], f"explicit order=2 should include orders [1, 2], got {orders2}"
    assert len(de2) == len(e1) + len(e2), "order=2 spectrum should concatenate order-1 and order-2 sticks"

    # auto mode never includes an order whose total probability mass is
    # below tol * order-1 mass; physically, higher-order shake-up should
    # carry less total probability than order 1
    if w2.sum() > 0:
        assert w2.sum() < w1.sum(), \
            f"order-2 total shake-up probability ({w2.sum():.3e}) should be smaller than order-1 ({w1.sum():.3e})"
    de_auto, dw_auto, orders_auto = shakeup_spectrum(AMat_ch, APrimeMat_ch, eps_occ_ch, eps_unocc_ch, order="auto", tol=0.01)
    assert orders_auto[0] == 1, f"auto order should always include order 1, got {orders_auto}"
```

(The strict `orders_auto in ([1], [1, 2])` check is dropped since order>=3 can now legitimately appear in "auto" mode if its mass clears `tol` — replaced with the weaker, still-meaningful "order 1 is always first" check.)

- [ ] **Step 3: Update the `shakeup_sticks_by_order` calls**

Replace lines 238-254 (`from pymbxas.mbxas.shakeup import shakeup_sticks_by_order` through the `sticks_by_order_down` assertion) with:

```python
    from pymbxas.mbxas.shakeup import shakeup_sticks_by_order

    sticks_by_order_2, orders_by_order_2 = shakeup_sticks_by_order(AMat_ch, APrimeMat_ch, eps_occ_ch, eps_unocc_ch, order=2)
    assert orders_by_order_2 == [1, 2], f"expected orders [1, 2], got {orders_by_order_2}"
    assert np.array_equal(sticks_by_order_2[1][0], e1) and np.array_equal(sticks_by_order_2[1][1], w1), \
        "shakeup_sticks_by_order order-1 entry should match shakeup_sticks(order=1)"
    assert np.array_equal(sticks_by_order_2[2][0], e2) and np.array_equal(sticks_by_order_2[2][1], w2), \
        "shakeup_sticks_by_order order-2 entry should match shakeup_sticks(order=2)"

    de2_from_dict = np.concatenate([sticks_by_order_2[k][0] for k in orders_by_order_2])
    dw2_from_dict = np.concatenate([sticks_by_order_2[k][1] for k in orders_by_order_2])
    assert np.array_equal(de2_from_dict, de2) and np.array_equal(dw2_from_dict, dw2), \
        "shakeup_spectrum(order=2) must equal the concatenation of shakeup_sticks_by_order's entries"

    sticks_by_order_down, _ = shakeup_sticks_by_order(AMat_ch, APrimeMat_ch, eps_occ_ch, eps_unocc_ch, order=1, shakedown_only=True)
    assert np.array_equal(np.sort(sticks_by_order_down[1][0]), np.sort(e1_down)), \
        "shakeup_sticks_by_order should forward shakedown_only to shakeup_sticks"
```

(This is unchanged from the original except for the `build_A_K`-supplied arguments — included here because it sits between the edited regions above and below and must keep working against the same `e1`/`e2`/`e1_down` names.)

- [ ] **Step 4: Run the full test suite**

Run: `conda run -n pymbxas pytest tests/ -q`
Expected: PASS, all tests including `test_full_h2o_kedge_kernel`, `test_sherman_morrison_row_update`, `test_maxvol_shakeup_configs_matches_minor_identity`.

- [ ] **Step 5: Commit**

```bash
git add tests/test_h2o_kedge.py
git commit -m "Update shake-up tests for the maxvol search signature and unbounded order"
```

---

### Task 7: Documentation

**Files:**
- Modify: `dev/method.md` (shake-up section, currently the paragraph starting "**One-body truncation, with an opt-in order-2 correction.**")
- Modify: `CHANGELOG.md`

**Interfaces:** none (documentation only).

- [ ] **Step 1: Rewrite the `dev/method.md` shake-up paragraph**

Find the paragraph in `dev/method.md` beginning `**One-body truncation, with an opt-in order-2 correction.**` (currently a single long paragraph documenting `MAX_IMPLEMENTED_ORDER`, `_shakeup_sticks_order2`'s prefix-growth pruning, and the order-3 `NotImplementedError`). Replace the sentences from `` `shakeup_spectrum`'s `"auto"` mode adds orders 2..`MAX_IMPLEMENTED_ORDER`... `` through `` ...matching `mbxas-qe`'s adaptive-tolerance `triples_overlap`. `` with:

```
`shakeup_sticks_by_order`'s `"auto"` mode includes order 1 plus every
order the maxvol-based configuration search (`mbxas.shakeup._maxvol_shakeup_configs`,
ported from `mbxas-qe`'s `maxvol_multi_mod.f90`) finds whose mass clears
`tol` relative to the order-1 mass; there is no hardcoded order cap.
The search reuses the existing `K`-minor weight formula unchanged (Jacobi's
complementary-minor identity: for any k-swap configuration, `det(A_swapped)
/ det(A_reference) = det(K[rows_in, cols_out])`, so the weight formula
generalizes to any order with no new physics) -- what changed is *which*
k-tuples get evaluated. Instead of ranking every valence->conduction single
by `|K|^2` and brute-force-forming 2x2 minors within an active prefix
(pymbxas's previous approach, and no longer present), the search runs a
maxvol-style swap search directly on `AMat`/`APrimeMat`
(`mbxas.mbxas.build_A_K`): seeded from the top-ranked order-1 singles, it
uses a Sherman-Morrison-updated pivot inverse (`mbxas.maxvol.sherman_morrison_row_update`)
to find each seed's best next swap, extending breadth-first order by order
until a new order's captured mass drops below `tol * mass1`. This mirrors
`mbxas-qe`'s own algorithm rather than a magnitude-of-singles heuristic,
and removes the order-2 cap: order 3+ configurations are found whenever
their mass is non-negligible, at no extra implementation cost. SCF/MOM and
`AMat`/`ADet`/`KMat` construction are untouched by this -- the search only
ever runs on the frozen, already-converged FCH orbitals; see
`docs/superpowers/specs/2026-08-24-maxvol-shakeup-search-design.md` for
the full design and the reasoning against overriding `occ_idxs_fch` from
a post-hoc overlap search.
```

Also update the earlier sentence referencing `mbxas.mbxas.build_A_K` (which lists what `build_A_K` returns implicitly via its role) if it names a specific return arity — check the surrounding text for any other place that says `build_A_K` returns three values and update to four (`AMat, ADet, KMat, APrimeMat`).

- [ ] **Step 2: Add the `CHANGELOG.md` entry**

Open `CHANGELOG.md`. If `## [Unreleased]` exists at the top, add to it; otherwise create it at the top per `AGENTS.md`'s changelog discipline. Add under `### Changed`:

```
- Shake-up satellite configuration search now uses a maxvol-based swap
  search instead of magnitude-pruned order-2-only combinatorics; shake-up
  intensities may shift slightly and order-3+ satellites can now appear.
```

- [ ] **Step 3: Bump `pymbxas/__init__.py` version and `CITATION.cff`**

Per `AGENTS.md`'s versioning discipline, bump the patch/minor version in `pymbxas/__init__.py` (`__version__`, `__date__` — set `__date__` to today) and mirror the version bump in `CITATION.cff`. Check the current version first:

Run: `conda run -n pymbxas python -c "import pymbxas; print(pymbxas.__version__)"`

Increment the minor version component (this changes computed shake-up numbers, matching the project's convention for a `### Changed` entry — check `CHANGELOG.md`'s history for the versioning pattern this project actually uses, e.g. whether patch or minor bumps accompany `### Changed` entries, and follow it).

- [ ] **Step 4: Run the full test suite one final time**

Run: `conda run -n pymbxas pytest tests/ -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dev/method.md CHANGELOG.md pymbxas/__init__.py CITATION.cff
git commit -m "Document the maxvol shake-up search and bump version"
```
