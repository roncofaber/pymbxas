# Shake-up Satellite Spectra Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in, order-generalized valence shake-up satellite correction to pymbxas's MBXAS spectra, generalizing the one-body truncation the same way `mbxas-qe`'s `singles_overlap`/`doubles_overlap` do, but with one order-agnostic formula instead of per-order code.

**Architecture:** New pure-function module `pymbxas/mbxas/shakeup.py` computes order-k valence-to-conduction shake-up sticks as minors of the existing `K = A' @ inv(A)` matrix (a shared `build_A_K` helper factored out of `run_MBXAS_pyscf`), and convolves them onto the broadened main spectrum. `Spectra` becomes self-sufficient (gains `mb_overlap`, FCH `mo_energy`, GS `mo_occ`, core orbital index) and becomes the single real implementation of `get_mbxas_spectra`; `PySCF_mbxas.get_mbxas_spectra` becomes a thin wrapper that sums per-excitation `Spectra` results.

**Tech Stack:** Python, NumPy, PySCF, pytest, HDF5 (h5py via `pymbxas/io/h5.py`).

**Spec:** `docs/superpowers/specs/2026-08-21-shakeup-satellites-design.md`

## Global Constraints

- Orders 1 and 2 only in this version. Order >=3 raises `NotImplementedError` naming the combinatorial-cost reason (spec: "Scope for this version").
- No backward-compatibility gating for old saved HDF5 files — the `Spectra` schema change is unconditional, no fallback path.
- Cross-spin convolution is out of scope, but `channel` is a first-class parameter everywhere shake-up math appears, so it needs no signature change later (spec: "Designed-in extension point").
- No new test files. `tests/test_h2o_kedge.py` is pymbxas's one integration test by explicit project convention (`CLAUDE.md`); every new assertion is appended to it, in order, following the existing style (manual reimplementation compared against the library's own output).
- `shakeup_order=None` (the default) must produce output byte-identical to current `get_mbxas_spectra` behavior — hard regression requirement, not just an expectation.
- `dev/method.md` is the physics authority and must be updated in the same change that alters what a returned array means (`CLAUDE.md` documentation-map rule).
- `CHANGELOG.md` gets a `### Added` entry under `[Unreleased]` (new capability, not a behavior change to existing default output).

---

## Task 1: Extract `build_A_K`, refactor `run_MBXAS_pyscf` to use it

**Files:**
- Modify: `pymbxas/mbxas/mbxas.py`
- Test: `tests/test_h2o_kedge.py` (existing assertions at lines 61-85 act as the regression guard — no new assertions needed for this task, see Step 1)

**Interfaces:**
- Produces: `build_A_K(mb_overlap_channel, occ_idxs_fch, occ_idxs_gs, uno_idxs_fch) -> (AMat, ADet, KMat)`, importable as `from pymbxas.mbxas.mbxas import build_A_K`. `mb_overlap_channel` is `mb_overlap[channel]`, shape `(norb_fch, norb_gs)`. Every later task that needs `K` for a given channel calls this.

This is a pure refactor — no behavior change. The existing test already independently recomputes `A`/`det_A`/`K` by hand (lines 61-78) and compares against `exc.mbxas["absorption"]` at 1e-12 (lines 80-85), so it is already a regression guard for this extraction; no new assertions are needed, but it must still pass unchanged.

- [ ] **Step 1: Confirm current behavior passes (baseline)**

Run: `conda run -n pymbxas pytest tests/test_h2o_kedge.py -v`
Expected: PASS (this is the pre-refactor baseline; if it doesn't pass first, stop and investigate before touching `mbxas.py`)

- [ ] **Step 2: Add `build_A_K` to `pymbxas/mbxas/mbxas.py`**

Insert this function above `run_MBXAS_pyscf` (after the `import numpy as np` / `#%%` header):

```python
def build_A_K(mb_overlap_channel, occ_idxs_fch, occ_idxs_gs, uno_idxs_fch):
    """Valence overlap determinant and K matrix for one spin channel.

    mb_overlap_channel: (norb_fch, norb_gs) overlap between one channel's
        FCH and GS orbitals, i.e. mb_overlap[channel].
    occ_idxs_fch, uno_idxs_fch: FCH occupied/unoccupied valence orbital
        indices for that channel (core orbital excluded from both).
    occ_idxs_gs: GS occupied valence orbital indices for that channel
        (excited core orbital excluded).

    Returns (AMat, ADet, KMat): AMat is the square valence overlap matrix,
    ADet its determinant, KMat = A'Mat @ inv(AMat) the matrix used both for
    the n=1 amplitude (Eq. 22, PRB 107,035146) and, at higher order, for
    shake-up minors (see mbxas.shakeup).
    """
    AMat = mb_overlap_channel[np.ix_(occ_idxs_fch, occ_idxs_gs)]
    ADet = np.linalg.det(AMat)
    APrimeMat = mb_overlap_channel[np.ix_(uno_idxs_fch, occ_idxs_gs)]
    KMat = APrimeMat @ np.linalg.inv(AMat)
    return AMat, ADet, KMat
```

- [ ] **Step 3: Replace the inline A/K construction in `run_MBXAS_pyscf` with a call to `build_A_K`**

Find this block in `run_MBXAS_pyscf`:

```python
    # Extract occupied block of the MB matrix (excited channel)
    AMat = mb_overlap[channel][np.ix_(occ_idxs_fch, occ_idxs_gs)]

    # Determinant of AMat
    ADet = np.linalg.det(AMat)

    # Extract unoccupied block of the MB matrix (excited channel)
    APrimeMat = mb_overlap[channel][np.ix_(uno_idxs_fch, occ_idxs_gs)]

    # Calculate KMat
    KMat = APrimeMat @ np.linalg.inv(AMat)
```

Replace it with:

```python
    # Extract A/K matrices for the excited channel
    AMat, ADet, KMat = build_A_K(mb_overlap[channel], occ_idxs_fch, occ_idxs_gs, uno_idxs_fch)
```

- [ ] **Step 4: Run the test to confirm the refactor is behavior-preserving**

Run: `conda run -n pymbxas pytest tests/test_h2o_kedge.py -v`
Expected: PASS, identical to Step 1 (the manual-reimplementation assertion at line 84-85 checks `max_diff < 1e-12` against `exc.mbxas["absorption"]`, which now flows through `build_A_K`)

- [ ] **Step 5: Commit**

```bash
git add pymbxas/mbxas/mbxas.py
git commit -m "Factor A/K matrix construction out of run_MBXAS_pyscf into build_A_K"
```

---

## Task 2: `shakeup_sticks` and `shakeup_spectrum` in a new `shakeup.py`

**Files:**
- Create: `pymbxas/mbxas/shakeup.py`
- Test: `tests/test_h2o_kedge.py`

**Interfaces:**
- Consumes: nothing from other tasks (pure NumPy on plain arrays).
- Produces: `shakeup_sticks(K, eps_occ, eps_unocc, order) -> (delta_e, weight)` and `shakeup_spectrum(K, eps_occ, eps_unocc, order="auto", tol=0.01) -> (delta_e, weight, orders_included)`, both importable as `from pymbxas.mbxas.shakeup import shakeup_sticks, shakeup_spectrum`. Task 3 imports `shakeup_spectrum`; Task 5 (`Spectra._shakeup_sticks`) calls it directly.
- `MAX_IMPLEMENTED_ORDER = 2` module constant, referenced by Task 5's docstrings/error messages if needed.

- [ ] **Step 1: Write the failing assertions**

Add this block to `tests/test_h2o_kedge.py`, immediately after the existing `assert amp_library.shape[1] == ...` line (currently line 132, right before the `h5_path = obj.save_object(...)` line):

```python
    from pymbxas.mbxas.shakeup import shakeup_sticks, shakeup_spectrum
    from pymbxas.mbxas.mbxas import build_A_K

    occ_idxs_gs_ch = np.setdiff1d(np.where(gs.mo_occ[ch] == 1)[0], [exc.orb_idx])
    occ_idxs_fch_ch = np.where(fch.mo_occ[ch] == 1)[0]
    uno_idxs_fch_ch = np.where(fch.mo_occ[ch] == 0)[0][1:]
    mb_overlap_ch = exc.mbxas["mb_overlap"][ch]
    _, _, K_ch = build_A_K(mb_overlap_ch, occ_idxs_fch_ch, occ_idxs_gs_ch, uno_idxs_fch_ch)
    eps_occ_ch = fch.mo_energy[ch][occ_idxs_fch_ch]
    eps_unocc_ch = fch.mo_energy[ch][uno_idxs_fch_ch]

    # order=1 shake-up recovers a plain |K_vc|^2 stick spectrum, one entry
    # per (valence, conduction) pair
    e1, w1 = shakeup_sticks(K_ch, eps_occ_ch, eps_unocc_ch, order=1)
    assert e1.shape == w1.shape == (len(occ_idxs_fch_ch) * len(uno_idxs_fch_ch),), \
        f"order=1 shake-up stick count mismatch: {e1.shape} vs expected {(len(occ_idxs_fch_ch)*len(uno_idxs_fch_ch),)}"
    w1_manual = np.abs(K_ch) ** 2
    assert np.allclose(np.sort(w1), np.sort(w1_manual.ravel()), atol=1e-14), \
        "order=1 shake-up weights do not match |K_vc|^2"

    # order=2: weight is the antisymmetrized 2x2 minor of K, matching
    # mbxas-qe's doubles_overlap formula exactly (K(v,c)*K(vp,cp)-K(v,cp)*K(vp,c))
    e2, w2 = shakeup_sticks(K_ch, eps_occ_ch, eps_unocc_ch, order=2)
    v0, v1_ = 0, 1
    c0, c1_ = 0, 1
    manual_minor = K_ch[v0, c0] * K_ch[v1_, c1_] - K_ch[v0, c1_] * K_ch[v1_, c0]
    assert any(abs(w - abs(manual_minor) ** 2) < 1e-14 for w in w2), \
        "no order=2 stick matches the hand-computed 2x2 minor for the first valence/conduction pair"

    # order=3 is explicitly out of scope for this version
    with pytest.raises(NotImplementedError):
        shakeup_sticks(K_ch, eps_occ_ch, eps_unocc_ch, order=3)

    # shakeup_spectrum: explicit order=1 includes only order 1; explicit
    # order=2 always includes both orders (no silent auto-downgrade)
    de1, dw1, orders1 = shakeup_spectrum(K_ch, eps_occ_ch, eps_unocc_ch, order=1)
    assert orders1 == [1], f"explicit order=1 should include only order 1, got {orders1}"
    de2, dw2, orders2 = shakeup_spectrum(K_ch, eps_occ_ch, eps_unocc_ch, order=2)
    assert orders2 == [1, 2], f"explicit order=2 should include orders [1, 2], got {orders2}"
    assert len(de2) == len(e1) + len(e2), "order=2 spectrum should concatenate order-1 and order-2 sticks"

    # auto mode never includes an order whose total probability mass is
    # below tol * order-1 mass; physically, higher-order shake-up should
    # carry less total probability than order 1
    assert w2.sum() < w1.sum(), \
        f"order-2 total shake-up probability ({w2.sum():.3e}) should be smaller than order-1 ({w1.sum():.3e})"
    de_auto, dw_auto, orders_auto = shakeup_spectrum(K_ch, eps_occ_ch, eps_unocc_ch, order="auto", tol=0.01)
    assert orders_auto in ([1], [1, 2]), f"auto order resolved to unexpected {orders_auto}"
    if w2.sum() > 0.01 * w1.sum():
        assert orders_auto == [1, 2]
    else:
        assert orders_auto == [1]
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n pymbxas pytest tests/test_h2o_kedge.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pymbxas.mbxas.shakeup'`

- [ ] **Step 3: Create `pymbxas/mbxas/shakeup.py`**

```python
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

    K: (n_occ, n_unocc) matrix for one spin channel (mbxas.mbxas.build_A_K).
    eps_occ: (n_occ,) orbital energies of the valence manifold indexing K's rows.
    eps_unocc: (n_unocc,) orbital energies of the conduction manifold indexing K's columns.
    order: number of simultaneous valence -> conduction excitations.

    Returns (delta_e, weight): flat 1D arrays, one entry per combination of
    `order` valence orbitals promoted to `order` conduction orbitals.
    weight = |det(K[v_combo, c_combo])|**2.
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

    # sub[i, j] is the (order, order) submatrix of K for valence combo i,
    # conduction combo j; numpy.linalg.det batches over leading dimensions
    sub = K[v_combos[:, None, :, None], c_combos[None, :, None, :]]
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `conda run -n pymbxas pytest tests/test_h2o_kedge.py -v`
Expected: PASS

If the `order=2` mass assertion (`w2.sum() < w1.sum()`) fails for H2O/O specifically, do not weaken the assertion — this is exactly the open normalization question the spec flags. Investigate whether `eps_occ`/`eps_unocc` or `K` orientation is transposed before concluding the physics itself is wrong.

- [ ] **Step 5: Commit**

```bash
git add pymbxas/mbxas/shakeup.py tests/test_h2o_kedge.py
git commit -m "Add order-generalized shake-up stick spectrum (shakeup.py)"
```

---

## Task 3: Broadening and convolution (`broaden_shakeup`, `convolve_shakeup`)

**Files:**
- Modify: `pymbxas/mbxas/shakeup.py`
- Test: `tests/test_h2o_kedge.py`

**Interfaces:**
- Consumes: `shakeup_spectrum` (Task 2); `broadened_spectrum(egrid, energies, intensities, sigma)` from `pymbxas.mbxas.broaden` (existing).
- Produces: `broaden_shakeup(delta_e, weight, egrid, sigma) -> kernel` and `convolve_shakeup(egrid, main_intensity, delta_e, weight, sigma) -> intensity`, both importable as `from pymbxas.mbxas.shakeup import broaden_shakeup, convolve_shakeup`. Task 6 (`Spectra.get_mbxas_spectra`) calls `convolve_shakeup`; Task 5 (`Spectra.get_shakeup_spectrum`) calls `broaden_shakeup`.

- [ ] **Step 1: Write the failing assertions**

Append to `tests/test_h2o_kedge.py`, right after the block added in Task 2 (after the `orders_auto` assertions, still before `h5_path = obj.save_object(...)`):

```python
    from pymbxas.mbxas.shakeup import broaden_shakeup, convolve_shakeup

    # broaden_shakeup with empty sticks reduces to a single normalized
    # Gaussian at delta_e=0 (the implicit n=0 "no extra shake-up" term)
    egrid_probe = np.linspace(-5, 5, 2001)
    kernel_empty = broaden_shakeup(np.empty(0), np.empty(0), egrid_probe, sigma=0.5)
    de_probe = egrid_probe[1] - egrid_probe[0]
    assert abs(kernel_empty.sum() * de_probe - 1.0) < 1e-6, \
        f"empty-sticks shake-up kernel should integrate to 1, got {kernel_empty.sum()*de_probe:.6f}"
    assert egrid_probe[np.argmax(kernel_empty)] == pytest.approx(0.0, abs=de_probe), \
        "empty-sticks shake-up kernel should peak at delta_e=0"

    # convolve_shakeup with empty sticks must leave the main spectrum
    # unchanged (this is the shakeup_order=None-equivalent limit)
    main_probe = np.exp(-0.5 * (egrid_probe / 1.0) ** 2)
    convolved_empty = convolve_shakeup(egrid_probe, main_probe, np.empty(0), np.empty(0), sigma=0.5)
    assert np.allclose(convolved_empty, main_probe, atol=1e-3), \
        "convolving with an empty shake-up spectrum should not change the main spectrum"

    # a single shake-up stick at a known offset should shift probability
    # mass to (roughly) that offset, and total integrated intensity should
    # be conserved (both terms sum to the original mass, up to the
    # normalization convention: main-only weight 1 vs shake-up weight w)
    stick_de = np.array([2.0])
    stick_w = np.array([1.0])  # equal weight to the n=0 term, for an easy 50/50 check
    convolved_one = convolve_shakeup(egrid_probe, main_probe, stick_de, stick_w, sigma=0.5)
    assert np.trapz(convolved_one, egrid_probe) == pytest.approx(
        np.trapz(main_probe, egrid_probe), rel=0.05), \
        "convolution should conserve total integrated intensity"
    # half the conserved intensity should now sit near delta_e=+2 rather
    # than at the original peak (equal-weight n=0 vs n=1 split)
    mass_near_peak = np.trapz(convolved_one[(egrid_probe > -1) & (egrid_probe < 1)], egrid_probe[(egrid_probe > -1) & (egrid_probe < 1)])
    mass_near_satellite = np.trapz(convolved_one[(egrid_probe > 1) & (egrid_probe < 3)], egrid_probe[(egrid_probe > 1) & (egrid_probe < 3)])
    assert mass_near_satellite > 0.3 * mass_near_peak, \
        "equal-weight single shake-up stick should move a comparable amount of intensity to the satellite"
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n pymbxas pytest tests/test_h2o_kedge.py -v`
Expected: FAIL with `ImportError: cannot import name 'broaden_shakeup'`

- [ ] **Step 3: Add `broaden_shakeup` and `convolve_shakeup` to `pymbxas/mbxas/shakeup.py`**

Append to the end of `pymbxas/mbxas/shakeup.py`:

```python
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
    kernel built from (delta_e, weight) sticks plus the implicit n=0 term.

    egrid: (npoints,) uniform grid main_intensity is defined on.
    main_intensity: (npoints,) or (naxes, npoints).
    Returns an array the same shape as main_intensity, on the same egrid.
    """
    de = egrid[1] - egrid[0]

    stick_extent = np.abs(delta_e).max() if len(delta_e) else 0.0
    half_width = stick_extent + 5 * sigma
    n_half = int(np.ceil(half_width / de))
    kgrid = np.arange(-n_half, n_half + 1) * de  # guaranteed symmetric, kgrid[n_half] == 0.0 exactly

    kernel = broaden_shakeup(delta_e, weight, kgrid, sigma)
    kernel = kernel / (kernel.sum() * de)  # normalize to unit probability

    def _convolve_1d(y):
        return np.convolve(y, kernel, mode="same") * de

    main_intensity = np.asarray(main_intensity)
    if main_intensity.ndim == 1:
        return _convolve_1d(main_intensity)
    return np.array([_convolve_1d(row) for row in main_intensity])
```

- [ ] **Step 4: Run to verify it passes**

Run: `conda run -n pymbxas pytest tests/test_h2o_kedge.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pymbxas/mbxas/shakeup.py tests/test_h2o_kedge.py
git commit -m "Add shake-up broadening and convolution onto the main spectrum"
```

---

## Task 4: Extend `Spectra` to store `mb_overlap`, FCH `mo_energy`, GS `mo_occ`, core orbital index

**Files:**
- Modify: `pymbxas/spectra.py`

**Interfaces:**
- Produces: new instance attributes `self._mb_overlap` (shape `(2, norb_fch, norb_gs)`), `self._fch_mo_energy` (shape `(2, norb_fch)`), `self._gs_mo_occ` (shape `(2, norb_gs)`), `self._core_orb_idx` (int). Task 5 (`Spectra._shakeup_sticks`) reads all four directly.
- These are also persisted via `_write_into`/`_read_from`, under a new `shakeup` HDF5 group.

- [ ] **Step 1: Write the failing assertions**

Append to `tests/test_h2o_kedge.py`, right after the Task 3 block (still before `h5_path = obj.save_object(...)`):

```python
    spectra_fields = obj.to_spectra(0)
    assert np.array_equal(spectra_fields._mb_overlap, exc.mbxas["mb_overlap"]), \
        "Spectra._mb_overlap does not match the excitation's mb_overlap"
    assert np.array_equal(spectra_fields._fch_mo_energy, fch.mo_energy), \
        "Spectra._fch_mo_energy does not match the FCH mo_energy"
    assert np.array_equal(spectra_fields._gs_mo_occ, gs.mo_occ), \
        "Spectra._gs_mo_occ does not match the GS mo_occ"
    assert spectra_fields._core_orb_idx == exc.orb_idx, \
        f"Spectra._core_orb_idx {spectra_fields._core_orb_idx} != exc.orb_idx {exc.orb_idx}"
```

Also append this near the existing spectra round-trip block at the end of the file (right after the existing `assert spectra_back.channel == spectra.channel, ...` line):

```python
    assert np.array_equal(spectra_back._mb_overlap, spectra._mb_overlap), \
        "Spectra mb_overlap changed across a save/load"
    assert np.array_equal(spectra_back._fch_mo_energy, spectra._fch_mo_energy), \
        "Spectra FCH mo_energy changed across a save/load"
    assert np.array_equal(spectra_back._gs_mo_occ, spectra._gs_mo_occ), \
        "Spectra GS mo_occ changed across a save/load"
    assert spectra_back._core_orb_idx == spectra._core_orb_idx, \
        "Spectra core_orb_idx changed across a save/load"
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n pymbxas pytest tests/test_h2o_kedge.py -v`
Expected: FAIL with `AttributeError: 'Spectra' object has no attribute '_mb_overlap'`

- [ ] **Step 3: Store the new fields in `Spectra.__initialize_spectra`**

In `pymbxas/spectra.py`, find:

```python
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
```

Replace with:

```python
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
```

- [ ] **Step 4: Persist the new fields in `_write_into` and `_read_from`**

Find:

```python
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
```

Replace with:

```python
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
```

Find:

```python
    def _read_from(self, group):
        self.structure     = h5.read_structure(group, "structure")
        self.calc_settings = h5.read_json(group, "calc_settings")

        xas = group["xas"]
        self._energies  = xas["energies"][()]
        self._amplitude = xas["amplitude"][()]
        self._el_labels = xas["el_labels"][()]

        exc_idx = int(group.attrs["exc_idx"])
```

Replace with:

```python
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
```

- [ ] **Step 5: Run to verify it passes**

Run: `conda run -n pymbxas pytest tests/test_h2o_kedge.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add pymbxas/spectra.py tests/test_h2o_kedge.py
git commit -m "Store mb_overlap/mo_energy/mo_occ/core_orb_idx on Spectra for shake-up math"
```

---

## Task 5: `Spectra._shakeup_sticks` (cached) and `Spectra.get_shakeup_spectrum`

**Files:**
- Modify: `pymbxas/spectra.py`

**Interfaces:**
- Consumes: `build_A_K` (Task 1), `shakeup_spectrum`, `broaden_shakeup` (Tasks 2-3), the fields from Task 4.
- Produces: `Spectra._shakeup_sticks(order, channel, tol) -> (delta_e_ev, weight, orders_included)` (private, cached) and `Spectra.get_shakeup_spectrum(order="auto", channel=None, sigma=0.5, npoints=3001, erange=None, tol=0.01) -> (egrid, kernel, orders_included)` (public). Task 6 calls `_shakeup_sticks` directly (not `get_shakeup_spectrum`, to reuse the main spectrum's own grid instead of building a second one).

- [ ] **Step 1: Write the failing assertions**

Append to `tests/test_h2o_kedge.py`, right after the Task 4 block:

```python
    egrid_shakeup, kernel_shakeup, orders_shakeup = spectra_fields.get_shakeup_spectrum(order=1, sigma=0.5)
    assert len(egrid_shakeup) == len(kernel_shakeup), "shake-up energy/kernel length mismatch"
    assert np.all(np.isfinite(kernel_shakeup)), "shake-up kernel contains non-finite values"
    assert orders_shakeup == [1], f"expected orders [1], got {orders_shakeup}"

    # caching: a second call with the same (channel, order, tol) must reuse
    # the cached sticks rather than recomputing (same object identity)
    cache_key = (spectra_fields._channel, 1, 0.01)
    assert cache_key in spectra_fields._shakeup_cache, "shake-up sticks were not cached"
    cached_before = spectra_fields._shakeup_cache[cache_key]
    spectra_fields.get_shakeup_spectrum(order=1, sigma=0.7)  # different sigma, same order/channel/tol
    assert spectra_fields._shakeup_cache[cache_key] is cached_before, \
        "changing sigma should not invalidate the cached shake-up sticks"

    # explicit channel argument must be accepted (designed-in extension
    # point for the future cross-spin feature), even though only the
    # excited channel is exercised meaningfully here
    _, _, orders_explicit_channel = spectra_fields.get_shakeup_spectrum(order=1, channel=spectra_fields._channel)
    assert orders_explicit_channel == [1]
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n pymbxas pytest tests/test_h2o_kedge.py -v`
Expected: FAIL with `AttributeError: 'Spectra' object has no attribute 'get_shakeup_spectrum'`

- [ ] **Step 3: Add `_shakeup_sticks` and `get_shakeup_spectrum` to `pymbxas/spectra.py`**

Insert this right after the existing `amp2int` method (after its `return energies * np.sum(amplitude**2, axis=0) / amplitude.shape[0]` line):

```python
    def _shakeup_sticks(self, order, channel, tol):
        """Cached (delta_e_ev, weight, orders_included) for one spin channel.
        `channel=None` defaults to the excited channel; an explicit channel
        is accepted so a future cross-spin feature can call this on the
        other channel without a signature change."""
        from pymbxas.mbxas.mbxas import build_A_K
        from pymbxas.mbxas.shakeup import shakeup_spectrum

        if channel is None:
            channel = self._channel

        if not hasattr(self, "_shakeup_cache"):
            self._shakeup_cache = {}

        key = (channel, order, tol)
        if key not in self._shakeup_cache:
            occ_idxs_gs = np.setdiff1d(
                np.where(self._gs_mo_occ[channel] == 1)[0], [self._core_orb_idx])
            occ_idxs_fch = np.where(self._mo_occ[channel] == 1)[0]
            uno_idxs_fch = np.where(self._mo_occ[channel] == 0)[0][1:]

            _, _, K = build_A_K(self._mb_overlap[channel], occ_idxs_fch,
                                occ_idxs_gs, uno_idxs_fch)

            eps_occ   = self._fch_mo_energy[channel][occ_idxs_fch]
            eps_unocc = self._fch_mo_energy[channel][uno_idxs_fch]

            delta_e, weight, orders = shakeup_spectrum(
                K, eps_occ, eps_unocc, order=order, tol=tol)
            self._shakeup_cache[key] = (Ha * delta_e, weight, orders)

        return self._shakeup_cache[key]

    def get_shakeup_spectrum(self, order="auto", channel=None, sigma=0.5,
                              npoints=3001, erange=None, tol=0.01):
        """Broadened valence shake-up probability spectrum P(dE), the
        f^(n) terms beyond the one-body truncation (see dev/method.md).
        Convolve this onto a main spectrum's own grid with
        pymbxas.mbxas.shakeup.convolve_shakeup, or use
        get_mbxas_spectra(shakeup_order=...) to do that automatically."""
        from pymbxas.mbxas.shakeup import broaden_shakeup

        delta_e_ev, weight, orders = self._shakeup_sticks(order, channel, tol)

        if erange is None:
            hi = (delta_e_ev.max() if len(delta_e_ev) else 0.0) + 5 * sigma
            erange = [-5 * sigma, hi]
        egrid = np.linspace(erange[0], erange[1], npoints)

        return egrid, broaden_shakeup(delta_e_ev, weight, egrid, sigma), orders
```

- [ ] **Step 4: Run to verify it passes**

Run: `conda run -n pymbxas pytest tests/test_h2o_kedge.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pymbxas/spectra.py tests/test_h2o_kedge.py
git commit -m "Add Spectra.get_shakeup_spectrum with per-channel caching"
```

---

## Task 6: Wire `shakeup_order` into `Spectra.get_mbxas_spectra`

**Files:**
- Modify: `pymbxas/spectra.py`

**Interfaces:**
- Consumes: `_shakeup_sticks` (Task 5), `convolve_shakeup` (Task 3).
- Produces: `Spectra.get_mbxas_spectra(..., shakeup_order=None)` — new keyword, default `None` preserves current behavior exactly.

- [ ] **Step 1: Write the failing assertions**

Append to `tests/test_h2o_kedge.py`, right after the Task 5 block:

```python
    # shakeup_order=None (the default) must be byte-identical to the
    # existing get_mbxas_spectra call already exercised above
    E_none, I_none = spectra_fields.get_mbxas_spectra(erange=[520, 560], sigma=0.5)
    assert np.array_equal(E_none, E_spectra) and np.array_equal(I_none, I_spectra), \
        "shakeup_order=None changed get_mbxas_spectra output"

    # shakeup_order=1 must change the spectrum shape (correction is applied)
    # but conserve total integrated intensity, since convolution conserves
    # the integral of a unit-normalized kernel
    E_sk, I_sk = spectra_fields.get_mbxas_spectra(erange=[520, 560], sigma=0.5, shakeup_order=1)
    assert E_sk.shape == E_none.shape
    assert not np.allclose(I_sk, I_none), \
        "shakeup_order=1 should change the spectrum (H2O/O has nonzero order-1 shake-up mass)"
    assert np.trapz(I_sk, E_sk) == pytest.approx(np.trapz(I_none, E_none), rel=0.1), \
        "shake-up convolution should approximately conserve total integrated intensity within the plotted erange"
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n pymbxas pytest tests/test_h2o_kedge.py -v`
Expected: FAIL with `TypeError: get_mbxas_spectra() got an unexpected keyword argument 'shakeup_order'`

- [ ] **Step 3: Add the `shakeup_order` parameter**

Find:

```python
    def get_mbxas_spectra(self, axis=None, sigma=0.5, npoints=3001, tol=0.01,
                          erange=None, el_label=None):
        
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
        
        return erange, spectra
```

Replace with:

```python
    def get_mbxas_spectra(self, axis=None, sigma=0.5, npoints=3001, tol=0.01,
                          erange=None, el_label=None, shakeup_order=None):

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

        if shakeup_order is not None:
            from pymbxas.mbxas.shakeup import convolve_shakeup
            delta_e_ev, weight, _ = self._shakeup_sticks(shakeup_order, None, tol)
            spectra = convolve_shakeup(erange, spectra, delta_e_ev, weight, sigma)

        return erange, spectra
```

Note: `erange` on the left of `erange, spectra = get_mbxas_spectra(...)` is reassigned to the actual energy grid array by that call (pre-existing behavior in this file) — this is exactly the grid `convolve_shakeup` needs, reused directly rather than rebuilt.

- [ ] **Step 4: Run to verify it passes**

Run: `conda run -n pymbxas pytest tests/test_h2o_kedge.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pymbxas/spectra.py tests/test_h2o_kedge.py
git commit -m "Add shakeup_order to Spectra.get_mbxas_spectra"
```

---

## Task 7: Replace `PySCF_mbxas.get_mbxas_spectra` with a thin wrapper over `Spectra`

**Files:**
- Modify: `pymbxas/calculators/pyscf.py`

**Interfaces:**
- Consumes: `self.to_spectra(i)` (existing), `Spectra.get_mbxas_spectra` (Task 6).
- Produces: `PySCF_mbxas.get_mbxas_spectra(ato_idx, axis=None, sigma=0.5, npoints=3001, tol=0.01, erange=None, shakeup_order=None)` — same public signature as before plus `shakeup_order`, same aggregation semantics (sum across every excitation matching `ato_idx`, matching current behavior for multi-atom site sums like N2's two symmetric nitrogens).

This task removes the third parallel implementation of `get_mbxas_spectra` math instead of extending it — do not add shake-up logic here directly.

- [ ] **Step 1: Write the failing assertions**

Append to `tests/test_h2o_kedge.py`, right after the Task 6 block:

```python
    # PySCF_mbxas.get_mbxas_spectra must still agree with Spectra's own
    # output after becoming a thin wrapper (extends the existing agreement
    # check above to the shakeup_order path too)
    E_pyscf_sk, I_pyscf_sk = obj.get_mbxas_spectra("O", erange=[520, 560], sigma=0.5, shakeup_order=1)
    assert np.array_equal(E_pyscf_sk, E_sk) and np.allclose(I_pyscf_sk, I_sk, atol=1e-12), \
        "PySCF_mbxas.get_mbxas_spectra(shakeup_order=1) disagrees with Spectra.get_mbxas_spectra"

    # shakeup_order=None through PySCF_mbxas must still match the original
    # pre-refactor baseline computed at the top of this test
    E_pyscf_none, I_pyscf_none = obj.get_mbxas_spectra("O", erange=[520, 560], sigma=0.5)
    assert np.array_equal(E_pyscf_none, E) and np.allclose(I_pyscf_none, I, atol=1e-12), \
        "PySCF_mbxas.get_mbxas_spectra regression after wrapper refactor"
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n pymbxas pytest tests/test_h2o_kedge.py -v`
Expected: FAIL with `TypeError: get_mbxas_spectra() got an unexpected keyword argument 'shakeup_order'` (the current `PySCF_mbxas.get_mbxas_spectra` doesn't accept it yet)

- [ ] **Step 3: Replace `PySCF_mbxas.get_mbxas_spectra`**

In `pymbxas/calculators/pyscf.py`, find:

```python
    def get_mbxas_spectra(self, ato_idx, axis=None, sigma=0.5, npoints=3001, tol=0.01,
                          erange=None):

        ato_idxs = atoms_to_indexes(self.structure, ato_idx)

        energies_ha = []
        intensities = []
        for exc in self.excitations:
            if exc.ato_idx not in ato_idxs:
                continue
            energies_ha.append(exc.mbxas["energies"])
            intensities.append(exc.mbxas["absorption"])

        energies_ha = np.concatenate(energies_ha)
        energies    = Ha*energies_ha
        # sigma(omega) ~ omega * |M|^2 (Eq. 4, PRB 107, 035146), weighted
        # in the same atomic units (Ha) as the amplitude
        intensities = energies_ha * np.concatenate(intensities, axis=1)**2  # |d|² per axis

        erange, spectras = get_mbxas_spectra(energies, intensities,
                                              sigma=sigma, npoints=npoints,
                                              tol=tol, erange=erange)

        if axis is None:
            spectras = np.mean(spectras, axis=0)
        else:
            spectras = spectras[axis]
```

Read a few lines further in the same method (it ends with `return erange, spectras`) and replace the whole method body with:

```python
    def get_mbxas_spectra(self, ato_idx, axis=None, sigma=0.5, npoints=3001, tol=0.01,
                          erange=None, shakeup_order=None):

        ato_idxs = atoms_to_indexes(self.structure, ato_idx)
        matched = [i for i, exc in enumerate(self.excitations) if exc.ato_idx in ato_idxs]
        if not matched:
            raise ValueError(f"No excitations found for atom index/label {ato_idx!r}")

        spectras = [self.to_spectra(i) for i in matched]

        if erange is None:
            all_energies = np.concatenate([sp.energies for sp in spectras])
            erange = [all_energies.min(), all_energies.max()]

        energy = None
        intensity_sum = None
        for sp in spectras:
            energy, intensity = sp.get_mbxas_spectra(axis=axis, sigma=sigma,
                                                      npoints=npoints, tol=tol,
                                                      erange=erange,
                                                      shakeup_order=shakeup_order)
            intensity_sum = intensity if intensity_sum is None else intensity_sum + intensity

        return energy, intensity_sum
```

Note: this matches the pre-existing site-summed (not averaged) semantics exactly — broadening is linear, so summing already-broadened per-excitation intensities on an identical shared grid is mathematically identical to the old approach of concatenating raw sticks across excitations and broadening once. The shared `erange` (computed once, up front, across every matched excitation) is what keeps the per-excitation grids identical so the sum is valid — this must not be left to each `Spectra.get_mbxas_spectra` call's own per-spectrum default.

- [ ] **Step 4: Run to verify it passes**

Run: `conda run -n pymbxas pytest tests/test_h2o_kedge.py -v`
Expected: PASS

- [ ] **Step 5: Run the full test suite**

Run: `conda run -n pymbxas pytest tests/ -q`
Expected: PASS (this is the only test file, but confirm nothing else in the repo imports/monkeypatches the old `get_mbxas_spectra` body)

- [ ] **Step 6: Commit**

```bash
git add pymbxas/calculators/pyscf.py tests/test_h2o_kedge.py
git commit -m "Replace PySCF_mbxas.get_mbxas_spectra with a thin wrapper over Spectra"
```

---

## Task 8: Documentation and changelog

**Files:**
- Modify: `dev/method.md`
- Modify: `CHANGELOG.md`
- Modify: `CLAUDE.md` (if the "get_mbxas_spectra exists on three classes" gotcha needs updating now that it's two)

**Interfaces:** none (documentation only).

- [ ] **Step 1: Update `dev/method.md`'s "One-body truncation" entry**

Find (this text was added earlier in this project's history when the shake-up investigation was first documented):

```
**One-body truncation.** Only single shake-up is kept, which is the standard determinant approximation, not a defect. Multi-electron shake-up and the continuum are outside the model. `mbxas-qe` treats this order by order per spin channel (`xas_mb1`/`xas_mb2`/`xas_mb3` = 1/2/3-body shake-up, matching the `f^(n)` terms of PRB 107, 035146 §III E) and cross-convolves the two channels' order-resolved spectra (`xas_f1_conv`, `xas_f2_conv` = mb2⊗0-body + mb1⊗singles, etc.), which is how it produces non-zero shake-up satellite intensity even for single-shake-up-dominated edges. Nothing like this is implemented in pymbxas; a scoped-down version (e.g. a small non-orthogonal CI over a handful of FCH/valence-excited determinants, using PySCF's exact two-electron integrals directly rather than a learned kernel) is a real candidate for a future feature — worth a design pass (see `SVD-NOCI-OL`, `shirley_noci_methodology.f90`, on `mbxas-qe`'s `noci-kpoint-shirley-aniso` branch) before starting, since it's a genuinely new capability, not a bug fix.
```

Replace with:

```
**One-body truncation, with an opt-in order-2 correction.** By default (`shakeup_order=None`) only single shake-up is kept. `get_mbxas_spectra(shakeup_order=1|2|"auto")` (`Spectra`, and `PySCF_mbxas` which delegates to it) additionally convolves in the order-k valence shake-up probability spectrum: a k-fold simultaneous valence-to-conduction excitation, weighted by `|det(K[v_combo, c_combo])|^2`, the k x k minor of the same `K = A' @ inv(A)` matrix already used for the n=1 amplitude (`pymbxas/mbxas/shakeup.py`, `mbxas.mbxas.build_A_K`). This is the exact non-interacting generalization of the `f^(n)` term (PRB 107, 035146, Eq. 32-35), matching `mbxas-qe`'s `singles_overlap`/`doubles_overlap` formula exactly (verified against `K(v,c)*K(vp,cp) - K(v,cp)*K(vp,c)` in `QE/SHIRLEY/src/mbxas_spectra.f90`). Order 3+ raises `NotImplementedError`: the combinatorics grow as `O(n_occ^3 * n_virt^3)`, and pymbxas has no pruning strategy like `mbxas-qe`'s adaptive-tolerance loop in `doubles_overlap`/`triples_overlap` to make that tractable. Cross-spin convolution (the *other*, non-excited channel's own shake-up, via `spin_convolve_spectrum` in `mbxas-qe`'s `spec.f90`) is not implemented; `Spectra._shakeup_sticks`/`get_shakeup_spectrum` accept an explicit `channel` argument specifically so that extension needs no signature change later. See `docs/superpowers/specs/2026-08-21-shakeup-satellites-design.md` for the full design, including the still-unverified Onishi/Fredholm-determinant-type normalization identity that would make "auto" convergence rigorous rather than heuristic.
```

- [ ] **Step 2: Add a verification entry**

In `dev/method.md`'s "Verification" table, add a row (numbers to be filled in from the actual test run once Task 2's assertions pass — use the printed `w1.sum()`/`w2.sum()` values, do not guess them):

```
| Order-1 / order-2 shake-up mass (H2O/O) | see `tests/test_h2o_kedge.py` order-2-mass assertion |
```

- [ ] **Step 3: Add the CHANGELOG entry**

In `CHANGELOG.md`, under `## [Unreleased]` -> `### Added`, add:

```
- `get_mbxas_spectra(shakeup_order=1|2|"auto")` convolves in order-k valence shake-up satellite intensity beyond the one-body truncation
```

- [ ] **Step 4: Update the `CLAUDE.md` "three classes" gotcha**

Find:

```
- **`get_mbxas_spectra` exists on three classes** with the same name and slightly different signatures: `PySCF_mbxas` (takes `ato_idx`), `Spectra` (takes `el_label`), `mbxas.broaden` (the free function both call). Verified to agree to 2e-19; keep them that way.
```

Replace with:

```
- **`get_mbxas_spectra` exists on two classes plus a free function**: `Spectra` (takes `el_label`, `shakeup_order`) is the one real implementation; `PySCF_mbxas` (takes `ato_idx`) is a thin wrapper that sums per-excitation `Spectra` results on a shared `erange`; `mbxas.broaden` is the free function `Spectra` calls for the base (non-shake-up) broadening. Do not reintroduce a third independent implementation.
```

- [ ] **Step 5: Commit**

```bash
git add dev/method.md CHANGELOG.md CLAUDE.md
git commit -m "Document shake-up satellite spectra"
```
