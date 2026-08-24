# Cross-Spin Shake-Up and Shake-Down Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the spectator (non-excited) spin channel's own shake-up satellite contribution and a shake-down (negative-electron-hole-energy) diagnostic mode to `get_mbxas_spectra`, matching `mbxas-qe`'s `spin_convolve_spectrum` and `shakedown`/`shakedown_only` concepts.

**Architecture:** A new `spectator_occ_unocc_indices` builds the spectator channel's own `K` matrix (no core hole, no removal); a new per-order stick dict (`shakeup_sticks_by_order`) exposes what `shakeup_spectrum` already computes internally; a new `combine_cross_channel_sticks` combines two channels' per-order sticks via outer-sum energy / outer-product weight for every `(i, j)` pair under a total-order cap, feeding the existing broadening/convolution unchanged. `shakedown_only` filters to `delta_e < 0` at both the low level (`shakeup_sticks`, per mbxas-qe naming) and, more usefully, on the final combined stick set (`Spectra` API).

**Tech Stack:** NumPy (vectorized outer sum/product, batched `det`), PySCF, existing `pymbxas.mbxas.broaden`.

**Spec:** `docs/superpowers/specs/2026-08-24-cross-spin-shakedown-design.md`

## Global Constraints

- Every new numerical routine operates on whole arrays (broadcasting, outer sum/product, batched `det`) — never a Python-level loop over individual valence/conduction combinations. Looping over the small number of `(i, j)` *order* pairs (at most `(MAX_IMPLEMENTED_ORDER+1)^2`) is fine; that is not what this constraint restricts.
- In `pymbxas/mbxas/mbxas.py` and `pymbxas/mbxas/shakeup.py` only, add a short comment at each new/changed function's core formula naming the physical quantity and, where one exists, the source equation ("Eq. 22, PRB 107,035146"-style). `CLAUDE.md`'s no-comments default is deliberately overridden for these two files, because the *why* — which formula, from which paper — is the non-obvious information a reader of physics code needs. Don't add comments elsewhere in the codebase for this feature.
- `spectator_order=None` and `max_total_order=None` (the defaults) must leave every existing call byte-identical to pre-feature output. Where a task's design achieves this by routing through a shared helper, that helper must short-circuit to literally the same call as before — not merely produce numerically-equal floats through a different code path.
- No backward-compatibility shims for old saved `.h5` files; none of the new fields are persisted (they're derived on demand from the already-persisted `mb_overlap`/`mo_occ`/`mo_energy`/`core_orb_idx`).
- `tests/test_h2o_kedge.py` is the project's one physics test file — add assertions to the existing `test_h2o_oxygen_kedge` function; do not create a new test file (per `CLAUDE.md`).
- Run `conda run -n pymbxas pytest tests/ -q` at the end of every task; all tests must pass before moving on.
- Run `conda run -n pymbxas python -c "import pymbxas"` after any import-order change to catch circular imports early (this codebase uses many function-local imports specifically to avoid them).
- After all coding tasks are done, Task 9 is a dedicated scientific-soundness review of the whole `mbxas/` implementation (not just this feature), separate from and in addition to the generic final whole-branch code review the subagent-driven-development process runs automatically afterward. Findings and improvement ideas from Task 9 are reported to the user, not auto-applied beyond what Task 9 itself fixes.

---

### Task 1: `spectator_occ_unocc_indices`

**Files:**
- Modify: `pymbxas/mbxas/mbxas.py` (add after `occ_unocc_indices`, before `build_A_K`, currently lines 13-34)
- Test: `tests/test_h2o_kedge.py` (append to `test_h2o_oxygen_kedge`, after the existing `occ_unocc_indices` assertions around line 270)

**Interfaces:**
- Produces: `spectator_occ_unocc_indices(gs_mo_occ_channel, fch_mo_occ_channel) -> (occ_idxs_gs, occ_idxs_fch, uno_idxs_fch)`, all `np.ndarray` of `int`. Raises `ValueError` if the channel's occupied count differs between GS and FCH (that channel had a core hole and isn't the spectator).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_h2o_kedge.py`, right after the existing block that ends with the `occ_unocc_indices` assertions (`assert np.array_equal(uno_fch_h, uno_fch), ...`):

```python
    from pymbxas.mbxas.mbxas import spectator_occ_unocc_indices

    spec_ch = 1 - ch
    occ_gs_spec_h, occ_fch_spec_h, uno_fch_spec_h = spectator_occ_unocc_indices(
        gs.mo_occ[spec_ch], fch.mo_occ[spec_ch])
    assert np.array_equal(occ_gs_spec_h, np.where(gs.mo_occ[spec_ch] == 1)[0]), \
        "spectator_occ_unocc_indices GS occupied indices mismatch (no core orbital should be removed)"
    assert np.array_equal(occ_fch_spec_h, np.where(fch.mo_occ[spec_ch] == 1)[0]), \
        "spectator_occ_unocc_indices FCH occupied indices mismatch"
    assert np.array_equal(uno_fch_spec_h, np.where(fch.mo_occ[spec_ch] == 0)[0]), \
        "spectator_occ_unocc_indices FCH unoccupied indices mismatch (no core-hole index should be dropped)"
    assert len(occ_gs_spec_h) == len(occ_fch_spec_h), \
        "spectator channel electron count should be unchanged between GS and FCH"

    with pytest.raises(ValueError):
        spectator_occ_unocc_indices(gs.mo_occ[ch], fch.mo_occ[ch])  # excited channel has a core hole
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n pymbxas pytest tests/test_h2o_kedge.py -v`
Expected: FAIL with `ImportError`/`AttributeError` — `spectator_occ_unocc_indices` does not exist yet.

- [ ] **Step 3: Implement**

In `pymbxas/mbxas/mbxas.py`, insert after `occ_unocc_indices` (after its closing `return` line, before `def build_A_K`):

```python
def spectator_occ_unocc_indices(gs_mo_occ_channel, fch_mo_occ_channel):
    """Occupied/unoccupied valence orbital indices for the spectator
    (non-excited) spin channel's own shake-up (mbxas.shakeup), the
    cross-spin contribution of mbxas-qe's spin_convolve_spectrum.

    Unlike occ_unocc_indices, there is no core orbital to remove and no
    core-hole index to drop from the unoccupied set: this channel keeps
    its full ground-state electron count in the FCH calculation, so its
    valence relaxation is a plain particle-hole excitation, not a
    core-hole one.

    Returns (occ_idxs_gs, occ_idxs_fch, uno_idxs_fch).
    """
    occ_idxs_gs  = np.where(gs_mo_occ_channel == 1)[0]
    occ_idxs_fch = np.where(fch_mo_occ_channel == 1)[0]
    if len(occ_idxs_gs) != len(occ_idxs_fch):
        raise ValueError(
            "Spectator channel electron count changed between GS and FCH "
            f"({len(occ_idxs_gs)} -> {len(occ_idxs_fch)}); this channel "
            "should be the one without a core hole. Pass the excited "
            "channel to occ_unocc_indices instead."
        )
    uno_idxs_fch = np.where(fch_mo_occ_channel == 0)[0]
    return occ_idxs_gs, occ_idxs_fch, uno_idxs_fch

```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n pymbxas pytest tests/test_h2o_kedge.py -v`
Expected: PASS

- [ ] **Step 5: Run the full suite and commit**

```bash
conda run -n pymbxas pytest tests/ -q
git add pymbxas/mbxas/mbxas.py tests/test_h2o_kedge.py
git commit -m "Add spectator_occ_unocc_indices for the non-excited channel's own shake-up"
```

---

### Task 2: `shakedown_only` on `shakeup_sticks`

**Files:**
- Modify: `pymbxas/mbxas/shakeup.py` (`shakeup_sticks`, lines 27-68)
- Test: `tests/test_h2o_kedge.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `shakeup_sticks(K, eps_occ, eps_unocc, order, shakedown_only=False)` — same return contract, now optionally filtered to `delta_e < 0`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_h2o_kedge.py`, right after the Task 1 assertions:

```python
    # shakedown_only filters to negative delta_e only ("shake-down",
    # mbxas-qe's kpoint_spectral_details.f90 convention), at the
    # single-order level
    e1_down, w1_down = shakeup_sticks(K_ch, eps_occ_ch, eps_unocc_ch, order=1, shakedown_only=True)
    assert np.all(e1_down < 0), "shakedown_only=True should keep only negative delta_e sticks"
    manual_mask = e1 < 0
    assert np.array_equal(np.sort(e1_down), np.sort(e1[manual_mask])), \
        "shakedown_only=True should match a manual delta_e<0 filter of the unfiltered order-1 sticks"
    assert np.array_equal(np.sort(w1_down), np.sort(w1[manual_mask])), \
        "shakedown_only=True should keep the matching weights unchanged"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n pymbxas pytest tests/test_h2o_kedge.py -v`
Expected: FAIL with `TypeError: shakeup_sticks() got an unexpected keyword argument 'shakedown_only'`.

- [ ] **Step 3: Implement**

In `pymbxas/mbxas/shakeup.py`, change the signature and docstring of `shakeup_sticks`:

```python
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
```

Then, immediately before the existing `return delta_e.ravel(), weight.ravel()` line, replace it with:

```python
    delta_e = delta_e.ravel()
    weight = weight.ravel()
    if shakedown_only:
        mask = delta_e < 0
        delta_e, weight = delta_e[mask], weight[mask]
    return delta_e, weight
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n pymbxas pytest tests/test_h2o_kedge.py -v`
Expected: PASS

- [ ] **Step 5: Run the full suite and commit**

```bash
conda run -n pymbxas pytest tests/ -q
git add pymbxas/mbxas/shakeup.py tests/test_h2o_kedge.py
git commit -m "Add shakedown_only diagnostic filter to shakeup_sticks"
```

---

### Task 3: `shakeup_sticks_by_order` (per-order dict, `shakeup_spectrum` refactored on top of it)

**Files:**
- Modify: `pymbxas/mbxas/shakeup.py` (`shakeup_spectrum`, lines 71-118)
- Test: `tests/test_h2o_kedge.py`

**Interfaces:**
- Consumes: `shakeup_sticks(K, eps_occ, eps_unocc, order, shakedown_only=False)` from Task 2.
- Produces: `shakeup_sticks_by_order(K, eps_occ, eps_unocc, order="auto", tol=0.01, shakedown_only=False) -> (sticks_by_order: dict[int, tuple[np.ndarray, np.ndarray]], orders_included: list[int])`. `shakeup_spectrum`'s existing public signature and return contract (`(delta_e, weight, orders_included)`) is unchanged — it now delegates to `shakeup_sticks_by_order`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_h2o_kedge.py`:

```python
    from pymbxas.mbxas.shakeup import shakeup_sticks_by_order

    sticks_by_order_2, orders_by_order_2 = shakeup_sticks_by_order(K_ch, eps_occ_ch, eps_unocc_ch, order=2)
    assert orders_by_order_2 == [1, 2], f"expected orders [1, 2], got {orders_by_order_2}"
    assert np.array_equal(sticks_by_order_2[1][0], e1) and np.array_equal(sticks_by_order_2[1][1], w1), \
        "shakeup_sticks_by_order order-1 entry should match shakeup_sticks(order=1)"
    assert np.array_equal(sticks_by_order_2[2][0], e2) and np.array_equal(sticks_by_order_2[2][1], w2), \
        "shakeup_sticks_by_order order-2 entry should match shakeup_sticks(order=2)"

    de2_from_dict = np.concatenate([sticks_by_order_2[k][0] for k in orders_by_order_2])
    dw2_from_dict = np.concatenate([sticks_by_order_2[k][1] for k in orders_by_order_2])
    assert np.array_equal(de2_from_dict, de2) and np.array_equal(dw2_from_dict, dw2), \
        "shakeup_spectrum(order=2) must equal the concatenation of shakeup_sticks_by_order's entries"

    sticks_by_order_down, _ = shakeup_sticks_by_order(K_ch, eps_occ_ch, eps_unocc_ch, order=1, shakedown_only=True)
    assert np.array_equal(np.sort(sticks_by_order_down[1][0]), np.sort(e1_down)), \
        "shakeup_sticks_by_order should forward shakedown_only to shakeup_sticks"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n pymbxas pytest tests/test_h2o_kedge.py -v`
Expected: FAIL — `shakeup_sticks_by_order` does not exist yet.

- [ ] **Step 3: Implement**

In `pymbxas/mbxas/shakeup.py`, replace the entire `shakeup_spectrum` function (lines 71-118) with:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n pymbxas pytest tests/test_h2o_kedge.py -v`
Expected: PASS. Also re-check the pre-existing `shakeup_spectrum` assertions from before this task (orders1/orders2/de2 length checks etc.) still pass — they must, since the public contract is unchanged.

- [ ] **Step 5: Run the full suite and commit**

```bash
conda run -n pymbxas pytest tests/ -q
git add pymbxas/mbxas/shakeup.py tests/test_h2o_kedge.py
git commit -m "Expose per-order shake-up sticks via shakeup_sticks_by_order"
```

---

### Task 4: `combine_cross_channel_sticks`

**Files:**
- Modify: `pymbxas/mbxas/shakeup.py` (add after `shakeup_sticks_by_order`/`shakeup_spectrum`, before `broaden_shakeup`)
- Test: `tests/test_h2o_kedge.py`

**Interfaces:**
- Consumes: dicts shaped like `shakeup_sticks_by_order`'s first return value.
- Produces: `combine_cross_channel_sticks(sticks_a_by_order, sticks_b_by_order, max_total_order) -> (delta_e, weight)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_h2o_kedge.py`:

```python
    from pymbxas.mbxas.shakeup import combine_cross_channel_sticks

    sticks_a = {1: (np.array([1.0, 2.0]), np.array([0.1, 0.2]))}
    sticks_b = {1: (np.array([3.0]), np.array([0.4]))}
    de_cross, dw_cross = combine_cross_channel_sticks(sticks_a, sticks_b, max_total_order=2)
    expected_e = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    expected_w = np.array([0.1, 0.2, 0.4, 0.04, 0.08])
    assert np.array_equal(np.sort(de_cross), np.sort(expected_e)), \
        f"combine_cross_channel_sticks energies mismatch: {sorted(de_cross)} vs {sorted(expected_e)}"
    assert np.allclose(np.sort(dw_cross), np.sort(expected_w), atol=1e-15), \
        f"combine_cross_channel_sticks weights mismatch: {sorted(dw_cross)} vs {sorted(expected_w)}"

    de_cap1, dw_cap1 = combine_cross_channel_sticks(sticks_a, sticks_b, max_total_order=1)
    assert len(de_cap1) == 3, \
        f"max_total_order=1 should keep 3 sticks (2 pure-a + 1 pure-b, dropping the (1,1) cross term), got {len(de_cap1)}"

    de_solo, dw_solo = combine_cross_channel_sticks(sticks_a, {}, max_total_order=1)
    assert np.array_equal(de_solo, sticks_a[1][0]) and np.array_equal(dw_solo, sticks_a[1][1]), \
        "combine_cross_channel_sticks with an empty spectator dict should reduce to the excited channel's own sticks"

    de_empty, dw_empty = combine_cross_channel_sticks({}, {}, max_total_order=0)
    assert len(de_empty) == 0 and len(dw_empty) == 0, \
        "combine_cross_channel_sticks with both dicts empty should return empty arrays"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n pymbxas pytest tests/test_h2o_kedge.py -v`
Expected: FAIL — `combine_cross_channel_sticks` does not exist yet.

- [ ] **Step 3: Implement**

In `pymbxas/mbxas/shakeup.py`, insert after `shakeup_spectrum`, before `def broaden_shakeup`:

```python
def combine_cross_channel_sticks(sticks_a_by_order, sticks_b_by_order, max_total_order):
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
            all_e.append((e_i[:, None] + e_j[None, :]).ravel())
            all_w.append((w_i[:, None] * w_j[None, :]).ravel())

    if not all_e:
        return np.empty(0), np.empty(0)
    return np.concatenate(all_e), np.concatenate(all_w)

```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n pymbxas pytest tests/test_h2o_kedge.py -v`
Expected: PASS

- [ ] **Step 5: Run the full suite and commit**

```bash
conda run -n pymbxas pytest tests/ -q
git add pymbxas/mbxas/shakeup.py tests/test_h2o_kedge.py
git commit -m "Add combine_cross_channel_sticks for cross-spin shake-up combination"
```

---

### Task 5: Wire cross-spin into `Spectra` and `PySCF_mbxas`

**Files:**
- Modify: `pymbxas/spectra.py` (`_shakeup_sticks` and surrounding block ~lines 351-410; `get_mbxas_spectra` ~lines 302-336)
- Modify: `pymbxas/calculators/pyscf.py` (`get_mbxas_spectra`, lines 373-396)
- Test: `tests/test_h2o_kedge.py`

**Interfaces:**
- Consumes: `shakeup_sticks_by_order`, `combine_cross_channel_sticks` (Tasks 3-4), `spectator_occ_unocc_indices` (Task 1), `build_A_K`/`occ_unocc_indices` (existing).
- Produces:
  - `Spectra._shakeup_sticks_by_order(self, order, channel, tol) -> dict[int, tuple[np.ndarray, np.ndarray]]` (eV), new cache `self._shakeup_cache_by_order`.
  - `Spectra._spectator_shakeup_sticks(self, order, tol) -> dict[int, tuple[np.ndarray, np.ndarray]]` (eV), new cache `self._spectator_shakeup_cache`.
  - `Spectra._combined_shakeup_sticks(self, shakeup_order, spectator_order, max_total_order, tol, shakedown_only) -> (delta_e_ev, weight)`.
  - `Spectra.get_mbxas_spectra(..., spectator_order=None, max_total_order=None, shakedown_only=False)`.
  - `Spectra.get_shakeup_spectrum(..., spectator_order=None, max_total_order=None, shakedown_only=False)` — raises `ValueError` if `spectator_order is not None and channel is not None`.
  - `PySCF_mbxas.get_mbxas_spectra(..., spectator_order=None, max_total_order=None, shakedown_only=False)`.
- `Spectra._shakeup_sticks` (existing) keeps its exact current signature, cache attribute name (`_shakeup_cache`), and return contract — refactored internally to call `_shakeup_sticks_by_order`, but observably unchanged.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_h2o_kedge.py` (after the existing shake-up block, following the already-defined `E_none`/`I_none`/`E_sk`/`I_sk` variables):

```python
    from pymbxas.mbxas.shakeup import combine_cross_channel_sticks as _combine_cc

    spec_ch2 = 1 - spectra_fields._channel
    spectator_sticks = spectra_fields._spectator_shakeup_sticks(order=1, tol=0.01)
    assert set(spectator_sticks.keys()) <= {1}, \
        f"spectator_shakeup_sticks(order=1) should only include order 1, got {set(spectator_sticks.keys())}"
    spec_e1, spec_w1 = spectator_sticks[1]
    assert np.all(np.isfinite(spec_e1)) and np.all(np.isfinite(spec_w1)), \
        "spectator channel order-1 shake-up sticks contain non-finite values"
    assert np.all(spec_w1 >= 0), "spectator channel shake-up weights must be non-negative"

    # spectator_order=None/max_total_order=None must remain byte-identical
    # to the pre-cross-spin behavior already exercised above
    E_none2, I_none2 = spectra_fields.get_mbxas_spectra(erange=[520, 560], sigma=0.5)
    assert np.array_equal(E_none2, E_none) and np.array_equal(I_none2, I_none), \
        "spectator_order=None regression: get_mbxas_spectra output changed"
    E_sk2, I_sk2 = spectra_fields.get_mbxas_spectra(erange=[520, 560], sigma=0.5, shakeup_order=1)
    assert np.array_equal(E_sk2, E_sk) and np.array_equal(I_sk2, I_sk), \
        "spectator_order=None regression: get_mbxas_spectra(shakeup_order=1) output changed"

    # spectator_order alone (shakeup_order=None) must apply a correction --
    # a spectator-only cross term reduces to that channel's own shake-up
    E_bare, I_bare = spectra_fields.get_mbxas_spectra(erange=[520, 560], sigma=0.5)
    E_spec_only, I_spec_only = spectra_fields.get_mbxas_spectra(
        erange=[520, 560], sigma=0.5, spectator_order=1)
    assert E_spec_only.shape == E_bare.shape
    if spec_w1.sum() > 0:
        assert not np.allclose(I_spec_only, I_bare), \
            "spectator_order=1 should change the spectrum when the spectator channel has nonzero shake-up mass"
    assert np.trapezoid(I_spec_only, E_spec_only) == pytest.approx(
        np.trapezoid(I_bare, E_bare), rel=0.1), \
        "spectator-only shake-up convolution should approximately conserve total integrated intensity"

    # combining both channels must agree with a manual combine_cross_channel_sticks call
    excited_sticks_by_order = spectra_fields._shakeup_sticks_by_order(1, None, 0.01)
    de_manual, dw_manual = _combine_cc(excited_sticks_by_order, spectator_sticks, max_total_order=2)
    de_from_spectra, dw_from_spectra = spectra_fields._combined_shakeup_sticks(1, 1, None, 0.01, False)
    assert np.array_equal(np.sort(de_from_spectra), np.sort(de_manual)) and \
           np.allclose(np.sort(dw_from_spectra), np.sort(dw_manual), atol=1e-15), \
        "_combined_shakeup_sticks(shakeup_order=1, spectator_order=1) disagrees with a manual combine_cross_channel_sticks call"

    # spectator_order combined with an explicit channel is a conflict --
    # channel identity is fixed by the cross-channel combination itself
    with pytest.raises(ValueError):
        spectra_fields.get_shakeup_spectrum(order=1, channel=spec_ch2, spectator_order=1)

    # PySCF_mbxas.get_mbxas_spectra must forward the new parameters and
    # agree with Spectra's own output, same pattern as the existing
    # shakeup_order agreement check
    E_pyscf_spec, I_pyscf_spec = obj.get_mbxas_spectra(
        "O", erange=[520, 560], sigma=0.5, spectator_order=1)
    assert np.array_equal(E_pyscf_spec, E_spec_only) and np.allclose(I_pyscf_spec, I_spec_only, atol=1e-12), \
        "PySCF_mbxas.get_mbxas_spectra(spectator_order=1) disagrees with Spectra.get_mbxas_spectra"

    # shakedown_only must not raise and must never increase total mass
    E_shakedown, I_shakedown = spectra_fields.get_mbxas_spectra(
        erange=[520, 560], sigma=0.5, shakeup_order=1, shakedown_only=True)
    assert np.all(np.isfinite(I_shakedown))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n pymbxas pytest tests/test_h2o_kedge.py -v`
Expected: FAIL — `_spectator_shakeup_sticks`/`_combined_shakeup_sticks`/`_shakeup_sticks_by_order` don't exist yet, and `get_mbxas_spectra` doesn't accept `spectator_order`.

- [ ] **Step 3: Implement**

In `pymbxas/spectra.py`, keep the existing `_shakeup_sticks` method's signature, docstring, and cache (`self._shakeup_cache`) exactly as they are, but replace its body so it delegates to a new `_shakeup_sticks_by_order` helper. Replace the current `_shakeup_sticks` method (and add the new helpers around it) with:

```python
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

            _, _, K = build_A_K(self._mb_overlap[channel], occ_idxs_fch,
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

            _, _, K = build_A_K(self._mb_overlap[channel], occ_idxs_fch,
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

            delta_e, weight = combine_cross_channel_sticks(sticks_a, sticks_b, max_total_order)

        if shakedown_only:
            mask = delta_e < 0
            delta_e, weight = delta_e[mask], weight[mask]

        return delta_e, weight
```

Then update `get_mbxas_spectra`'s signature and shake-up branch (replace lines 302-336):

```python
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
```

Also update `get_shakeup_spectrum`'s signature and body:

```python
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
```

In `pymbxas/calculators/pyscf.py`, update `get_mbxas_spectra` (lines 373-396):

```python
    def get_mbxas_spectra(self, ato_idx, axis=None, sigma=0.5, npoints=3001, tol=0.01,
                          erange=None, shakeup_order=None, spectator_order=None,
                          max_total_order=None, shakedown_only=False):

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
                                                      shakeup_order=shakeup_order,
                                                      spectator_order=spectator_order,
                                                      max_total_order=max_total_order,
                                                      shakedown_only=shakedown_only)
            intensity_sum = intensity if intensity_sum is None else intensity_sum + intensity

        return energy, intensity_sum
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n pymbxas pytest tests/test_h2o_kedge.py -v`
Expected: PASS. Re-check every pre-existing assertion involving `_shakeup_cache`/`get_shakeup_spectrum`/`get_mbxas_spectra` still passes unchanged (the cache-identity check, the explicit-`channel` check, the `shakeup_order=None`/`shakeup_order=1` agreement checks).

- [ ] **Step 5: Run the full suite and commit**

```bash
conda run -n pymbxas pytest tests/ -q
conda run -n pymbxas python -c "import pymbxas"
git add pymbxas/spectra.py pymbxas/calculators/pyscf.py tests/test_h2o_kedge.py
git commit -m "Wire spectator-channel cross-spin shake-up through Spectra and PySCF_mbxas"
```

---

### Task 6: `get_shakeup_summary` cross-spin/shake-down + plotting support

**Files:**
- Modify: `pymbxas/spectra.py` (add `import logging` near the top; `get_shakeup_summary`, ~lines 412-447)
- Modify: `pymbxas/plotting.py` (`plot_shakeup_summary`)
- Test: `tests/test_h2o_kedge.py`

**Interfaces:**
- Consumes: `Spectra._combined_shakeup_sticks`, `Spectra.get_mbxas_spectra`, `Spectra.get_shakeup_spectrum` (Task 5).
- Produces: `Spectra.get_shakeup_summary(order=2, sigma=0.5, npoints=3001, erange=None, tol=0.01, spectator_order=None, max_total_order=None, shakedown_only=False)` — return dict gains a `"cross"` key in `"spectra"`/`"integrated"` when `spectator_order` is given, and always gains `"shakedown_fraction"`. `plot_shakeup_summary` renders a `"cross"` entry if present, unchanged otherwise.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_h2o_kedge.py`, after the existing `get_shakeup_summary`/plotting assertions:

```python
    summary2 = spectra_fields.get_shakeup_summary(order=1, sigma=0.5, erange=[520, 560])
    assert "shakedown_fraction" in summary2, "get_shakeup_summary should report shakedown_fraction"
    assert 0.0 <= summary2["shakedown_fraction"] <= 1.0, \
        f"shakedown_fraction should be a probability fraction in [0, 1], got {summary2['shakedown_fraction']}"

    summary_cross = spectra_fields.get_shakeup_summary(
        order=1, sigma=0.5, erange=[520, 560], spectator_order=1)
    assert set(summary_cross["spectra"].keys()) == {0, 1, "cross"}, \
        f"spectator_order should add a 'cross' entry, got {set(summary_cross['spectra'].keys())}"
    assert np.array_equal(summary_cross["spectra"][0], summary2["spectra"][0]), \
        "spectator_order should not change the existing bare spectrum entry"
    assert np.array_equal(summary_cross["spectra"][1], summary2["spectra"][1]), \
        "spectator_order should not change the existing order-1 spectrum entry"

    E_cross_direct, I_cross_direct = spectra_fields.get_mbxas_spectra(
        erange=[520, 560], sigma=0.5, shakeup_order=1, spectator_order=1)
    assert np.array_equal(summary_cross["energy"], E_cross_direct)
    assert np.array_equal(summary_cross["spectra"]["cross"], I_cross_direct), \
        "get_shakeup_summary's 'cross' entry should match a direct get_mbxas_spectra(shakeup_order=order, spectator_order=...) call"
    assert summary_cross["integrated"]["cross"] == pytest.approx(np.trapezoid(I_cross_direct, E_cross_direct)), \
        "get_shakeup_summary's cross integrated intensity mismatch"

    # shakedown_fraction warning: temporarily seed the real (channel, 1, tol)
    # cache entry with a synthetic negative-heavy stick set to exercise the
    # warning path without needing a molecule that actually shakes down
    warn_cache_key = (spectra_fields._channel, 1, 0.01)
    real_cached_value = spectra_fields._shakeup_cache[warn_cache_key]
    spectra_fields._shakeup_cache[warn_cache_key] = (
        np.array([-10.0, 3.0]), np.array([0.9, 0.1]), [1])

    class _RecordCollector2(logging.Handler):
        def __init__(self):
            super().__init__()
            self.records = []
        def emit(self, record):
            self.records.append(record)

    collector2 = _RecordCollector2()
    spectra_logger = logging.getLogger("pymbxas.spectra")
    spectra_logger.addHandler(collector2)
    spectra_logger.setLevel(logging.WARNING)
    try:
        summary_warn = spectra_fields.get_shakeup_summary(
            order=1, sigma=0.5, erange=[520, 560], tol=0.01)
    finally:
        spectra_logger.removeHandler(collector2)
        spectra_fields._shakeup_cache[warn_cache_key] = real_cached_value

    assert summary_warn["shakedown_fraction"] == pytest.approx(0.45), \
        f"expected shakedown_fraction 0.9/(0.9+0.1+1)=0.45 with the seeded sticks, got {summary_warn['shakedown_fraction']}"
    assert any("shake-down" in r.getMessage() for r in collector2.records), \
        "get_shakeup_summary should warn when shakedown_fraction exceeds tol"

    # plot_shakeup_summary must also handle a summary with a "cross" entry
    fig_cross, axes_cross = plot_shakeup_summary(summary_cross, show_probability=True)
    assert len(axes_cross) == 2
    _plt.close(fig_cross)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n pymbxas pytest tests/test_h2o_kedge.py -v`
Expected: FAIL — `get_shakeup_summary` doesn't accept `spectator_order`, no `"shakedown_fraction"` key, `plot_shakeup_summary` crashes on a `"cross"` key with `TypeError` (mixed `int`/`str` comparison in `max()`/`sorted()`).

- [ ] **Step 3: Implement**

In `pymbxas/spectra.py`, add `import logging` near the top (with the other stdlib imports) and, after the class's other module-level setup, add `logger = logging.getLogger(__name__)` (module scope, alongside `Ha = units.Ha`).

Replace `get_shakeup_summary` (lines 412-447) with:

```python
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
```

In `pymbxas/plotting.py`, replace the body of `plot_shakeup_summary` from `energy = summary["energy"]` through the main-plot loop with:

```python
    energy = summary["energy"]
    spectra = summary["spectra"]
    order_keys = sorted(k for k in spectra if isinstance(k, int))
    max_order = max(order_keys) if order_keys else 0
    has_cross = "cross" in spectra
    plot_keys = order_keys + (["cross"] if has_cross else [])

    if show_probability:
        fig, (ax_main, ax_prob) = plt.subplots(
            2, 1, constrained_layout=True,
            gridspec_kw={"height_ratios": [3, 1]})
        axes = [ax_main, ax_prob]
    else:
        fig, ax_main = plt.subplots(constrained_layout=True)
        axes = [ax_main]

    labels = {0: "no shake-up"}
    labels.update({k: "shakeup_order={}".format(k) for k in order_keys if k > 0})
    styles = {0: dict(color="crimson", lw=1.8)}
    for k in order_keys:
        if k > 0:
            styles[k] = dict(lw=1.6, ls="--" if k == 1 else ":")
    if has_cross:
        labels["cross"] = "cross-spin + shake-up"
        styles["cross"] = dict(color="teal", lw=1.6, ls="-.")

    for k in plot_keys:
        ax_main.plot(energy, spectra[k], label=labels[k], **styles[k])
```

Leave the rest of the function (axis limits, labels, legend, probability panel) unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n pymbxas pytest tests/test_h2o_kedge.py -v`
Expected: PASS

- [ ] **Step 5: Run the full suite and commit**

```bash
conda run -n pymbxas pytest tests/ -q
git add pymbxas/spectra.py pymbxas/plotting.py tests/test_h2o_kedge.py
git commit -m "Add cross-spin and shake-down reporting to get_shakeup_summary"
```

---

### Task 7: Real-molecule demo script (Cu complex, write only)

**Files:**
- Create: `/home/roncofaber/WORK/MBXAS/shakeup/shakeup/cu_complex_shakeup_compare.py`

**Interfaces:**
- Consumes: `PySCF_mbxas.get_mbxas_spectra(..., spectator_order=..., shakedown_only=...)`, `Spectra.get_shakeup_summary(..., spectator_order=...)`, `pymbxas.plotting.plot_shakeup_summary` (Tasks 5-6).
- Produces: nothing consumed by later tasks. This script is never executed as part of this plan — it is a deliverable for the user to run later themselves.

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
"""
Copper K-edge MBXAS spectrum of an open-shell Cu(II) complex, showcasing
cross-spin shake-up (spectator_order) and shake-down (shakedown_only)
alongside the existing single-channel shake-up (shakeup_order).

Source structure: ~/Downloads/17949957.mol (heavy atoms only, no explicit
H) -- RDKit adds hydrogens before conversion to an ASE Atoms object. This
molecule was picked over H2O/N2 specifically because it has an
intrinsically unpaired spin (Cu(II), d9), so the spectator channel's own
valence relaxation is a real physical effect here rather than a
near-symmetric echo of the excited channel.

RDKit is not a pymbxas dependency; install it yourself before running
this script, e.g. `conda install -c conda-forge rdkit`.

Charge/spin/basis/excited-atom choices below are starting points, not
verified against this specific complex -- check them before running.
"""

import os

import numpy as np
import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

from rdkit import Chem
import ase

from pymbxas.calculators.pyscf import PySCF_mbxas
from pymbxas.plotting import plot_shakeup_summary

HERE = os.path.dirname(os.path.abspath(__file__))
MOL_FILE = os.path.expanduser("~/Downloads/17949957.mol")

SIGMA = 0.6
ORDER = 2
SPECTATOR_ORDER = 1
PRE_EDGE = 4.0
POST_EDGE = 40.0

rdmol = Chem.MolFromMolFile(MOL_FILE, sanitize=False, removeHs=False)
rdmol = Chem.AddHs(rdmol, addCoords=True)
conf = rdmol.GetConformer()

symbols = [atom.GetSymbol() for atom in rdmol.GetAtoms()]
positions = conf.GetPositions()
structure = ase.Atoms(symbols=symbols, positions=positions)

# Cu(II), d9: one unpaired electron. Neutral complex (two monoanionic
# bidentate ligands balance the +2 metal charge). Verify against the
# actual ligand before running.
CHARGE = 0
SPIN = 1

obj = PySCF_mbxas(
    structure   = structure,
    charge      = CHARGE,
    spin        = SPIN,
    xc          = "b3lyp",
    basis       = "def2-svpd",
    calc_type   = "UKS",
    do_xch      = True,
    loc_type    = "ibo",
    target_dir  = HERE,
    xas_verbose = 3,
    dft_verbose = 3,
    dft_output  = False,
    dft_logfile = "pyscf.log",
    xas_logfile = "pymbxas.log",
    save        = True,
    save_name   = "cu_complex.h5",
    gpu         = True,
)

# Cu K-edge (~8979 eV) is far above where non-relativistic DFT is
# quantitatively reliable -- this run demonstrates the code path
# (cross-spin + shake-down), not a publication-quality spectrum.
obj.kernel("Cu")

spectra = obj.to_spectra(0)

onset = spectra.energies.min()
erange = [onset - PRE_EDGE, onset + POST_EDGE]

summary = spectra.get_shakeup_summary(
    order=ORDER, sigma=SIGMA, erange=erange,
    spectator_order=SPECTATOR_ORDER)

print("Excited atom               : Cu#{}".format(spectra.exc_idx))
print("First transition           : {:.2f} eV".format(onset))
print("Plot range                 : [{:.1f}, {:.1f}] eV".format(*erange))
print("Shake-up orders included   : {}".format(summary["probability"][2]))
print("Shake-down fraction        : {:.4f}".format(summary["shakedown_fraction"]))
for k in sorted(summary["integrated"], key=str):
    if k == 0:
        label = "no shake-up"
    elif k == "cross":
        label = "cross-spin (shakeup={}, spectator={})".format(ORDER, SPECTATOR_ORDER)
    else:
        label = "shakeup_order={}".format(k)
    print("  integrated intensity ({:<40s}): {:.6e}".format(label, summary["integrated"][k]))

# shake-down-only diagnostic: isolate delta_e < 0 combinations
_, I_shakedown_only = spectra.get_mbxas_spectra(
    erange=erange, sigma=SIGMA, shakeup_order=ORDER,
    spectator_order=SPECTATOR_ORDER, shakedown_only=True)
print("Shake-down-only integrated intensity: {:.6e}".format(
    np.trapezoid(I_shakedown_only, summary["energy"])))

fig, axes = plot_shakeup_summary(summary)
axes[0].set_title(r"Cu(II) complex, copper K-edge, B3LYP/def2-SVPD")

oname = os.path.join(HERE, "cu_complex_shakeup_compare.png")
fig.savefig(oname, dpi=200)
print("Figure saved to            : {}".format(oname))

if os.environ.get("DISPLAY"):
    import matplotlib.pyplot as plt
    plt.show()
```

- [ ] **Step 2: Do not run it**

This script is a deliverable for the user, not verified by this plan. Do not execute it (RDKit is not installed in the `pymbxas` env, and the user explicitly asked not to run this calculation).

- [ ] **Step 3: Commit**

```bash
git -C /home/roncofaber/software/pymbxas add -A -- /home/roncofaber/WORK/MBXAS/shakeup/shakeup/cu_complex_shakeup_compare.py 2>/dev/null || true
```

This file lives outside the `pymbxas` git repository (`/home/roncofaber/WORK/MBXAS/shakeup/shakeup/`), matching where `n2_shakeup_compare.py`/`h2o_shakeup_compare.py` already live — nothing to commit in the `pymbxas` repo for this task.

---

### Task 8: Documentation

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `dev/method.md` (paragraphs starting "Spectator-spin determinant is omitted" and "One-body truncation, with an opt-in order-2 correction")
- Modify: `CLAUDE.md` (the `get_mbxas_spectra` gotcha bullet)

**Interfaces:** none (documentation only).

- [ ] **Step 1: Update `CHANGELOG.md`**

Under `## [Unreleased]` / `### Added`, add:

```markdown
- `spectator_order` on `get_mbxas_spectra` adds shake-up satellites from the non-excited spin channel's own valence relaxation
- `shakedown_only` isolates negative-energy shake-down combinations in shake-up spectra
- `get_shakeup_summary` now reports a `shakedown_fraction` for the shake-up probability distribution
```

- [ ] **Step 2: Update `dev/method.md`**

Replace the sentence in the "Spectator-spin determinant is omitted" paragraph that currently reads (starting "...but that codebase goes further..."):

> "...but that codebase goes further: `spec.f90`'s `spin_convolve_spectrum` (called from `mbxas_spectra.f90` on the `noci-kpoint-shirley*` branches) energy-convolves each spin channel's own order-resolved shake-up spectrum with the *other* channel's relaxation-overlap spectrum, so the spectator channel contributes genuine satellite structure, not just a constant scale factor. See "One-body truncation" below."

with:

> "...but that codebase goes further: `spec.f90`'s `spin_convolve_spectrum` (called from `mbxas_spectra.f90` on the `noci-kpoint-shirley*` branches) energy-convolves each spin channel's own order-resolved shake-up spectrum with the *other* channel's relaxation-overlap spectrum, so the spectator channel contributes genuine satellite structure, not just a constant scale factor. PyMBXAS implements this: `spectator_order` on `get_mbxas_spectra`/`get_shakeup_spectrum`/`get_shakeup_summary` combines the excited channel's own shake-up (built from `mbxas.mbxas.occ_unocc_indices`) with the spectator channel's own shake-up (built from the equivalent `mbxas.mbxas.spectator_occ_unocc_indices`, which has no core orbital to remove or core-hole index to drop) via `mbxas.shakeup.combine_cross_channel_sticks` -- the outer sum of electron-hole energies and outer product of weights for every `(i, j)` order pair under a `max_total_order` cap, the discrete-stick form of convolving two independent probability spectra. See "One-body truncation" below."

Replace, in the "One-body truncation, with an opt-in order-2 correction" paragraph, the sentence:

> "Cross-spin convolution (the *other*, non-excited channel's own shake-up, via `spin_convolve_spectrum` in `mbxas-qe`'s `spec.f90`) is not implemented; `Spectra._shakeup_sticks`/`get_shakeup_spectrum` accept an explicit `channel` argument specifically so that extension needs no signature change later."

with:

> "Cross-spin convolution (the *other*, non-excited channel's own shake-up, via `spin_convolve_spectrum` in `mbxas-qe`'s `spec.f90`) is implemented as `spectator_order`/`max_total_order` on `get_mbxas_spectra`/`get_shakeup_spectrum`/`get_shakeup_summary`; `spectator_order=None` (the default) reproduces the pre-cross-spin behavior exactly. `shakedown_only` isolates combinations with negative electron-hole energy ("shake-down", `mbxas-qe`'s `kpoint_spectral_details.f90` naming); `get_shakeup_summary` always reports the corresponding `shakedown_fraction` and warns above `tol`."

- [ ] **Step 3: Update `CLAUDE.md`**

In the gotcha bullet starting "`get_mbxas_spectra` exists on two classes plus a free function", update:

> "...`Spectra` (takes `el_label`, `shakeup_order`) is the one real implementation..."

to:

> "...`Spectra` (takes `el_label`, `shakeup_order`, `spectator_order`, `max_total_order`, `shakedown_only`) is the one real implementation..."

- [ ] **Step 4: Commit**

```bash
git add CHANGELOG.md dev/method.md CLAUDE.md
git commit -m "Document cross-spin shake-up and shake-down"
```

---

### Task 9: Final scientific review of `mbxas/`

**Files:**
- Create: `docs/superpowers/reviews/2026-08-24-mbxas-scientific-review.md`

**Interfaces:** none — this task produces a written report, not code consumed by later tasks.

- [ ] **Step 1: Review the implementation**

Read, in full: `pymbxas/mbxas/mbxas.py`, `pymbxas/mbxas/shakeup.py`, `pymbxas/mbxas/broaden.py`, and the physics-relevant parts of `pymbxas/spectra.py` (`get_mbxas_spectra`, `_shakeup_sticks`, `_shakeup_sticks_by_order`, `_spectator_shakeup_sticks`, `_combined_shakeup_sticks`, `get_shakeup_spectrum`, `get_shakeup_summary`, `amp2int`) and `pymbxas/calculators/pyscf.py`'s `get_mbxas_spectra`. Read `dev/method.md` in full.

Check, and record a finding for any that fails:
1. Every documented invariant in `dev/method.md`'s "Method invariants" section and `CLAUDE.md`'s "Method invariants" section still holds (spin channel convention, core-hole index location, GS orbital indexing by MO number, Hartree-internally/eV-at-boundary units, XCH alignment, transition-dipole origin independence, spectator channel omitted from the *amplitude* specifically — the new shake-up cross term is a separate, downstream correction and must not be confused with this).
2. `spectator_occ_unocc_indices`'s no-removal, no-drop indexing is actually correct for a spin channel with no core hole (cross-check against `occ_unocc_indices`'s reasoning).
3. `combine_cross_channel_sticks`'s outer-sum/outer-product identity is the correct discrete form of convolving two independent probability distributions — verify by hand on a small numeric example that total probability mass is conserved appropriately (i.e. that summing weights over all `(i,j)` pairs up to a given cap, plus the implicit trivial term, gives the expected total).
4. The two-level `shakedown_only` design (per-channel filter in `mbxas.shakeup.shakeup_sticks`, whole-combination filter in the `Spectra` API) is self-consistent and each level's docstring correctly describes which one it is.
5. Every "byte-identical when off" claim (both new `Spectra`/`PySCF_mbxas` parameters at their defaults) actually holds by re-reading the merged code's control flow, not by re-trusting each task's own tests in isolation.
6. `plot_shakeup_summary`'s handling of the `"cross"` key has no edge case left broken (an empty `order_keys`, a summary with only order 0, etc.).
7. Skim `pymbxas/mbxas/mbxas.py` and `pymbxas/mbxas/shakeup.py` end to end for any other correctness issue unrelated to this feature, encountered incidentally while reading.

- [ ] **Step 2: Write the report**

Create `docs/superpowers/reviews/2026-08-24-mbxas-scientific-review.md` with sections `## Findings` (one entry per issue found: file, location, description, severity, and — only if it is an unambiguous, narrowly-scoped bug — a fix applied directly with the commit noted) and `## Suggestions (not applied)` (anything that would improve the implementation but is a judgment call, reported for the user to decide, not applied). If nothing is found in a section, say so explicitly rather than omitting the section.

- [ ] **Step 3: Commit**

```bash
conda run -n pymbxas pytest tests/ -q
git add docs/superpowers/reviews/2026-08-24-mbxas-scientific-review.md
# also add any files touched by a narrowly-scoped fix from Step 1, if any
git commit -m "Add scientific review of the mbxas cross-spin/shake-down implementation"
```

After this task, report the review's findings and suggestions directly to the user in chat — this is the point of the task, and the report file alone does not satisfy it.
