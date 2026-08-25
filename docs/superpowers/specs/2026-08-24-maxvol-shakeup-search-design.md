# Maxvol-based shake-up configuration search

## Problem

`mbxas/shakeup.py` prunes order-2 shake-up combinatorics by a magnitude-rank-and-grow heuristic (`_shakeup_sticks_order2`): sort all valence->conduction singles by `|K|**2`, geometrically grow an active prefix, form 2x2 minors within it, stop by mass convergence. It is hard-capped at `MAX_IMPLEMENTED_ORDER=2` because brute-force k-tuple enumeration is combinatorial and nothing prunes it past pairs.

`mbxas-qe`'s reference implementation (`QE/SHIRLEY/src/maxvol_multi_mod.f90`, invoked from `overlap_binding.f90:722-776`) does not enumerate by magnitude-of-singles at all. It runs the maxvol algorithm directly on the FCH/GS overlap matrix: a greedy swap search that finds the locally maximal-determinant `nocc`-orbital subset, then repeats with exclusions (`degree` parameter) to find other local maxima. Each local maximum found this way is a genuine multi-electron configuration (a "swap set" relative to the reference occupation) with no order cap and no reliance on ranking individual transitions first.

This spec ports that search strategy — not QE's SCF/MOM machinery, which stays untouched (see Non-goals).

## The unifying identity

For any k-swap configuration (k reference-occupied orbitals replaced by k virtual ones), Jacobi's complementary-minor theorem gives:

```
det(A_swapped) / det(A_reference) = ± det(K[rows_in, cols_out])
```

a plain k x k minor of the existing `K = A' @ inv(A)` matrix, for *any* k — not just k=2. This is exactly the formula `shakeup_sticks` already uses for order 2 (`weight = |det(K[c_combo, v_combo])|**2`), previously hardcoded to pairs via `itertools.combinations`.

Consequence: **maxvol's role here is purely combinatorial search** — deciding which `(rows_in, cols_out)` index-sets are worth evaluating. The weight formula itself is unchanged, already validated (`dev/method.md`), and requires no new physics. We are not porting QE's determinant-tracking bookkeeping (`current_det`, `det_values`); once a configuration's index sets are known, its weight is a direct `np.linalg.det` call on the corresponding submatrix of `K`, same as today.

## Scope

- **In scope**: replacing the order>=2 search strategy in `mbxas/shakeup.py` (currently `_shakeup_sticks_order2`, hard-capped at order 2) with a maxvol-based swap search, generalized to any order.
- **Out of scope, deliberately**:
  - SCF/MOM (`pyscf.scf.addons.mom_occ`) is untouched. It determines the actual converged FCH density; maxvol only ever runs post-hoc on the frozen, converged `K`.
  - `AMat`/`ADet`/`KMat` construction (`mbxas.py:build_A_K`) is untouched — `occ_idxs_fch` remains exactly what MOM/`mo_occ` converged to. No override, no orbital-correspondence reassignment.
  - Order-1 sticks stay exactly as today: full exact enumeration (`n_occ * n_unocc`, cheap, no combinatorics to prune). Maxvol is not used to (re)discover order-1 — it would regress from "exact, all singles" to "only the few maxvol's search happens to surface."
  - Cross-channel combination (`_prune_outer_product`, `combine_cross_channel_sticks`) is unchanged — it is already exact-mass-based and does not need maxvol.
  - No dependency on `maxvolpy` (evaluated and rejected: no clean PyPI release, install is git-clone-from-Bitbucket with an optional Cython compile step, and it only implements the single-submatrix search — the multi-degree exclusion-based exploration we actually need is QE-specific logic we'd have to write regardless). Hand-rolled on top of existing `numpy`.
  - "Stick" terminology is unchanged — it is standard spectroscopy jargon (a single delta-function transition pre-broadening) inherited directly from `mbxas-qe`'s own `spec.f90:stick_to_spec`, not a name we introduced arbitrarily.

## Algorithm

New function, replacing `_shakeup_sticks_order2`:

```python
def _maxvol_shakeup_configs(K, eps_occ, eps_unocc, tol):
    """Swap-search shake-up configurations via maxvol on K.

    Returns {order: (delta_e, weight)} for order >= 2 configurations
    discovered by the search (order 1 is handled separately, by exact
    enumeration, and is not this function's job).
    """
```

Steps:

1. **Reference state** = the actual occupied valence set (0 swaps = "no shake-up"). Its maxvol "B matrix" (`A' @ inv(A_r)` restricted to non-pivot rows) *is* `K` itself for this specific starting point — no extra inversion needed to seed the search, since `K` is already computed by `build_A_K`.
2. **Standard maxvol iteration**: find `(i, j)` maximizing `|K(i,j)|` where `i` indexes a virtual (candidate-in) orbital and `j` indexes a currently-occupied (candidate-out) orbital. If `|K(i,j)| > 1 + tol`, swap orbital `j` out for orbital `i`; recompute the swapped submatrix's inverse via a **rank-1 (Sherman-Morrison) update**, not a full re-inversion (QE's `zgetrf`/`zgetri`-per-swap is the thing we deliberately do *not* copy — Sherman-Morrison on a single-row change is the established approach, e.g. what `maxvolpy`'s own core loop does). Iterate to convergence (no `|K(i,j)| > 1+tol` remains). This typically converges in 0-1 swaps if MOM already found a near-optimal occupation, which is the expected case.
3. **Multi-degree exploration** (QE's `maxvol_multi`): from B-matrix entries that were large but did not trigger a swap, seed additional constrained searches — excluding orbitals already used as pivots in previously-found configurations — to surface other local maxima. Each is a genuine alternate configuration (shake-up satellite).
4. **Order labeling**: for each discovered pivot set, `order = |symmetric difference from the reference set| / 2` (a k-swap changes k occupied and k virtual slots, symmetric difference size `2k`).
5. **Weight**: `|det(K[rows_in, cols_out])|**2` — the existing minor formula, generalized to whatever `rows_in`/`cols_out` sizes the discovered configuration has (no longer hardcoded to 2x2).
6. **Stopping**: adaptive, matching the existing `tol`/mass convention (`_prune_outer_product`, `_shakeup_sticks_order2`) — not QE's fixed `degree=10`. Keep exploring for additional local maxima while newly-found weight is still a non-negligible fraction of already-captured mass; stop once it drops below `tol` relative to the running total (order-1 mass, consistent with `shakeup_sticks_by_order`'s existing "auto" convergence rule).

## Integration

- `shakeup_sticks_by_order`: order 1 unchanged (exact enumeration). For order >= 2, call `_maxvol_shakeup_configs` once; bucket its results into the same `{order: (delta_e, weight)}` dict already returned today. Dedupe against the exact order-1 set for any 1-swap configs maxvol's search also happens to surface (it may, since the search isn't order-restricted — a 1-swap "config" found during multi-degree exploration is just a less-important single that the exact order-1 pass already has).
- `shakeup_sticks`: the `order > MAX_IMPLEMENTED_ORDER` guard and `MAX_IMPLEMENTED_ORDER = 2` constant are removed. Order becomes unbounded, limited only by the search's own convergence.
- `combine_cross_channel_sticks`, `_prune_outer_product`, `broaden_shakeup`, `convolve_shakeup`: unchanged. They already operate on the `{order: (delta_e, weight)}` contract this produces.
- `shakedown_only` filtering (`delta_e < 0`): applied as a post-filter exactly as today, no change needed — it's independent of how the configurations were found.

## Testing

Extends `tests/test_h2o_kedge.py` (per the existing "one calculation, add assertions, no new test files" rule):

- Regression: with the new search, order-2 mass for H2O should match (or very closely bound) the existing `_shakeup_sticks_order2` result — same underlying weight formula, different search strategy, so any large discrepancy indicates a bug in the swap search rather than a physical difference.
- Order-3 (or higher, if the search happens to find any for H2O/def2-SVPD) no longer raises `NotImplementedError` — assert it either produces a smaller-than-order-2 mass (physical expectation) or is simply absent (search found nothing beyond order 2), not that it errors.
- Unit-level check (can live in the test file per the existing single-file scope, or as an assertion within the physics test): for a hand-constructed small `K`, verify the swap search's discovered weight for a known 2-swap configuration matches `np.abs(np.linalg.det(K[rows_in, cols_out]))**2` computed independently — confirms the Jacobi complementary-minor identity holds in the actual swap-index bookkeeping, not just in derivation.
- Sherman-Morrison update correctness: after each swap, the incrementally-updated inverse should match a from-scratch `np.linalg.inv` on the new pivot submatrix within numerical tolerance (guards against a bookkeeping bug in the rank-1 update silently drifting).

## Documentation

- `dev/method.md`: update the shake-up section to describe the maxvol-based search and the Jacobi complementary-minor identity underpinning it; remove references to the order-2-only limitation and the magnitude-rank-and-grow heuristic it replaces.
- `CHANGELOG.md`: `### Changed` entry — shake-up satellite search now uses a maxvol-based configuration search instead of magnitude-pruned order-2-only combinatorics, and can surface order-3+ configurations. This changes computed shake-up spectra (per `CLAUDE.md`'s "a change that alters computed numbers always gets a Changed entry" rule), so state plainly that shake-up intensities may shift slightly and higher-order satellites may now appear.
