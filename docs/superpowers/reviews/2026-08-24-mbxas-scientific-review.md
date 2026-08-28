# Scientific Review: Cross-Spin Shake-Up and Shake-Down Implementation

> **Superseded for shake-up conclusions.** The deeper reference comparison in
> `2026-08-25-shakeup-scientific-review.md` found important differences from
> `mbxas-qe` that this self-consistency review missed. Retain this file as a
> historical record, not as the current scientific assessment.

**Date:** 2026-08-24  
**Reviewer:** Claude (Haiku 4.5)  
**Scope:** Physics and mathematical correctness of the entire `mbxas/` implementation, with focus on the new cross-spin shake-up and shake-down feature.

---

## Summary

A comprehensive review of `pymbxas/mbxas/mbxas.py`, `pymbxas/mbxas/shakeup.py`, `pymbxas/mbxas/broaden.py`, physics-relevant methods in `pymbxas/spectra.py` and `pymbxas/calculators/pyscf.py`, plus the new `plot_shakeup_summary` function. All documented method invariants from `dev/method.md` and `AGENTS.md` were verified to hold. The implementation is mathematically sound and passes all physics checks.

---

## Findings

### No Critical Issues Found

Every documented method invariant was verified to hold:

1. **Spin channel convention** (`channel=1` beta default) - correctly enforced throughout. Every orbital operation indexes `mo_coeff[channel]`, `mo_occ[channel]`, `mo_energy[channel]` without hardcoding channel 1. ✓

2. **Core-hole index location** - consistently uses `np.where(mo_occ[channel]==0)[0][0]` (line 117 of `mbxas.py`), never hardcoded 0. The virtual manifold correctly drops this index with `[1:]`. ✓

3. **GS orbital indexing by MO number** - correctly uses `np.setdiff1d(gs_occ_idxs, [core_orb_idx])` (line 30 of `mbxas.py`) to remove the excited core orbital by MO index, never by position in the occupied list. ✓

4. **Hartree-internal / eV-at-boundary units** - verified across all files:
   - `mbxas.py` returns energies in Hartree (line 135)
   - Spectra properties convert to eV via `Ha * self._energies` (line 94 of `spectra.py`)
   - Shake-up energies converted to eV: `Ha * e` (lines 384, 443)
   - Amplitude-to-intensity conversion uses Hartree energies (`energies_ha`, line 327)
   - All conversions consistent with unit formula `sigma(omega) ~ omega * |M|^2` with `omega` in Hartree. ✓

5. **XCH alignment** - formula verified at line 139 of `mbxas.py`:
   ```
   energies += xch_calc.e_tot - gs_calc.e_tot - np.min(energies)
   ```
   Correctly shifts FCH virtual eigenvalues so minimum sits at `E_XCH - E_GS`. ✓

6. **Transition dipole origin independence** - both orbitals come from FCH calculation:
   - Line 127: `chb_xmat = dipole_KS[channel][:, :, exc_orb_idx]` (FCH orbitals)
   - Line 129: `chb_xmat_uno = chb_xmat[:, uno_idxs_fch]` (FCH virtual orbitals)
   - Both from same `dipole_KS[channel]` block, therefore orthogonal (verified to 1e-15 in test line 109). ✓

7. **Spectator channel omitted from core amplitude** - CRITICAL: verified that only excited channel enters the amplitude:
   - Line 124: `AMat, ADet, KMat = build_A_K(mb_overlap[channel], ...)` (excited channel only)
   - Line 127: `dipole_KS[channel]` (excited channel only)
   - Line 132: `absorption = ADet*(chb_xmat_uno - (KMat @ chb_xmat_occ.T).T)` (excited channel only)
   - The spectator channel's MB overlap and dipoles are computed (lines 106-114) but not used in the amplitude. This is correct per the method invariant. The spectator channel contribution enters only through downstream cross-spin combination in `combine_cross_channel_sticks`, which is a separate correction and is properly isolated. ✓

8. **`spectator_occ_unocc_indices` correctness** - correctly handles spectator (non-excited) channel:
   - No core-hole removal: all GS occupied orbitals kept (line 49)
   - No core-hole index drop from unoccupied: uses full `np.where(...==0)[0]` without `[1:]` (line 58)
   - Electron count consistency check (lines 51-57) ensures this function is only called on the non-excited channel
   - Verified against `occ_unocc_indices` logic: former requires core-hole removal/dropping, latter forbids it. ✓

9. **`combine_cross_channel_sticks` discrete probability correctness** - verified by hand with concrete example:
   - Energies: outer sum `e_i[:, None] + e_j[None, :]` correctly adds independent electron-hole costs
   - Weights: outer product `w_i[:, None] * w_j[None, :]` correctly multiplies independent probabilities
   - Trivial (0,0) term excluded per design (probabilities already implicit in stick format)
   - Test case: Channel A with order-1 mass 1.0, Channel B with order-1 mass 1.0 → combined explicit mass 3.0 (two pure terms + one cross), conserved correctly. ✓

10. **Two-level `shakedown_only` design self-consistency** - verified:
    - **Level 1 (mbxas.shakeup)**: `shakeup_sticks` has `shakedown_only` parameter (line 27, docstring lines 34-38) that filters individual channel's combinations by `delta_e < 0`. Available for direct use.
    - **Level 2 (Spectra API)**: `_combined_shakeup_sticks` applies `shakedown_only` filter AFTER cross-channel combination (lines 474-476). This is the only shakedown filter exposed at the Spectra API level.
    - **Design rationale**: Per-channel filter in `shakeup_sticks_by_order` is available but not threaded through `_shakeup_sticks_by_order` (which is Spectra's internal method). Instead, Spectra filters the final combined result globally. This is self-consistent: choosing to filter before or after combination gives different physics results (different resonance manifolds), and the current design chooses to filter after (capturing combined resonance effects). Docstrings are accurate. ✓

11. **"Byte-identical when off" claims verified by control flow analysis**:
    - `get_mbxas_spectra(shakeup_order=None, spectator_order=None)`: Line 335 condition `shakeup_order is not None or spectator_order is not None` evaluates to False, convolve_shakeup NOT called, spectra returned unchanged. ✓
    - `get_shakeup_spectrum(spectator_order=None, max_total_order=None, shakedown_only=False)`: Line 507 condition `spectator_order is None and max_total_order is None and not shakedown_only` evaluates to True, calls `self._shakeup_sticks` directly (pre-cross-spin code path). ✓
    - `_combined_shakeup_sticks(spectator_order=None)`: Line 459 condition evaluates to True, returns via `self._shakeup_sticks` only (lines 461-462). ✓

12. **`plot_shakeup_summary` edge cases**:
    - Empty `order_keys`: Not possible; spectra dict always has at least key 0 (bare spectrum).
    - Summary with only order 0: `order_keys = [0]`, `plot_keys = [0]`, plots correctly.
    - `show_probability=False`: Creates single-axis figure, iterates over plot_keys, all labels and styles defined appropriately. ✓
    - `"cross"` key handling: Lines 29-49 correctly add cross label/style if present. No edge case broken. ✓

---

## Incidental Checks (No Issues Found)

Skim through `mbxas.py` and `shakeup.py` for unrelated correctness issues:

- **Gaussian broadening normalization** (broaden.py): `gaussian_broadening(x, sigma) = exp(-0.5*(x/sigma)^2) / (sigma * sqrt(2*pi))` is the standard normal kernel, confirmed to integrate to 1.0 numerically. ✓

- **Indexing robustness**: All fancy indexing uses `np.ix_` appropriately (lines 77, 79 in `mbxas.py`). No off-by-one errors detected. ✓

- **Empty-array handling**: `shakeup_sticks` correctly returns `np.empty(0)` when order > n_occ or order > n_unocc (line 59). `combine_cross_channel_sticks` correctly handles empty dicts and returns empty arrays when no combinations exist (lines 194-195). ✓

- **Error messages**: Descriptive and actionable (e.g., line 52-57 in `spectator_occ_unocc_indices` clearly states "this channel should be the one without a core hole"). ✓

---

## Suggestions (Not Applied)

These are judgment calls or design improvements, not bugs:

1. **Docstring clarity for `_combined_shakeup_sticks`**: The current docstring (lines 448-457) does not explicitly state WHEN shakedown_only is applied (i.e., after combination). Adding a sentence like "shakedown_only filters the final combined result after cross-channel combination" would align with the two-level design description in `dev/method.md`. This is not a bug (the code is correct), but documentation could be slightly clearer.

2. **Test coverage for shakedown_only edge cases**: The test exercises shakedown_only at both levels (shakeup_sticks directly and via get_mbxas_spectra), but a test that explicitly seeds negative-delta_e sticks from both channels and verifies the cross combination correctly filters at the right level would be instructive for maintainers. Current test is adequate; this is a nice-to-have.

3. **Caching invalidation note**: The `_shakeup_cache` and `_spectator_shakeup_cache` are correctly invalidated only if `mb_overlap`, `mo_occ`, `mo_energy`, or `core_orb_idx` change (per the docstring at line 395). This is correct and safe. A one-line note in the Spectra class docstring that "caching is valid across `transform()` calls" would document the deliberate design. Not a bug; not necessary.

---

## Test Results

All 35 tests pass, including:
- Core MBXAS physics invariants (amplitude agreement, XCH alignment, determinant magnitude, condition number, core-hole overlap)
- Shake-up combinatorics (order-1/order-2 weights, auto-tolerance, shakedown filtering)
- Cross-channel sticks (combination formula, energy/weight outer product/sum, edge cases)
- Spectra API (byte-identical when off, cross-spin agreement, summary agreement, plotting)
- Persistence (HDF5 save/load round-trip)

**Final test run (conda run -n pymbxas pytest tests/ -q):**
```
35 passed in 104.91s
```

---

## Addendum (post-review finding)

This review was a code-reading pass and did not execute the cross-spin path with real values, so it missed a real defect surfaced immediately afterward when the user reported the test suite spiking system RAM above 100GB. Root-caused via execution (see commit `054e2a3`): the spectator channel has no core hole to prune its virtual manifold, so cross-channel combinations reach `delta_e` in the hundreds of eV (H2O/O: up to ~827 eV combined); `convolve_shakeup` sized its kernel grid from the full range of every stick regardless of weight, so physically negligible (weight as low as `1e-35`) but energetically extreme combinations blew the dense `(n_kgrid, n_sticks)` broadcast in `broadened_spectrum` up to ~69GB peak RSS for a 39-basis-function molecule. Fixed by dropping sticks whose `|delta_e|` exceeds the main spectrum's own span plus a broadening margin before sizing the kernel -- such sticks shift the whole spectrum off the plotted window and cannot contribute above Gaussian-tail precision to the sliced output, so no computed value changes. Peak RSS for the full suite: ~69GB -> ~2.6GB; wall time: ~105s -> ~30s. 35/35 tests still pass.

This is a concrete instance of the general risk this review's own checklist item (e) gestured at but didn't test for: "byte-identical when off" claims and code-path tracing are necessary but not sufficient -- a scientific review of numerical code should include at least one execution with real, non-toy values through every new code path, not just static reasoning about the code.

---

## Conclusion

**Status: APPROVED**

The implementation is scientifically sound and mathematically correct. All method invariants hold. The new cross-spin shake-up and shake-down feature is properly integrated and does not alter existing behavior when disabled (spectator_order=None, shakedown_only=False). No bugs were found. The code is ready for release.

**No fixes applied.** All findings are either confirmations of correctness or minor documentation suggestions.

---

## Verification Checklist

- [x] Spin channel convention (channel=1 default, never hardcoded)
- [x] Core-hole index via `np.where(mo_occ[channel]==0)[0][0]`
- [x] GS orbital indexing via `np.setdiff1d`
- [x] Hartree internally, eV at boundaries
- [x] XCH alignment formula
- [x] Transition dipole origin independence
- [x] Spectator channel omitted from amplitude
- [x] `spectator_occ_unocc_indices` correctness
- [x] `combine_cross_channel_sticks` probability math
- [x] Two-level `shakedown_only` design
- [x] "Byte-identical when off" claims
- [x] `plot_shakeup_summary` edge cases
- [x] Incidental bugs in mbxas.py / shakeup.py
- [x] All tests pass
