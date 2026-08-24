# Cross-spin shake-up and shake-down

## Problem

The shake-up satellite correction added earlier (`get_mbxas_spectra(shakeup_order=...)`)
only accounts for extra valence excitations in the *excited* spin channel. The
spectator (non-excited) channel also relaxes in the core-hole field and can
itself contribute shake-up satellite structure -- `mbxas-qe`'s
`spin_convolve_spectrum` (`spec.f90`) captures this by convolving each
channel's own order-resolved shake-up spectrum with the *other* channel's,
building explicit per-order cross terms up to a total-order budget.

Separately, `mbxas-qe` treats **shake-down** as a named case of the same
k-fold-minor formula: a combination whose electron-hole energy `delta_e` comes
out negative (`kpoint_spectral_details.f90`: `shakedown = any(de < 0)`), with a
dedicated `shakedown_only` diagnostic mode that isolates those contributions.
pymbxas's `shakeup_sticks` already represents negative `delta_e` correctly
(the `get_shakeup_spectrum` erange fix already accounts for it) but never
labels, isolates, or reports it.

This spec adds both, and folds them together since cross-spin naturally
produces the stick-combination machinery shake-down also needs.

## Math, in matrix/vector notation

All formulas below operate on whole arrays; no element-wise Python loops
over combinations.

**Per-channel order-k sticks** (already implemented, `shakeup_sticks`):
for row-subset (conduction) indices `C` and column-subset (valence) indices
`V`, `|C| = |V| = k`, weight is the squared-magnitude of the k x k minor
`det(K[C, V])`. Implemented as a batched determinant over all combinations
at once (`np.linalg.det` on a stacked array of submatrices) -- this pattern
continues unchanged.

**Spectator-channel `K`**: identical `K = A' @ inv(A)` construction
(`build_A_K`, unchanged), fed a different valence-index partition (no core
orbital removed, since there is no core hole in this channel).

**Cross-channel combination.** Given the excited channel's order-`i` sticks
`(e_i, w_i)` (vectors) and the spectator channel's order-`j` sticks `(e_j, w_j)`,
the order-`(i,j)` cross term is the *outer* combination:

```
E_ij = e_i[:, None] + e_j[None, :]      # outer sum of energies  (n_i x n_j)
W_ij = w_i[:, None] * w_j[None, :]      # outer product of weights (n_i x n_j)
```

ravelled to flat stick lists. This is the discrete-stick form of convolving
the two channels' probability spectra -- physically, the two channels'
shake-up processes are treated as independent (sudden-approximation
factorization), so joint probability is a product and joint energy cost is
a sum. All `(i, j)` pairs with `i + j <= max_total_order` are accumulated
into one combined stick list (including the trivial `(0, 0)` pair: energy 0,
weight 1), which then feeds the *existing* `convolve_shakeup`/
`broaden_shakeup` unchanged -- the `(0, 0)` pair is exactly the delta term
those functions already special-case.

**Shake-down.** No new formula: it is `shakeup_sticks`'s existing output
where `delta_e < 0`. Adds a boolean mask/filter, not a new computation.

## Architecture

**New in `pymbxas/mbxas/mbxas.py`:**
- `spectator_occ_unocc_indices(gs_mo_occ_channel, fch_mo_occ_channel) ->
  (occ_idxs_gs, occ_idxs_fch, uno_idxs_fch)`: the no-core-hole sibling of
  `occ_unocc_indices` -- no `core_orb_idx` argument, no `setdiff1d` removal,
  no `[1:]` drop on the unoccupied side. Feeds `build_A_K` unchanged.

**Changed in `pymbxas/mbxas/shakeup.py`:**
- `shakeup_sticks(..., shakedown_only=False)`: filters to `delta_e < 0` when
  set. Name matches `mbxas-qe`'s flag exactly.
- `shakeup_spectrum` refactored to build and return its per-order sticks as
  a dict (`{0: (e0, w0), 1: (e1, w1), ...}`) internally, so specific orders
  are addressable for cross-term construction; its existing tuple-returning
  public contract is unchanged.
- New `combine_cross_channel_sticks(sticks_a_by_order, sticks_b_by_order,
  max_total_order) -> (delta_e, weight)`: the outer-sum/outer-product
  combination above, one call per `(i, j)` pair, `np.concatenate`d.

**Changed in `pymbxas/spectra.py`:**
- New sibling method `_spectator_shakeup_sticks(self, order, tol)`, parallel
  to `_shakeup_sticks` but building `K` via `spectator_occ_unocc_indices`
  on channel `1 - self._channel`, with its own cache dict keyed
  `(order, tol)`. Kept separate rather than an added parameter on
  `_shakeup_sticks`, since the two differ in index construction
  (`spectator_occ_unocc_indices` vs `occ_unocc_indices`), not just in
  which channel number is passed.
- `get_mbxas_spectra`/`get_shakeup_spectrum`/`get_shakeup_summary` gain
  `spectator_order=None` (default off, byte-identical to today) and
  `max_total_order=None` (defaults to `shakeup_order + spectator_order`
  when both are set -- no additional truncation beyond each channel's own
  cap unless explicitly requested).
- `get_shakeup_summary` additionally reports `shakedown_fraction` (mass of
  `delta_e < 0` sticks / total mass) unconditionally, and logs (via the
  existing TRACE-capable logger) when it is non-negligible -- the
  equivalent of `mbxas-qe`'s stdout shakedown warning.

## Global constraints (bind every task)

- **Matrix/vector notation.** Every new numerical routine operates on whole
  arrays (numpy broadcasting/outer products/batched `det`), never an
  explicit Python-level loop over individual combinations. This continues
  the existing style in `shakeup_sticks`.
- **Physics comments.** In `pymbxas/mbxas/mbxas.py` and
  `pymbxas/mbxas/shakeup.py` specifically (not elsewhere), add a short
  comment at each new/changed function's core formula naming the physical
  quantity and, where one exists, the source equation/reference (matching
  the style already used in `build_A_K`'s docstring, e.g. "Eq. 22, PRB
  107,035146"). A handful of targeted comments, not a rewrite of existing
  ones -- `CLAUDE.md`'s no-comments default is deliberately overridden
  here, for this directory only, because the *why* (which formula, from
  which paper) is exactly the non-obvious information a reader of physics
  code needs.
- **Final scientific review.** After implementation, a dedicated task
  reviews the whole `mbxas/` implementation (not just this feature) against
  `dev/method.md`'s documented invariants and the two reference papers
  (PRB 106,075133; PRB 107,035146) for correctness -- separate from, and in
  addition to, the standard final whole-branch code-quality review the
  subagent-driven-development process already runs. Findings and any
  further improvement suggestions are reported back, not auto-applied.
- No backward-compatibility constraint on old saved files (same as the
  original shake-up feature).

## Testing

H2O/N2 are aufbau-like, symmetric, and don't naturally produce negative
`delta_e` or large cross-channel terms, so automated verification (in
`tests/test_h2o_kedge.py`, per the project's one-file test convention) again
leans on cache-seeded synthetic sticks and hand-computed outer-sum/outer-
product arithmetic on small arrays -- same pattern as the original feature's
erange fix. `spectator_order=None`/`max_total_order=None` must remain
byte-identical to current output (hard regression requirement, mirroring the
original shake-up feature's `shakeup_order=None` guarantee).

**Real-molecule demo script (not part of the automated suite).** An
open-shell Cu(II) complex is a better showcase for this feature than H2O/N2:
it has an intrinsically unpaired spin, so the spectator channel's own
valence relaxation is physically meaningful rather than a near-symmetric
echo of the excited channel. The implementation plan adds one script,
following the existing `n2_shakeup_compare.py`/`h2o_shakeup_compare.py`
pattern, built from `~/Downloads/17949957.mol` (a Cu bis-diketonate-type
complex, heavy atoms only -- C/O/Cu, no explicit H in the source file).
The script uses RDKit (`Chem.MolFromMolFile` +
`Chem.AddHs(mol, addCoords=True)`) to add hydrogens before converting to an
ASE `Atoms` object for `PySCF_mbxas`, and exercises `spectator_order`,
`max_total_order`, and `shakedown_only` alongside the existing
`shakeup_order`. Like the two existing scripts, it is a runnable file the
user executes later themselves -- the plan's task only writes it; it is
never invoked as part of implementation or review. RDKit is not currently
installed in the `pymbxas` conda env and is not added as a pymbxas
dependency -- it is only imported inside this one demo script, for the
user to install themselves (e.g. `conda install -c conda-forge rdkit`)
before running it.
