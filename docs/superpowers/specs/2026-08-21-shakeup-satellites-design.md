# Shake-up satellite spectra

## Problem

pymbxas's determinant amplitude (`run_MBXAS_pyscf`, `mbxas.py`) keeps only the
one-body term of the exact many-body expansion: a single valence electron
promoted from the core to the conduction orbital `f`, everything else frozen.
This is documented in `dev/method.md` as "one-body truncation" and treated as
a standard, deliberate approximation.

The determinant framework this package implements (Liang & Prendergast;
Roychoudhury & Prendergast, PRB 107, 035146) has an exact expansion beyond
that term (PRB 107,035146 Eq. 32-35, the `f^(n)` series): additional,
non-self-consistent valence-hole/conduction-electron pairs on top of the core
transition. `mbxas-qe` (the group's periodic/plane-wave implementation,
`/home/roncofaber/software/mbxas-qe`) implements this order by order
(`singles_overlap`/`doubles_overlap`/`triples_overlap` in
`QE/SHIRLEY/src/mbxas_spectra.f90`) and reports it can produce non-zero
shake-up satellite intensity that the one-body truncation misses entirely.

This spec adds that capability to pymbxas: an opt-in, energy-convolved
shake-up satellite correction to the broadened spectrum.

## The math

Checking `mbxas-qe`'s actual weight formula for the two-body term:

```fortran
delta_k = abssq( K(v,c)*K(vp,cp) - K(v,cp)*K(vp,c) )
```

This is the 2x2 minor determinant of `K` (rows = valence orbitals being
shaken up, columns = conduction orbitals receiving them) — the exact
generalization of the matrix-determinant-lemma identity pymbxas's own
`K = A' @ inv(A)` already relies on for the n=1 term (`mbxas.py`,
`run_MBXAS_pyscf`): swapping k rows of the (N x N) valence-overlap matrix `A`
for k unoccupied ones, relative to `det(A)`, is exactly the k x k minor of
`K` over those rows/columns. This is not a special-cased QE trick; it holds
for any order k, using the same `K` pymbxas already computes.

So the generalized weight for a k-fold shake-up (k valence orbitals
`{v_1..v_k}` promoted to k conduction orbitals `{c_1..c_k}`, on top of the
ordinary core-hole -> f transition) is:

```
weight(v_1..v_k -> c_1..c_k) = |det( K[{v_1..v_k}, {c_1..c_k}] )|^2
energy_shift               = sum_i (eps[c_i] - eps[v_i])
```

`k=1` recovers `mbxas-qe`'s `singles_overlap` (and is consistent with the
existing n=1 amplitude formula). This is fully general in `k` — no
per-order formula needed, see Implementation below.

`mbxas-qe` builds this as a **valence-only probability spectrum**, decoupled
from which core-edge transition `f` is being considered (energy axis is
purely `sum(eps_c - eps_v)`, independent of `f`). The physical picture: the
sudden creation of the core hole has some probability of *also* kicking
valence electrons into an excited configuration, and that probability
(as a function of the extra energy it costs) convolves onto every transition
in the main spectrum, producing replica satellites at `E_main + delta_E`.

Total spectrum with shake-up: `main_spectrum ⊛ (delta(0) + P_1(dE) + P_2(dE) + ...)`
where `P_k` is the order-k probability spectral function above, broadened the
same way the main spectrum is.

### Open item, not yet settled

I expect (Onishi/Fredholm-determinant-type identity) that
`sum over all k of (sum of order-k minors squared)`, normalized by `det(A)^2`,
sums to exactly 1 — the rigorous version of "the series converges". I have
not hand-verified the exact normalization. This needs numerical verification
during implementation (sum order-1 + order-2 probability mass for H2O or N2,
check it trends toward the expected bound) before the auto-convergence
tolerance is trusted as more than a heuristic.

## Scope for this version

- Orders 1 and 2 only. Order 2 is O(n_occ^2 * n_virt^2) minors, fully
  vectorizable in numpy, cheap on pymbxas's typical (small-molecule) system
  sizes.
- Order 3+ is O(n_occ^3 * n_virt^3) — for a mid-sized molecule this can reach
  10^9+ combinations. `mbxas-qe` needs adaptive-tolerance pruning
  (`doubles_overlap`/`triples_overlap`'s iterative tolerance-narrowing loop)
  to make this tractable at that scale; pymbxas has nothing like that yet.
  Requesting order 3 raises `NotImplementedError` naming this reason.
- Cross-spin convolution (the *other*, non-excited channel's own shake-up,
  which `mbxas-qe`'s `spin_convolve_spectrum` in `spec.f90` computes and
  convolves in) is explicitly out of scope for this version, but the data
  model and function signatures below are designed so adding it later needs
  no breaking change (see "Designed-in extension point").
- No backward compatibility constraint on old saved `Spectra`/`PySCF_mbxas`
  HDF5 files — the schema change described below is not gated behind a
  fallback path.

## Architecture

**`Spectra` becomes self-sufficient and becomes the single implementation.**
Today, `get_mbxas_spectra` exists on three places (`PySCF_mbxas`, `Spectra`,
and the free function in `mbxas/broaden.py`) that must be kept numerically
identical by hand (`dev/method.md`, `CLAUDE.md` gotchas). Adding a fourth
piece of logic (shake-up) into that pattern makes it worse. Instead:

- `Spectra` gains new stored fields at construction (from data
  `Excitation`/`pyscf_obj` already has, just not currently retained):
  `mb_overlap` (both spin channels, `(2, norb_fch, norb_gs)`), FCH
  `mo_energy` (both channels), GS `mo_occ` (both channels), and the excited
  core orbital's GS MO index (`exc.orb_idx`). This is everything needed to
  rebuild `A`/`K` for either channel without touching PySCF or rerunning
  anything.
- `PySCF_mbxas.get_mbxas_spectra` becomes a thin wrapper around
  `self.to_spectra(...).get_mbxas_spectra(...)` — one real implementation
  instead of two. (`to_spectra()` already exists and has no side effects.)
- HDF5 persistence (`io/h5.py`) gains the new `Spectra` fields, following the
  existing append-only group pattern.

## New module: `pymbxas/mbxas/shakeup.py`

Pure functions, no class state, mirroring the existing style of
`mbxas/mbxas.py` and `mbxas/broaden.py`.

**One generic minor function, not one function per order** — this is the
concrete answer to "generalize the nomenclature instead of hardcoding each
element":

```python
def shakeup_sticks(K, eps_occ, eps_unocc, order):
    """order-fold simultaneous valence-to-conduction shake-up.
    weight = |det(K[v_combo, c_combo])|^2, summed over all combinations."""
    v_combos = np.array(list(itertools.combinations(range(len(eps_occ)), order)))
    c_combos = np.array(list(itertools.combinations(range(len(eps_unocc)), order)))
    sub = K[v_combos[:, None, :, None], c_combos[None, :, None, :]]  # batched (order, order) blocks
    weights = np.abs(np.linalg.det(sub)) ** 2   # numpy batches det over leading dims
    delta_e = eps_unocc[c_combos].sum(axis=1)[None, :] - eps_occ[v_combos].sum(axis=1)[:, None]
    return delta_e.ravel(), weights.ravel()
```

`order=1` is not a special case: a 1x1 "minor" is just `K[v,c]` itself, and
`np.linalg.det` on a batch of 1x1 matrices returns that value directly — the
formula recovers `mbxas-qe`'s `singles_overlap` weight with no branching.
Adding order 3 later, once a pruning strategy exists, is calling this same
function with `order=3` and adding whatever pre-filtering keeps the
combinatorics bounded — not writing a new function.

- `K` construction (`APrimeMat @ inv(AMat)`) is factored out of
  `run_MBXAS_pyscf` into a small shared helper (`build_A_K(mb_overlap,
  channel, occ_idxs_gs, occ_idxs_fch, uno_idxs_fch)`) so the n=1 amplitude
  and the shake-up module build `K` identically, once, instead of twice.
- `shakeup_spectrum(K, eps_occ, eps_unocc, order, tol=0.01)`: always includes
  order 1; includes order 2 only if its total probability mass
  (`sum(weights)`) exceeds `tol * order_1_mass`. `order="auto"` resolves to
  `1` or `2` via this rule. `order=3` (or higher) raises `NotImplementedError`
  with the reason above. `order=2` explicitly requested always runs
  (no silent downgrade), only `"auto"` uses the tolerance to decide.

## `Spectra` API

- `get_shakeup_spectrum(order="auto", channel=None, sigma=0.5, npoints=3001,
  erange=None, tol=0.01)` — `channel` defaults to the excited channel.
  Returns the broadened `P_k(dE)` probability curve (summed over included
  orders). Cached per `(channel, order, tol)` on the instance (plain dict);
  changing `sigma`/`erange` re-broadens cached sticks rather than
  recomputing the combinatorics.
- `get_mbxas_spectra(..., shakeup_order=None)` — when set, computes (cached)
  the shake-up spectrum and convolves it with the existing broadened main
  spectrum before returning. `shakeup_order=None` (the default) must be
  byte-identical to current behavior — this is a hard regression test, not
  just an expectation.

### Designed-in extension point (cross-spin, not built now)

`get_shakeup_spectrum`'s `channel` parameter is why this is ready for the
`spin_convolve_spectrum`-equivalent later: that feature would call
`get_shakeup_spectrum(channel=<the other channel>)` — using the exact same
function, no signature change — and convolve *that* result into the main
spectrum as well. It isn't built now because the *other* channel's `A`
matrix has a structurally different shape (no core orbital excluded, since
that channel isn't excited — full occupied count, not N-1), which is a
distinct-enough piece of construction logic to be its own follow-up rather
than bolted on here.

## Testing

Extends `tests/test_h2o_kedge.py` (per the existing "one calculation, add
assertions, no new test files" scope rule in `CLAUDE.md`):

- `shakeup_order=None` produces output identical to the pre-existing
  `get_mbxas_spectra` call already in the test (regression guard).
- Order-1 probability mass is positive and finite; order-2 mass is smaller
  than order-1 mass (physical expectation: higher-order shake-up is rarer),
  checkable without any hardcoded external reference value.
- `order=3` raises `NotImplementedError`.
- Numerically probe the open normalization question above and record
  whatever bound is actually observed as a new reference value in
  `dev/method.md`'s verification table, the same way `det(A)`, `cond(A)`,
  etc. are recorded today.

## Documentation

- `dev/method.md`: new section under "Known approximations" is replaced /
  extended — "one-body truncation" no longer describes current behavior when
  `shakeup_order` is set; document the formula, the `mbxas-qe` provenance,
  and the order-3 limitation.
- `CHANGELOG.md`: `### Added` entry (new capability, not a behavior change to
  existing output when unused).
