# Scientific review: shake-up, cross-spin, shake-down, and maxvol-style search

**Date:** 2026-08-25
**Scope:** `pymbxas/mbxas/{mbxas,shakeup,maxvol,broaden}.py`, the corresponding
`Spectra` APIs, tests, and the local `mbxas-qe` reference checkout.
**Status:** Historical pre-fix review. Its findings motivated the implementation
now documented in `dev/shakeup.md`: the spectral maxvol heuristic and normalized
excited-channel overlap convolution were removed, explicit determinant
amplitudes were added, and QE-compatible normalization and shake-down flags
were restored. The findings below describe the code before those fixes.

This review supersedes the shake-up conclusions in
`2026-08-24-mbxas-scientific-review.md`. That review checked internal
self-consistency but did not compare the algorithms and selection semantics
closely enough against their claimed QE counterparts.

## Reference material inspected

- PyMBXAS `run_MBXAS_pyscf`, `build_A_K`, `shakeup_sticks`,
  `_maxvol_shakeup_configs`, `combine_cross_channel_sticks`,
  `convolve_shakeup`, and their `Spectra` integration.
- `mbxas-qe/QE/SHIRLEY/src/mbxas_spectra.f90`: overlap singles, doubles,
  triples, `mb1_spectrum`, `mb2_spectrum`, and `mb3_spectrum`.
- `mbxas-qe/QE/SHIRLEY/src/spec.f90`: `spin_convolve_spectrum`.
- `mbxas-qe/QE/SHIRLEY/src/maxvol_multi_mod.f90` and its call from
  `overlap_binding.f90`.
- `mbxas-qe/QE/SHIRLEY/src/convseq.f90` and
  `kpoint_spectral_details.f90`: shake-down detection and reporting.
- PRB 107, 035146 for the one-body core-hole-basis amplitude and the
  higher-order determinant expansion.

## Findings

### 1. Determinant-minor overlap weights are correct

**Severity:** confirmed invariant.

For any fixed occupied reference matrix `A`, replacing occupied rows `V` by
virtual rows `C` gives a determinant ratio equal, up to sign, to
`det(K[C,V])`, where `K=A' A^-1`. Squaring its magnitude produces the relative
determinant-overlap probability used by QE's overlap singles/doubles/triples.
The order-2 and order-3 formulas in QE are explicit expansions of the same
minor. The all-electron versus plane-wave representation does not alter this
linear-algebra identity.

The excited and spectator orbital partitions in PyMBXAS are also consistent:
only the excited channel removes the GS core orbital and FCH core-hole row.

### 2. The Python search is not a port of `maxvol_multi`

**Severity:** high documentation and validation issue.

QE's `maxvol_multi` first finds a locally maximal full `n_occ`-row submatrix,
then runs constrained searches from entries exceeding one percent of the
converged `B` maximum. It returns a fixed number of whole pivot sets ranked by
their full determinant. Its visible call in `overlap_binding.f90` is an
orbital-subset search.

PyMBXAS instead starts up to 64 branches from the largest exact single-swap
weights, extends each branch by only its largest available `B` entry when that
entry exceeds `1+tol`, buckets the resulting swaps by order, and stops by mass
relative to the order-1 sum. These rules are not QE's initialization,
candidate generation, exclusions, ranking, or stopping rules.

QE's spectral doubles/triples do not call `maxvol_multi`; they use adaptive
element-threshold enumeration in `mbxas_spectra.f90`. The Python algorithm is
therefore neither a port of the full-pivot search nor of the spectral search.

An exhaustive 4-occupied/5-virtual synthetic check (`default_rng(7)`,
`A = I + 0.03*N(0,1)`, `A' = 0.2*N(0,1)`, `tol=1e-6`) found 7 of 60 order-2
configurations and captured 36.8% of the exact order-2 mass. The largest
configuration happened to be found, and each returned weight was correct, but
the current tests only verify the latter property. There is no completeness or
error guarantee.

**Action in this change:** documentation now consistently says
"maxvol-style heuristic" and records the missing validation. The algorithm was
not silently replaced because choosing between QE's two different searches is
a scientific design decision.

### 3. Excited-channel convolution is not the explicit `f^(n)` MBXAS series

**Severity:** high scientific-scope issue.

PyMBXAS builds valence-overlap sticks and convolves them with the already
computed one-body MBXAS spectrum. This is a sudden-approximation satellite
model. A minor weight is an overlap probability, not a higher-order dipole
transition amplitude.

QE separately computes `mb1_spectrum`, `mb2_spectrum`, and `mb3_spectrum`.
It then combines an excited channel's order-resolved XAS terms with the
opposite spin channel's order-resolved overlap terms. Its `f^(2)` spectrum,
for example, contains both the explicit `mb2` term times the spectator
zero-overlap spectrum and the `mb1` term times spectator singles.

Calling the current PyMBXAS result an exact implementation of PRB 107's
higher-order `f^(n)` amplitudes is therefore incorrect. It is still a
well-defined opt-in approximation for producing overlap-weighted replicas of
the main spectrum.

**Action in this change:** public and maintainer documentation now calls it an
experimental overlap-satellite convolution and distinguishes it from explicit
higher-order MBXAS amplitudes.

### 4. `shakedown_only` does not match QE selection semantics

**Severity:** medium-to-high parity issue.

PyMBXAS retains a stick only when its summed energy shift is negative. After
cross-spin combination it applies the test to the sum of both channel shifts.

QE defines a shake-down condition when any elementary electron-hole energy is
negative. Its doubles and triples retain a configuration if any constituent
promotion is negative, even when the total configuration energy is positive.
The per-channel selection occurs before spin convolution.

The current `(delta_e, weight)` arrays discard the constituent-level boolean,
so exact QE semantics require a data-contract change rather than a different
one-line mask.

**Action in this change:** the different behavior is documented. No numerical
behavior changed.

### 5. Cross-spin outer convolution is structurally correct but applied to a
different input model

**Severity:** medium scope qualification.

Adding independent channel energy shifts and multiplying their probabilities
is the correct discrete convolution for factorized spin determinants. The
`max_total_order` bookkeeping also correctly includes pure and mixed terms and
excludes the implicit `(0,0)` term supplied by the convolution kernel.

The cross-product pruning has a meaningful per-block mass bound because the
exact product mass is `sum(w_i) * sum(w_j)`. This is stronger than the
single-channel maxvol-style search.

However, QE convolves explicit excited-channel XAS spectra with opposite-spin
overlap spectra. PyMBXAS convolves its one-body main spectrum with overlap
kernels from both channels. The convolution operation matches; the inputs and
order decomposition do not.

### 6. Unit normalization changes the meaning of truncated weights

**Severity:** medium quantitative issue.

`convolve_shakeup` normalizes the retained kernel to unit area. This preserves
the integrated intensity of the input main spectrum and makes the feature a
redistribution model.

For a complete orbital manifold, Cauchy-Binet relates the reference
determinant and the sum over all squared minors. With truncated orders and an
incomplete configuration search, normalizing the retained subset reallocates
the missing probability among the configurations that survived. QE carries
the reference overlap determinant into each overlap spectrum and sums
order-resolved contributions without this retained-subset renormalization.

Absolute shake-up and spectator-spin intensities are therefore not comparable
between the two implementations as currently written.

### 7. Numerical safeguards are appropriate but do not validate the physics

**Severity:** confirmed implementation quality with a testing gap.

The Sherman-Morrison row update is correct and raises on a near-singular
denominator. Uniform-grid broadening avoids the former dense memory explosion,
and far-off-window stick removal is justified by the finite Gaussian support
used numerically. Default-off behavior remains isolated.

The existing maxvol test proves only that a discovered configuration has the
right minor weight and energy. It does not compare selection, captured mass,
or spectra with exhaustive enumeration or QE. The H2O test is aufbau-like and
does not exercise a physical shake-down case.

## All-electron versus plane-wave considerations

The overlap determinant, complementary-minor identity, spin factorization,
and constituent shake-down definition transfer unchanged between bases.
The following QE details do not transfer automatically and must be aligned in
any numerical comparison:

- k-point and spin weights;
- the finite Gaussian virtual space versus the Shirley/plane-wave band window;
- occupation changes and any maxvol-selected reference occupation;
- QE's `eshift_swap`, XPS shifts, and energy-window truncation;
- pseudopotential/core reconstruction and periodic gauge choices;
- basis-set completeness and continuum discretization.

These differences justify different data plumbing, not different determinant
algebra.

## Recommendation

Do not use the current optional shake-up output as a quantitative claim of QE
parity. It is scientifically interpretable as a normalized,
overlap-weighted satellite model, with exact order-1 overlap sticks and
heuristically selected higher-order sticks.

Before changing numerical behavior, choose one explicit target:

1. port QE's adaptive spectral enumeration and explicit `mb2`/`mb3`
   transition amplitudes; or
2. implement a faithful `maxvol_multi` pivot-set search for a clearly defined
   orbital-selection use case, separate from spectral enumeration.

The first target is the route to higher-order MBXAS spectral parity. The
second solves a different occupation/submatrix-search problem and should not
be presented as a substitute for the first.
