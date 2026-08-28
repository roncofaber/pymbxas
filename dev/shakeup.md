# Many-body shake-up, spectator-spin, and shake-down spectra

This document describes the current implementation in
`pymbxas/mbxas/shakeup.py` and `Spectra`. Historical design documents under
`docs/superpowers/` are not current specifications.

## Scientific scope

The optional path now implements the same determinant algebra as the molecular
form of `mbxas-qe`'s order-resolved spectra. The representation differs:
PyMBXAS uses a finite all-electron Gaussian virtual space and no k points,
whereas QE uses periodic plane-wave/Shirley states and band/k-point machinery.
Formula-level agreement therefore does not imply identical numerical spectra.

The ordinary spectrum is spin-complete f1/MB1 (`10`): it includes the
spectator zero-order determinant. Explicit `spectator_order=None` is retained
only as a single-channel diagnostic.

## Orbital partitions

For either spin channel, let

```text
A  = <FCH occupied | GS occupied>
A' = <FCH virtual  | GS occupied>
K  = A' A^-1
```

The excited channel excludes the GS core orbital and the FCH core-hole orbital.
The spectator channel keeps its full electron count and virtual manifold.

## Explicit excited-channel amplitudes

The public `f_order` follows QE's direct naming. Internally,
`max_extra_order = f_order - 1` counts extra valence particle-hole pairs:

- `0`: the existing f1/MB1 transition;
- `1`: f2/MB2, one additional pair;
- `2`: f3/MB3, two additional pairs.

The stored f1 amplitude already contains `det(A)`. For one additional pair,
occupied hole `v`, shake virtual `c`, and final virtual `f > c`, the code uses

```text
F2(f,c,v) = F1(f) K(c,v) - F1(c) K(f,v)
E2(f,c,v) = E1(f) + eps(c) - eps(v).
```

Higher orders use the corresponding Laplace expansion: for `k` occupied holes
and `k+1` final virtuals, each f1 amplitude is multiplied by the complementary
`k x k` minor of `K` with its alternating determinant sign. This reproduces
the explicit MB3 expression without a separate hard-coded formula.

The low-level algebra helper can enumerate configurations exactly for tests and
small systems. The production MB2 path follows QE's adaptive screening:

1. discard electron-hole pairs whose individual promotion energy exceeds the
   requested spectral span plus six Gaussian widths;
2. keep the dipole-final orbital manifold independent of that pair selection,
   discarding a final determinant only when its photon energy lies outside the
   requested output range plus six Gaussian widths;
3. sort `|det(A) K(c,v)|²` in descending order;
4. begin at an element-amplitude threshold of `0.1 * 0.7 = 0.07`; QE's
   inclusive Fortran loop also admits the first ranked element immediately
   below each threshold;
5. lower the threshold by `0.7` until the first 104,857-configuration
   spectral buffer fills, then use QE's slower count-dependent update
   `t_next = t / (1 + 0.7 * 104857 / Nconfig)`;
6. monitor `|det(A)|² sum(|A2|²) / sum(|A1|²)` and stop when QE's
   delta, second-difference, and curvature tests meet `tol`, or when every
   energy-relevant pair has been exhausted.

The loop is capped at 200 iterations, matching QE. Each iteration and its stop
reason are logged. Reaching the iteration cap before convergence emits a
warning; reaching `max_configurations` raises rather than returning a
silently underconverged spectrum. Final-configuration construction is chunked
using QE's 104,857-entry spectral buffer size.

Excited-channel MB3 follows QE's `mb3_spectrum` product search. It ranks the
same determinant-scaled elementary K entries, admits canonical pairs with
`v < v'` and `c' < c < f`, and starts at the same `0.07` element threshold.
Every retained transition uses the complete three-term MB3 amplitude; the
screen only decides which candidate pairs reach that evaluation. Its
count-adaptive threshold update uses QE's 104,857-configuration spectral
buffer and the same normalized-weight convergence tests as MB2. The final
orbital `f` remains independent of the energy-windowed K-pair manifold.

There are no positional occupied/virtual active-space cutoffs: f1 and f2 use
the complete energy-relevant FCH manifold, and MB3 searches that same manifold
adaptively. Orders above MB3 still require guarded exact enumeration. A
default two-million retained-configuration guard raises an actionable error
rather than silently changing the orbital manifold or appearing to hang.
`order="auto"` remains unsupported.

## Spectator overlaps and spin assembly

In `mbxas-qe`, spectator convolution is not a user-selectable physical switch:
the overlap singles, doubles, and triples flags are enabled in
`process_input_details`, and `many_body_spectra` assembles f1=`10`,
f2=`20+11`, and f3=`30+21+12`. QE computes and writes these resolved orders;
selection of an accumulated order is effectively a post-processing choice.

For a spectator configuration replacing occupied set `V` by virtual set `C`,

```text
W(C,V) = |det(A)|^2 |det(K[C,V])|^2
dE(C,V) = sum(eps(C)) - sum(eps(V)).
```

Order zero is `(dE=0, W=|det(A)|^2)`. Excited-channel XAS order `i` and
spectator-overlap order `j` are combined by adding energies and multiplying
intensities. By default `i+j <= f_order-1`, producing f1=`10`,
f2=`20+11`, and f3=`30+21+12`, matching QE's organization by total extra-pair
order. Explicit spectator and total-order arguments remain available for
component diagnostics. Spectator singles use the full energy-windowed
manifold. Spectator doubles follow QE's `doubles_overlap` search: rank
`|det(A)K(c,v)|²`, admit canonical pair products through a decreasing
threshold starting at `0.7*tol`, and apply the energy bound both to each
constituent and their summed shift. Once QE's 1,048,576-configuration overlap
buffer fills, the threshold update becomes
`t_next = t / sqrt(1 + 0.7*1048576/Nconfig)`. The accumulated determinant
weight uses the same delta, second-difference, and curvature convergence tests
as QE. Retained weights remain exact 2x2 minors; screening only selects which
configurations are evaluated. Construction and the later Cartesian product
with XAS sticks are processed in bounded chunks.

No retained-stick or convolution kernel is normalized to unit area. The
excited-channel determinant is already present in f1 and its higher-order
amplitudes; the spectator determinant is present in every overlap order.

## Captured overlap convergence

For one channel, stack every eligible FCH row against the GS occupied
subspace as `B = [A; A']`. Cauchy--Binet gives the overlap mass available in
that row manifold without enumerating determinants:

```text
P_available = det(B^dagger B).
```

The retained mass at order `k` is the sum of the corresponding determinant
weights. For two spins, masses multiply and are grouped by total order `i+j`,
using the same `10`, `20+11`, and `30+21+12` organization as the spectrum.
`get_mbxas_decomposition()` reports the per-total-order mass, its captured sum,
the product of both channels' available masses, and their ratio under
`data["overlap"]`. Screened configurations contribute only when retained, so
the ratio diagnoses both order truncation and screening convergence.

This is not an absorption-intensity sum rule. Dipole interference, transition
energies, electronic-structure error, and basis completeness remain separate
sources of spectral error even when the captured overlap fraction is near one.

## Broadening convention

PyMBXAS combines the discrete excited-channel and spectator sticks first and
applies one normalized Gaussian to each complete final many-body transition.
The public `sigma` is therefore the width of the final spectrum. This treats
core-hole lifetime and instrumental resolution as properties of the complete
final state rather than independent uncertainties attached to its two spin
factors.

`mbxas-qe` instead broadens the excited-channel spectrum and spectator-overlap
spectrum separately with `sigma`, then convolves the two grids. For Gaussian
lineshapes, the resulting final width is `sqrt(2) * sigma`; QE explicitly uses
that wider value for its non-convolved single-particle comparison. PyMBXAS
does not reproduce this numerical convention because applying the same
lifetime or instrumental broadening twice would double-count that source of
uncertainty.

For like-for-like Gaussian comparisons, use

```text
sigma_PyMBXAS = sqrt(2) * sigma_QE.
```

This mapping changes peak widths and maxima, not stick energies, determinant
weights, or integrated intensity. A future physical lineshape may combine a
Lorentzian core-hole lifetime with Gaussian instrumental resolution, but it
should still be applied once to the complete final-state sticks.

## Photon-energy prefactor

PyMBXAS uses position-dipole amplitudes, so every absorption stick carries the
cross-section prefactor `E_photon |M|^2`. For a spectator-assisted transition,
the photon creates the complete final state and its energy is

```text
E_photon = E_XAS + delta_E_spectator.
```

The summed energy is used both to place the stick and to weight its intensity.
Using the pre-convolution XAS energy would make those two parts of the same
transition inconsistent. This differs from QE's spectral assembly, which does
not multiply the broadened tensor by an explicit final photon energy at this
stage and therefore follows a different transition-matrix/output convention.
The correction affects spectator-assisted `11`, `21`, and `12` terms; f1 and
pure excited-channel terms are unchanged.

## Shake-down selection

Occupied holes are ordered ascending and shake virtuals descending, following
QE's canonical pairing. A configuration is marked shake-down when **any**
constituent promotion has negative energy:

```text
any(eps_virtual[c_i] - eps_occupied[v_i] < 0).
```

Physical spectra always retain every configuration. The public summary
decomposes each higher-order f contribution into `"shakeup"` and
`"shakedown"` arrays; f1 is the order-zero reference and therefore has no
shake-down component. Internally, the shake-down component contains only
higher-order configurations carrying the flag. For cross-spin terms, channel
flags are combined with logical OR. A positive summed energy does not discard
a configuration containing a negative constituent.

## Maxvol

QE uses maximum-volume selection in two distinct places. During SCF it can
replace ordinary occupation assignment with determinant-based tracking against
reference or previous-iteration orbitals. After SCF, `overlap_binding.f90`
uses `maxvol_multi` again to enumerate occupied-size GS--FCH reference
submatrices and their configurational energy costs.

`pymbxas/calculators/maxvol.py` implements fixed-reference, degree-one UHF/UKS occupation selection through
PySCF's per-iteration `get_occ` hook. The overlap matrix and linear solves stay
on NumPy for CPU PySCF or CuPy for GPU4PySCF. Its initial subset comes from
projection-overlap ranking before determinant-improving swaps because an
all-electron FCH calculation retains the deliberately empty deep-core orbital;
QE's core-hole pseudopotential does not. `occupation="maxvol"` applies
it consistently to production FCH and XCH calculations; `"mom"` remains the
default and restores the historical path. It does not implement QE's evolving
or two-reference SCF policies.

`pymbxas/mbxas/maxvol.py` remains a tested Sherman–Morrison row-update utility,
but it is no longer used to choose spectral configurations. QE's
`maxvol_multi` is used for full occupied-subspace pivot selection in
`overlap_binding.f90`, not for MB2/MB3 enumeration. Reusing that name for the
old Python spectral heuristic was scientifically misleading.

PyMBXAS does **not** yet port the post-SCF full-reference pivot selection. It builds
`A` directly from the MOM occupation labels. QE can instead select a
better-conditioned occupied-size FCH row subset, track which orbitals were
swapped into the reference, and carry the corresponding `eshift_swap` into
spectral energies. This is deferred because changing the reference alters
`det(A)`, `K`, and every truncated f-order contribution; it is not merely a
performance optimization. A future port must keep reference selection
separate from spectral candidate screening and validate ordinary, swapped,
and non-Aufbau cases before becoming the default.

## Remaining QE comparison work

The determinant amplitudes, f1/f2/f3 spin assembly, adaptive spectral searches,
and constituent shake-down classification are now implemented. A deterministic
complex-valued fixture independently evaluates QE's MB1/MB2/MB3 and overlap
zero/singles/doubles formulas from common synthetic overlaps, orbital energies,
and dipoles. It verifies the resolved `10`, `20`, `11`, `30`, `21`, and `12`
terms before broadening. PyMBXAS's intentional final photon-energy prefactor is
applied and tested only after that common factorization.

The remaining comparison work is:

1. port and validate `maxvol_multi` reference selection and `eshift_swap`;
2. eventually compare matched molecular and periodic calculations after
   accounting for basis, band-window, k-point, and core-reconstruction effects.

Absolute alignment is deliberately outside the current comparison. The
single-final-state Gaussian broadening and final-energy photon prefactor are
documented intentional conventions, not unresolved ports.

## Public interfaces

- `Spectra.get_mbxas_spectra(f_order=..., spectator_order=...,
  max_total_order=...)` returns the energy-windowed
  cumulative spectrum through f1, f2, or f3.
- `Spectra.get_mbxas_decomposition(f_order=...)` returns resolved f-order
  contributions, cumulative sums, the final total, and the automatic
  shake-up/shake-down decomposition of every order above f1, plus captured
  determinant-overlap convergence by total order.
- `Spectra.print_mbxas_summary(data)` prints the integrated values from that
  decomposition without recomputing spectra.
- `overlap_sticks(...)` exposes exact determinant-weighted overlap sticks and
  their constituent-level shake-down flags.
- `mbxas_sticks_by_order(...)` exposes explicit order-resolved energies,
  amplitudes, and flags for validation and development.
- `get_shakeup_spectrum()` is an overlap-distribution diagnostic; it is not an
  XAS spectrum.

Order-2 overlap diagnostics use the same screened doubles search for both spin
factors. Cross-spin products are broadened in bounded blocks rather than
materialized, and their total and shake-down masses are evaluated from factor
sums. `get_mbxas_decomposition()` constructs these factors once and reuses
them for its probability curve and `shakedown_fraction`.

The implementation is covered by exhaustive small-system tests for overlap
completeness, determinant normalization, MB2 and MB3 formulas, QE's adaptive
threshold control flow for MB2, MB3, and spectator doubles, energy-window
screening, batching invariance, count guards, shake-down selection, and the
full order-resolved two-spin parity fixture. Molecular
benchmarks against QE remain desirable, with orbital
manifold and energy-alignment differences documented explicitly.
