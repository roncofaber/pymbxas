# Method reference

The physics PyMBXAS implements, the conventions the code relies on, and what is approximated. Fast-reference summary in `CLAUDE.md`; this file is the authority.

## What MBXAS computes

X-ray absorption from a core level is not a one-electron transition. Removing a core electron changes the potential every remaining electron sees, so the final state is not a single Slater determinant built from ground-state orbitals. The valence electrons relax, and that relaxation redistributes spectral weight from the main line into shake-up satellites and suppresses the total.

The determinant approach of Liang and Prendergast treats this by computing the many-body overlap between the ground-state determinant and each core-excited final-state determinant explicitly, keeping the one-body (single shake-up) term. PyMBXAS implements that for molecules with PySCF providing the two ΔSCF calculations.

## Workflow

Three SCF calculations per excited atom, all unrestricted.

| Step | Charge | Spin | Occupation | Purpose |
|---|---|---|---|---|
| **GS** | `q` | `s` | aufbau | Reference determinant. Run once, shared by all excitations. |
| **FCH** | `q + 1` | `s ± 1` | core orbital of the excited channel emptied, MOM-constrained | Full core hole. Supplies the final-state orbitals, eigenvalues and transition dipoles. |
| **XCH** | `q` | `s` | FCH occupation plus one electron in the FCH LUMO, MOM-constrained | Excited core hole. Supplies the absolute energy alignment only. |

Sign convention for the FCH spin: `spin_FCH = spin_GS + 2*channel - 1`. For the default `channel=1` (beta) this is `spin + 1`, since removing a beta electron raises `n_alpha - n_beta`. For `channel=0` it is `spin - 1`, which PySCF accepts as a negative `mol.spin`.

The core hole is localized on one atom. When the excited element has several symmetry-equivalent 1s orbitals they come out of the SCF delocalized over all of them, and a determinant built from those is meaningless. `_run_localization` detects this and runs IBO (default) or Boys over the core manifold before any excitation, replacing `gs_data.mo_coeff`. Verified on C2H2: the two carbons then give spectra agreeing to 4e-8 eV in energy and 8e-14 in intensity.

## Determinant amplitude

Let `h` be the core-hole orbital, `i, j` the occupied FCH orbitals of the excited channel, `f` the unoccupied ones, and `n` the ground-state occupied orbitals of that channel with the excited core orbital removed.

The many-body overlap matrix between the two SCF solutions is

```
S_MB = C_FCH^T · S_AO · C_GS
```

from which two blocks are taken:

```
A      = S_MB[occ_FCH, occ_GS]     square, (N-1) x (N-1)
A'     = S_MB[uno_FCH, occ_GS]     (n_virt) x (N-1)
K      = A' A^-1
```

The transition amplitude to the final state with the electron in `f` is

```
amp[x, f] = det(A) * ( <f|x|h>  -  sum_i K[f, i] <i|x|h> )
```

for each Cartesian component `x`. The first term is the bare one-electron transition; the second subtracts the part already accounted for by relaxation of the occupied manifold; `det(A)` is the orthogonality-catastrophe suppression of the whole spectrum.

Excitation energies are the FCH eigenvalues of the virtual manifold, shifted:

```
E[f] = eps_FCH[f] + (E_XCH - E_GS) - min(eps_FCH[virt])
```

so the lowest transition lands exactly at the XCH total-energy difference and the rest follow the FCH eigenvalue spacing.

## Index conventions

These are the easiest thing in the package to get subtly wrong.

- `exc_orb_idx = np.where(mo_occ[channel] == 0)[0][0]` is the core hole. It is the *lowest-energy unoccupied* MO, which is index 0 only when the excited atom is the heaviest present. For the C K-edge of CO the O 1s is occupied at index 0 and the hole sits at index 1. Verified.
- `uno_idxs_fch = np.where(mo_occ[channel] == 0)[0][1:]` is the virtual manifold. The `[1:]` drops the hole, which is not a valid final state for its own transition.
- `occ_idxs_gs` removes the excited core orbital from the ground-state occupied set **by MO index**, via `np.setdiff1d`. Using `np.delete` treats the argument as a position in the occupied list, which coincides with the MO index only for a strictly aufbau ground state.
- `A` must come out square. `n_occ_FCH == n_occ_GS - 1` is what guarantees that; if `A` is rectangular something upstream mis-identified the core orbital.
- `find_1s_orbitals_pyscf` returns global MO indices. Anything that enumerates the occupied subset must map back through `occ_idxs` before indexing `mo_energy`.

## Units and array shapes

| Quantity | Unit | Shape |
|---|---|---|
| `mbxas["energies"]` | Hartree | `(n_transitions,)` |
| `mbxas["absorption"]` | atomic units, **amplitude** | `(3, n_transitions)` |
| `mbxas["mb_overlap"]` | dimensionless | `(2, n_orb, n_orb)`, both spin channels |
| `mbxas["dipole_KS"]` | atomic units | `(2, 3, n_orb, n_orb)` |
| `Spectra.energies`, `get_mbxas_spectra` output | eV | - |

Intensity is `energy * amplitude**2`, the photon-energy-weighted absorption cross section `sigma(omega) ~ omega * |M|^2` (Eq. 4 and Eq. 27, Roychoudhury & Prendergast, PRB 107, 035146). `energy` is the transition energy in Hartree, matching the atomic-unit amplitude; `Spectra.amp2int` and `PySCF_mbxas.get_mbxas_spectra` both apply it at the same point (converting amplitude to intensity), never inside `mbxas["absorption"]` itself. The isotropic spectrum is the **mean** over the three Cartesian components before the energy weighting, `energy * sum(amp**2, axis=0) / 3`, matching an orientation average of `|e·d|^2`. Summing instead of averaging inflates every spectrum by exactly 3.

## Known approximations

Ordered by how much they matter.

**Spectator-spin determinant is omitted.** The full many-body overlap is the product over both spin channels, `det(A_alpha) * det(A_beta)`. The code computes `mb_overlap` for both channels but only the excited one enters the amplitude. The non-excited channel also relaxes in the core-hole field, and its determinant is a constant multiplicative factor on every transition. For H2O/O it is 0.916, so absolute intensities are high by `1/0.916^2 = 1.19`. **Spectrum shapes are unaffected**, which is why this has never shown up in a comparison. Applying it would change the absolute intensity of every previously computed result, so it is documented rather than fixed. If you do apply it, it needs a `### Changed` changelog entry. This is also the convention in the reference papers themselves (e.g. the CrO2 majority/minority spin spectra in PRB 106, 075133 Fig. 4 are reported as separate curves, never multiplied) and in the production QE implementation at [`mbxas-qe`](https://gitlab.com/mbxas/mbxas-qe) — but that codebase goes further: `spec.f90`'s `spin_convolve_spectrum` (called from `mbxas_spectra.f90` on the `noci-kpoint-shirley*` branches) energy-convolves each spin channel's own order-resolved shake-up spectrum with the *other* channel's relaxation-overlap spectrum, so the spectator channel contributes genuine satellite structure, not just a constant scale factor. See "One-body truncation" below.

**One-body truncation, with an opt-in order-2 correction.** By default (`shakeup_order=None`) only single shake-up is kept. `get_mbxas_spectra(shakeup_order=1|2|"auto")` (`Spectra`, and `PySCF_mbxas` which delegates to it) additionally convolves in the order-k valence shake-up probability spectrum: a k-fold simultaneous valence-to-conduction excitation, weighted by `|det(K[v_combo, c_combo])|^2`, the k x k minor of the same `K = A' @ inv(A)` matrix already used for the n=1 amplitude (`pymbxas/mbxas/shakeup.py`, `mbxas.mbxas.build_A_K`). This is the exact non-interacting generalization of the `f^(n)` term (PRB 107, 035146, Eq. 32-35), matching `mbxas-qe`'s `singles_overlap`/`doubles_overlap` formula exactly (verified against `K(v,c)*K(vp,cp) - K(v,cp)*K(vp,c)` in `QE/SHIRLEY/src/mbxas_spectra.f90`). Order 3+ raises `NotImplementedError`: the combinatorics grow as `O(n_occ^3 * n_virt^3)`, and pymbxas has no pruning strategy like `mbxas-qe`'s adaptive-tolerance loop in `doubles_overlap`/`triples_overlap` to make that tractable. Cross-spin convolution (the *other*, non-excited channel's own shake-up, via `spin_convolve_spectrum` in `mbxas-qe`'s `spec.f90`) is not implemented; `Spectra._shakeup_sticks`/`get_shakeup_spectrum` accept an explicit `channel` argument specifically so that extension needs no signature change later. See `docs/superpowers/specs/2026-08-21-shakeup-satellites-design.md` for the full design, including the still-unverified Onishi/Fredholm-determinant-type normalization identity that would make "auto" convergence rigorous rather than heuristic.

**No self-interaction correction.** pymbxas runs plain LDA/B3LYP ΔSCF; core-hole self-interaction error is fully present and uncorrected. `mbxas-qe`'s `sic-functional-dev` / `density-matrix-sic` / `sic-projector-augmentation` / `integration/pSIC-core` branches implement a state-dependent SVD-SIC potential (MaxVol-projector-based, split into occupied-removal and virtual-addition channels) to correct this for strongly-correlated and narrow-gap systems. See `SVD_SIC_METHODOLOGY_GUIDE.md` on those branches. Not implemented here; porting it to a molecular ΔSCF/PySCF context would be a substantial project of its own.

**Final states are the FCH virtual manifold.** Their number and quality are set entirely by the basis. Diffuse functions (`def2-svpd` rather than `def2-svp`) matter for near-edge structure; the discrete pseudo-continuum above the edge is a basis artifact and should not be read as physical.

**XCH alignment is a single rigid shift.** It fixes the lowest transition to `E_XCH - E_GS` and assumes the FCH eigenvalue spacing is right above it. The absolute edge position inherits the functional's error: LDA/def2-SVPD puts the H2O oxygen edge at 529.1 eV against 534.5 eV measured.

**Broadening is a normalized Gaussian.** `gaussian_broadening` includes the `1/(σ√2π)` factor, so the kernel integrates to 1 and the integrated area of a broadened spectrum equals the sum of the transition intensities regardless of `sigma`. Spectra broadened with different `sigma` are therefore directly comparable. This was not true previously: the kernel was unnormalized, and absolute y-values were larger by a factor `σ√2π`.

**K-edge, all-electron only.** The 1s AO label is matched literally, so ECP bases and L-edges find no core orbital.

**Two hand-tuned thresholds** in `find_1s_orbitals_pyscf`, neither validated: `0.3 * max(coeff^2)` decides whether an orbital has weight on the target atom, and `1e-1` Hartree (2.7 eV) decides whether two core orbitals are degenerate enough to localize together. The degeneracy window is wide; chemically inequivalent cores within 2.7 eV are localized as one manifold, which is usually what you want but is not what the number says.

**No periodic boundary conditions.** `int1e_r` is not periodic. A lattice sum of it returns finite numbers that are not transition dipoles, so `pbc=True` raises rather than producing them. A periodic implementation needs the velocity gauge (`int1e_ipovlp`) throughout.

## Failure modes

The ΔSCF core-hole steps fail in ways that do not raise on their own.

- **Variational collapse.** MOM is a maximum-overlap constraint, not a projection. The FCH can relax back to the ground state and report `converged = True`. Detect it by projecting the converged hole orbital onto the ground-state core orbital; healthy is above roughly 0.99. For H2O/O the overlap is 0.99993 and the hole is 99.2% O 1s by Mulliken weight.
- **Non-convergence.** A non-converged FCH still has `mo_coeff` and still produces a spectrum.
- **Ill-conditioned `A`.** A near-singular `A` makes `K = A' A^-1` blow up. Healthy is `det(A)` around 0.9 and `cond(A)` near 1; H2O/O gives 0.9486 and 1.016. A `det(A)` far below 0.5 means the two SCF solutions describe different states.
- **Delocalized cores.** If localization did not run or did not separate the core orbitals, `find_1s_orbitals_pyscf` returns more than one index for a single atom and the excitation is aborted rather than guessed at.

## Verification

`tests/test_h2o_kedge.py` runs H2O oxygen K-edge at LDA/def2-SVPD and checks the invariants above. Reference values from that system, which is small enough to reason about:

| Quantity | Value |
|---|---|
| `det(A)`, `cond(A)` | 0.9486, 1.016 |
| Hole orbital character | 99.2% O 1s |
| Hole overlap with GS core | 0.99993 |
| `det(A_spectator)`, unused | 0.916 |
| First transition | 529.136 eV |
| `E_XCH - E_GS` | 529.136 eV (identical by construction) |
| FCH ionization potential | 534.47 eV |
| Transitions retained | 34 of 39 AOs |
| Order-1 / order-2 shake-up mass (H2O/O) | see `tests/test_h2o_kedge.py` order-2-mass assertion |

Two independent checks worth re-running by hand after any change to `run_MBXAS_pyscf`:

- Reimplement the amplitude from the stored `gs_data` and `data["fch"]` and compare. Agreement should be at machine precision (1e-16).
- Translate the molecule by an arbitrary vector and recompute the dipoles. They must not change (1e-15), since both orbitals come from the same FCH calculation and are orthogonal. A nonzero change means orbitals from different SCF runs have leaked into the dipole expression.

## References

- Liang, Vinson, Pemmaraju, Drisdell, Shirley, Prendergast, *Accurate x-ray spectral predictions: an advanced self-consistent-field approach inspired by many-body perturbation theory*, [PRL 118, 096402 (2017)](https://doi.org/10.1103/PhysRevLett.118.096402)
- Roychoudhury, Prendergast, *Efficient core-excited state orbital perspective on calculating x-ray absorption transitions in determinant framework*, [PRB 107, 035146 (2023)](https://doi.org/10.1103/PhysRevB.107.035146) — the core-hole-basis (CHB) reformulation `run_MBXAS_pyscf` implements (Eq. 22), and the source of the `omega` intensity prefactor (Eq. 4)
- Liang, Prendergast, *Quantum many-body effects in x-ray spectra efficiently computed using a basic graph algorithm*, [PRB 97, 205127 (2018)](https://doi.org/10.1103/PhysRevB.97.205127)
- Liang, Prendergast, *Taming convergence in the determinant approach for x-ray excitation spectra*, [PRB 100, 075121 (2019)](https://doi.org/10.1103/PhysRevB.100.075121)
- Reference implementation this package started from: [CleaRIXS](https://github.com/subhayanroychoudhury/CleaRIXS)
- Active plane-wave/QE implementation of the same MBXAS method (periodic, Shirley optimal basis, SIC, NOCI shake-up satellites, GPU): [`mbxas-qe`](https://gitlab.com/mbxas/mbxas-qe), local checkout at `/home/roncofaber/software/mbxas-qe`. `main` is a stable but older snapshot; the interesting development is on unmerged feature branches (`sic-*`, `noci-kpoint-shirley*`, `l-edges`, `gpu-port-shirley*`). Architecturally too different to share code with pymbxas, but useful for tracking where the method is headed.
