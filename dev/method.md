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

Intensity is `amplitude**2`. The isotropic spectrum is the **mean** over the three Cartesian components, `sum(amp**2, axis=0) / 3`, matching an orientation average of `|e·d|^2`. Summing instead of averaging inflates every spectrum by exactly 3.

## Known approximations

Ordered by how much they matter.

**Spectator-spin determinant is omitted.** The full many-body overlap is the product over both spin channels, `det(A_alpha) * det(A_beta)`. The code computes `mb_overlap` for both channels but only the excited one enters the amplitude. The non-excited channel also relaxes in the core-hole field, and its determinant is a constant multiplicative factor on every transition. For H2O/O it is 0.916, so absolute intensities are high by `1/0.916^2 = 1.19`. **Spectrum shapes are unaffected**, which is why this has never shown up in a comparison. Applying it would change the absolute intensity of every previously computed result, so it is documented rather than fixed. If you do apply it, it needs a `### Changed` changelog entry.

**One-body truncation.** Only single shake-up is kept, which is the standard determinant approximation, not a defect. Multi-electron shake-up and the continuum are outside the model.

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

Two independent checks worth re-running by hand after any change to `run_MBXAS_pyscf`:

- Reimplement the amplitude from the stored `gs_data` and `data["fch"]` and compare. Agreement should be at machine precision (1e-16).
- Translate the molecule by an arbitrary vector and recompute the dipoles. They must not change (1e-15), since both orbitals come from the same FCH calculation and are orthogonal. A nonzero change means orbitals from different SCF runs have leaked into the dipole expression.

## References

- Liang, Vinson, Pemmaraju, Drisdell, Shirley, Prendergast, *Accurate x-ray spectral predictions: an advanced self-consistent-field approach inspired by many-body perturbation theory*, [PRL 118, 096402 (2017)](https://doi.org/10.1103/PhysRevLett.118.096402)
- Liang, Prendergast, *Quantum many-body effects in x-ray spectra efficiently computed using a basic graph algorithm*, [PRB 97, 205127 (2018)](https://doi.org/10.1103/PhysRevB.97.205127)
- Liang, Prendergast, *Taming convergence in the determinant approach for x-ray excitation spectra*, [PRB 100, 075121 (2019)](https://doi.org/10.1103/PhysRevB.100.075121)
- Reference implementation this package started from: [CleaRIXS](https://github.com/subhayanroychoudhury/CleaRIXS)
