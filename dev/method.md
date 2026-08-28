# Method reference

The physics PyMBXAS implements, the conventions the code relies on, and what
is approximated. The fast-reference summary is in `AGENTS.md`; this file is
the authority for the established MBXAS path. The optional shake-up paths are
specified separately in `dev/shakeup.md`.

## What MBXAS computes

X-ray absorption from a core level is not a one-electron transition. Removing
a core electron changes the potential seen by every remaining electron, so
the final state is not a single Slater determinant built from ground-state
orbitals. Valence relaxation redistributes spectral weight and suppresses the
main line.

The determinant approach of Liang and Prendergast evaluates the many-body
overlap between the ground-state determinant and core-excited final-state
determinants. PyMBXAS implements the one-body core-hole-basis amplitude for
molecules, with PySCF supplying the Delta-SCF states. Experimental
overlap-kernel satellites can be added during post-processing, but they are not
the explicit higher-order transition amplitudes of the full `f^(n)` series.

## Workflow

Three SCF calculations are available per excited atom, all unrestricted.

| Step | Charge | Spin | Occupation | Purpose |
|---|---|---|---|---|
| **GS** | `q` | `s` | aufbau | Reference determinant, shared by all excitations. |
| **FCH** | `q + 1` | `s +/- 1` | excited-channel core orbital emptied; MOM, maxvol, or mixed constrained | Supplies final-state orbitals, eigenvalues, and dipoles. |
| **XCH** | `q` | `s` | FCH occupation plus one electron in its lowest-energy ordinary virtual; same constraint as FCH | Supplies absolute energy alignment only. |

The ordinary virtual excludes the core hole, which is identified by maximum
overlap with the selected GS core orbital. This matters for MOM solutions:
occupation zeros need not be Aufbau ordered, and the orbital whose numerical
index equals the electron count may already be occupied. After XCH convergence,
PyMBXAS verifies the electron counts and checks that both the core hole and the
added-electron orbital survived the optimization. `occupation="maxvol"` is
the production default and enables fixed-reference, degree-one determinant
tracking from the first occupation call. `"mom"` and `"mixed"` remain useful
for controlled state comparisons; mixed applies MOM for
`mom_warmup_calls` occupation calls before the same fixed-reference maxvol
selection. Every excitation persists its complete configuration, including
state-specific SCF overrides.

FCH and XCH use a bounded adaptive SCF sequence. Ordinary DIIS receives the
first cycle budget. If it fails, a second DIIS attempt starts from the
lowest-gradient orbitals observed, delays DIIS for two damped iterations, and
applies a virtual level shift. Both stages continue to invoke the configured
occupation controller. A remaining failure is reported rather than handed to
second-order SCF: Newton orbital rotations hold a numerical occupation array
fixed but do not reapply MOM/maxvol, and can therefore leave the intended
diabatic state. `SCFConfig(second_order=True)` is an explicit diagnostic override,
not the production default.

After FCH convergence, PyMBXAS measures `<S^2>` against the ideal value for
the requested spin and compares the complete occupied subspace with the
GS-derived target using its determinant overlap and minimum singular value.
These checks complement the core-hole overlap, which alone cannot detect a
rearrangement elsewhere in the determinant.

### SCF reference policies

The three available controllers express two implemented reference policies:

- **MOM** ranks individual current orbitals by their projection onto the fixed
  target orbitals.
- **Maxvol** selects the current occupied subspace collectively by maximizing
  its determinant overlap with that fixed target.
- **Mixed** changes the selection rule, not the reference: it starts with MOM,
  then maxvol tracks the occupied subspace collectively against the original
  fixed target. It is an optional diagnostic controller, not a preferred
  warm-up stage.

For FCH, the fixed target is the localized GS orbital set with the selected
core occupation removed. It therefore means "remain connected to this
specific GS core excitation." For XCH, the target is the converged FCH orbital
set with one electron placed in the selected spectator orbital. This includes
the spectator explicitly; the unmodified GS occupied space is not a complete
XCH reference.

The Prendergast QE implementation can instead freeze an intermediate SCF
snapshot and later track that snapshot, optionally multiplying its maxvol
criterion by one against the original reference. A snapshot supplies local
SCF continuity, while the original reference supplies the identity of the
requested excitation. QE's core-hole pseudopotential makes its early snapshot
safer than it is in an all-electron calculation. Here, freezing after one MOM
call converged an early low-cost 6fda O-site test to a state about 2.2 eV above
the MOM state, and the literal dual-reference handoff failed within 100 cycles.
Consequently, snapshot-only and dual-reference policies are documented but not
exposed.

A corrected PBE/def2-SVPD O8 comparison on the production geometry found two
stationary FCH branches. MOM and mixed followed by second-order SCF converged
to the same lower-energy state (`<S^2>` about 1.060), while direct maxvol
converged in 21 DIIS cycles to a state 0.261 eV higher with `<S^2>` 0.801 and a
larger target determinant overlap (0.767 versus 0.676). Newton rotations did
not reapply either occupation selector and therefore defeated the intended
diabatic tracking. The 6fda workflow consequently defaults to direct maxvol
and disables second-order recovery.

After each maxvol or mixed SCF, the high-level log reports aggregate selector
wall time, internal row swaps, calls with occupation changes, total selected
orbital changes, and the last changing maxvol call. Internal swaps measure work
needed to improve the current MOM-ranked seed. Occupation changes instead
measure evolution of the SCF eigenvectors and can remain nonzero even when the
current seed already satisfies maxvol. The latter is the relevant warning sign
for an oscillating constrained SCF. Per-call, per-spin details and mixed MOM
warm-up calls are retained in raw PySCF output only at debug verbosity;
maxvol call numbering begins at the mixed handoff.

The FCH spin is `spin_GS + 2*channel - 1`. For the default `channel=1`
(beta), removing a beta electron raises `n_alpha - n_beta` and gives
`spin + 1`. For `channel=0`, PySCF accepts the resulting `spin - 1`, including
negative spin.

When the excited element has several symmetry-equivalent 1s orbitals, the GS
core manifold is localized with IBO (default) or Boys before any excitation.
`gs_data.mo_coeff` is replaced by the localized coefficients and the original
coefficients are retained as `mo_coeff_del`. A determinant built from an
unlocalized degenerate core manifold is not atom specific.

## One-body determinant amplitude

Let `h` be the FCH core-hole orbital, `i` the occupied FCH valence orbitals,
`f` the FCH virtual orbitals excluding the core hole, and `n` the occupied GS
orbitals after removing the excited core orbital.

```text
S_MB = C_FCH^T S_AO C_GS
A    = S_MB[occ_FCH, occ_GS]
A'   = S_MB[virt_FCH, occ_GS]
K    = A' A^-1
```

For Cartesian component `x`, the implemented amplitude is

```text
M_xf = det(A) [ <f|x|h> - sum_i K[f,i] <i|x|h> ].
```

This is Eq. 22 of Roychoudhury and Prendergast, PRB 107, 035146, in the
core-hole basis. Both orbitals in every dipole integral come from the same FCH
calculation, so their orthogonality makes the transition dipole origin
independent.

FCH virtual eigenvalues are rigidly aligned when XCH is enabled:

```text
E[f] = eps_FCH[f] + (E_XCH - E_GS) - min(eps_FCH[virtual]).
```

The lowest returned transition therefore sits exactly at `E_XCH - E_GS`.

## Index conventions

- `channel=0` is alpha and `channel=1` is beta. Every downstream orbital
  array must be indexed with the requested channel.
- The FCH core hole is the unoccupied FCH MO with the largest absolute overlap
  with the selected GS core orbital. It is not inferred from the position of
  an occupation zero.
- The excited-channel virtual manifold contains every zero-occupation FCH MO
  except that overlap-identified core hole.
- The excited GS core orbital is removed by MO number with `np.setdiff1d`, not
  by position with `np.delete`.
- `A` must be square: `n_occ_FCH == n_occ_GS - 1` in the excited channel.
- `find_1s_orbitals_pyscf` returns global MO indices. Occupied-subset loops
  must map through their MO-index arrays before reading orbital energies.
- Unrestricted occupations are `1.0` or `0.0`; comparisons against `== 1`
  are intentional and incompatible with a restricted `2.0/0.0` path.

## Units and array shapes

| Quantity | Unit | Shape |
|---|---|---|
| `mbxas["energies"]` | Hartree | `(n_transitions,)` |
| `mbxas["absorption"]` | atomic units, amplitude | `(3, n_transitions)` |
| `mbxas["mb_overlap"]` | dimensionless | `(2, n_orb, n_orb)` |
| `mbxas["dipole_KS"]` | atomic units | `(2, 3, n_orb, n_orb)` |
| `Spectra.energies`, spectrum energy grids | eV | one-dimensional |

The photon-energy-weighted intensity is

```text
I_f = E_f * mean_x(|M_xf|^2),
```

with `E_f` in Hartree at the point of multiplication. The Cartesian mean is
the isotropic orientation average; using a sum would inflate every spectrum by
three. `Spectra.amp2int` is the implementation, and calculator-level spectrum
generation delegates to `Spectra`.
For spectator-assisted terms, `E_f` is the complete energy after adding the
spectator excitation. The same final energy therefore controls both the peak
position and the photon-energy prefactor.

## Optional order-resolved many-body spectra

`f_order` directly selects the highest cumulative many-body order: 1 returns
f1, 2 returns f1+f2, and 3 returns f1+f2+f3. The spectator channel follows
that order automatically: f1=`10`, f2=`20+11`, and f3=`30+21+12`. Spin
channels are combined by adding energies and multiplying intensities, with
their determinant normalization retained; final sticks are broadened once.
`spectator_order` and `max_total_order` are diagnostic overrides rather than
normal physical controls.

The formulas match `mbxas-qe`, including constituent-level shake-down flags.
Physical spectra include both flagged and unflagged configurations;
`get_mbxas_decomposition()` returns their per-order decomposition, while
`print_mbxas_summary()` reports its integrated values.
Production f2 also uses QE's energy-windowed, count-adaptive `K`-element
screening and convergence test. Excited-channel MB3 and spectator doubles use
QE's corresponding adaptive product-threshold searches. PyMBXAS nevertheless
has a finite molecular Gaussian virtual manifold and no k points or plane-wave
band machinery. Diagnostic overlap distributions and orders above MB3 remain
exact and combinatorial, so an explicit small order is required.

See `dev/shakeup.md` for formulas, exact control flow, validation status, and
the QE comparison.

### Orbital rearrangement diagrams

The GS-to-FCH diagram is a wavefunction diagnostic, not another MBXAS
approximation. Rows of `mb_overlap` are final FCH orbitals and columns are GS
orbitals. PyMBXAS maximizes the total squared overlap with a Hungarian
one-to-one assignment. This prevents several GS orbitals from being assigned
to the same FCH orbital, which independent maximum-overlap choices can do.

MOM and maxvol determine which occupied subspace is followed during SCF; they
do not define a unique final orbital-by-orbital history. Rotations within
degenerate or strongly mixed subspaces are physically equivalent, so weak or
competing connectors indicate ambiguity rather than literal electron paths.
Connector opacity reports squared-overlap confidence. The default view uses
the global GS HOMO as a common zero, retaining real GS/FCH shifts. FCH HOMO and
LUMO are derived from actual occupations, so the lowest FCH unoccupied orbital
can correctly be the deep core hole in a non-Aufbau state.

## Known approximations

Ordered roughly by their effect on interpretation.

**Order truncation.** The default MBXAS spectrum contains spin-complete f1.
The optional path adds explicit determinant amplitudes only through the
requested order; omitted higher orders remain an approximation.

**Spectator-spin truncation.** The default f1 includes the full
`det(A_alpha) * det(A_beta)` factor. At higher requested order, spectator
overlap terms are included through the same total-order cap. Explicitly
passing `spectator_order=None` is a single-channel diagnostic and is not the
physical default.

**Selectable fixed SCF reference.** Production FCH and XCH calculations use
the same selected controller: PySCF MOM by default or fixed-reference maxvol
when requested. PyMBXAS builds the spectral occupied reference directly from
the resulting converged occupations. QE additionally
supports evolving SCF references and post-SCF reference reselection with an
`eshift_swap` correction. Those integration changes remain deferred in
`dev/shakeup.md`.

**No self-interaction correction.** PyMBXAS runs ordinary LDA, hybrid DFT, or
HF Delta-SCF. The state-dependent SIC developments on `mbxas-qe` branches are
not implemented here.

**Finite FCH virtual manifold.** Final states are determined by the molecular
basis. Diffuse functions matter near the edge, and the discrete high-energy
pseudo-continuum is a basis artifact.

**Rigid XCH alignment.** XCH fixes only the lowest transition. All higher
spacing comes from FCH eigenvalues and inherits the functional and basis error.

**Normalized Gaussian broadening.** The Gaussian includes
`1/(sigma*sqrt(2*pi))`, so its area is one and ordinary broadened-spectrum area
does not depend on `sigma`.
For spin-complete many-body spectra, PyMBXAS combines discrete spin-channel
sticks before applying this final-state broadening once. QE broadens both
factors before convolution, so equivalent Gaussian widths require
`sigma_PyMBXAS = sqrt(2) * sigma_QE`; see `dev/shakeup.md`.

**K-edge, all-electron only.** Core identification matches the `1s` AO label
literally. ECP bases and L-edges are unsupported.

**Hand-tuned core thresholds.** `find_1s_orbitals_pyscf` uses
`0.3 * max(coeff^2)` for atomic character and `0.1` Hartree for core
degeneracy. Neither threshold has a systematic validation study.

**No periodic boundary conditions.** The molecular position operator
`int1e_r` is not periodic. `pbc=True` raises; a periodic implementation needs
a consistent velocity-gauge formulation.

## Failure modes

- **Variational collapse:** either occupation constraint can converge to the wrong state. Check SCF
  convergence, electron counts, overlap of the converged hole with the target
  core orbital, and overlap of the XCH electron with its intended FCH virtual.
- **Non-convergence:** unconverged SCF objects still contain plausible-looking
  orbitals; PyMBXAS rejects them.
- **Ill-conditioned `A`:** `K=A'A^-1` becomes unstable. H2O/O has
  `det(A) ~= 0.9486` and `cond(A) ~= 1.016`; a much smaller determinant or much
  larger condition number signals incompatible SCF states.
- **Delocalized core:** if localization cannot isolate one core orbital for an
  atom, the excitation is rejected rather than guessed.
- **Heuristic satellite completeness:** every returned higher-order minor may
  be correct while important configurations are absent. This affects only the
  optional path and is detailed in `dev/shakeup.md`.

## Verification

`tests/test_h2o_kedge.py` runs the H2O oxygen K-edge at LDA/def2-SVPD and
checks SCF convergence, core-hole retention, determinant conditioning,
one-body amplitude reconstruction, XCH alignment, origin independence,
shake-up plumbing, and default-off behavior.

| Quantity | Reference |
|---|---|
| `det(A)`, `cond(A)` | approximately 0.9486, 1.016 |
| Hole orbital character | approximately 99.2% O 1s |
| Hole overlap with GS core | approximately 0.99993 |
| Spectator `det(A)` | approximately 0.916, omitted from base amplitude |
| First transition | approximately 529.136 eV |
| `E_XCH - E_GS` | identical to the first transition by construction |
| FCH ionization potential | approximately 534.47 eV |
| Transitions retained | 34 of 39 AOs |

Synthetic tests exhaustively validate overlap completeness and normalization,
the explicit MB2 and MB3 formulas, and constituent-level shake-down selection.
The molecular integration test validates the all-electron execution path;
direct numerical comparison to a matched QE calculation remains future work.

## References

- Liang et al., *Accurate x-ray spectral predictions: an advanced
  self-consistent-field approach inspired by many-body perturbation theory*,
  PRL 118, 096402 (2017), https://doi.org/10.1103/PhysRevLett.118.096402
- Roychoudhury and Prendergast, *Efficient core-excited state orbital
  perspective on calculating x-ray absorption transitions in determinant
  framework*, PRB 107, 035146 (2023),
  https://doi.org/10.1103/PhysRevB.107.035146
- Liang and Prendergast, *Quantum many-body effects in x-ray spectra
  efficiently computed using a basic graph algorithm*, PRB 97, 205127 (2018),
  https://doi.org/10.1103/PhysRevB.97.205127
- Liang and Prendergast, *Taming convergence in the determinant approach for
  x-ray excitation spectra*, PRB 100, 075121 (2019),
  https://doi.org/10.1103/PhysRevB.100.075121
- Molecular starting point: https://github.com/subhayanroychoudhury/CleaRIXS
- Periodic reference implementation: https://gitlab.com/mbxas/mbxas-qe
