# PyMBXAS - Project Instructions

Python implementation of Many-Body X-ray Absorption Spectroscopy (MBXAS) for molecules, built on PySCF and ASE. Computes core-level XAS via a ΔSCF core-hole workflow plus the one-body determinant approximation to the many-body transition amplitude.

- **Package** `pymbxas` · **Main branch** `main` · **Working branch** `dev` · **Remote** GitLab (`roncofaber/pymbxas`)
- **Stack** PySCF 2.x, ASE 3.23+, NumPy 2.x, h5py, psutil, scikit-learn
- **Optional** gpu4pyscf (GPU), MOKIT (fchk output), gpflow + TensorFlow (`explorer/`), sea_urchin (structure alignment)
- **Method** GS → FCH → XCH ΔSCF with MOM, then determinant-based amplitudes. See `dev/method.md`.

## Documentation map

This file is the fast-reference layer; `dev/*.md` is the deep reference. **Change something a `dev/` file describes, update that file in the same change.**

| File | Covers | Update when you... |
|---|---|---|
| `dev/method.md` | The physics: ΔSCF workflow, determinant amplitude, index conventions, XCH alignment, units, known approximations and their numerical size | Touch anything in `mbxas/`, `calculators/`, or `utils/orbitals.py`; change what is approximated; change what a returned array means |
| `dev/architecture.md` | Module map, object graph, data flow, pickle/restart formats, GPU path, extension points | Add or rename a module, change a stored attribute, change the `PySCF_mbxas` / `Excitation` / `Spectra` contract |

## Environment

Conda env `pymbxas`. PySCF, ASE and NumPy are already installed there; gpu4pyscf is **not**.

```bash
conda run -n pymbxas python -c "import pymbxas; print(pymbxas.__version__)"
```

## Tests

There are two test files. `tests/test_h2o_kedge.py` runs the full H2O oxygen K-edge (LDA/def2-SVPD, roughly 15 s) and asserts the method invariants listed in `dev/method.md`: SCF convergence, core-hole retention, `det(A)` sanity, the XCH alignment identity, and a non-empty spectrum. `tests/test_h5_io.py` covers HDF5 save/load round-trip fidelity for `Spectra`, `Spectras`, and calculation checkpoints - construction/serialization plumbing, not physics invariants.

```bash
conda run -n pymbxas pytest tests/ -q
```

**Run the full suite before every commit that touches `mbxas/`, `calculators/`, `build/`, or `utils/orbitals.py`.** The physics test is the thing standing between a refactor and a silently wrong spectrum. There is no CI; nothing runs it for you.

Scope of `test_h2o_kedge.py` is deliberately one calculation. Do not grow it into a broad unit-test suite. If you need to check a new physics invariant, add an assertion to that test rather than creating a new file.

## Versioning

`pymbxas/__init__.py` (`__version__`, `__date__`) is the single source of truth; `setup.cfg` reads it via `attr:`. **Bump `CITATION.cff` in the same change** - it has drifted before.

## Changelog discipline

Add a `CHANGELOG.md` entry under `## [Unreleased]` **in the same change** that makes something visible to someone using the package, not at release time. Entries are never backfilled from git history; unwritten means lost. Create `## [Unreleased]` at the top if absent.

Group under `### Added`/`### Changed`/`### Fixed`, only the sections that apply. One line per entry, under ~15 words, hard cap one sentence. The audience is a researcher calling the API, so state *what* changed in observable behavior, never *why* or *how* - no root cause, no file/function/variable names, no parentheticals; that belongs in the commit message. Internal refactors get no entry, but a bug fix that fell out of one does.

- Bad: "Fixed `find_1s_orbitals_pyscf` enumerating `coefficients.T[occ_idxs]` so that `cc` indexed the occupied subset rather than the global MO index, which broke degeneracy lookup for non-aufbau occupations."
- Good: "Fixed core orbital identification for non-aufbau occupations."

**A change that alters computed numbers always gets a `### Changed` entry**, even if it is a bug fix, so a spectrum can be traced back to a version. Say so plainly: "Absolute intensities now include the spectator-spin determinant."

## Method invariants

These hold across the whole package. Breaking one produces a plausible-looking wrong spectrum, not an error. Derivations and verification numbers in `dev/method.md`.

- **Spin channel** `channel=1` is beta and is the default excited channel. `channel=0` (alpha) works. Everything downstream indexes `mo_coeff[channel]`, `mo_occ[channel]`, `mo_energy[channel]`; never assume channel 1.
- **Unrestricted only.** `mo_occ` is tested against `== 1`. An RKS/RHF path would silently match nothing. `calc_type` must be `UKS` or `UHF`.
- **The core hole is the lowest-energy unoccupied MO of the excited channel**, i.e. `np.where(mo_occ[channel] == 0)[0][0]`. This is index 0 only when the excited atom is the heaviest; for C in CO it is index 1. Always locate it by that expression, never hardcode 0. The virtual manifold is the same array `[1:]`.
- **Ground-state orbitals must be indexed by MO number, not by position** in the occupied list. Use `np.setdiff1d`, not `np.delete`, when removing the excited core orbital.
- **Units** are Hartree everywhere internally. Conversion to eV happens only at the presentation boundary (`get_mbxas_spectra`, `Spectra.energies`). Do not return eV from anything in `mbxas/`.
- **`absorption` is an amplitude, shape `(3, n_transitions)`, not an intensity.** Intensity is `energy * amplitude**2` (Hartree energy, matching the atomic-unit amplitude), the `sigma(omega) ~ omega * |M|^2` cross section of Eq. 4/27 in PRB 107, 035146. Isotropic intensity is the *mean* over the three Cartesian components, not the sum, applied before the energy weighting. `Spectra.amp2int` and `PySCF_mbxas.get_mbxas_spectra` are the only two places this conversion happens; they must stay numerically identical.
- **XCH alignment** shifts the FCH virtual eigenvalues so the lowest one sits at `E_XCH - E_GS`. The electron added in the XCH step must be the same orbital that defines that minimum.
- **Transition dipoles are origin-independent** because both orbitals come from the same FCH calculation and are therefore orthogonal. Verified to 4e-15. Any change that mixes orbitals from different SCF runs into the dipole breaks this.
- **Only the excited spin channel enters the amplitude.** The spectator channel determinant is deliberately omitted; see the "Known approximations" section of `dev/method.md` before changing it.

## Key architecture decisions

- **`PySCF_mbxas` is the entry point and owns the ground state.** One GS, then N `Excitation` objects in `_excitations`, one per excited atom. `kernel()` = GS + all excitations + save.
- **`Excitation` runs FCH then XCH then MBXAS** in `__init__`. It is not lazy. A failure raises rather than producing a partial object; `_single_excite` catches and logs so one bad atom does not kill a batch.
- **`pyscf_data` is a snapshot, not a calculator.** It holds `mol`, `mo_coeff`, `mo_occ`, `mo_energy`, `e_tot`, `nelec` and nothing else. This is what makes the object picklable. `to_cpu()`/`to_gpu()` move the arrays; `mol` is never converted.
- **Localization is only applied when needed.** If the excited element has one 1s orbital, GS coefficients are used as-is. Otherwise IBO (default) or Boys runs over the core manifold and `gs_data.mo_coeff` is replaced, with the delocalized set kept as `mo_coeff_del`. `_used_loc` records which happened. A determinant built from unlocalized degenerate cores is meaningless. This decision is made once at the GS stage, before any excitation chooses its channel, so it checks both spin channels and localizes if either needs it — never gate this on one hardcoded channel.
- **`Spectra` / `Spectras` are the post-processing layer**, decoupled from the calculators. They carry the FCH MOs and the amplitudes, rebuild `mol` from the ASE structure on load, and never re-run SCF.
- **Persistence is HDF5 via `io/h5.py`**, which is the only module that imports h5py. The layout mirrors PySCF's chkfile shape, so `chkfile.load_scf` works on a checkpoint, but the writer is ours so that arrays can be gzipped. `.pkl` files from 0.5.x and earlier cannot be read; support was removed in 0.6.0.
- **Checkpoint writes are append-only.** `save_object()` writes the header and ground state once, then adds one `/excitations/NNN` group per finished atom. The `complete` attribute is written last, and a group without it is treated as absent by both the loader and the next write.
- **PBC is blocked, not supported.** The position operator is not periodic, so lattice-summed `int1e_r` returns numbers that are not transition dipoles. `pbc=True` raises. Reviving it needs the velocity gauge, not a flag flip.
- **`explorer/` is WIP and optional.** It requires gpflow and TensorFlow, which are not installed and not declared as hard dependencies. Nothing in the core path imports it.

## Gotchas

- **`mo_occ` values are 1.0/0.0 (unrestricted), not 2.0/0.0.** Comparisons against `== 1` are load-bearing and appear in `mbxas.py`, `orbitals.py` and `spectra.py`.
- **MOM does not guarantee the core hole survives.** `pyscf.scf.addons.mom_occ` is a maximum-overlap constraint, not a projection. Variational collapse to the ground state is a real failure mode and it converges cleanly when it happens. Check `.converged` *and* that the hole orbital still overlaps the target core orbital.
- **`ase_to_mole` forwards unknown `**kwargs` to `gto.Mole`, which silently ignores them.** `Spectra.make_mol` passes the whole `calc_settings` dict, so `xc`, `calc_type`, `loc` and `xch` are handed to `Mole` and dropped. Harmless today, but do not rely on a kwarg reaching `Mole` without checking.
- **`np.array(mol.ao_labels(fmt=False))` upcasts the atom index to a string.** Comparing a row against `(idx, symbol, "1s", "")` still works because NumPy stringifies the tuple too. Do not "fix" this by casting; do check it if you change the label matching.
- **The 1s AO label is hardcoded**, so the package is K-edge and all-electron only. An ECP basis or an L-edge finds no orbital and raises.
- **`find_1s_orbitals_pyscf` uses two magic thresholds**: `0.3 * max(coeff^2)` for "this orbital has weight on that atom", and `1e-1` Hartree (2.7 eV) for "these cores are degenerate". Both are tuned by hand and neither is validated. Widening the degeneracy window pulls more orbitals into the localization.
- **`self.mol.stdout.close()` runs at the end of `run_ground_state`.** Anything that writes to the mol logger afterwards fails. Each of FCH and XCH builds its own `mol` with `append=True` for this reason.
- **`kernel()` chdirs into `_tdir` and back.** Relative paths inside a calculation resolve against the target directory, not the caller's cwd.
- **`pbc` defaults to `None`, which infers from `structure.get_pbc()`.** A periodic ASE structure therefore raises `NotImplementedError` rather than being silently treated as an isolated molecule. Passing `pbc=False` explicitly still forces molecular treatment of a periodic cell, which is occasionally what you want for a cluster cut out of a solid.
- **`get_mbxas_spectra` exists on two classes plus a free function**: `Spectra` (takes `el_label`, `shakeup_order`, `spectator_order`, `max_total_order`, `shakedown_only`) is the one real implementation; `PySCF_mbxas` (takes `ato_idx`) is a thin wrapper that sums per-excitation `Spectra` results on a shared `erange`; `mbxas.broaden` is the free function `Spectra` calls for the base (non-shake-up) broadening. Do not reintroduce a third independent implementation.
- **`make_pyscf_calculator` takes `is_gpu`, not `gpu`.** Calling it with `gpu=True` silently disabled the GPU for the entire package until this was caught; it no longer has a `**kwargs` catch-all, so an unrecognized keyword now raises instead of being dropped (`ase_to_mole` still forwards `**kwargs` to `gto.Mole`/`Cell` and logs a warning when it does, but does not raise). When adding a parameter to either function, check every call site by name rather than assuming it arrives.
