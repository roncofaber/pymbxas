# Contributing to PyMBXAS

## Getting started

PyMBXAS uses a conda environment named `pymbxas` with PySCF, ASE, and NumPy pre-installed. Verify your environment:

```bash
conda run -n pymbxas python -c "import pymbxas; print(pymbxas.__version__)"
```

## Testing

There is one end-to-end test, `tests/test_h2o_kedge.py`. It runs the full H2O oxygen K-edge workflow (roughly 15 seconds) and verifies the method invariants: SCF convergence, core-hole retention, determinant sanity, XCH alignment, and spectral shape.

```bash
conda run -n pymbxas pytest tests/ -q
```

**Before committing any changes to `mbxas/`, `calculators/`, `build/`, or `utils/orbitals.py`, this test must pass.** It is the only thing preventing silently wrong spectra from reaching users. Do not grow the test suite - if you need a new physics invariant, add an assertion to the existing test.

## Changelog

Add an entry under `## [Unreleased]` in `CHANGELOG.md` in the same change that makes something visible to users. Never backfill from git history. Group under `### Added` / `### Changed` / `### Fixed` as applicable.

Each entry is one line, under 15 words, and states what changed in observable behavior (never why or how). Reference no files, functions, or variables. Any change that alters computed numbers must be marked `### Changed` so spectra can be traced back to the version that produced them.

## Versioning

`pymbxas/__init__.py` (`__version__`, `__date__`) is the single source of truth. Bump `CITATION.cff` in the same change.

## Method invariants

Breaking one silently produces a wrong spectrum, not an error. The method invariants are documented in `CLAUDE.md` and derived in `dev/method.md`. The key ones:
- Only unrestricted calculations (UKS/UHF).
- `mo_occ` is 1.0/0.0, not 2.0/0.0; comparisons against `== 1` are load-bearing.
- The core hole is the lowest-energy unoccupied MO: `np.where(mo_occ[channel] == 0)[0][0]`, never hardcoded as index 0.
- Ground-state orbitals indexed by MO number, not position; use `np.setdiff1d`, not `np.delete`.
- Units Hartree internally, eV only at presentation.
- Amplitude is shape `(3, n_transitions)` and is squared before broadening.
- Transition dipoles are origin-independent because both orbitals come from the same FCH calculation.

See `dev/method.md` for the full physics and `dev/architecture.md` for the object graph and persistence design.
