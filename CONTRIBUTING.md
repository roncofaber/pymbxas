# Contributing to PyMBXAS

## Getting started

PyMBXAS uses a conda environment named `pymbxas` with PySCF, ASE, and NumPy pre-installed. Verify your environment:

```bash
conda run -n pymbxas python -c "import pymbxas; print(pymbxas.__version__)"
```

## Testing

The test suite combines focused unit tests with a full H2O oxygen K-edge
workflow. It covers SCF convergence and recovery, MOM/maxvol occupation
tracking, core-hole retention, XCH alignment, determinant algebra, HDF5
restart behavior, shake-up/down assembly, logging, and plotting.

```bash
conda run -n pymbxas pytest tests/ -q
```

Before committing changes to `mbxas/`, `calculators/`, `build/`, or
`utils/orbitals.py`, run the complete suite. New behavior and physics
invariants should receive focused regression tests.

GPU-specific tests skip when CUDA is unavailable. On a GPU host, also run them
explicitly with `pytest -q tests/test_maxvol_occ.py -k gpu` so a skip in the
general suite is not mistaken for GPU coverage.

## Changelog

Add an entry under `## [Unreleased]` in `CHANGELOG.md` in the same change that makes something visible to users. Never backfill from git history. Group under `### Added` / `### Changed` / `### Fixed` as applicable.

Each entry is one line, under 15 words, and states what changed in observable behavior (never why or how). Reference no files, functions, or variables. Any change that alters computed numbers must be marked `### Changed` so spectra can be traced back to the version that produced them.

## Versioning

`pymbxas/__init__.py` (`__version__`, `__date__`) is the single source of truth. Bump `CITATION.cff` in the same change.

## Method invariants

Breaking one silently produces a wrong spectrum, not an error. The method
invariants are summarized in `AGENTS.md` and derived in `dev/method.md`. The
experimental satellite path and its current limitations are documented in
`dev/shakeup.md`. The key established invariants are:
- Only unrestricted calculations (UKS/UHF).
- `mo_occ` is 1.0/0.0, not 2.0/0.0; comparisons against `== 1` are load-bearing.
- Identify the FCH core hole by maximum overlap with the selected GS 1s
  orbital; non-Aufbau occupations make the first unoccupied index unreliable.
- Ground-state orbitals indexed by MO number, not position; use `np.setdiff1d`, not `np.delete`.
- Units Hartree internally, eV only at presentation.
- Amplitude is shape `(3, n_transitions)` and is squared before broadening.
- Transition dipoles are origin-independent because both orbitals come from the same FCH calculation.

See `dev/method.md` for the full physics and `dev/architecture.md` for the object graph and persistence design.
