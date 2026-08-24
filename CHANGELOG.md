# Changelog

Entries describe what changed for someone using the package. Conventions in `CLAUDE.md`.

## [Unreleased]

### Added
- The `mbxas` command line tool is now installed with the package
- Optional install extras for the machine learning and GPU paths, `pymbxas[ml]` and `pymbxas[gpu]`
- Saved calculations can be reopened with `.load()` and continued without repeating the ground state
- Saved calculations are readable by PySCF as chkfiles
- `shakeup_order` on `get_mbxas_spectra` adds valence shake-up satellite intensity beyond the one-body truncation
- `Spectra.get_shakeup_summary()` returns bare and shake-up-corrected spectra together with the underlying probability curve
- Optional `pymbxas[plot]` extra adds `pymbxas.plotting.plot_shakeup_summary()` for a ready-made shake-up comparison figure
- HDF5 file creation, file reads, and SCF snapshot writes now log at debug level
- `spectator_order` on `get_mbxas_spectra` adds shake-up satellites from the non-excited spin channel's own valence relaxation
- `shakedown_only` isolates negative-energy shake-down combinations in shake-up spectra
- `get_shakeup_summary` now reports a `shakedown_fraction` for the shake-up probability distribution

### Changed
- Spectral intensity now includes the photon-energy prefactor of the absorption cross section, so relative peak heights shift slightly across an edge
- Broadened spectra are now area-normalized, so integrated intensity no longer depends on `sigma`
- A structure's own periodicity now decides whether a calculation is treated as periodic
- Periodic calculations now raise an error instead of returning unphysical intensities
- Calculations now stop with an error instead of continuing from an unconverged SCF
- An excitation whose core hole collapses to the ground state now fails instead of returning a spectrum
- `do_xch=False` now skips the XCH step and returns unaligned energies, instead of being ignored
- Verbosity level 5 is now the most detailed setting instead of the quietest
- PyMBXAS logging no longer clears logging handlers set up by your own program
- Temporary file cleanup is now limited to PySCF scratch files
- Minimum supported Python is now declared as 3.9
- Calculations and spectra are now saved as HDF5 files, and `.pkl` files can no longer be loaded
- Reloading a calculation no longer reads orbital coefficients until they are used

### Fixed
- Verbosity level 5 now logs more detail than level 4, instead of the two being identical
- The shake-up probability plot range no longer clips satellites at negative energy shifts
- `ase_to_mole` now warns when forwarding an unrecognized keyword to PySCF's `Mole`/`Cell` constructor
- Failed temporary-file cleanup now logs a warning instead of failing silently
- Batch calculation failures now identify which structure and excitation target failed, instead of a generic message
- Ground-state localization is now decided from both spin channels, not just beta, so exciting the alpha channel no longer risks skipping needed localization
- Fixed identification of the core orbital to excite when occupations are not aufbau
- Fixed near-degenerate core orbitals being listed twice, which could break localization
- Excitations with still-delocalized core orbitals are now skipped instead of left unusable
- Fixed the highest core orbital being dropped when localizing with a `loc_type` ending in `m`
- An unsupported `loc_type` or `calc_type` now raises a clear error instead of failing later
- `Spectra.copy()` and `Spectras.copy()` now return independent objects
- Failed excitations now keep their PySCF output and close the log file instead of leaking it
- Batch and command line runs now report the full traceback when a calculation fails
- Batch runs no longer default to requesting a GPU
- Fixed `gpu=True` having no effect; GPU calculations previously ran on the CPU or crashed
- Passing an unknown keyword to the calculator builder now raises instead of being ignored
- Fixed shake-up convolution using tens of GB of memory when combining with a spectator channel reaching high-lying diffuse virtuals
- Fixed spectral broadening exhausting memory for spectra built from millions of shake-up sticks
