# Changelog

Entries describe what changed for someone using the package. Conventions in `AGENTS.md`.

## [Unreleased]

### Added
- Immutable SCF and excitation configurations support validated reusable settings.
- Spectrum decompositions can plot cumulative and shake-up/down-resolved panels.
- The 6fda example accepts any positive f order and plots every contribution.
- `Spectras` can aggregate and plot order-resolved decompositions across sites
  as either a mean or sum.
- `Spectra` can plot two-spin GS-to-FCH orbital rearrangements with occupied,
  frontier, and core-hole levels plus one-to-one MO-overlap connectors.
- New `Spectra` HDF5 files retain GS orbital energies; historical files remain
  readable and explain how to recover plotting data from calculation files.
- The `mbxas` command line tool is now installed with the package
- Optional install extras for the machine learning and GPU paths, `pymbxas[ml]` and `pymbxas[gpu]`
- Saved calculations can be reopened with `.load()` and continued without repeating the ground state
- Saved calculations are readable by PySCF as chkfiles
- `f_order` directly requests cumulative f1, f2, or f3 spectra
- `Spectra.get_mbxas_decomposition()` returns resolved contributions, cumulative spectra, shake-up/shake-down decomposition, total, and overlap probability
- `Spectra.print_mbxas_summary()` prints integrated many-body contributions
- MBXAS decomposition data report captured determinant overlap by total order
- Optional `pymbxas[plot]` extra adds `pymbxas.plotting.plot_mbxas_decomposition()` for a ready-made many-body comparison figure
- HDF5 file creation, file reads, and SCF snapshot writes now log at debug level
- `spectator_order` on `get_mbxas_spectra` adds shake-up satellites from the non-excited spin channel's own valence relaxation
- MBXAS decomposition data report the overlap `shakedown_fraction`
- `scf_max_cycle`, `scf_conv_tol`, and `dft_grid_level` expose shared PySCF controls for GS, FCH, and XCH calculations.
- FCH and XCH can select CPU/GPU maximum-volume occupation tracking instead of MOM.
- Maxvol SCF logs report selector timing, determinant swaps, and occupation changes.
- Mixed SCF tracking can warm up with MOM before fixed-reference maxvol selection.
- Excitation requests expose per-site success, failure, and skip outcomes.
- Raw PySCF logs identify each GS, FCH, and XCH calculation block.
- SCF logs summarize convergence, validation overlaps, and occupation tracking.
- Console and file logs use readable multiline scientific summary blocks.
- Constrained FCH/XCH SCF now recovers from DIIS plateaus with stabilized
  mixing followed by CPU/GPU second-order orbital optimization.
- Live PySCF logs report maxvol occupation changes as they occur.
- Second-order SCF is now opt-in for constrained states because it bypasses
  iterative MOM/maxvol selection; controller-preserving stabilized DIIS is
  the default recovery path.
- FCH summaries report spin contamination and full occupied-subspace drift.
- The 6fda production example now defaults to direct maxvol tracking.
- The 6fda example can select PBE or B3LYP without sharing checkpoints.

### Changed
- Direct maxvol is now the default constrained-SCF occupation controller;
  MOM and mixed MOM/maxvol tracking remain explicit diagnostic choices.
- Calculations now use `PySCFMBXAS`, configuration objects, and explicit `run()` execution.
- Checkpoints now store immutable calculation and per-excitation configurations with runtime provenance.
- Loading checkpoints accepts runtime settings only; scientific settings remain unchanged.
- Orbital-rearrangement diagrams now use thin energy-level strokes instead of
  filled and hollow boxes, with quieter background overlap connectors.
- FCH LUMO highlighting excludes the constrained core hole, which remains a
  separately identified level in the excited spin channel.
- Orbital diagrams optionally add compact Gaussian-broadened level-density
  profiles in the existing margins beside GS and FCH columns.
- Explicit orbital-diagram energy windows now set identical displayed limits
  across sites instead of merely filtering levels before autoscaling.
- The 6fda example groups outputs by type and writes per-site orbital diagrams.
- Run summaries now identify partial failures instead of reporting unconditional success.
- Fresh calculation logs overwrite old runs while HDF5 restarts append.
- HDF5 restarts can override the saved PySCF verbosity for SCF diagnostics.
- Routine maxvol calls and screening iterations now require debug verbosity.
- Physical spectra now always include shake-up and shake-down configurations.
- Order-2 overlap diagnostics now use screened factors and streamed cross-spin products.
- XCH alignment now fills the lowest-energy ordinary FCH virtual identified from the actual MOM occupation.
- Many-body order is now a single physical control: the default f1 includes
  spectator S0, `f_order=2` assembles f2=`20+11`, and `f_order=3` assembles
  f3=`30+21+12`. Opposite-spin omission now requires the explicit diagnostic
  override `spectator_order=None`.
- Shake-up spectra now use the explicit order-resolved determinant amplitudes from `mbxas-qe`; f2 adds MB2 transitions and f3 adds MB3 transitions.
- Spectator overlap weights now retain `|det(A)|²`, shake-down selection uses any negative-energy constituent promotion, and optional many-body spectra are no longer unit-normalized.
- `order="auto"` was removed from overlap-stick helpers because the former maxvol stopping rule had no completeness guarantee; choose an explicit order.
- Production MB2 spectra now use QE-style energy-window and adaptive
  `|det(A)K|²` screening over the complete FCH orbital manifold. Pair
  screening is independent of the dipole-final orbital selection, spectator
  products are streamed in bounded chunks, and guarded higher orders fail
  explicitly instead of silently truncating by orbital position.
- MB2 screening now uses QE's count-adaptive threshold reduction and reports convergence.
- MB3 now uses QE's adaptive product-threshold search.
- Spectator doubles now use QE's adaptive product-threshold search.
- Spectator-assisted intensities now use the complete final-state photon energy.
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
- Full and shake-down spectra now reuse one screened many-body configuration set.
- Many-body screening logs now identify their site and convergence evidence.
- Library utilities no longer print errors or silently discard requested log files.
- Invalid geometry metric inputs now raise clear errors.
- Logger reconfiguration now preserves caller handlers and closes replaced files.
- Fixed invalid XCH states when the electron-count MO index was already occupied.
- Core holes are now identified by overlap with the selected ground-state core orbital.
- Verbosity level 5 now logs more detail than level 4, instead of the two being identical
- The shake-up probability plot range no longer clips satellites at negative energy shifts
- `ase_to_mole` now validates molecule keywords against the installed PySCF
  `Mole.build`/`Cell.build` signatures and raises for unsupported arguments.
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
- Fixed excited-channel shake-up corrections using overlap-probability convolution in place of explicit higher-order XAS amplitudes.
- Fixed the maxvol-style spectral search omitting valid configurations without a captured-mass guarantee.
- Fixed valid PySCF molecule-build arguments such as `magmom` being reported as unrecognized; unsupported arguments now fail immediately, and calculation-only settings are no longer forwarded into `Mole`.
- Fixed fresh calculations appending into an older HDF5 target and silently
  reusing stale excitation indices; only a calculation resumed from the same
  backing file now appends.
