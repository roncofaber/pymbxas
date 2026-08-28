# Architecture

Module layout, object contracts, and data flow. Established MBXAS physics lives
in `dev/method.md`; experimental satellite post-processing lives in
`dev/shakeup.md`. This file describes where those operations happen and what
their objects promise each other.

## Package layout

```
pymbxas/
  __init__.py         version, date, re-exports Spectra and Spectras
  config.py           calculation, SCF, excitation, runtime, and I/O configs
  calculators/
    pyscf.py          PySCFMBXAS      entry point, owns the ground state
    excitation.py     Excitation       one excited atom: FCH + XCH + MBXAS
    maxvol.py         CPU/GPU maximum-volume SCF occupation utility
  mbxas/
    mbxas.py          run_MBXAS_pyscf, overlap partitions, A/K construction
    broaden.py        get_mbxas_spectra, broadened_spectrum, gaussian_broadening
    shakeup.py        explicit many-body amplitudes and overlap sticks
    maxvol.py         standalone Sherman-Morrison row-update utility
  build/
    structure.py      ase_to_mole, mole_to_ase, rotate_structure
    input_pyscf.py    make_pyscf_calculator, make_density_fitter
  utils/
    orbitals.py       find_1s_orbitals_pyscf
    boys.py           do_localization_pyscf  (IBO / Boys)
    basis.py          get_AO_permutation, get_l_val
    indexing.py       atoms_to_indexes
    auxiliary.py      as_list, get_available_memory
    check_keywords.py check_pbc
    metrics.py        scalers and distance features, used only by explorer/
  io/
    data.py           pyscf_data       serializable SCF snapshot
    h5.py             HDF5 schema, lazy arrays, compression primitives
    config.py         configure_logger
    logger.py         Logger           PySCF stdout capture
    write.py          write_data_to_fchk (MOKIT, optional)
    cleanup.py        remove_tmp_files
  spectra.py          Spectra          one excited atom, post-processing
  spectras.py         Spectras         a collection, averaging and alignment
  drivers/
    acquisitor.py     pyscf_acquire    structure in, Spectra out
  cli/pyscf.py        argparse front end
  tools/geometry.py   COM, inertia tensor, angular momentum (used by md/)
  md/                 AIMD and geometry optimization, fork of pyscf.md
  explorer/           Gaussian-process spectrum prediction, WIP, optional deps
  examples/
bin/mbxas.py          console entry point
tests/                physics/integration and HDF5 tests
dev/                  maintained method and architecture references
```

## Core flow

```
ase.Atoms
   │  ase_to_mole
   ▼
gto.Mole ──► make_pyscf_calculator ──► UKS/UHF ──► PySCF kernel()
   │                                                  │
   │                                          pyscf_data(gs)
   │                                                  │
   │                          find_1s_orbitals_pyscf ─┤
   │                                                  ▼
   │                       do_localization_pyscf (only if degenerate cores)
   │                                                  │
   │                                        gs_data.mo_coeff replaced
   ▼                                                  │
Excitation(ato_idx) ◄─────────────────────────────────┘
   ├── _run_fch   charge+1, spin±1, MOM/maxvol ──► pyscf_data(fch)
   ├── _run_xch   charge,   spin,   MOM/maxvol ──► pyscf_data(xch)
   └── _run_mbxas ──► run_MBXAS_pyscf ──► self.mbxas dict
                                              │
                            PySCFMBXAS.to_spectra()
                                              ▼
                                      Spectra / Spectras
                                              │
                          base broadening ─────┤
                                              │ optional
                         explicit determinant amplitudes, spectator overlaps,
                         order-resolved spin combination and broadening
                                              ▼
                                        (energy_eV, intensity)
```

The ground state runs once. Each `Excitation` is independent given `gs_data`, so the loop in `excite()` is embarrassingly parallel in principle, though it is serial today.

`calculators/maxvol.py` owns the reversible occupation-controller dispatch.
The default `occupation="maxvol"` replaces a
UHF/UKS object's `get_occ` callback with fixed-reference, determinant-based
occupied-subspace selection on NumPy or CuPy arrays. `"mom"` applies PySCF
MOM. The diagnostic `"mixed"` controller wraps both
callbacks, dispatching the first `mom_warmup_calls` occupation calls to MOM
and every later call to maxvol without changing the reference. FCH and XCH
always use the same persisted method and warm-up count.

## Object contracts

### Configuration

Public configurations are validated templates. `.set()` mutates a user-owned template transactionally and `.copy()` creates an independent variant. When a template crosses into a calculator or execution call, PyMBXAS stores an immutable snapshot. Direct field assignment is rejected, as is `.set()` on a calculator-owned snapshot. `to_dict()` and `from_dict()` form the strict JSON/HDF5 boundary; unknown keys are rejected.

The classes are shallow and lifecycle-specific. `CalculationConfig` contains only the electronic model. Independent `SCFConfig` values are passed as `gs_scf`, `fch_scf`, and `xch_scf`. `ExcitationConfig` contains channel, XCH, and occupation tracking. `RuntimeConfig` contains device and work directory; `LoggingConfig` is separate. A checkpoint is a direct HDF5 path, `None`, or an advanced `CheckpointConfig` carrying artifact policy.

### `PySCFMBXAS`

Owns one electronic model, one ground state, and a list of excitations. Two ways in: a structure plus construction configuration, or `.load()` to restore a checkpoint. Runtime and logging are supplied at execution boundaries.

| Attribute | Meaning |
|---|---|
| `structure` | the ASE `Atoms` |
| `mol` | the ground-state `gto.Mole` |
| `gs_data` | `pyscf_data` for the ground state, with `mo_coeff` possibly localized |
| `gs_data.mo_coeff_del` | the pre-localization coefficients, **only present if localization ran** |
| `_excitations` | list of successful `Excitation` results; one site/channel may have multiple configurations |
| `calculation`, `gs_scf` | immutable electronic-model and GS-solver snapshots |
| `checkpoint` | immutable checkpoint policy, or `None` |
| `runtime`, `logging` | immutable snapshots from the latest execution |
| `_ran_GS`, `_used_loc` | state flags consulted by `run_gs` and `_print_fchk_files` |

`run_gs()` runs only GS. `excite()` requires a completed GS. `run()` runs a missing GS and then the requested excitations. None changes the process working directory, and no excitation call accepts electronic-model overrides.

### `Excitation`

`__init__` records and validates the request and locates the GS core orbital.
The explicit `.run()` method performs FCH, optional XCH, and determinant math.

| Attribute | Meaning |
|---|---|
| `ato_idx`, `symbol`, `channel` | which atom, which spin |
| `config` | exact physical excitation configuration |
| `fch_scf`, `xch_scf` | exact constrained-state solver snapshots |
| `orb_idx` | global MO index of the ground-state core orbital being emptied |
| `data["fch"]`, `data["xch"]` | `pyscf_data` snapshots |
| `output["fch"]`, `output["xch"]` | captured PySCF stdout as a string |
| `mbxas` | dict: `energies`, `absorption`, `mb_overlap`, `dipole_KS`, `basis_ovlp` |

A failure raises rather than leaving a partial object. `PySCFMBXAS._single_excite` catches and logs so one bad atom does not abort a batch, which means **`len(obj.excitations)` can be smaller than the number of atoms requested**. Check it.

### `pyscf_data`

A snapshot, not a calculator: `mol`, `mo_coeff`, `mo_occ`, `mo_energy`,
`e_tot`, `nelec`. This narrow surface makes the object graph serializable and
lets `to_cpu()`/`to_gpu()` be a simple array walk. `mol` is never converted
between devices.

Anything that needs a real calculator has to rebuild one with `make_pyscf_calculator`.

### `Spectra` and `Spectras`

The post-processing layer, deliberately decoupled from the calculators.
`Spectra` carries the FCH MOs, amplitudes, structure, both-spin MB overlap,
GS/FCH orbital energies, GS occupations, and the GS core-orbital index. Those
extra fields make the optional shake-up and spectator-channel kernels
self-contained after save/load. `Spectra` rebuilds `mol` from the structure on
load and never re-runs SCF.

`_mo_coeff` is `(2, nao, nmo)` and `_mo_occ` is `(2, nmo)`, with `_channel` selecting the excited one through the `_active_mo_coeff` / `_active_mo_occ` properties. Both are always stored with the spin axis present, so those properties index it unconditionally.

`CMO` is the virtual manifold with the core hole dropped, matching the transition list one-to-one. `_el_labels` labels those virtuals for clustering and must stay the same length.

`Spectra.get_mbxas_spectra` is the single spectrum implementation. Its default
is spin-complete f1 (`10`). `f_order` uses direct physical notation: 2 adds
f2=`20+11`, and 3 adds f3=`30+21+12`. It builds the
explicit excited-channel amplitudes, combines them with the required spectator
overlap sticks, and broadens the final sticks once. Explicit spectator and
total-order overrides exist for diagnostics. The calculator-level method only
builds one or more `Spectra` objects and sums their results on a shared energy
grid. See `dev/shakeup.md` before changing or interpreting this path.

`Spectras` is a list wrapper with averaging, atomic and electronic labels, and structure alignment through sea_urchin. Indexing with an int returns a `Spectra`; with a slice, list or boolean mask it returns a new `Spectras`. Its decomposition API aggregates every numerical field on a common grid, using a mean by default or a sum for independent absorbing sites, while retaining site-level overlap reports.

`pymbxas.plotting` contains optional, data-oriented Matplotlib helpers.
Matplotlib is imported only inside plotting calls. Spectrum rendering consumes
decomposition dictionaries from either post-processing class. Orbital
rendering consumes `Spectra.get_orbital_rearrangement()`, whose GS-FCH
correspondence is a maximum-weight one-to-one assignment of squared MO
overlaps. The pure assignment helper lives in `utils/orbitals.py` so its
scientific behavior can be tested independently of plotting.

## Persistence

HDF5, written and read by `pymbxas/io/h5.py`. That module owns the schema, the compression policy (gzip level 4 above 64 KiB) and every h5py call; no other module imports h5py directly. Root attributes are `kind` (`calculation`, `spectra` or `spectras`), `schema_version` and `pymbxas_version`.

The layout mirrors PySCF's chkfile shape: a `mol` dataset holding `mol.dumps()` beside an `scf/` group of `e_tot`, `mo_coeff`, `mo_occ`, `mo_energy` and `nelec`. PySCF's own `save_mol`/`load_scf` wrappers hardcode the root keys `mol` and `scf` and take a filename rather than a group, so only the ground state, which sits at the root, is loadable through them; `chkfile.load_scf(path)` and `mf.from_chk(path)` both work on a checkpoint. Excitation snapshots repeat the same shape at `/excitations/NNN/{fch,xch}/` and are read with `chkfile.load(path, key)` plus `gto.loads`, which accept arbitrary keys.

- `PySCFMBXAS.save()` writes `calculation_config`, `gs_scf_config`, checkpoint policy, structure, and ground state on first call, then appends one `/excitations/NNN` group per finished excitation. Every group stores its excitation, FCH-SCF, and XCH-SCF snapshots; all participate in identity. The `complete` attribute is written last, and incomplete groups are skipped and safely rewritten. `load()` restores these configurations exactly; a later execution may supply new runtime and logging only.
- `Spectra.save()` stores the structure through `ase.io.jsonio`, the settings as JSON, `mol.dumps()`, and GS/FCH orbital energies, so `make_mol()` is now only the fallback for a file without a stored mol. `Spectras.save()` writes each member into `/spectras/NNN` using the same `_write_into` method. `shakeup/gs_mo_energy` is an additive optional dataset: historical files remain readable, but orbital-diagram requests explain how to recover the missing data.
- Orbital coefficients load on first access, not at open. `pyscf_data` and `Spectra` both keep an `_h5_source` tuple of `(path, group)` and read through `__getattr__`; `materialize()` forces everything in. The ground state is read eagerly because a restart needs it immediately.

**`.pkl` files cannot be read.** Support was removed in 0.6.0 along with the dill dependency.
Calculation HDF5 files using the nested pre-schema-3 configuration layout require regeneration; no runtime-default inference manufactures missing scientific settings.
The additive historical readers for `Spectra` and `Spectras` remain available.

## GPU path

`RuntimeConfig(device="gpu")` causes `make_pyscf_calculator` to call
`.to_gpu()` on the SCF object, turning a `pyscf.dft.uks.UKS` into a
`gpu4pyscf.dft.uks.UKS`. `pyscf_data.to_gpu()`/`to_cpu()` move the stored
arrays with CuPy; `mol` stays on the host.

**The division of labour is: SCF on the GPU, determinant math on the CPU.** `run_MBXAS_pyscf` is called with `.to_cpu()` snapshots on both sides and must stay pure NumPy. Do not move the amplitude code onto the device.

The conversion boundary is narrow and has to be respected explicitly. The two places that mix device and host arrays, and therefore need a `.get()`, are the core-hole retention check in `_run_fch` (`mol.intor` returns a host array while the MO coefficients are on the device) and the `pyscf_data` snapshot itself. Anything new that contracts an integral against an MO coefficient is a candidate for the same bug.

Verified on an RTX 4090 with gpu4pyscf 1.8.1 and cupy-cuda12x 14.2.0: H2O oxygen K-edge at LDA/def2-SVPD gives the same answer on both paths to 5e-10 eV in transition energy and 1e-8 in summed intensity, in 1.4 s on the GPU against 11.3 s on the CPU. CuPy needs CUDA headers; if it reports "Failed to find CUDA headers", install them into the env with `pip install "cupy-cuda12x[ctk]"`.

## Logging

Two independent channels, which is deliberate.

- **PySCF output** goes through `io/logger.py`'s `Logger`, assigned to `mol.stdout`. It tees to the terminal, to an in-memory `StringIO` (retrievable as `output` / `output["fch"]`), and optionally to a file. FCH and XCH build their `mol` with `append=True` because the ground-state `mol.stdout` is closed at the end of `run_gs`.
- **PyMBXAS progress** goes through the standard `logging` module, configured by `io/config.py`.

Both logfile paths are resolved against `RuntimeConfig.work_directory` before they are opened
and are stored as absolute paths. A fresh `PySCFMBXAS` calculation opens its
PyMBXAS log in write mode; loading an HDF5 calculation opens that log in append
mode so restart activity follows the original run. The raw PySCF stream uses
the same lifecycle: GS begins a fresh file and subsequent FCH/XCH calculations
append. Its UTF-8 file is line buffered and `flush()` forwards to both active
destinations, allowing a long SCF to be monitored while it runs.

`configure_logger()` owns only handlers it creates. Reconfiguration removes,
closes, and replaces those handlers, while handlers installed by the calling
application remain attached to the `pymbxas` logger. It does not clear the root
logger. This prevents leaked logfile descriptors and avoids deleting a host
application's logging setup. Configuration remains package-global for now;
simultaneous calculation objects cannot yet select independent PyMBXAS
handlers.

Every raw SCF block begins with a structured boundary before PySCF writes its
own output:

```text
================================================================================
BEGIN PyMBXAS SCF
timestamp        : 2026-08-25T22:05:57-07:00
site             : O:8
stage            : FCH
channel          : beta
calculator       : UKS
xc               : pbe
occupation       : maxvol
charge           : 1
spin             : 1
basis            : def2-svpd
device           : GPU
================================================================================
```

GS omits site, channel, and occupation fields. FCH/XCH headers are written to
the combined raw logfile and the stage-local `StringIO`, so the same context
is retained in the HDF5 `output` fields. Generic `ase_to_mole` calls outside
the SCF workflow do not receive an SCF header.

High-level workflow records use `[site stage]` context, for example
`[O:8 FCH]`. A GS record has only `[GS]`, and an atom-level record has only
`[O:8]`. `excite()` returns an immutable `ExcitationOutcome` tuple for the
current request and exposes the same tuple as `last_excitation_outcomes`.
Expected per-site `ValueError`/`RuntimeError` failures are recorded and do not
stop later sites; unexpected exceptions still propagate. The request summary
reports succeeded, failed, and skipped counts and lists each failure, so a
partial calculation is never logged as fully successful.

`dft_verbose` is PySCF's own 0-9 scale. `xas_verbose` is PyMBXAS's 1-5 scale,
mapped in `configure_logger`: 1-4 map onto the standard
`ERROR`/`WARNING`/`INFO`/`DEBUG` levels, while 5 reserves a custom `TRACE`
level (`pymbxas.io.config.TRACE`) below `DEBUG` for future very large numerical
diagnostics.

Logging levels follow this runtime policy:

- `INFO`: stage starts, one SCF convergence record, one aggregate occupation
  record when applicable, contextual screening stopping summaries, paths, and
  run totals. Screening summaries include the final threshold, iteration,
  accumulated series, delta, and derivative.
- `DEBUG`: individual maxvol calls, MOM warm-up calls, screening threshold
  iterations, orbital indices, and file operations.
- `TRACE`: exceptionally large numerical diagnostics; the level is reserved
  but currently has no production call sites.
- `WARNING`: recoverable scientific or numerical conditions, including a
  screening loop that reaches its iteration limit.
- `ERROR`: a failed requested result or an incomplete run.

Library code does not write directly to standard output or standard error.
Operational and recoverable conditions use the package logger; invalid inputs
and requested-output I/O failures raise exceptions rather than printing and
continuing. The two deliberate presentation boundaries are the command-line
interface, which writes its final result or failure to the corresponding
stream, and `Spectra.print_mbxas_summary()`, whose purpose is explicitly to
render a report. Optional dependency notices such as a missing MOKIT install
are warnings in the package log.

PyMBXAS does not call `logging.captureWarnings()`: warnings emitted by PySCF,
ASE, gpu4pyscf, or another dependency remain on Python's warnings channel, so
applications and test suites retain control through their normal warning
filters. New PyMBXAS operational notices should use logging; reserve
`warnings.warn()` for API deprecations or other warning categories callers
need to filter programmatically.

FCH completion includes the retained core-hole MO and overlap. XCH completion
includes both core-hole and spectator overlaps. Maxvol aggregation reads the
callback history without redoing any determinant work.

Many-body screening records carry the excited site. A decomposition screens
each requested order once, then accumulates the physical spectrum and its
shake-down subset from the same flagged sticks in one traversal. A completed
`get_mbxas_decomposition()` call emits one site-level block containing its
order-resolved integrals, captured-overlap fraction, and shake-down fraction.
The numerical screening helpers accept an optional contextual logger and use
their module logger by default.

Console and file handlers deliberately use different record headers. Console
records are compact:

```text
22:05:57 INFO    [O:8 FCH] Converged
	cycles                  : 29
	energy                  : -2004.359232936700 Ha
	electrons (alpha, beta) : (120, 119)
	core hole MO            : 8
	core hole overlap       : 0.99871
	elapsed                 : 267.1 s
```

The corresponding file record begins with a full local timestamp and module:

```text
2026-08-25 22:05:57 | INFO    | pymbxas.calculators.excitation | [O:8 FCH] Converged
```

`MultilineFormatter` emits the record header once and tab-indents every
continuation line. `format_log_fields` aligns labels and wraps long values
under their value column, targeting 88 characters before the handler prefix.
An empty INFO record becomes a genuinely blank separator line rather than a
timestamp with no message. Run metadata, SCF validation, occupation tracking,
localization, screening summaries, and run totals all use this block format.

## Secondary subsystems

Neither is on the MBXAS path and neither is imported by it.

**`md/`** is a fork of `pyscf.md.integrators` (`VelocityVerlet`, `NVTBerendson`) with `_zero_rotation` and `_zero_translation` added, driven by `md/solvers.py` (`Geometry_optimizer`, `AIMD_solver`) and `md/callback.py` for trajectory output. It is the one place in the package that already checks SCF convergence. Being a fork, it does not track upstream PySCF fixes.

**`explorer/`** is the WIP Gaussian-process layer: `MBXASplorer` trains on `Spectras` to predict spectra from structure, with `node.py` holding the spectral representation, `gpmodels.py` the GPflow and GPyTorch backends, `features.py` and `pca.py` the dimensionality reduction. It needs gpflow and TensorFlow, which are optional extras and not installed. Five modules here fail to import in a default environment; that is expected.

## Extension points

- **Another electronic-structure backend.** Write a `<code>_data` snapshot with the same six attributes as `pyscf_data` and a `run_MBXAS_<code>` that returns the same five-tuple. `Spectra` onward is backend-agnostic already.
- **Another localization scheme.** Add a branch in `do_localization_pyscf` keyed off `loc_type`. It must return a full `(2, nao, nmo)` coefficient array with only the core columns replaced.
- **Another broadening.** `broaden.py` is pure and takes `(energies, intensities)`. Gaussian line shapes have unit area, but many-body stick weights are not rescaled to force the completed spectrum to match the f1 area.
- **Higher-order MBXAS.** Explicit `f^(2)` and higher determinant amplitudes live in `mbxas/shakeup.py`, separated from spectator overlap probabilities. Preserve that distinction; see `dev/shakeup.md`.
- **Batch drivers.** `drivers/acquisitor.py` is the structure-in, `Spectra`-out wrapper the CLI uses. It swallows exceptions and returns `None`, so a caller cannot distinguish a crash from a bad structure.
