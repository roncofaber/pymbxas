# HDF5 persistence for PyMBXAS

Replace dill/`.pkl` persistence with HDF5 across the calculator and the post-processing layer. Target version 0.6.0.

## Motivation

`PySCF_mbxas` can already reload a finished calculation and continue exciting new atoms: `pkl_file=` restores `gs_data`, `_ran_GS` and `_excitations`, and `excite()` skips atoms already in `excited_idxs`. The capability is not new. Its implementation is the problem.

The pickle stores live objects: a `Mole` whose `stdout` was closed at the end of `run_ground_state`, a `logging.Logger`, a `df_obj`, and `_cdir`/`_tdir` frozen as absolute paths from the machine that ran the job. Reloading therefore requires the class layout, the PySCF version and the directory tree to all still match. When one of them does not, dill fails with an attribute error rather than a missing-key error, and there is nothing to shim.

The real change is not the container. It is forcing a split between state that is stored (arrays, scalars, the ASE structure, the parameter dicts) and objects that are reconstructed on load (`Mole`, logger, calculators). HDF5 makes that split mandatory because it cannot hold anything else. Once it exists, a version skew produces a named missing dataset, which is recoverable.

Secondary gains: no arbitrary-code-execution on load, compression on the MO coefficients, partial reads, append-only writes, and a file other tools can open.

## Decisions

| Decision | Choice |
|---|---|
| Scope | Calculator and `Spectra`/`Spectras` together, one shared module |
| Existing `.pkl` files | Clean break. No read path, no migration script. `dill` leaves the package |
| Compression | gzip on array datasets above a size floor |
| Lazy loading | MO coefficients read on first access, not at open |
| Captured PySCF stdout | Retained in the file |
| XCH snapshot | Stored in full, symmetric with FCH |
| `Mole` serialization | PySCF's `mol.dumps()` / `gto.loads()` |
| On-disk layout | Mirrors PySCF's chkfile schema |
| Load API | `classmethod .load(path)`; the constructor kwarg is removed |

## Constraint discovered during design

PySCF's convenience wrappers hardcode the root keys `mol` and `scf`. `lib.chkfile.save_mol`, `load_mol`, `scf.chkfile.dump_scf` and `load_scf` all take a filename rather than a group and write to the root. A checkpoint holds 1 + 2N snapshots, so at most one of them can be chkfile-compatible in the strict sense.

Resolution, verified against PySCF 2.12.1:

- Adopt PySCF's layout (`mol` as a `dumps()` string beside an `scf/` group of `e_tot`, `mo_coeff`, `mo_occ`, `mo_energy`), but write it with our own h5py writer.
- Put the ground state at the root. `scf.chkfile.load_scf(path)` then works unmodified, and `mf.from_chk(path)` builds an SCF initial guess directly from a pymbxas checkpoint.
- Repeat the same shape at `/excitations/NNN/{fch,xch}/`. These are reachable through `lib.chkfile.load(path, "excitations/000/fch/scf")` and `gto.loads()`, which accept arbitrary keys; only the root-only wrappers do not apply.

Writing the file ourselves is also what makes compression possible: `lib.chkfile.save` performs a bare `root[key] = value` with no compression hook. Compressed datasets read back through PySCF's loaders transparently, since h5py decompresses on read. Both facts were confirmed experimentally before this spec was written.

## Schema

Root attributes on every file:

| Attribute | Value |
|---|---|
| `kind` | `calculation`, `spectra` or `spectras` |
| `schema_version` | integer, starts at 1 |
| `pymbxas_version` | `pymbxas.__version__` at write time |

### kind = "calculation"

Written by `PySCF_mbxas`. Default name `pymbxas_obj.h5`.

```
/mol                       str    GS mol.dumps()
/scf/e_tot                 f8
/scf/mo_coeff              (2, nao, nmo) f8, gzip
/scf/mo_occ                (2, nmo) f8
/scf/mo_energy             (2, nmo) f8
/scf/nelec                 (2,) i8
/scf/mo_coeff_del          (2, nao, nmo) f8, gzip   present only when _used_loc
/structure/numbers         (natm,) i4
/structure/positions       (natm, 3) f8
/structure/cell            (3, 3) f8
/structure/pbc             (3,) bool
/structure/initial_magmoms (natm,) f8               present only when set
/parameters                str    JSON
/output_settings           str    JSON
/output                    str    GS captured stdout, gzip
/excitations/NNN                  attrs: ato_idx, symbol, channel, orb_idx, complete
/excitations/NNN/fch/mol   str
/excitations/NNN/fch/scf/{e_tot,mo_coeff,mo_occ,mo_energy,nelec}
/excitations/NNN/fch/output
/excitations/NNN/xch/...          same shape; absent when do_xch is False
/excitations/NNN/mbxas/{energies,absorption,mb_overlap,dipole_KS,basis_ovlp}
```

Root flags `ran_GS` and `used_loc` are stored as attributes on `/`.

Three points about this layout:

**Excitation groups are keyed by zero-padded sequence index, not by `ato_idx`.** `to_spectra(excitation=2)` indexes positionally, so the list order is part of the contract. Zero-padding makes h5py's alphabetical group iteration reproduce insertion order. `ato_idx` is a group attribute, and resume matches against it.

**The settings dicts are JSON strings, not HDF5 attributes.** `solvent` is `None` or a float, `xas_logfile` is `None` or a string, `pbc` is a bool. Mapping that onto HDF5 attribute types invites round-trip bugs. A single JSON string round-trips exactly and stays readable under `h5dump`.

**`complete` is written last on each excitation group.** A crash mid-write leaves a group without the flag; the loader warns and skips it rather than returning a truncated `mo_coeff`. This is a cheap substitute for transactional writes, and it matters because writes now happen incrementally during long batch runs.

### kind = "spectra"

```
/mol                       str
/structure/...             as above
/calc_settings             str    JSON
/scf/mo_coeff              (2, nao, nmo) f8, gzip
/scf/mo_occ                (2, nmo) f8
/xas/energies              (n,) f8      Hartree
/xas/amplitude             (3, n) f8
/xas/el_labels             (n,) i8
```

Root attributes: `channel`, `exc_idx`, `label`, `gs_energy`.

Storing `/mol` here is an upgrade over current behavior. `Spectra.make_mol()` rebuilds the `Mole` with `ase_to_mole(structure, **calc_settings)`, and `calc_settings` carries `xc`, `calc_type`, `loc` and `xch`, which `Mole` silently discards. A stored `dumps()` string is the mol that actually ran. `make_mol()` remains the fallback when `/mol` is absent.

`transform()` mutates `self.mol` and `self._mo_coeff` in place, so `save()` always re-dumps the live object rather than echoing what was read.

### kind = "spectras"

`/spectras/NNN/...`, each entry identical in shape to a standalone spectra payload, plus `/labels` and a root `aligned` attribute. `_erange` is recomputed in `__init__` and is not stored.

## Module design

New module `pymbxas/io/h5.py`. Everything else calls into it; no other module touches h5py directly.

| Function | Contract |
|---|---|
| `write_snapshot(group, data)` | `pyscf_data` to a PySCF-shaped sub-tree |
| `read_snapshot(group, lazy=False)` | inverse; `lazy=True` defers coefficient reads |
| `write_structure(group, atoms)` / `read_structure(group)` | ASE `Atoms` |
| `write_json(group, key, obj)` / `read_json(group, key)` | settings dicts |
| `write_array(group, key, arr)` | applies the compression policy |
| `check_schema(f, expected_kind)` | version and kind guard |

Compression policy lives in one place: gzip level 4 on arrays above 64 KiB, uncompressed below. Compressing small arrays costs more in chunk overhead than it saves.

### Lazy loading

The object does not hold an open h5py handle. An open handle breaks `deepcopy`, breaks multiprocessing, and dangles if the file is moved. Instead a lazily-backed object stores the file path and its group path, and the coefficient attributes are properties over a private cache. On first access with an empty cache, the file is opened, the dataset is read, the cache is filled, and the file is closed. One open per cache miss is negligible against the cost of reading a multi-megabyte dataset.

`copy()` and `transform()` force materialization before proceeding.

This requires a change to `pyscf_data`. Its `to_cpu` and `to_gpu` currently reflect over `vars(self)` and `setattr` by name, which breaks the moment `mo_coeff` becomes a property backed by `_mo_coeff`. `pyscf_data` gains an explicit `_FIELDS` tuple, and both methods iterate that instead. The public surface (`.mol`, `.mo_coeff`, `.mo_occ`, `.mo_energy`, `.e_tot`, `.nelec`) is unchanged.

Laziness is applied where it pays: the ground state loads eagerly, since a restart needs it immediately, while each excitation's FCH and XCH coefficients load on demand. Reopening a 40-atom job to add a 41st atom therefore reads a few megabytes rather than a gigabyte.

## Restart semantics

`PySCF_mbxas.load(path)` produces an object on which `excite()` works with no ground-state recomputation.

| Category | Handling |
|---|---|
| Restored | `structure`, `_parameters`, `_output_settings`, `gs_data` (eager), `_used_loc`, `_ran_GS`, `_excitations` (lazy coefficients) |
| Rebuilt | `logger` via `configure_logger`; `mol` via `gto.loads()` |
| Re-derived | `_cdir` = current working directory; `_tdir` = directory containing the file |
| Dropped | `df_obj`, always `None` because PBC is blocked; `self.data`, initialized empty and never written |

Re-deriving the directories rather than restoring them fixes a latent bug: today `_tdir` is a stale absolute path, so a checkpoint moved between machines chdirs somewhere unintended.

The reloaded `mol` gets `verbose` and `output` reapplied from the current output settings rather than the stored ones, because the original mol's `stdout` was closed at the end of `run_ground_state`.

The loaded object remembers its path, so subsequent saves append to the same file.

### Append-only writes

`save_object()` currently re-dills the entire object after every excitation, including all previous ones. Under HDF5 the first save writes the header, structure, settings and ground state; each completed excitation then opens the file in append mode and writes only its own group. A 40-atom batch that dies on atom 31 leaves 30 intact groups on disk, and resume is a matter of which groups exist.

`excite()` after a restart must not rewrite groups already present. Group existence is the authority; no separate counter is kept.

`run_ground_state(force=True)` on a loaded object that already has excitations raises. The stored excitations were computed against the old ground state and silently keeping them would produce a mixed, wrong checkpoint. The message directs the user to a new file.

## What a checkpoint must guarantee

**A loaded checkpoint contains everything needed to re-derive the spectrum without running SCF.** This is the invariant that pins the schema, and it is testable.

It is why the XCH snapshot is stored in full and symmetrically with FCH. `run_MBXAS_pyscf` consumes only `xch_calc.e_tot` (`pymbxas/mbxas/mbxas.py:85`), so the current code path does not force storing the XCH wavefunction. It is stored deliberately, so that the alignment can be re-derived and the added electron inspected after the fact. Recorded here so it is not later mistaken for an oversight and removed.

## Error handling

| Condition | Behavior |
|---|---|
| `schema_version` newer than supported | Refuse, naming both versions. No guessing |
| `kind` mismatch | Refuse, naming expected and actual |
| Missing required dataset | `KeyError` quoting the full HDF5 path |
| Excitation group lacking `complete` | Warn through the logger, skip the group, continue |
| File not HDF5 | Clear error. A `.pkl` path gets a message saying support was removed in 0.6.0 |

The last row is the only concession to the clean break: a one-line diagnostic, not a read path.

## Files changed

Core:

| File | Change |
|---|---|
| `pymbxas/io/h5.py` | New. Schema, primitives, compression policy, version guard |
| `calculators/pyscf.py` | `save_object` becomes an append-only writer; `_restart_from_pickle` becomes `load()`; `pkl_file` kwarg, `dill` import and vestigial `self.data` removed |
| `spectra.py` | `save`, `__restart`, `__pkl_to_dict` reworked; `Spectra.load()`; `make_mol` prefers stored `/mol`; 0.4.x key shims removed |
| `spectras.py` | Same, over `/spectras/NNN` |
| `io/data.py` | `pyscf_data` gains `_FIELDS` and lazy slots; `to_cpu`/`to_gpu` stop reflecting over `vars(self)` |
| `explorer/mbxasplorer.py` | Last remaining `dill.dump(self)` at line 404 |
| `utils/auxiliary.py` | `change_key` removed; it exists only to serve the two shims |

Defaults, packaging and docs: `setup.cfg` (`dill` out, `h5py` declared), `cli/pyscf.py` output default, `drivers/acquisitor.py`, `examples/example_H2O_molecule.py`, `__init__.py` docstring and version, `README.md`, `dev/architecture.md` persistence section, `CLAUDE.md` stack line and persistence bullet, `CHANGELOG.md`, `CITATION.cff`.

## Testing

Per project policy the end-to-end test stays one file. Assertions are appended to `tests/test_h2o_kedge.py`, reusing the objects already in scope:

1. Save, `PySCF_mbxas.load()`, and assert `gs.mo_coeff`, `fch.mo_coeff`, `xch.e_tot` and every `mbxas` array match bit-for-bit.
2. The reloaded object reports `_ran_GS is True`, and `excite(0)` returns without a new SCF.
3. `pyscf.scf.chkfile.load_scf(path)` succeeds on the file. The native-compatibility claim is asserted, not assumed.
4. MBXAS re-derives from the loaded snapshots with no SCF, matching the original amplitudes.
5. A `Spectra` round-trip preserves `energies`, `amplitude` and `CMO`.

## Changelog

Under `## [Unreleased]`:

```
### Changed
- Calculations and spectra are now saved as HDF5 files; `.pkl` files can no longer be loaded.
- Reloading a calculation no longer reads molecular orbital coefficients until they are used.

### Added
- Saved calculations can be reopened with `.load()` and continued without repeating the ground state.
- Checkpoints are readable by PySCF as chkfiles.
```

## Out of scope

- Reading legacy `.pkl` files, in any form.
- Parallel or SWMR writes. Excitations are written sequentially by one process.
- Changing what is computed. No numerical result changes; the round-trip test asserts bit-for-bit equality.
