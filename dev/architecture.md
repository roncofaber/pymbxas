# Architecture

Module layout, object contracts, and data flow. The physics lives in `dev/method.md`; this file is about where things are and what they promise each other.

## Package layout

```
pymbxas/
  __init__.py         version, date, re-exports Spectra and Spectras
  calculators/
    pyscf.py          PySCF_mbxas      entry point, owns the ground state
    excitation.py     Excitation       one excited atom: FCH + XCH + MBXAS
  mbxas/
    mbxas.py          run_MBXAS_pyscf  the determinant amplitude
    broaden.py        get_mbxas_spectra, broadened_spectrum, gaussian_broadening
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
    data.py           pyscf_data       picklable SCF snapshot
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
tests/                one end-to-end check
dev/                  this reference layer
```

## Core flow

```
ase.Atoms
   │  ase_to_mole
   ▼
gto.Mole ──► make_pyscf_calculator ──► UKS/UHF ──► kernel()
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
   ├── _run_fch   charge+1, spin±1, mom_occ ──► pyscf_data(fch)
   ├── _run_xch   charge,   spin,   mom_occ ──► pyscf_data(xch)
   └── _run_mbxas ──► run_MBXAS_pyscf ──► self.mbxas dict
                                              │
                            PySCF_mbxas.to_spectra()
                                              ▼
                                      Spectra / Spectras
                                              │
                                     get_mbxas_spectra()
                                              ▼
                                        (energy_eV, intensity)
```

The ground state runs once. Each `Excitation` is independent given `gs_data`, so the loop in `excite()` is embarrassingly parallel in principle, though it is serial today.

## Object contracts

### `PySCF_mbxas`

Owns the ground state and the list of excitations. Two ways in: a structure plus parameters, or `.load()` to restore from a saved checkpoint.

| Attribute | Meaning |
|---|---|
| `structure` | the ASE `Atoms` |
| `mol` | the ground-state `gto.Mole` |
| `gs_data` | `pyscf_data` for the ground state, with `mo_coeff` possibly localized |
| `gs_data.mo_coeff_del` | the pre-localization coefficients, **only present if localization ran** |
| `_excitations` | list of `Excitation`, one per successfully excited atom |
| `_parameters` | charge, spin, xc, basis, solvent, pbc, loc, xch, calc_type |
| `_output_settings` | verbosity, logging, saving, GPU flags. Read through the `oset` property, which returns a copy |
| `_ran_GS`, `_used_loc` | state flags consulted by `run_ground_state` and `_print_fchk_files` |

`_cdir` and `_tdir` are the caller's directory and the calculation directory. `kernel()` chdirs into `_tdir` and back out, so relative paths inside a calculation resolve against the target directory.

Both `oset` and `parameters` return **copies**. Mutating what they hand back does nothing; assign to the underlying dict if you need to change a setting.

### `Excitation`

Everything happens in `__init__`: locate the core orbital, run FCH, run XCH, run the determinant math. There is no lazy mode and no `.run()`.

| Attribute | Meaning |
|---|---|
| `ato_idx`, `symbol`, `channel` | which atom, which spin |
| `orb_idx` | global MO index of the ground-state core orbital being emptied |
| `data["fch"]`, `data["xch"]` | `pyscf_data` snapshots |
| `output["fch"]`, `output["xch"]` | captured PySCF stdout as a string |
| `mbxas` | dict: `energies`, `absorption`, `mb_overlap`, `dipole_KS`, `basis_ovlp` |

A failure raises rather than leaving a partial object. `PySCF_mbxas._single_excite` catches and logs so one bad atom does not abort a batch, which means **`len(obj.excitations)` can be smaller than the number of atoms requested**. Check it.

### `pyscf_data`

A snapshot, not a calculator: `mol`, `mo_coeff`, `mo_occ`, `mo_energy`, `e_tot`, `nelec`. This narrow surface is what makes the whole object graph picklable and what lets `to_cpu()`/`to_gpu()` be a simple array walk. `mol` is never converted between devices.

Anything that needs a real calculator has to rebuild one with `make_pyscf_calculator`.

### `Spectra` and `Spectras`

The post-processing layer, deliberately decoupled from the calculators. `Spectra` carries the FCH MOs, the amplitudes and the structure, and rebuilds `mol` from the structure on load. It never re-runs SCF.

`_mo_coeff` is `(2, nao, nmo)` and `_mo_occ` is `(2, nmo)`, with `_channel` selecting the excited one through the `_active_mo_coeff` / `_active_mo_occ` properties. Both are always stored with the spin axis present, so those properties index it unconditionally.

`CMO` is the virtual manifold with the core hole dropped, matching the transition list one-to-one. `_el_labels` labels those virtuals for clustering and must stay the same length.

`Spectras` is a list wrapper with averaging, atomic and electronic labels, and structure alignment through sea_urchin. Indexing with an int returns a `Spectra`; with a slice, list or boolean mask it returns a new `Spectras`.

## Persistence

HDF5, written and read by `pymbxas/io/h5.py`. That module owns the schema, the compression policy (gzip level 4 above 64 KiB) and every h5py call; no other module imports h5py directly. Root attributes are `kind` (`calculation`, `spectra` or `spectras`), `schema_version` and `pymbxas_version`.

The layout mirrors PySCF's chkfile shape: a `mol` dataset holding `mol.dumps()` beside an `scf/` group of `e_tot`, `mo_coeff`, `mo_occ`, `mo_energy` and `nelec`. PySCF's own `save_mol`/`load_scf` wrappers hardcode the root keys `mol` and `scf` and take a filename rather than a group, so only the ground state, which sits at the root, is loadable through them; `chkfile.load_scf(path)` and `mf.from_chk(path)` both work on a checkpoint. Excitation snapshots repeat the same shape at `/excitations/NNN/{fch,xch}/` and are read with `chkfile.load(path, key)` plus `gto.loads`, which accept arbitrary keys.

- `PySCF_mbxas.save_object()` writes the header and ground state on first call, then appends one `/excitations/NNN` group per finished excitation, keyed by zero-padded sequence index with `ato_idx` as a group attribute. The `complete` attribute is written last; a group missing it is skipped by the loader and overwritten by the next save. `PySCF_mbxas.load()` restores the structure, parameters, ground state and excitations, rebuilds `mol` with `gto.loads` and the logger with `configure_logger`, re-derives `_cdir` and `_tdir` from the current directory and the file's location, and sets `df_obj` to `None`.
- `Spectra.save()` stores the structure through `ase.io.jsonio`, the settings as JSON, and `mol.dumps()`, so `make_mol()` is now only the fallback for a file without a stored mol. `Spectras.save()` writes each member into `/spectras/NNN` using the same `_write_into` method.
- Orbital coefficients load on first access, not at open. `pyscf_data` and `Spectra` both keep an `_h5_source` tuple of `(path, group)` and read through `__getattr__`; `materialize()` forces everything in. The ground state is read eagerly because a restart needs it immediately.

**`.pkl` files cannot be read.** Support was removed in 0.6.0 along with the dill dependency.

## GPU path

`gpu=True` threads `is_gpu` through `_output_settings`, and `make_pyscf_calculator` calls `.to_gpu()` on the SCF object, turning a `pyscf.dft.uks.UKS` into a `gpu4pyscf.dft.uks.UKS`. `pyscf_data.to_gpu()`/`to_cpu()` move the stored arrays with CuPy; `mol` stays on the host.

**The division of labour is: SCF on the GPU, determinant math on the CPU.** `run_MBXAS_pyscf` is called with `.to_cpu()` snapshots on both sides and must stay pure NumPy. Do not move the amplitude code onto the device.

The conversion boundary is narrow and has to be respected explicitly. The two places that mix device and host arrays, and therefore need a `.get()`, are the core-hole retention check in `_run_fch` (`mol.intor` returns a host array while the MO coefficients are on the device) and the `pyscf_data` snapshot itself. Anything new that contracts an integral against an MO coefficient is a candidate for the same bug.

Verified on an RTX 4090 with gpu4pyscf 1.8.1 and cupy-cuda12x 14.2.0: H2O oxygen K-edge at LDA/def2-SVPD gives the same answer on both paths to 5e-10 eV in transition energy and 1e-8 in summed intensity, in 1.4 s on the GPU against 11.3 s on the CPU. CuPy needs CUDA headers; if it reports "Failed to find CUDA headers", install them into the env with `pip install "cupy-cuda12x[ctk]"`.

## Logging

Two independent channels, which is deliberate.

- **PySCF output** goes through `io/logger.py`'s `Logger`, assigned to `mol.stdout`. It tees to the terminal, to an in-memory `StringIO` (retrievable as `output` / `output["fch"]`), and optionally to a file. FCH and XCH build their `mol` with `append=True` because the ground-state `mol.stdout` is closed at the end of `run_ground_state`.
- **PyMBXAS progress** goes through the standard `logging` module, configured by `io/config.py`.

`dft_verbose` is PySCF's own 0-9 scale. `xas_verbose` is PyMBXAS's 1-5 scale, mapped in `configure_logger`.

## Secondary subsystems

Neither is on the MBXAS path and neither is imported by it.

**`md/`** is a fork of `pyscf.md.integrators` (`VelocityVerlet`, `NVTBerendson`) with `_zero_rotation` and `_zero_translation` added, driven by `md/solvers.py` (`Geometry_optimizer`, `AIMD_solver`) and `md/callback.py` for trajectory output. It is the one place in the package that already checks SCF convergence. Being a fork, it does not track upstream PySCF fixes.

**`explorer/`** is the WIP Gaussian-process layer: `MBXASplorer` trains on `Spectras` to predict spectra from structure, with `node.py` holding the spectral representation, `gpmodels.py` the GPflow and GPyTorch backends, `features.py` and `pca.py` the dimensionality reduction. It needs gpflow and TensorFlow, which are optional extras and not installed. Five modules here fail to import in a default environment; that is expected.

## Extension points

- **Another electronic-structure backend.** Write a `<code>_data` snapshot with the same six attributes as `pyscf_data` and a `run_MBXAS_<code>` that returns the same five-tuple. `Spectra` onward is backend-agnostic already.
- **Another localization scheme.** Add a branch in `do_localization_pyscf` keyed off `loc_type`. It must return a full `(2, nao, nmo)` coefficient array with only the core columns replaced.
- **Another broadening.** `broaden.py` is pure and takes `(energies, intensities)`. Note the existing kernel is unnormalized; see `dev/method.md`.
- **Batch drivers.** `drivers/acquisitor.py` is the structure-in, `Spectra`-out wrapper the CLI uses. It swallows exceptions and returns `None`, so a caller cannot distinguish a crash from a bad structure.
