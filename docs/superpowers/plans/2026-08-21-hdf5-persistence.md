# HDF5 Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace dill/`.pkl` persistence in PyMBXAS with HDF5, so a finished calculation can be reopened and continued without repeating the ground state.

**Architecture:** One new module, `pymbxas/io/h5.py`, owns every h5py call and the on-disk schema. The layout mirrors PySCF's chkfile shape (`mol` as a `dumps()` string beside an `scf/` group), so `pyscf.scf.chkfile.load_scf()` works on our files, but the writer is ours in order to get gzip. `PySCF_mbxas`, `Spectra` and `Spectras` call into that module and gain `.load()` classmethods. Molecular orbital coefficients load on first access rather than at open.

**Tech Stack:** Python 3.9+, h5py (already present as a PySCF dependency), PySCF 2.x (`gto.loads`, `mol.dumps`, `lib.chkfile`), ASE (`ase.io.jsonio`), NumPy 2.x, pytest.

**Spec:** `docs/superpowers/specs/2026-08-21-hdf5-persistence-design.md`

## Global Constraints

- **Conda env is `pymbxas`.** Every command runs as `conda run -n pymbxas ...`. Nothing is installed globally.
- **Working branch is `dev`.** Do not commit to `main`.
- **`dill` must not appear anywhere in the package when this plan is done.** No read path, no migration script, no import.
- **`SCHEMA_VERSION = 1`.** Root attributes on every file: `kind`, `schema_version`, `pymbxas_version`.
- **`kind` values are exactly** `"calculation"`, `"spectra"`, `"spectras"`.
- **Compression policy: gzip level 4 on arrays of 65536 bytes or more, uncompressed below.** Defined once in `io/h5.py`; no other module chooses.
- **Units stay Hartree everywhere in stored data.** Conversion to eV happens only in `Spectra.energies` and `get_mbxas_spectra`.
- **`mo_occ` is 1.0/0.0, not 2.0/0.0.** The package is unrestricted-only.
- **`channel=1` (beta) is the default excited channel.** Never assume it; always read it from the stored attribute.
- **No comments in code unless the *why* is non-obvious.** No multi-line rationale blocks, no references to this plan or the spec from inside source files.
- **No em dashes** in code, comments, docs or commit messages. Use a plain hyphen.
- **In markdown, do not hard-wrap prose.** One line per paragraph or list item.
- **Version target is 0.6.0**, `__date__ = "21 Aug. 2026"`.
- **Run the end-to-end test before any commit that touches `mbxas/`, `calculators/`, `build/` or `utils/orbitals.py`:** `conda run -n pymbxas pytest tests/ -q`.

### Deviations from the spec, already decided

Two details were settled by experiment after the spec was approved. Follow the plan, not the spec, where they differ.

1. **`/structure` is a single JSON string dataset produced by `ase.io.jsonio.encode`, not a group of `numbers`/`positions`/`cell`/`pbc` arrays.** The spec's hand-rolled layout silently drops `Atoms.info`, tags, momenta and constraints. ASE's own serializer round-trips all of them and is what `ase.db` uses.
2. **Captured PySCF stdout is stored as a gzipped `uint8` array, not a string dataset.** HDF5 scalar string datasets cannot be chunked, so `compression="gzip"` on one raises `TypeError: Scalar datasets don't support chunk/filter options`. Small strings (`mol.dumps()`, the JSON settings) stay as plain uncompressed string datasets.
3. **The spec's "missing required dataset raises a `KeyError` quoting the full HDF5 path" is relaxed to h5py's own error.** h5py already raises `KeyError: "Unable to synchronously open object (object 'mo_coeff' doesn't exist)"`, which names the dataset. Wrapping every read to prepend the group path would add a guard to roughly thirty call sites for a marginal gain in the message. Nothing else in the error-handling table is relaxed.
4. **Deferred attributes are served by `__getattr__`, not by one property per field** as the spec's module design sketched. `__getattr__` fires only when normal lookup fails, so eagerly built objects never pay for it, and it covers `mo_coeff`, `mo_occ`, `mo_energy` and `mo_coeff_del` without four near-identical property blocks.

Also note: the spec puts the round-trip assertions in `tests/test_h2o_kedge.py`, per the project policy of one end-to-end test. The serialization primitives in Task 1 through Task 3 additionally get `tests/test_h5_io.py`, which runs no SCF and finishes in well under a second. This is a second test file, which the project policy discourages; it is justified because the policy is about not diluting the one physics-invariant test, and pure I/O round-trips cannot be driven by TDD otherwise.

## File Structure

| File | Responsibility |
|---|---|
| `pymbxas/io/h5.py` | **New.** Schema constants, version guard, typed read/write primitives, `pyscf_data` snapshot round-trip, lazy field reads. The only module that imports h5py. |
| `pymbxas/io/data.py` | `pyscf_data` gains an explicit field list and a lazy-backed construction path. |
| `pymbxas/calculators/pyscf.py` | `PySCF_mbxas` append-only writer and `.load()`. |
| `pymbxas/calculators/excitation.py` | `Excitation.from_h5()`. |
| `pymbxas/spectra.py` | `Spectra` save/load, lazy coefficients, constructor narrowed. |
| `pymbxas/spectras.py` | `Spectras` save/load over `/spectras/NNN`. |
| `tests/test_h5_io.py` | **New.** Primitive and snapshot round-trips, no SCF. |
| `tests/test_h2o_kedge.py` | End-to-end round-trip and restart assertions appended. |

---

### Task 1: HDF5 primitives module

**Files:**
- Create: `pymbxas/io/h5.py`
- Test: `tests/test_h5_io.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `SCHEMA_VERSION: int`, `KIND_CALCULATION/KIND_SPECTRA/KIND_SPECTRAS: str`, `write_array(group, key, value) -> None`, `write_str(group, key, text) -> None`, `read_str(group, key) -> str`, `write_text(group, key, text) -> None`, `read_text(group, key) -> str`, `write_json(group, key, obj) -> None`, `read_json(group, key) -> object`, `write_structure(group, key, atoms) -> None`, `read_structure(group, key) -> ase.Atoms`, `read_attr_str(obj, key) -> str`, `stamp(f, kind) -> None`, `check_schema(f, expected_kind) -> None`, `open_read(path, expected_kind) -> h5py.File`, `create(path, kind) -> h5py.File`, `append(path) -> h5py.File`, `open_plain(path, mode="r") -> h5py.File`.

`create`, `append` and `open_plain` exist so that no module outside `io/h5.py` imports h5py. Task 10 documents that as a package invariant; honour it in every later task.

- [ ] **Step 1: Write the failing test**

Create `tests/test_h5_io.py`:

```python
import json

import ase.build
import h5py
import numpy as np
import pytest

from pymbxas.io import h5


def test_write_array_compresses_only_large_arrays(tmp_path):
    path = tmp_path / "a.h5"
    small = np.arange(8, dtype=np.float64)
    large = np.zeros((256, 256), dtype=np.float64)
    with h5py.File(path, "w") as f:
        h5.write_array(f, "small", small)
        h5.write_array(f, "large", large)
    with h5py.File(path, "r") as f:
        assert f["small"].compression is None
        assert f["large"].compression == "gzip"
        assert np.array_equal(f["small"][()], small)
        assert np.array_equal(f["large"][()], large)


def test_str_and_text_roundtrip(tmp_path):
    path = tmp_path / "b.h5"
    short = "verbose = 4"
    long_log = "scf cycle line\n" * 5000
    with h5py.File(path, "w") as f:
        h5.write_str(f, "short", short)
        h5.write_text(f, "log", long_log)
        h5.write_text(f, "empty", "")
    with h5py.File(path, "r") as f:
        assert h5.read_str(f, "short") == short
        assert h5.read_text(f, "log") == long_log
        assert h5.read_text(f, "empty") == ""
        assert f["log"].compression == "gzip"


def test_json_roundtrip_preserves_none_and_bool(tmp_path):
    path = tmp_path / "c.h5"
    settings = {"charge": 0, "spin": 0, "xc": "lda", "basis": "def2-svpd",
                "solvent": None, "pbc": False, "loc": "ibo", "xch": True,
                "calc_type": "UKS"}
    with h5py.File(path, "w") as f:
        h5.write_json(f, "parameters", settings)
    with h5py.File(path, "r") as f:
        assert h5.read_json(f, "parameters") == settings


def test_structure_roundtrip_preserves_info_and_tags(tmp_path):
    path = tmp_path / "d.h5"
    atoms = ase.build.molecule("H2O")
    atoms.set_initial_magnetic_moments([1.0, 0.0, 0.0])
    atoms.set_tags([3, 0, 0])
    atoms.info["origin"] = "test"
    with h5py.File(path, "w") as f:
        h5.write_structure(f, "structure", atoms)
    with h5py.File(path, "r") as f:
        back = h5.read_structure(f, "structure")
    assert back == atoms
    assert back.info == {"origin": "test"}
    assert np.array_equal(back.get_tags(), [3, 0, 0])
    assert np.array_equal(back.get_initial_magnetic_moments(), [1.0, 0.0, 0.0])


def test_open_read_rejects_wrong_kind(tmp_path):
    path = tmp_path / "e.h5"
    with h5py.File(path, "w") as f:
        h5.stamp(f, h5.KIND_SPECTRA)
    with pytest.raises(ValueError, match="expected 'calculation'"):
        h5.open_read(path, h5.KIND_CALCULATION)


def test_open_read_rejects_future_schema(tmp_path):
    path = tmp_path / "f.h5"
    with h5py.File(path, "w") as f:
        h5.stamp(f, h5.KIND_CALCULATION)
        f.attrs["schema_version"] = h5.SCHEMA_VERSION + 1
    with pytest.raises(ValueError, match="schema version"):
        h5.open_read(path, h5.KIND_CALCULATION)


def test_open_read_rejects_pkl_path(tmp_path):
    path = tmp_path / "old.pkl"
    path.write_bytes(b"not hdf5")
    with pytest.raises(ValueError, match="0.6.0"):
        h5.open_read(path, h5.KIND_CALCULATION)


def test_open_read_accepts_good_file(tmp_path):
    path = tmp_path / "g.h5"
    with h5py.File(path, "w") as f:
        h5.stamp(f, h5.KIND_CALCULATION)
        h5.write_str(f, "hello", "world")
    with h5.open_read(path, h5.KIND_CALCULATION) as f:
        assert h5.read_str(f, "hello") == "world"
        assert h5.read_attr_str(f, "kind") == "calculation"


def test_create_stamps_the_file(tmp_path):
    import pymbxas

    path = tmp_path / "h.h5"
    with h5.create(path, h5.KIND_SPECTRAS) as f:
        h5.write_str(f, "x", "y")
    with h5.append(path) as f:
        h5.write_str(f, "z", "w")
    with h5.open_plain(path) as f:
        assert h5.read_attr_str(f, "kind") == "spectras"
        assert h5.read_attr_str(f, "pymbxas_version") == pymbxas.__version__
        assert h5.read_str(f, "x") == "y"
        assert h5.read_str(f, "z") == "w"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n pymbxas pytest tests/test_h5_io.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'pymbxas.io.h5'`.

- [ ] **Step 3: Write the implementation**

Create `pymbxas/io/h5.py`:

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HDF5 persistence primitives for pymbxas.
"""

import json

import h5py
import numpy as np
from ase.io import jsonio

#%%

SCHEMA_VERSION = 1

KIND_CALCULATION = "calculation"
KIND_SPECTRA     = "spectra"
KIND_SPECTRAS    = "spectras"

_COMPRESS_MIN_BYTES = 65536
_GZIP_LEVEL         = 4


def write_array(group, key, value):
    arr = np.ascontiguousarray(value)
    if arr.nbytes >= _COMPRESS_MIN_BYTES:
        group.create_dataset(key, data=arr, compression="gzip",
                             compression_opts=_GZIP_LEVEL)
    else:
        group.create_dataset(key, data=arr)
    return


def write_str(group, key, text):
    group.create_dataset(key, data=text)
    return


def read_str(group, key):
    raw = group[key][()]
    return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)


def write_text(group, key, text):
    buf = np.frombuffer((text or "").encode("utf-8"), dtype=np.uint8)
    write_array(group, key, buf)
    return


def read_text(group, key):
    return group[key][()].tobytes().decode("utf-8")


def write_json(group, key, obj):
    write_str(group, key, json.dumps(obj))
    return


def read_json(group, key):
    return json.loads(read_str(group, key))


def write_structure(group, key, atoms):
    write_str(group, key, jsonio.encode(atoms))
    return


def read_structure(group, key):
    return jsonio.decode(read_str(group, key))


def read_attr_str(obj, key):
    raw = obj.attrs[key]
    return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)


def stamp(f, kind):
    import pymbxas
    f.attrs["kind"]            = kind
    f.attrs["schema_version"]  = SCHEMA_VERSION
    f.attrs["pymbxas_version"] = pymbxas.__version__
    return


def check_schema(f, expected_kind):
    if "schema_version" not in f.attrs:
        raise ValueError("{} is not a pymbxas HDF5 file: no schema_version attribute".format(
            f.filename))

    version = int(f.attrs["schema_version"])
    if version > SCHEMA_VERSION:
        raise ValueError("{} has schema version {}, this pymbxas reads up to {}".format(
            f.filename, version, SCHEMA_VERSION))

    kind = read_attr_str(f, "kind")
    if kind != expected_kind:
        raise ValueError("{} holds a '{}' payload, expected '{}'".format(
            f.filename, kind, expected_kind))
    return


def open_read(path, expected_kind):
    path = str(path)

    if path.endswith(".pkl"):
        raise ValueError(
            "pymbxas 0.6.0 removed pickle support, so {} cannot be loaded. "
            "Re-run the calculation to produce an HDF5 file.".format(path))

    if not h5py.is_hdf5(path):
        raise ValueError("{} is not an HDF5 file".format(path))

    f = h5py.File(path, "r")
    try:
        check_schema(f, expected_kind)
    except Exception:
        f.close()
        raise
    return f


def create(path, kind):
    f = h5py.File(path, "w")
    stamp(f, kind)
    return f


def append(path):
    return h5py.File(path, "a")


def open_plain(path, mode="r"):
    return h5py.File(path, mode)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n pymbxas pytest tests/test_h5_io.py -q`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add pymbxas/io/h5.py tests/test_h5_io.py
git commit -m "Add HDF5 persistence primitives"
```

---

### Task 2: Lazy-capable `pyscf_data`

**Files:**
- Modify: `pymbxas/io/data.py` (whole file)
- Test: `tests/test_h5_io.py` (append)

**Interfaces:**
- Consumes: nothing.
- Produces: `pyscf_data._FIELDS: tuple`, `pyscf_data.from_arrays(mol, e_tot, nelec, **arrays) -> pyscf_data` (classmethod), `pyscf_data.from_h5_source(mol, e_tot, nelec, path, key) -> pyscf_data` (classmethod, defers array reads), `pyscf_data.materialize() -> None`. `.mol`, `.mo_coeff`, `.mo_occ`, `.mo_energy`, `.e_tot`, `.nelec`, `.mo_coeff_del`, `.to_cpu()`, `.to_gpu()`, `.copy()` keep their current meaning.

Background for the implementer: `to_cpu` and `to_gpu` currently reflect over `vars(self)` and `setattr` by name. That breaks as soon as an attribute is deferred, because a deferred attribute is absent from `vars(self)` and would be silently skipped. Both methods must iterate the explicit `_FIELDS` tuple instead. `mo_coeff_del` is not set in `__init__`; it is attached later by `PySCF_mbxas._run_localization`, so every read of it must tolerate absence.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_h5_io.py`:

```python
def test_pyscf_data_from_arrays_and_fields():
    from pyscf import gto
    from pymbxas.io.data import pyscf_data

    mol = gto.M(atom="He 0 0 0", basis="sto-3g", verbose=0)
    data = pyscf_data.from_arrays(
        mol, e_tot=-2.8, nelec=(1, 1),
        mo_coeff=np.eye(2), mo_occ=np.ones((2, 2)), mo_energy=np.zeros((2, 2)))

    assert data.e_tot == -2.8
    assert data.nelec == (1, 1)
    assert np.array_equal(data.mo_coeff, np.eye(2))
    assert getattr(data, "mo_coeff_del", None) is None
    assert "mo_coeff" in pyscf_data._FIELDS
    assert data.to_cpu().e_tot == -2.8


def test_pyscf_data_lazy_defers_array_read(tmp_path):
    import h5py as _h5py
    from pyscf import gto
    from pymbxas.io.data import pyscf_data

    mol = gto.M(atom="He 0 0 0", basis="sto-3g", verbose=0)
    coeff = np.arange(4, dtype=np.float64).reshape(2, 2)

    path = tmp_path / "lazy.h5"
    with _h5py.File(path, "w") as f:
        scf = f.create_group("snap/scf")
        h5.write_array(scf, "mo_coeff", coeff)
        h5.write_array(scf, "mo_occ", np.ones((2, 2)))
        h5.write_array(scf, "mo_energy", np.zeros((2, 2)))

    data = pyscf_data.from_h5_source(mol, -2.8, (1, 1), str(path), "snap")

    assert "mo_coeff" not in vars(data)
    assert np.array_equal(data.mo_coeff, coeff)
    assert "mo_coeff" in vars(data)
    assert getattr(data, "mo_coeff_del", None) is None

    data.materialize()
    assert "mo_energy" in vars(data)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n pymbxas pytest tests/test_h5_io.py -q -k pyscf_data`
Expected: FAIL with `AttributeError: type object 'pyscf_data' has no attribute 'from_arrays'`.

- [ ] **Step 3: Write the implementation**

Replace the body of `pymbxas/io/data.py` below the imports. Keep the module docstring and the `cupy` import guard exactly as they are. The class becomes:

```python
class pyscf_data():
    """
    This class provides a convenient wrapper for storing and manipulating 
    data extracted from a PySCF calculation. It supports conversion between 
    NumPy (CPU) and CuPy (GPU) array formats for potential acceleration.
    """

    _FIELDS = ("mol", "mo_coeff", "mo_occ", "mo_energy", "e_tot", "nelec",
               "mo_coeff_del")
    _LAZY_FIELDS = ("mo_coeff", "mo_occ", "mo_energy", "mo_coeff_del")

    def __init__(self, calculator):
        """
        Initializes the pyscf_data object.
        
        Args:
            calculator (pyscf.gto.Mole, pyscf.scf.HF, etc.): A PySCF calculator 
                object that has already been run. If None, an empty object is created. 
        """
        
        self.mol       = calculator.mol
        self.mo_coeff  = calculator.mo_coeff
        self.mo_occ    = calculator.mo_occ
        self.mo_energy = calculator.mo_energy
        self.e_tot     = calculator.e_tot
        self.nelec     = calculator.nelec
        
        # Convert arrays in-place to ensure np.array
        if cp is not None:
            for attr_name in vars(self):
                attr_value = getattr(self, attr_name)
                if isinstance(attr_value, cp.ndarray):
                    setattr(self, attr_name, attr_value.get())
                
        self._is_gpu = False
        
        return

    @classmethod
    def from_arrays(cls, mol, e_tot, nelec, **arrays):
        data = cls.__new__(cls)
        data.mol   = mol
        data.e_tot = e_tot
        data.nelec = nelec
        for name, value in arrays.items():
            setattr(data, name, value)
        data._is_gpu = False
        return data

    @classmethod
    def from_h5_source(cls, mol, e_tot, nelec, path, key):
        data = cls.__new__(cls)
        data.mol        = mol
        data.e_tot      = e_tot
        data.nelec      = nelec
        data._h5_source = (path, key)
        data._is_gpu    = False
        return data

    def __getattr__(self, name):
        if name in type(self)._LAZY_FIELDS:
            source = self.__dict__.get("_h5_source")
            if source is not None:
                from pymbxas.io.h5 import read_lazy_field
                value = read_lazy_field(source, name)
                self.__dict__[name] = value
                return value
        raise AttributeError(name)

    def materialize(self):
        """Force every deferred array to be read from disk."""
        if self.__dict__.get("_h5_source") is None:
            return
        for name in type(self)._LAZY_FIELDS:
            getattr(self, name, None)
        self.__dict__.pop("_h5_source", None)
        return

    def copy(self):
        return copy.deepcopy(self)
    
    def to_cpu(self):
        """Converts all internal arrays to NumPy format"""
        
        result = self.copy()
        
        if not result._is_gpu:
            return result

        for attr_name in type(self)._FIELDS:
            attr_value = getattr(result, attr_name, None)
            if cp is not None and isinstance(attr_value, cp.ndarray):
                setattr(result, attr_name, attr_value.get())
                
        result._is_gpu = False

        return result
    
    def to_gpu(self):
        """Converts all internal arrays to CuPy format"""
        
        result = self.copy()
        
        if result._is_gpu:
            return result

        result.materialize()

        for attr_name in type(self)._FIELDS:
            attr_value = getattr(result, attr_name, None)
            if isinstance(attr_value, np.ndarray):
                setattr(result, attr_name, cp.asarray(attr_value))
                
        result._is_gpu = True
    
        return result
```

Then add `read_lazy_field` to `pymbxas/io/h5.py`, after `open_read`:

```python
def read_lazy_field(source, name):
    path, key = source
    with h5py.File(path, "r") as f:
        scf = f[key]["scf"]
        if name not in scf:
            raise AttributeError(name)
        return scf[name][()]
```

Note for the implementer: `__getattr__` runs only when normal attribute lookup fails, so eagerly built objects never enter it. Raising `AttributeError` for an absent `mo_coeff_del` is deliberate: it keeps `getattr(data, "mo_coeff_del", None)` returning `None`, which is what `_print_fchk_files` relies on. Do not return `None` directly from `read_lazy_field`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n pymbxas pytest tests/test_h5_io.py -q`
Expected: 11 passed.

Run: `conda run -n pymbxas pytest tests/ -q`
Expected: all pass. `pyscf_data`'s public surface is unchanged, so the end-to-end test must still be green.

- [ ] **Step 5: Commit**

```bash
git add pymbxas/io/data.py pymbxas/io/h5.py tests/test_h5_io.py
git commit -m "Give pyscf_data an explicit field list and a lazy read path"
```

---

### Task 3: Snapshot round-trip in PySCF chkfile layout

**Files:**
- Modify: `pymbxas/io/h5.py` (append)
- Test: `tests/test_h5_io.py` (append)

**Interfaces:**
- Consumes: `write_array`, `write_str`, `read_str` from Task 1; `pyscf_data.from_arrays`, `pyscf_data.from_h5_source` from Task 2.
- Produces: `write_snapshot(group, data) -> None`, `read_snapshot(path, key, lazy=False) -> pyscf_data`.

`write_snapshot` writes `mol` (a `mol.dumps()` string) and an `scf/` group holding `e_tot`, `mo_coeff`, `mo_occ`, `mo_energy`, `nelec`, and `mo_coeff_del` when present. Written at the root of a file, this is exactly what `pyscf.scf.chkfile.load_scf` expects.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_h5_io.py`:

```python
def _tiny_uks():
    from pyscf import gto, scf
    mol = gto.M(atom="O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24",
                basis="sto-3g", spin=0, verbose=0)
    mf = scf.UKS(mol)
    mf.xc = "lda"
    mf.kernel()
    return mf


def test_snapshot_roundtrip_at_root_is_chkfile_readable(tmp_path):
    from pyscf.scf import chkfile as pyscf_chkfile
    from pymbxas.io.data import pyscf_data

    mf = _tiny_uks()
    data = pyscf_data(mf)

    path = tmp_path / "snap.h5"
    with h5py.File(path, "w") as f:
        h5.stamp(f, h5.KIND_CALCULATION)
        h5.write_snapshot(f, data)

    back = h5.read_snapshot(str(path), "/")
    assert np.array_equal(back.mo_coeff, np.asarray(mf.mo_coeff))
    assert np.array_equal(back.mo_occ, np.asarray(mf.mo_occ))
    assert np.array_equal(back.mo_energy, np.asarray(mf.mo_energy))
    assert back.e_tot == mf.e_tot
    assert back.nelec == tuple(mf.nelec)
    assert back.mol.natm == mf.mol.natm
    assert np.allclose(back.mol.atom_coords(), mf.mol.atom_coords())

    mol_chk, scf_chk = pyscf_chkfile.load_scf(str(path))
    assert mol_chk.natm == 3
    assert np.array_equal(scf_chk["mo_coeff"], np.asarray(mf.mo_coeff))


def test_snapshot_roundtrip_in_nested_group_and_lazy(tmp_path):
    from pymbxas.io.data import pyscf_data

    mf = _tiny_uks()
    data = pyscf_data(mf)
    data.mo_coeff_del = np.asarray(mf.mo_coeff) * 2.0

    path = tmp_path / "nested.h5"
    with h5py.File(path, "w") as f:
        h5.stamp(f, h5.KIND_CALCULATION)
        h5.write_snapshot(f.create_group("excitations/000/fch"), data)

    eager = h5.read_snapshot(str(path), "excitations/000/fch")
    assert np.array_equal(eager.mo_coeff_del, np.asarray(mf.mo_coeff) * 2.0)

    lazy = h5.read_snapshot(str(path), "excitations/000/fch", lazy=True)
    assert "mo_coeff" not in vars(lazy)
    assert lazy.e_tot == mf.e_tot
    assert np.array_equal(lazy.mo_coeff, np.asarray(mf.mo_coeff))


def test_snapshot_without_mo_coeff_del_reports_absent(tmp_path):
    from pymbxas.io.data import pyscf_data

    mf = _tiny_uks()
    path = tmp_path / "nodel.h5"
    with h5py.File(path, "w") as f:
        h5.stamp(f, h5.KIND_CALCULATION)
        h5.write_snapshot(f, pyscf_data(mf))

    for lazy in (False, True):
        back = h5.read_snapshot(str(path), "/", lazy=lazy)
        assert getattr(back, "mo_coeff_del", None) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n pymbxas pytest tests/test_h5_io.py -q -k snapshot`
Expected: FAIL with `AttributeError: module 'pymbxas.io.h5' has no attribute 'write_snapshot'`.

- [ ] **Step 3: Write the implementation**

Append to `pymbxas/io/h5.py`:

```python
def write_snapshot(group, data):
    data = data.to_cpu()

    write_str(group, "mol", data.mol.dumps())

    scf = group.create_group("scf")
    scf.create_dataset("e_tot", data=float(data.e_tot))
    write_array(scf, "nelec", np.asarray(data.nelec, dtype=np.int64))
    for name in ("mo_coeff", "mo_occ", "mo_energy"):
        write_array(scf, name, np.asarray(getattr(data, name)))

    mo_coeff_del = getattr(data, "mo_coeff_del", None)
    if mo_coeff_del is not None:
        write_array(scf, "mo_coeff_del", np.asarray(mo_coeff_del))
    return


def read_snapshot(path, key, lazy=False):
    from pyscf import gto
    from pymbxas.io.data import pyscf_data

    with h5py.File(path, "r") as f:
        group = f[key]
        mol   = gto.loads(read_str(group, "mol"))
        scf   = group["scf"]
        e_tot = float(scf["e_tot"][()])
        nelec = tuple(int(x) for x in scf["nelec"][()])

        if lazy:
            mol.verbose = 0
            return pyscf_data.from_h5_source(mol, e_tot, nelec, str(path), key)

        arrays = {name: scf[name][()] for name in
                  ("mo_coeff", "mo_occ", "mo_energy") if name in scf}
        if "mo_coeff_del" in scf:
            arrays["mo_coeff_del"] = scf["mo_coeff_del"][()]

    mol.verbose = 0
    return pyscf_data.from_arrays(mol, e_tot, nelec, **arrays)
```

Note for the implementer: `gto.loads` cannot restore `mol.stdout`, which was a pymbxas `Logger`. It substitutes `sys.stdout` and keeps the stored `verbose`, so a reloaded mol at verbose 4 would print SCF chatter to the terminal. Setting `mol.verbose = 0` here is what prevents that; callers that want output raise it themselves.

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n pymbxas pytest tests/test_h5_io.py -q`
Expected: 14 passed.

- [ ] **Step 5: Commit**

```bash
git add pymbxas/io/h5.py tests/test_h5_io.py
git commit -m "Round-trip pyscf_data snapshots in PySCF chkfile layout"
```

---

### Task 4: `Spectra` save and load

**Files:**
- Modify: `pymbxas/spectra.py`
- Test: `tests/test_h5_io.py` (append)

**Interfaces:**
- Consumes: everything from Tasks 1 and 3.
- Produces: `Spectra.save(filename="spectra.h5") -> None`, `Spectra.load(filename) -> Spectra` (classmethod), `Spectra._write_into(group) -> None`, `Spectra._read_from(group) -> None`, `Spectra.materialize() -> None`. `_write_into` and `_read_from` operate on any h5py group, which is what lets `Spectras` reuse them in Task 5.

The constructor narrows: `Spectra(pyscf_obj, excitation=None)` accepts only an object with an `excitations` attribute. The `str` and `dict` branches, `__restart`, `__pkl_to_dict`, `_prepare_for_save` and the `change_key` import all go.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_h5_io.py`:

```python
def _hand_built_spectra():
    from pymbxas.spectra import Spectra

    mf  = _tiny_uks()
    nao = mf.mol.nao

    sp = Spectra.__new__(Spectra)
    sp.mol           = mf.mol
    sp.structure     = ase.build.molecule("H2O")
    sp._exc_idx      = 0
    sp.calc_settings = {"charge": 0, "spin": 0, "xc": "lda", "basis": "sto-3g",
                        "solvent": None, "pbc": False, "loc": "ibo",
                        "xch": True, "calc_type": "UKS"}
    sp._gs_energy = -76.0
    sp._energies  = np.linspace(19.0, 20.0, 4)
    sp._amplitude = np.arange(12, dtype=np.float64).reshape(3, 4)
    sp._mo_coeff  = np.asarray(mf.mo_coeff)
    sp._mo_occ    = np.asarray(mf.mo_occ)
    sp._channel   = 1
    sp._el_labels = np.array([-1, -1, 2, 2])
    sp._label     = 7
    return sp


def test_spectra_roundtrip(tmp_path):
    from pymbxas.spectra import Spectra

    sp   = _hand_built_spectra()
    path = tmp_path / "spectra.h5"
    sp.save(str(path))

    back = Spectra.load(str(path))

    assert np.array_equal(back._energies, sp._energies)
    assert np.array_equal(back.energies, sp.energies)
    assert np.array_equal(back._amplitude, sp._amplitude)
    assert np.array_equal(back._mo_coeff, sp._mo_coeff)
    assert np.array_equal(back._mo_occ, sp._mo_occ)
    assert np.array_equal(back._el_labels, sp._el_labels)
    assert np.array_equal(back.CMO, sp.CMO)
    assert back._channel == 1
    assert back._exc_idx == 0
    assert back._label == 7
    assert back._gs_energy == -76.0
    assert back.calc_settings == sp.calc_settings
    assert back.structure == sp.structure
    assert back.mol.natm == 3
    assert np.allclose(back.mol.atom_coords(), sp.mol.atom_coords())


def test_spectra_load_defers_mo_coeff(tmp_path):
    from pymbxas.spectra import Spectra

    sp   = _hand_built_spectra()
    path = tmp_path / "lazyspectra.h5"
    sp.save(str(path))

    back = Spectra.load(str(path))
    assert "_mo_coeff" not in vars(back)
    assert np.array_equal(back._mo_coeff, sp._mo_coeff)
    assert "_mo_coeff" in vars(back)

    other = Spectra.load(str(path))
    other.materialize()
    assert "_mo_occ" in vars(other)


def test_spectra_copy_is_independent(tmp_path):
    from pymbxas.spectra import Spectra

    sp   = _hand_built_spectra()
    path = tmp_path / "copyme.h5"
    sp.save(str(path))

    back = Spectra.load(str(path))
    dup  = back.copy()
    dup._amplitude[0, 0] = 999.0

    assert back._amplitude[0, 0] != 999.0
    assert isinstance(dup, Spectra)


def test_spectra_constructor_rejects_a_path(tmp_path):
    from pymbxas.spectra import Spectra

    with pytest.raises(TypeError, match="Spectra.load"):
        Spectra(str(tmp_path / "whatever.h5"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n pymbxas pytest tests/test_h5_io.py -q -k spectra`
Expected: FAIL with `AttributeError: type object 'Spectra' has no attribute 'load'`.

- [ ] **Step 3: Write the implementation**

In `pymbxas/spectra.py`, replace the `dill` import with the new ones. The import block becomes:

```python
# data manipulation
import numpy as np
from functools import reduce
import copy

# pymbxas utils
from pymbxas.build.structure import rotate_structure, ase_to_mole
from pymbxas.utils.basis import get_AO_permutation, get_l_val
from pymbxas.mbxas.broaden import get_mbxas_spectra
from pymbxas.io.write import write_data_to_fchk
from pymbxas.io import h5

# pyscf stuff
from pyscf import gto, lo
from pyscf.lo import iao, orth
```

Replace `__init__` (lines 35 to 42) with:

```python
    def __init__(self, pyscf_obj, excitation=None):
        if isinstance(pyscf_obj, (str, dict)):
            raise TypeError("Spectra() no longer loads files; use Spectra.load(path).")
        if not hasattr(pyscf_obj, "excitations"):
            raise TypeError("Invalid pyscf_obj type. Must be a pyscf object with excitations.")
        self.__initialize_spectra(pyscf_obj, excitation)
```

Delete `__restart` (lines 97 to 134) and `__pkl_to_dict` (lines 136 to 139) outright, and add in their place:

```python
    @classmethod
    def load(cls, filename):
        with h5.open_read(filename, h5.KIND_SPECTRA) as f:
            return cls._from_group(f)

    @classmethod
    def _from_group(cls, group):
        obj = cls.__new__(cls)
        obj._read_from(group)
        return obj

    def _write_into(self, group):
        h5.write_str(group, "mol", self.mol.dumps())
        h5.write_structure(group, "structure", self.structure)
        h5.write_json(group, "calc_settings", self.calc_settings)

        scf = group.create_group("scf")
        h5.write_array(scf, "mo_coeff", np.asarray(self._mo_coeff))
        h5.write_array(scf, "mo_occ", np.asarray(self._mo_occ))

        xas = group.create_group("xas")
        h5.write_array(xas, "energies", np.asarray(self._energies))
        h5.write_array(xas, "amplitude", np.asarray(self._amplitude))
        h5.write_array(xas, "el_labels", np.asarray(self._el_labels))

        group.attrs["channel"]   = int(self._channel)
        group.attrs["exc_idx"]   = -1 if self._exc_idx is None else int(self._exc_idx)
        group.attrs["label"]     = int(self._label)
        group.attrs["gs_energy"] = float(self._gs_energy)
        return

    def _read_from(self, group):
        self.structure     = h5.read_structure(group, "structure")
        self.calc_settings = h5.read_json(group, "calc_settings")

        xas = group["xas"]
        self._energies  = xas["energies"][()]
        self._amplitude = xas["amplitude"][()]
        self._el_labels = xas["el_labels"][()]

        exc_idx = int(group.attrs["exc_idx"])
        self._channel   = int(group.attrs["channel"])
        self._exc_idx   = None if exc_idx < 0 else exc_idx
        self._label     = int(group.attrs["label"])
        self._gs_energy = float(group.attrs["gs_energy"])

        self._h5_source = (group.file.filename, group.name)

        if "mol" in group:
            self.mol = gto.loads(h5.read_str(group, "mol"))
            self.mol.verbose = 0
        else:
            self.make_mol()
        return

    def __getattr__(self, name):
        if name in ("_mo_coeff", "_mo_occ"):
            source = self.__dict__.get("_h5_source")
            if source is not None:
                value = h5.read_lazy_field(source, name[1:])
                self.__dict__[name] = value
                return value
        raise AttributeError(name)

    def materialize(self):
        """Force deferred orbital coefficients to be read from disk."""
        if self.__dict__.get("_h5_source") is None:
            return
        self._mo_coeff
        self._mo_occ
        self.__dict__.pop("_h5_source", None)
        return
```

Replace `save` (lines 351 to 360) and delete `_prepare_for_save` (lines 342 to 349):

```python
    def save(self, filename="spectra.h5"):
        """Saves the object to an HDF5 file."""
        with h5.create(filename, h5.KIND_SPECTRA) as fout:
            self._write_into(fout)
        return
```

Replace `copy` (lines 367 to 369):

```python
    def copy(self):
        self.materialize()
        return copy.deepcopy(self)
```

Finally, `make_mol` keeps its current body; it is now only the fallback when a file has no stored `/mol`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n pymbxas pytest tests/test_h5_io.py -q`
Expected: 18 passed.

- [ ] **Step 5: Commit**

```bash
git add pymbxas/spectra.py tests/test_h5_io.py
git commit -m "Save and load Spectra as HDF5"
```

---

### Task 5: `Spectras` save and load

**Files:**
- Modify: `pymbxas/spectras.py`
- Test: `tests/test_h5_io.py` (append)

**Interfaces:**
- Consumes: `Spectra._write_into`, `Spectra._read_from`, `Spectra._from_group`, `Spectra.materialize` from Task 4.
- Produces: `Spectras.save(filename="spectras.h5") -> None`, `Spectras.load(filename) -> Spectras` (classmethod), `Spectras.materialize() -> None`.

The constructor narrows to lists and single `Spectra` objects. `__restart`, `__pkl_to_dict` and `_prepare_for_save` go.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_h5_io.py`:

```python
def test_spectras_roundtrip(tmp_path):
    from pymbxas.spectras import Spectras

    first  = _hand_built_spectra()
    second = _hand_built_spectra()
    second._energies = second._energies + 1.0

    coll = Spectras([first, second], labels=[4, 9])
    path = tmp_path / "coll.h5"
    coll.save(str(path))

    back = Spectras.load(str(path))

    assert len(back) == 2
    assert back.labels == [4, 9]
    assert back[0].label == 4
    assert back[1].label == 9
    assert np.array_equal(back[1]._energies, second._energies)
    assert np.array_equal(back[0]._mo_coeff, first._mo_coeff)
    assert np.allclose(back._erange, coll._erange)


def test_spectras_constructor_rejects_a_path(tmp_path):
    from pymbxas.spectras import Spectras

    with pytest.raises(TypeError, match="Spectras.load"):
        Spectras(str(tmp_path / "whatever.h5"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n pymbxas pytest tests/test_h5_io.py -q -k spectras`
Expected: FAIL with `AttributeError: type object 'Spectras' has no attribute 'load'`.

- [ ] **Step 3: Write the implementation**

In `pymbxas/spectras.py`, replace `import dill` with `from pymbxas.io import h5`.

Replace the `__init__` dispatch (lines 32 to 39) with:

```python
        if isinstance(spectra_list, Spectra):
            spectra_list = [spectra_list]
        
        if not isinstance(spectra_list, list):
            raise TypeError("Spectras() takes a list of Spectra; use Spectras.load(path) for files.")

        self.__initialize_collection(spectra_list, labels, post_align, alignment)
```

Delete `__restart` (lines 64 to 76) and `__pkl_to_dict` (lines 78 to 82), and add:

```python
    @classmethod
    def load(cls, filename):
        with h5.open_read(filename, h5.KIND_SPECTRAS) as f:
            spectras = [Spectra._from_group(f["spectras"][key])
                        for key in sorted(f["spectras"])]
            labels = [int(x) for x in f["labels"][()]]
            aligned = bool(f.attrs["aligned"])

        obj = cls(spectras, labels=labels)
        obj._aligned = aligned
        return obj

    def materialize(self):
        """Force every member's deferred coefficients to be read from disk."""
        for spectra in self.spectras:
            spectra.materialize()
        return
```

Replace `_prepare_for_save` (lines 310 to 318) and `save` (lines 320 to 329) with:

```python
    def save(self, filename="spectras.h5"):
        """Saves the collection to an HDF5 file."""
        with h5.create(filename, h5.KIND_SPECTRAS) as fout:
            fout.attrs["aligned"] = bool(self._aligned)
            h5.write_array(fout, "labels", np.asarray(self.labels, dtype=np.int64))
            root = fout.create_group("spectras")
            for cc, spectra in enumerate(self.spectras):
                spectra._write_into(root.create_group("{:03d}".format(cc)))
        return
```

Replace `copy` (lines 332 to 334):

```python
    def copy(self):
        self.materialize()
        return copy.deepcopy(self)
```

Note for the implementer: `__initialize_collection` deep-copies its input, and `assign_atomic_labels` sets `_aligned = False`, which is why `load` restores `_aligned` after construction rather than before. `_erange` is recomputed by `__init__` and is deliberately not stored.

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n pymbxas pytest tests/test_h5_io.py -q`
Expected: 20 passed.

- [ ] **Step 5: Commit**

```bash
git add pymbxas/spectras.py tests/test_h5_io.py
git commit -m "Save and load Spectras as HDF5"
```

---

### Task 6: `PySCF_mbxas` append-only writer

**Files:**
- Modify: `pymbxas/calculators/pyscf.py`
- Test: `tests/test_h5_io.py` (append)

**Interfaces:**
- Consumes: `write_snapshot`, `write_structure`, `write_json`, `write_text`, `write_array`, `stamp` from Tasks 1 and 3.
- Produces: `PySCF_mbxas.save_object(oname=None, save_path=None) -> str` (now returns the path it wrote), `PySCF_mbxas._resolve_save_path(oname, save_path) -> str`, `PySCF_mbxas._write_header(path) -> None`, `PySCF_mbxas._append_excitations(path) -> None`, and the attribute `PySCF_mbxas._h5_path`. Default `save_name` becomes `"pymbxas_obj.h5"`.

This task introduces the session fixture that Tasks 6 and 7 share. It runs one real H2O/sto-3g calculation, about 7 seconds, once per test session.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_h5_io.py`:

```python
@pytest.fixture(scope="session")
def tiny_calc(tmp_path_factory):
    from pymbxas.calculators.pyscf import PySCF_mbxas

    workdir = tmp_path_factory.mktemp("tiny")
    obj = PySCF_mbxas(
        structure=ase.build.molecule("H2O"),
        charge=0, spin=0, xc="lda", basis="sto-3g", calc_type="UKS",
        loc_type="ibo", xas_verbose=1, dft_verbose=0, dft_output=False,
        save=False, target_dir=str(workdir))
    obj.kernel("O")
    return obj


def test_save_object_writes_chkfile_shaped_root(tiny_calc, tmp_path):
    from pyscf.scf import chkfile as pyscf_chkfile

    path = tiny_calc.save_object(oname="calc.h5", save_path=str(tmp_path))
    assert path == str(tmp_path / "calc.h5")

    with h5py.File(path, "r") as f:
        assert h5.read_attr_str(f, "kind") == "calculation"
        assert int(f.attrs["schema_version"]) == h5.SCHEMA_VERSION
        assert bool(f.attrs["ran_GS"]) is True
        assert bool(f.attrs["used_loc"]) is False
        assert set(f["scf"]) >= {"e_tot", "mo_coeff", "mo_occ", "mo_energy", "nelec"}
        assert "mo_coeff_del" not in f["scf"]
        assert sorted(f["excitations"]) == ["000"]

        exc = f["excitations/000"]
        assert int(exc.attrs["ato_idx"]) == 0
        assert h5.read_attr_str(exc, "symbol") == "O"
        assert int(exc.attrs["channel"]) == 1
        assert int(exc.attrs["orb_idx"]) == tiny_calc.excitations[0].orb_idx
        assert bool(exc.attrs["complete"]) is True
        assert set(exc) == {"fch", "xch", "mbxas"}
        assert set(exc["mbxas"]) == {"energies", "absorption", "mb_overlap",
                                     "dipole_KS", "basis_ovlp"}
        assert h5.read_text(exc["fch"], "output").startswith("")
        assert h5.read_json(f, "parameters")["xc"] == "lda"
        assert h5.read_structure(f, "structure") == tiny_calc.structure

    mol_chk, scf_chk = pyscf_chkfile.load_scf(path)
    assert mol_chk.natm == 3
    assert np.array_equal(scf_chk["mo_coeff"], np.asarray(tiny_calc.gs_data.mo_coeff))


def test_save_object_is_append_only(tiny_calc, tmp_path):
    path = tiny_calc.save_object(oname="append.h5", save_path=str(tmp_path))

    with h5py.File(path, "r+") as f:
        f["excitations/000"].attrs["sentinel"] = 42

    tiny_calc.save_object(oname="append.h5", save_path=str(tmp_path))

    with h5py.File(path, "r") as f:
        assert int(f["excitations/000"].attrs["sentinel"]) == 42


def test_save_object_rewrites_incomplete_groups(tiny_calc, tmp_path):
    path = tiny_calc.save_object(oname="partial.h5", save_path=str(tmp_path))

    with h5py.File(path, "r+") as f:
        del f["excitations/000"].attrs["complete"]
        f["excitations/000"].attrs["sentinel"] = 42

    tiny_calc.save_object(oname="partial.h5", save_path=str(tmp_path))

    with h5py.File(path, "r") as f:
        assert "sentinel" not in f["excitations/000"].attrs
        assert bool(f["excitations/000"].attrs["complete"]) is True


def test_save_object_normalizes_the_extension(tiny_calc, tmp_path):
    path = tiny_calc.save_object(oname="named.pkl", save_path=str(tmp_path))
    assert path == str(tmp_path / "named.h5")


def test_save_object_requires_a_ground_state(tmp_path):
    from pymbxas.calculators.pyscf import PySCF_mbxas

    obj = PySCF_mbxas(
        structure=ase.build.molecule("H2O"), basis="sto-3g",
        xas_verbose=1, dft_verbose=0, dft_output=False, save=False,
        target_dir=str(tmp_path))

    with pytest.raises(RuntimeError, match="ground state"):
        obj.save_object()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n pymbxas pytest tests/test_h5_io.py -q -k save_object`
Expected: FAIL. `save_object` still writes a pickle, so `assert path == ...` fails on the returned `None`.

- [ ] **Step 3: Write the implementation**

In `pymbxas/calculators/pyscf.py`, drop `import dill` and add `from pymbxas.io import h5`. Do not import h5py here; use the `h5.create` and `h5.append` helpers from Task 1. Change `save_name = "pyscf_obj.pkl"` to `save_name = "pymbxas_obj.h5"` in the `__init__` signature (line 68).

In `_initialize_from_scratch`, after `self._excitations = []` (line 151), add:

```python
        self._h5_path      = None
```

Replace `save_object` (lines 402 to 431) with:

```python
    def save_object(self, oname=None, save_path=None):
        """
        Write the calculation to an HDF5 file, appending any excitation that
        is not on disk yet.

        Returns:
        str: path of the file written.
        """

        if not self._ran_GS:
            raise RuntimeError("Cannot save before the ground state has been run.")

        path = self._resolve_save_path(oname, save_path)

        if not os.path.exists(path):
            self._write_header(path)

        self._append_excitations(path)
        self._h5_path = path

        return path

    def _resolve_save_path(self, oname, save_path):

        if oname is None:
            oname = self.oset["save_name"]

        root = self._tdir if save_path is None else save_path

        if not oname.endswith(".h5"):
            oname = os.path.splitext(oname)[0] + ".h5"

        return os.path.join(root, oname)

    def _write_header(self, path):

        with h5.create(path, h5.KIND_CALCULATION) as f:
            f.attrs["ran_GS"]   = bool(self._ran_GS)
            f.attrs["used_loc"] = bool(self._used_loc)

            h5.write_structure(f, "structure", self.structure)
            h5.write_json(f, "parameters", self._parameters)
            h5.write_json(f, "output_settings", self._output_settings)
            h5.write_text(f, "output", self.output if isinstance(self.output, str) else "")
            h5.write_snapshot(f, self.gs_data)

            f.create_group("excitations")

        return

    def _append_excitations(self, path):

        with h5.append(path) as f:
            root = f["excitations"]

            for idx, exc in enumerate(self._excitations):
                key = "{:03d}".format(idx)

                if key in root:
                    if root[key].attrs.get("complete", False):
                        continue
                    del root[key]

                group = root.create_group(key)
                group.attrs["ato_idx"] = int(exc.ato_idx)
                group.attrs["symbol"]  = exc.symbol
                group.attrs["channel"] = int(exc.channel)
                group.attrs["orb_idx"] = int(exc.orb_idx)

                for name in ("fch", "xch"):
                    if name not in exc.data:
                        continue
                    sub = group.create_group(name)
                    h5.write_snapshot(sub, exc.data[name])
                    h5.write_text(sub, "output", exc.output.get(name, ""))

                mbxas = group.create_group("mbxas")
                for name, value in exc.mbxas.items():
                    h5.write_array(mbxas, name, np.asarray(value))

                group.attrs["complete"] = True

        return
```

Note for the implementer: `complete` is written last on purpose. A process killed mid-group leaves the group without that attribute, and both the loader in Task 7 and `_append_excitations` above treat such a group as absent. Do not hoist it to the top of the block.

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n pymbxas pytest tests/test_h5_io.py -q`
Expected: 25 passed.

- [ ] **Step 5: Commit**

```bash
git add pymbxas/calculators/pyscf.py tests/test_h5_io.py
git commit -m "Write calculations to HDF5 one excitation at a time"
```

---

### Task 7: `PySCF_mbxas.load` and restart semantics

**Files:**
- Modify: `pymbxas/calculators/pyscf.py`
- Modify: `pymbxas/calculators/excitation.py`
- Test: `tests/test_h5_io.py` (append)

**Interfaces:**
- Consumes: `read_snapshot`, `read_structure`, `read_json`, `read_text`, `read_attr_str`, `open_read` from Tasks 1 and 3; the writer from Task 6.
- Produces: `PySCF_mbxas.load(filename) -> PySCF_mbxas` (classmethod), `Excitation.from_h5(path, key) -> Excitation` (classmethod). The `pkl_file` constructor keyword is removed.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_h5_io.py`:

```python
def test_load_restores_the_ground_state(tiny_calc, tmp_path):
    from pymbxas.calculators.pyscf import PySCF_mbxas

    path = tiny_calc.save_object(oname="restart.h5", save_path=str(tmp_path))
    back = PySCF_mbxas.load(path)

    assert back._ran_GS is True
    assert back._used_loc is False
    assert back.df_obj is None
    assert back.parameters == tiny_calc.parameters
    assert back.structure == tiny_calc.structure
    assert np.array_equal(back.gs_data.mo_coeff, tiny_calc.gs_data.mo_coeff)
    assert np.array_equal(back.gs_data.mo_occ, tiny_calc.gs_data.mo_occ)
    assert back.gs_data.e_tot == tiny_calc.gs_data.e_tot
    assert back.gs_data.nelec == tiny_calc.gs_data.nelec
    assert back.mol.natm == 3
    assert back._tdir == str(tmp_path)
    assert back._cdir == os.getcwd()
    assert back.logger is not None


def test_load_restores_excitations_lazily(tiny_calc, tmp_path):
    from pymbxas.calculators.pyscf import PySCF_mbxas

    path = tiny_calc.save_object(oname="lazyexc.h5", save_path=str(tmp_path))
    back = PySCF_mbxas.load(path)

    assert len(back.excitations) == 1
    assert back.excited_idxs == [0]

    exc = back.excitations[0]
    ref = tiny_calc.excitations[0]

    assert exc.symbol == "O"
    assert exc.channel == ref.channel
    assert exc.orb_idx == ref.orb_idx
    assert set(exc.data) == {"fch", "xch"}
    assert "mo_coeff" not in vars(exc.data["fch"])
    assert np.array_equal(exc.data["fch"].mo_coeff, ref.data["fch"].mo_coeff)
    assert exc.data["xch"].e_tot == ref.data["xch"].e_tot

    for name in ref.mbxas:
        assert np.array_equal(exc.mbxas[name], ref.mbxas[name])


def test_loaded_object_skips_a_finished_atom(tiny_calc, tmp_path):
    from pymbxas.calculators.pyscf import PySCF_mbxas

    path = tiny_calc.save_object(oname="skip.h5", save_path=str(tmp_path))
    back = PySCF_mbxas.load(path)

    back.excite(0)
    assert len(back.excitations) == 1


def test_loaded_object_produces_a_spectra(tiny_calc, tmp_path):
    from pymbxas.calculators.pyscf import PySCF_mbxas
    from pymbxas.spectra import Spectra

    path = tiny_calc.save_object(oname="tospectra.h5", save_path=str(tmp_path))
    spectra = PySCF_mbxas.load(path).to_spectra()

    assert isinstance(spectra, Spectra)
    assert np.array_equal(spectra._energies, tiny_calc.excitations[0].mbxas["energies"])


def test_load_skips_incomplete_excitation_groups(tiny_calc, tmp_path):
    from pymbxas.calculators.pyscf import PySCF_mbxas

    path = tiny_calc.save_object(oname="broken.h5", save_path=str(tmp_path))
    with h5py.File(path, "r+") as f:
        del f["excitations/000"].attrs["complete"]

    back = PySCF_mbxas.load(path)
    assert back.excitations == []


def test_force_ground_state_rerun_is_refused_after_load(tiny_calc, tmp_path):
    from pymbxas.calculators.pyscf import PySCF_mbxas

    path = tiny_calc.save_object(oname="refuse.h5", save_path=str(tmp_path))
    back = PySCF_mbxas.load(path)

    with pytest.raises(RuntimeError, match="excitations"):
        back.run_ground_state(force=True)


def test_pkl_file_keyword_is_gone():
    import inspect
    from pymbxas.calculators.pyscf import PySCF_mbxas

    assert "pkl_file" not in inspect.signature(PySCF_mbxas.__init__).parameters
```

Add `import os` to the top of `tests/test_h5_io.py` if it is not already there.

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n pymbxas pytest tests/test_h5_io.py -q -k "load or force or pkl_file"`
Expected: FAIL with `AttributeError: type object 'PySCF_mbxas' has no attribute 'load'`.

- [ ] **Step 3: Write the implementation**

In `pymbxas/calculators/excitation.py`, add `from pymbxas.io import h5` to the imports. Do not import h5py here. Then add this classmethod to `Excitation` immediately after `__init__`:

```python
    @classmethod
    def from_h5(cls, path, key):
        exc = cls.__new__(cls)

        with h5.open_plain(path) as f:
            group = f[key]
            exc.ato_idx = int(group.attrs["ato_idx"])
            exc.symbol  = h5.read_attr_str(group, "symbol")
            exc.channel = int(group.attrs["channel"])
            exc.orb_idx = int(group.attrs["orb_idx"])

            names      = [name for name in ("fch", "xch") if name in group]
            exc.output = {name: h5.read_text(group[name], "output") for name in names}
            exc.mbxas  = {name: group["mbxas"][name][()] for name in group["mbxas"]}

        exc.data = {name: h5.read_snapshot(path, "{}/{}".format(key, name), lazy=True)
                    for name in names}

        return exc
```

In `pymbxas/calculators/pyscf.py`, remove the `pkl_file` parameter from `__init__` (line 55) and replace the whole restart branch (lines 73 to 95) with:

```python
        # store directories and path
        self._cdir = os.getcwd() # current directory
        self._tdir = os.getcwd() if target_dir is None \
            else os.path.abspath(target_dir) # target directory

        if not os.path.exists(self._tdir):
            os.makedirs(self._tdir)

        self._initialize_from_scratch(structure, charge, spin,
                                      xc, basis, pbc, solvent, calc_type,
                                      do_xch, xas_verbose, xas_logfile,
                                      dft_verbose, dft_logfile, dft_output,
                                      print_fchk, save, loc_type,
                                      save_name, save_path, save_chk, gpu)
        
        return
```

Replace `_restart_from_pickle` (lines 434 to 453) with:

```python
    @classmethod
    def load(cls, filename):
        """Reopen a calculation previously written with save_object()."""

        obj = cls.__new__(cls)
        obj._load_h5(filename)
        return obj

    def _load_h5(self, filename):

        path = os.path.abspath(filename)

        with h5.open_read(path, h5.KIND_CALCULATION) as f:
            self.structure        = h5.read_structure(f, "structure")
            self._parameters      = h5.read_json(f, "parameters")
            self._output_settings = h5.read_json(f, "output_settings")
            self.output           = h5.read_text(f, "output")
            self._ran_GS          = bool(f.attrs["ran_GS"])
            self._used_loc        = bool(f.attrs["used_loc"])

            complete   = []
            incomplete = []
            for key in sorted(f["excitations"]):
                if f["excitations"][key].attrs.get("complete", False):
                    complete.append("excitations/" + key)
                else:
                    incomplete.append(key)

        configure_logger(self._output_settings["xas_verbose"],
                         log_file=self._output_settings["xas_logfile"])
        self.logger = logging.getLogger(__name__)

        for key in incomplete:
            self.logger.warning("Skipping incomplete excitation {} in {}".format(key, path))

        self.gs_data     = h5.read_snapshot(path, "/")
        self.mol         = self.gs_data.mol
        self.mol.verbose = self._output_settings["dft_verbose"]
        self.df_obj      = None

        self._cdir        = os.getcwd()
        self._tdir        = os.path.dirname(path) or os.getcwd()
        self._h5_path     = path
        self._excitations = [Excitation.from_h5(path, key) for key in complete]

        return
```

In `run_ground_state`, insert this immediately after the `if self._ran_GS and not force:` block (after line 250):

```python
        if force and self._excitations:
            raise RuntimeError(
                "Cannot re-run the ground state: {} excitations were computed against "
                "the current one. Start a new calculation instead.".format(
                    len(self._excitations)))
```

Finally, in `kernel`, change the save log line (line 195) to report the returned path:

```python
        # save object if needed
        if self.oset["save"]:
            self.logger.info("Saved everything as {}".format(self.save_object()))
```

and delete the now-duplicated `self.save_object()` call on line 194.

Note for the implementer: `_cdir` and `_tdir` are deliberately re-derived rather than restored. The pickle used to carry absolute paths from the machine that ran the job, so `kernel()` would chdir somewhere unintended after the file moved. `read_snapshot(path, "/")` reads the root group, which is where `_write_header` put the ground state.

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n pymbxas pytest tests/test_h5_io.py -q`
Expected: 32 passed.

Run: `conda run -n pymbxas pytest tests/ -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add pymbxas/calculators/pyscf.py pymbxas/calculators/excitation.py tests/test_h5_io.py
git commit -m "Reopen calculations from HDF5 and continue exciting atoms"
```

---

### Task 8: End-to-end round-trip assertions

**Files:**
- Modify: `tests/test_h2o_kedge.py` (append to the existing test)

**Interfaces:**
- Consumes: `PySCF_mbxas.save_object`, `PySCF_mbxas.load`, `Spectra.save`, `Spectra.load` from Tasks 4, 6 and 7.
- Produces: nothing consumed by later tasks.

The project keeps one end-to-end test. These assertions extend `test_h2o_oxygen_kedge` rather than adding a file, and reuse the `obj`, `exc`, `gs`, `fch`, `xch`, `ch` and `tmp_path` names already in scope.

- [ ] **Step 1: Write the failing test**

Append to the end of `test_h2o_oxygen_kedge` in `tests/test_h2o_kedge.py`:

```python
    h5_path = obj.save_object(oname="roundtrip.h5", save_path=str(tmp_path))

    from pyscf.scf import chkfile as pyscf_chkfile
    from pymbxas.calculators.pyscf import PySCF_mbxas as _PySCF_mbxas
    from pymbxas.mbxas.mbxas import run_MBXAS_pyscf
    from pymbxas.spectra import Spectra

    mol_chk, scf_chk = pyscf_chkfile.load_scf(h5_path)
    assert mol_chk.natm == 3, "Checkpoint is not readable as a PySCF chkfile"
    assert np.array_equal(scf_chk["mo_coeff"], np.asarray(gs.mo_coeff)), \
        "chkfile-read GS coefficients differ from the in-memory ones"

    back = _PySCF_mbxas.load(h5_path)
    assert back._ran_GS is True, "Reloaded object does not report a finished ground state"
    assert back.excited_idxs == [0], f"Expected excited atom [0], got {back.excited_idxs}"

    b_gs  = back.gs_data
    b_exc = back.excitations[0]
    b_fch = b_exc.data["fch"]
    b_xch = b_exc.data["xch"]

    assert np.array_equal(b_gs.mo_coeff, gs.mo_coeff), "GS coefficients changed across a save/load"
    assert np.array_equal(b_gs.mo_occ, gs.mo_occ), "GS occupations changed across a save/load"
    assert b_gs.e_tot == gs.e_tot, "GS energy changed across a save/load"
    assert b_gs.nelec == gs.nelec, f"GS nelec changed across a save/load: {b_gs.nelec} vs {gs.nelec}"
    assert np.array_equal(b_fch.mo_coeff, fch.mo_coeff), "FCH coefficients changed across a save/load"
    assert np.array_equal(b_fch.mo_energy, fch.mo_energy), "FCH eigenvalues changed across a save/load"
    assert b_xch.e_tot == xch.e_tot, "XCH energy changed across a save/load"
    assert b_exc.orb_idx == exc.orb_idx, "Core orbital index changed across a save/load"
    assert b_exc.channel == ch, f"Excited channel changed across a save/load: {b_exc.channel} vs {ch}"

    for key in exc.mbxas:
        assert np.array_equal(b_exc.mbxas[key], exc.mbxas[key]), \
            f"mbxas['{key}'] changed across a save/load"

    n_before = len(back.excitations)
    back.excite(0)
    assert len(back.excitations) == n_before, "Reloaded object re-ran an excitation it already had"

    energies_rt, absorption_rt, _, _, _ = run_MBXAS_pyscf(
        b_gs.mol, b_gs.to_cpu(), b_fch.to_cpu(), b_exc.orb_idx,
        channel=b_exc.channel, xch_calc=b_xch.to_cpu())

    assert np.allclose(energies_rt, exc.mbxas["energies"], atol=1e-12), \
        "MBXAS re-derived from the checkpoint gives different energies"
    assert np.allclose(absorption_rt, amp_library, atol=1e-12), \
        "MBXAS re-derived from the checkpoint gives different amplitudes"

    spectra_path = str(tmp_path / "spectra.h5")
    spectra = obj.to_spectra()
    spectra.save(spectra_path)
    spectra_back = Spectra.load(spectra_path)

    assert np.array_equal(spectra_back.energies, spectra.energies), \
        "Spectra energies changed across a save/load"
    assert np.array_equal(spectra_back.amplitude, spectra.amplitude), \
        "Spectra amplitudes changed across a save/load"
    assert np.array_equal(spectra_back.CMO, spectra.CMO), \
        "Spectra CMO changed across a save/load"
    assert spectra_back.exc_idx == spectra.exc_idx, "Spectra excited index changed across a save/load"
    assert spectra_back.channel == spectra.channel, "Spectra channel changed across a save/load"
```

- [ ] **Step 2: Run test to verify it fails**

Before running, confirm the test currently ends at the `amp_library.shape[1]` assertion, and that `amp_library` is still in scope where you appended. It is assigned on line 80.

Run: `conda run -n pymbxas pytest tests/test_h2o_kedge.py -q`
Expected: PASS if Tasks 1 to 7 are all complete. If any is missing this fails with a clear `AttributeError`, which is the point of running it here.

- [ ] **Step 3: No implementation needed**

This task adds coverage over behavior built in Tasks 1 to 7. If the test fails, fix the module it points at rather than weakening the assertion.

- [ ] **Step 4: Run the full suite**

Run: `conda run -n pymbxas pytest tests/ -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_h2o_kedge.py
git commit -m "Assert the HDF5 round-trip preserves the full calculation"
```

---

### Task 9: Remove dill from the package

**Files:**
- Modify: `setup.cfg:25`
- Modify: `pymbxas/cli/pyscf.py:35-36`
- Modify: `pymbxas/drivers/acquisitor.py:35,44`
- Modify: `pymbxas/examples/example_H2O_molecule.py:52`
- Modify: `pymbxas/explorer/mbxasplorer.py:402-406`
- Modify: `pymbxas/utils/auxiliary.py:59-64`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_h5_io.py`:

```python
def test_dill_is_gone_from_the_package():
    import pathlib
    import pymbxas

    root = pathlib.Path(pymbxas.__file__).parent
    offenders = [str(p) for p in root.rglob("*.py")
                 if "dill" in p.read_text()]
    assert offenders == [], f"dill still referenced in: {offenders}"


def test_change_key_is_gone():
    from pymbxas.utils import auxiliary

    assert not hasattr(auxiliary, "change_key")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n pymbxas pytest tests/test_h5_io.py -q -k "dill or change_key"`
Expected: FAIL listing `explorer/mbxasplorer.py`, and `change_key` still present.

- [ ] **Step 3: Make the changes**

`setup.cfg`, in `install_requires`, replace the `dill>=0.3.8` line with:

```
    h5py
```

`pymbxas/cli/pyscf.py`, lines 35 and 36:

```python
        "-o", "--output_file", default="spectrum.h5",
        help="Path to save the spectrum (default: spectrum.h5)"
```

`pymbxas/drivers/acquisitor.py`: delete the `"pkl_file": None,` entry from the `defaults` dict (line 35), and change line 44 to:

```python
        "save_name": "pymbxas_obj.h5",
```

`pymbxas/examples/example_H2O_molecule.py`, line 52:

```python
    save_name    = "pymbxas_obj.h5", # name of saved file
```

`pymbxas/explorer/mbxasplorer.py`: delete `_save_self` entirely, lines 402 to 406. It is defined but never called anywhere in the tree, and the explorer has no HDF5 writer.

`pymbxas/utils/auxiliary.py`: delete `change_key`, lines 59 to 64. Its only two call sites were the pickle compatibility shims, both removed in Tasks 4 and 7.

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n pymbxas pytest tests/ -q`
Expected: all pass.

Run: `conda run -n pymbxas python -c "import pymbxas.explorer.mbxasplorer" 2>&1 | tail -1`
Expected: a gpflow or tensorflow `ImportError`. That is the pre-existing state, not a regression. If it imports cleanly, also fine.

- [ ] **Step 5: Commit**

```bash
git add setup.cfg pymbxas/cli/pyscf.py pymbxas/drivers/acquisitor.py \
        pymbxas/examples/example_H2O_molecule.py pymbxas/explorer/mbxasplorer.py \
        pymbxas/utils/auxiliary.py tests/test_h5_io.py
git commit -m "Drop dill from the package"
```

---

### Task 10: Documentation and version bump

**Files:**
- Modify: `pymbxas/__init__.py:51,76-77`
- Modify: `CITATION.cff:8-9`
- Modify: `CHANGELOG.md`
- Modify: `README.md:80,82,117,133-136`
- Modify: `CLAUDE.md:6,75`
- Modify: `dev/architecture.md:124,131-138`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing.

- [ ] **Step 1: Bump the version**

`pymbxas/__init__.py`, lines 76 and 77:

```python
__version__ = '0.6.0'
__date__ = "21 Aug. 2026"
```

Line 51, in the docstring example:

```python
        save_name    = "pymbxas_obj.h5", # name of saved file
```

`CITATION.cff`, lines 8 and 9:

```yaml
version: 0.6.0
date-released: 2026-08-21
```

- [ ] **Step 2: Add the changelog entries**

In `CHANGELOG.md`, add these lines to the existing `### Added` and `### Changed` sections under `## [Unreleased]`. Do not create new sections and do not reorder what is there.

Under `### Added`:

```markdown
- Saved calculations can be reopened with `.load()` and continued without repeating the ground state
- Saved calculations are readable by PySCF as chkfiles
```

Under `### Changed`:

```markdown
- Calculations and spectra are now saved as HDF5 files, and `.pkl` files can no longer be loaded
- Reloading a calculation no longer reads orbital coefficients until they are used
```

- [ ] **Step 3: Update the README**

`README.md`, line 80:

```
    save         = True,  # save object as an HDF5 file
```

Line 82:

```
    save_name    = "pymbxas_obj.h5", # name of saved file
```

Line 117, inside the console output block:

```
[16:02:02] |(I) Saved everything as pymbxas_obj.h5
```

Lines 133 to 136, replace the paragraph and code block with:

````markdown
Calculations are stored as HDF5 files. You can reload one, and keep exciting further atoms without re-running the ground state, with:

```python
obj = PySCF_mbxas.load("pymbxas_obj.h5")
obj.excite("N")
```

The file follows PySCF's chkfile layout for the ground state, so `pyscf.scf.chkfile.load_scf` and `mf.from_chk` also work on it.
````

- [ ] **Step 4: Update CLAUDE.md**

Line 6, replace `dill` with `h5py` in the stack list.

Line 75, replace the persistence bullet with:

```markdown
- **Persistence is HDF5 via `io/h5.py`**, which is the only module that imports h5py. The layout mirrors PySCF's chkfile shape, so `chkfile.load_scf` works on a checkpoint, but the writer is ours so that arrays can be gzipped. `.pkl` files from 0.5.x and earlier cannot be read; support was removed in 0.6.0.
```

Add a new bullet directly after it:

```markdown
- **Checkpoint writes are append-only.** `save_object()` writes the header and ground state once, then adds one `/excitations/NNN` group per finished atom. The `complete` attribute is written last, and a group without it is treated as absent by both the loader and the next write.
```

- [ ] **Step 5: Update dev/architecture.md**

Line 124, replace `Older pickles stored only the excited channel` with `Older files stored only the excited channel`.

Replace the entire `## Persistence` section, lines 131 to 138, with:

```markdown
HDF5, written and read by `pymbxas/io/h5.py`. That module owns the schema, the compression policy (gzip level 4 above 64 KiB) and every h5py call; no other module imports h5py directly. Root attributes are `kind` (`calculation`, `spectra` or `spectras`), `schema_version` and `pymbxas_version`.

The layout mirrors PySCF's chkfile shape: a `mol` dataset holding `mol.dumps()` beside an `scf/` group of `e_tot`, `mo_coeff`, `mo_occ`, `mo_energy` and `nelec`. PySCF's own `save_mol`/`load_scf` wrappers hardcode the root keys `mol` and `scf` and take a filename rather than a group, so only the ground state, which sits at the root, is loadable through them; `chkfile.load_scf(path)` and `mf.from_chk(path)` both work on a checkpoint. Excitation snapshots repeat the same shape at `/excitations/NNN/{fch,xch}/` and are read with `chkfile.load(path, key)` plus `gto.loads`, which accept arbitrary keys.

- `PySCF_mbxas.save_object()` writes the header and ground state on first call, then appends one `/excitations/NNN` group per finished excitation, keyed by zero-padded sequence index with `ato_idx` as a group attribute. The `complete` attribute is written last; a group missing it is skipped by the loader and overwritten by the next save. `PySCF_mbxas.load()` restores the structure, parameters, ground state and excitations, rebuilds `mol` with `gto.loads` and the logger with `configure_logger`, re-derives `_cdir` and `_tdir` from the current directory and the file's location, and sets `df_obj` to `None`.
- `Spectra.save()` stores the structure through `ase.io.jsonio`, the settings as JSON, and `mol.dumps()`, so `make_mol()` is now only the fallback for a file without a stored mol. `Spectras.save()` writes each member into `/spectras/NNN` using the same `_write_into` method.
- Orbital coefficients load on first access, not at open. `pyscf_data` and `Spectra` both keep an `_h5_source` tuple of `(path, group)` and read through `__getattr__`; `materialize()` forces everything in. The ground state is read eagerly because a restart needs it immediately.

**`.pkl` files cannot be read.** Support was removed in 0.6.0 along with the dill dependency.
```

- [ ] **Step 6: Verify and commit**

Run: `conda run -n pymbxas pytest tests/ -q`
Expected: all pass.

Run: `conda run -n pymbxas python -c "import pymbxas; print(pymbxas.__version__)"`
Expected: `0.6.0`.

Run: `grep -rn "pkl\|dill" README.md CLAUDE.md dev/ pymbxas/ setup.cfg | grep -v "\.pyc"`
Expected: only the deliberate mentions, namely the removal notes in `CLAUDE.md`, `dev/architecture.md` and the `open_read` error message in `pymbxas/io/h5.py`.

```bash
git add pymbxas/__init__.py CITATION.cff CHANGELOG.md README.md CLAUDE.md dev/architecture.md
git commit -m "Document HDF5 persistence and bump to 0.6.0"
```

---

## Done when

- `conda run -n pymbxas pytest tests/ -q` is green.
- `grep -rn dill pymbxas/ setup.cfg` returns nothing.
- `PySCF_mbxas.load(path).excite(new_atom)` extends an existing file without re-running the ground state.
- `pyscf.scf.chkfile.load_scf(path)` returns the ground state from a checkpoint.
- `pymbxas.__version__` is `0.6.0` and `CITATION.cff` agrees.
