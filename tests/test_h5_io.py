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
