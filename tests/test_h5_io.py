import json
import os

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
    sp._mb_overlap    = np.arange(2 * nao * nao, dtype=np.float64).reshape(2, nao, nao)
    sp._fch_mo_energy = np.linspace(-20.0, 5.0, 2 * nao).reshape(2, nao)
    sp._gs_mo_energy  = np.asarray(mf.mo_energy)
    sp._gs_mo_occ     = np.asarray(mf.mo_occ)
    sp._core_orb_idx  = 0
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
    assert np.array_equal(back._gs_mo_energy, sp._gs_mo_energy)
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


def test_historical_spectra_without_gs_mo_energy_still_loads(tmp_path):
    from pymbxas.spectra import Spectra

    sp = _hand_built_spectra()
    path = tmp_path / "historical-spectra.h5"
    sp.save(str(path))
    with h5py.File(path, "r+") as f:
        del f["shakeup/gs_mo_energy"]

    back = Spectra.load(str(path))
    assert back._gs_mo_energy is None


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


@pytest.fixture(scope="session")
def tiny_calc(tmp_path_factory):
    from pymbxas.calculators.pyscf import PySCFMBXAS
    from pymbxas.config import (
        CalculationConfig, CheckpointConfig, LoggingConfig, RuntimeConfig,
    )

    workdir = tmp_path_factory.mktemp("tiny")
    obj = PySCFMBXAS(
        ase.build.molecule("H2O"),
        config=CalculationConfig(xc="lda", basis="sto-3g"),
        runtime=RuntimeConfig(
            work_directory=workdir,
            logging=LoggingConfig(
                pymbxas_verbosity=1, pyscf_verbosity=0,
                pyscf_console=False),
            checkpoint=CheckpointConfig(enabled=False)))
    obj.run("O")
    return obj


def test_save_writes_configs_and_chkfile_shaped_root(tiny_calc, tmp_path):
    from pyscf.scf import chkfile as pyscf_chkfile

    path = tiny_calc.save(tmp_path / "calc.h5")
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
        assert set(exc) == {
            "config", "provenance", "fch", "xch", "mbxas"}
        assert set(exc["mbxas"]) == {"energies", "absorption", "mb_overlap",
                                     "dipole_KS", "basis_ovlp"}
        assert h5.read_text(exc["fch"], "output").startswith("")
        assert h5.read_json(f, "calculation_config") == tiny_calc.config.to_dict()
        assert h5.read_json(exc, "config") == tiny_calc.excitations[0].config.to_dict()
        assert h5.read_json(f, "ground_state_provenance")["device"] == "cpu"
        assert h5.read_json(exc, "provenance")["device"] == "cpu"
        assert h5.read_structure(f, "structure") == tiny_calc.structure

    mol_chk, scf_chk = pyscf_chkfile.load_scf(path)
    assert mol_chk.natm == 3
    assert np.array_equal(scf_chk["mo_coeff"], np.asarray(tiny_calc.gs_data.mo_coeff))


def test_save_is_append_only(tiny_calc, tmp_path):
    path = tiny_calc.save(tmp_path / "append.h5")

    with h5py.File(path, "r+") as f:
        f["excitations/000"].attrs["sentinel"] = 42

    tiny_calc.save(path)

    with h5py.File(path, "r") as f:
        assert int(f["excitations/000"].attrs["sentinel"]) == 42


def test_fresh_save_replaces_stale_target(tiny_calc, tmp_path):
    path = tiny_calc.save(tmp_path / "fresh.h5")

    with h5py.File(path, "r+") as f:
        f["excitations"].create_group("999").attrs["complete"] = True

    # Simulate a new in-memory calculation choosing an existing output name.
    tiny_calc._h5_path = None
    tiny_calc.save(path)

    with h5py.File(path, "r") as f:
        assert sorted(f["excitations"]) == ["000"]


def test_save_rewrites_incomplete_groups(tiny_calc, tmp_path):
    path = tiny_calc.save(tmp_path / "partial.h5")

    with h5py.File(path, "r+") as f:
        del f["excitations/000"].attrs["complete"]
        f["excitations/000"].attrs["sentinel"] = 42

    tiny_calc.save(path)

    with h5py.File(path, "r") as f:
        assert "sentinel" not in f["excitations/000"].attrs
        assert bool(f["excitations/000"].attrs["complete"]) is True


def test_save_requires_a_ground_state(tmp_path):
    from pymbxas.calculators.pyscf import PySCFMBXAS
    from pymbxas.config import CalculationConfig, RuntimeConfig

    obj = PySCFMBXAS(
        ase.build.molecule("H2O"),
        config=CalculationConfig(basis="sto-3g"),
        runtime=RuntimeConfig(work_directory=tmp_path))

    with pytest.raises(RuntimeError, match="ground state"):
        obj.save()


def test_load_restores_the_ground_state(tiny_calc, tmp_path):
    from pymbxas.calculators.pyscf import PySCFMBXAS

    path = tiny_calc.save(tmp_path / "restart.h5")
    back = PySCFMBXAS.load(path)

    assert back._ran_GS is True
    assert back._used_loc is False
    assert back.df_obj is None
    assert back.config == tiny_calc.config
    assert back.structure == tiny_calc.structure
    assert np.array_equal(back.gs_data.mo_coeff, tiny_calc.gs_data.mo_coeff)
    assert np.array_equal(back.gs_data.mo_occ, tiny_calc.gs_data.mo_occ)
    assert back.gs_data.e_tot == tiny_calc.gs_data.e_tot
    assert back.gs_data.nelec == tiny_calc.gs_data.nelec
    assert back.mol.natm == 3
    assert back.runtime.work_directory == str(tmp_path)
    assert back.logger is not None


def test_load_accepts_runtime_but_not_scientific_overrides(tiny_calc, tmp_path):
    from pymbxas.calculators.pyscf import PySCFMBXAS
    from pymbxas.config import LoggingConfig, RuntimeConfig

    path = tiny_calc.save(tmp_path / "verbosity.h5")
    analysis_log = tmp_path / "analysis.log"
    runtime = RuntimeConfig(
        work_directory=tmp_path,
        logging=LoggingConfig(
            pymbxas_logfile=analysis_log, pyscf_verbosity=4,
            pyscf_console=False))
    back = PySCFMBXAS.load(path, runtime=runtime)

    assert back.mol.verbose == 4
    assert back.runtime.logging.pymbxas_logfile == str(analysis_log)
    assert back.config == tiny_calc.config


def test_load_restores_excitations_lazily(tiny_calc, tmp_path):
    from pymbxas.calculators.pyscf import PySCFMBXAS

    path = tiny_calc.save(tmp_path / "lazyexc.h5")
    back = PySCFMBXAS.load(path)

    assert len(back.excitations) == 1
    assert back.excited_idxs == [0]

    exc = back.excitations[0]
    ref = tiny_calc.excitations[0]

    assert exc.symbol == "O"
    assert exc.channel == ref.channel
    assert exc.orb_idx == ref.orb_idx
    assert exc.config == ref.config
    assert exc.provenance == ref.provenance
    assert set(exc.data) == {"fch", "xch"}
    assert "mo_coeff" not in vars(exc.data["fch"])
    assert np.array_equal(exc.data["fch"].mo_coeff, ref.data["fch"].mo_coeff)
    assert exc.data["xch"].e_tot == ref.data["xch"].e_tot

    for name in ref.mbxas:
        assert np.array_equal(exc.mbxas[name], ref.mbxas[name])


def test_loaded_object_skips_a_finished_atom(tiny_calc, tmp_path):
    from pymbxas.calculators.pyscf import PySCFMBXAS

    path = tiny_calc.save(tmp_path / "skip.h5")
    back = PySCFMBXAS.load(path)

    back.excite(0)
    assert len(back.excitations) == 1


def test_same_site_with_different_config_is_retained(tiny_calc, tmp_path,
                                                     monkeypatch):
    from dataclasses import replace
    from pymbxas.config import CheckpointConfig, ExcitationConfig
    from pymbxas.calculators.pyscf import PySCFMBXAS
    from pymbxas.calculators.excitation import Excitation

    path = tiny_calc.save(tmp_path / "variants.h5")
    back = PySCFMBXAS.load(path)
    original = back.excitations[0]
    variant = ExcitationConfig(channel="alpha")
    back.runtime = replace(
        back.runtime, checkpoint=CheckpointConfig(enabled=False))
    monkeypatch.setattr(Excitation, "run", lambda self, *args: self)
    outcomes = back.excite(0, config=variant)
    assert outcomes[0].status == "succeeded"
    assert back.excitations[-1].config == variant
    assert original.config.channel.value == "beta"
    with pytest.raises(ValueError, match="Multiple excitation configurations"):
        back.get_mbxas_spectra(0)


def test_loaded_object_produces_a_spectra(tiny_calc, tmp_path):
    from pymbxas.calculators.pyscf import PySCFMBXAS
    from pymbxas.spectra import Spectra

    path = tiny_calc.save(tmp_path / "tospectra.h5")
    spectra = PySCFMBXAS.load(path).to_spectra()

    assert isinstance(spectra, Spectra)
    assert np.array_equal(spectra._energies, tiny_calc.excitations[0].mbxas["energies"])
    assert np.array_equal(spectra._gs_mo_energy, tiny_calc.gs_data.mo_energy)


def test_real_spectra_builds_orbital_rearrangement(tiny_calc):
    spectra = tiny_calc.to_spectra()
    data = spectra.get_orbital_rearrangement(
        energy_window=None, min_overlap=0.0)

    assert len(data["channels"]) == 2
    for channel in data["channels"]:
        matches = channel["all_matches"]
        assert len(np.unique(matches["gs_index"])) == len(matches["gs_index"])
        assert len(np.unique(matches["fch_index"])) == len(matches["fch_index"])
    assert data["channels"][spectra.channel]["core_fch"] == (
        spectra._fch_core_hole_index())


def test_load_skips_incomplete_excitation_groups(tiny_calc, tmp_path):
    from pymbxas.calculators.pyscf import PySCFMBXAS

    path = tiny_calc.save(tmp_path / "broken.h5")
    with h5py.File(path, "r+") as f:
        del f["excitations/000"].attrs["complete"]

    back = PySCFMBXAS.load(path)
    assert back.excitations == []


def test_force_ground_state_rerun_is_refused_after_load(tiny_calc, tmp_path):
    from pymbxas.calculators.pyscf import PySCFMBXAS

    path = tiny_calc.save(tmp_path / "refuse.h5")
    back = PySCFMBXAS.load(path)

    with pytest.raises(RuntimeError, match="excitations"):
        back.run_ground_state(force=True)


def test_flat_constructor_keywords_are_gone():
    import inspect
    from pymbxas.calculators.pyscf import PySCFMBXAS

    assert tuple(inspect.signature(PySCFMBXAS.__init__).parameters) == (
        "self", "structure", "config", "runtime")


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
