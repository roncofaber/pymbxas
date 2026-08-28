import json

import ase
import pytest

from pymbxas.calculators.pyscf import PySCFMBXAS
from pymbxas.config import (
    CalculationConfig, CheckpointConfig, Device, ExcitationConfig,
    LoggingConfig, OccupationMethod, RuntimeConfig, SCFConfig, SpinChannel,
)
from pymbxas.io import h5


def test_scf_config_validates_and_normalizes_values():
    config = SCFConfig(
        max_cycles=150, convergence_tolerance=1e-8, grid_level=4,
        diis_cycles=40, mixing_cycles=20, damping=1, level_shift=0,
        second_order=True)
    assert config.max_cycles == 150
    assert config.convergence_tolerance == pytest.approx(1e-8)
    assert config.damping == pytest.approx(1.0)
    assert config.level_shift == pytest.approx(0.0)


@pytest.mark.parametrize("kwargs", [
    {"max_cycles": 0}, {"max_cycles": True},
    {"convergence_tolerance": 0}, {"grid_level": -1},
    {"grid_level": True}, {"diis_cycles": 0},
    {"mixing_cycles": -1}, {"damping": -0.1},
    {"second_order": "yes"},
])
def test_scf_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        SCFConfig(**kwargs)


@pytest.mark.parametrize("value, expected, index", [
    ("alpha", SpinChannel.ALPHA, 0), ("BETA", SpinChannel.BETA, 1),
    (0, SpinChannel.ALPHA, 0), (1, SpinChannel.BETA, 1),
])
def test_excitation_config_normalizes_readable_channel_values(
        value, expected, index):
    config = ExcitationConfig(channel=value)
    assert config.channel is expected
    assert config.channel_index == index


def test_excitation_config_has_no_solver_settings_and_defaults_to_maxvol():
    config = ExcitationConfig()
    assert config.occupation is OccupationMethod.MAXVOL
    assert set(config.to_dict()) == {
        "channel", "xch", "occupation", "mom_warmup_calls"}


def test_configs_round_trip_as_json_and_hdf5(tmp_path):
    original = ExcitationConfig(
        channel="alpha", occupation="mixed", mom_warmup_calls=3)
    payload = json.loads(json.dumps(original.to_dict()))
    assert ExcitationConfig.from_dict(payload) == original

    path = tmp_path / "config.h5"
    with h5.create(path, h5.KIND_CALCULATION) as handle:
        h5.write_json(handle, "excitation_config", original.to_dict())
    with h5.open_read(path, h5.KIND_CALCULATION) as handle:
        restored = ExcitationConfig.from_dict(
            h5.read_json(handle, "excitation_config"))
    assert restored == original


def test_set_is_transactional_and_copy_is_independent():
    theory = CalculationConfig(xc="pbe")
    result = theory.set(xc="lda")
    assert result is theory
    assert theory.xc == "lda"

    variant = theory.copy(xc="b3lyp")
    assert variant.xc == "b3lyp"
    assert theory.xc == "lda"

    with pytest.raises(ValueError):
        theory.set(method="RKS", xc="pbe")
    assert theory.method == "UKS"
    assert theory.xc == "lda"
    with pytest.raises(TypeError):
        hash(theory)


def test_direct_assignment_is_rejected_and_snapshot_cannot_be_set():
    theory = CalculationConfig(xc="pbe")
    with pytest.raises(Exception):
        theory.xc = "lda"

    frozen = theory.snapshot()
    with pytest.raises(TypeError, match="calculator-owned snapshot"):
        frozen.set(xc="lda")
    assert frozen.copy().set(xc="lda").xc == "lda"


def test_calculator_snapshots_theory_and_ground_state_solver():
    theory = CalculationConfig(xc="pbe", basis="sto-3g")
    gs_scf = SCFConfig(max_cycles=80)
    calc = PySCFMBXAS(
        ase.Atoms("H"), calculation=theory, gs_scf=gs_scf,
        checkpoint=None)

    theory.set(xc="lda")
    gs_scf.set(max_cycles=20)

    assert calc.calculation.xc == "pbe"
    assert calc.gs_scf.max_cycles == 80
    assert calc.calculation.is_snapshot
    assert calc.gs_scf.is_snapshot
    with pytest.raises(AttributeError):
        calc.calculation = CalculationConfig(xc="lda")


def test_default_calculator_has_complete_stage_configs():
    calc = PySCFMBXAS(ase.Atoms("H"), checkpoint=None)
    assert isinstance(calc.calculation, CalculationConfig)
    assert isinstance(calc.gs_scf, SCFConfig)

    excitation = ExcitationConfig()
    fch_scf = SCFConfig()
    xch_scf = SCFConfig()
    assert excitation.xch is True
    assert fch_scf is not xch_scf


def test_from_dict_and_set_reject_unknown_fields():
    with pytest.raises(ValueError, match="Unknown SCFConfig field"):
        SCFConfig.from_dict({"max_cycle": 100})
    with pytest.raises(ValueError, match="Unknown ExcitationConfig field"):
        ExcitationConfig().set(occupation_method="mom")


def test_flat_configs_round_trip(tmp_path):
    calculation = CalculationConfig(
        charge=1, spin=1, xc="pbe", basis="def2-svpd", method="uks",
        localization="IBO")
    runtime = RuntimeConfig(work_directory=tmp_path, device="GPU")
    logging = LoggingConfig(
        pymbxas_verbosity=4, pymbxas_logfile="app.log",
        pyscf_verbosity=5, pyscf_logfile="raw.log", pyscf_console=False)
    checkpoint = CheckpointConfig(
        path=tmp_path / "calculation.hdf5", pyscf_chkfiles=True)

    assert CalculationConfig.from_dict(calculation.to_dict()) == calculation
    assert RuntimeConfig.from_dict(runtime.to_dict()) == runtime
    assert LoggingConfig.from_dict(logging.to_dict()) == logging
    assert CheckpointConfig.from_dict(checkpoint.to_dict()) == checkpoint
    assert runtime.device is Device.GPU


@pytest.mark.parametrize("factory, kwargs", [
    (CalculationConfig, {"method": "RKS"}),
    (CalculationConfig, {"localization": "pipek"}),
    (LoggingConfig, {"pyscf_verbosity": True}),
    (CheckpointConfig, {"path": "calculation.data"}),
    (RuntimeConfig, {"device": "cuda"}),
])
def test_high_level_configs_reject_invalid_values(factory, kwargs):
    with pytest.raises((TypeError, ValueError)):
        factory(**kwargs)


@pytest.mark.parametrize("value", [True, False, 2, "up", None])
def test_invalid_spin_channels_are_rejected(value):
    with pytest.raises(ValueError, match="channel"):
        ExcitationConfig(channel=value)
