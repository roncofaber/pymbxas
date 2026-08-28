import json

import pytest

from pymbxas.config import (
    CalculationConfig, CheckpointConfig, Device, ExcitationConfig,
    LoggingConfig, OccupationMethod, RuntimeConfig, SCFConfig, SpinChannel,
    resolve_excitation_config,
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


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_cycles": 0},
        {"max_cycles": True},
        {"convergence_tolerance": 0},
        {"grid_level": -1},
        {"grid_level": True},
        {"diis_cycles": 0},
        {"mixing_cycles": -1},
        {"damping": -0.1},
        {"second_order": "yes"},
    ],
)
def test_scf_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        SCFConfig(**kwargs)


@pytest.mark.parametrize(
    "value, expected, index",
    [
        ("alpha", SpinChannel.ALPHA, 0),
        ("BETA", SpinChannel.BETA, 1),
        (0, SpinChannel.ALPHA, 0),
        (1, SpinChannel.BETA, 1),
    ],
)
def test_excitation_config_normalizes_readable_channel_values(
        value, expected, index):
    config = ExcitationConfig(channel=value)
    assert config.channel is expected
    assert config.channel_index == index


def test_excitation_config_resolves_shared_and_state_specific_scf():
    shared = SCFConfig(max_cycles=100)
    fch = SCFConfig(max_cycles=200)
    config = ExcitationConfig(scf=shared, fch_scf=fch)

    assert config.resolved_fch_scf is fch
    assert config.resolved_xch_scf is shared


def test_excitation_config_defaults_to_direct_maxvol():
    assert ExcitationConfig().occupation is OccupationMethod.MAXVOL


def test_config_round_trip_is_json_serializable():
    original = ExcitationConfig(
        channel="alpha", occupation="mixed", mom_warmup_calls=3,
        scf=SCFConfig(max_cycles=120, convergence_tolerance=1e-9),
        xch_scf=SCFConfig(max_cycles=80),
    )

    payload = json.loads(json.dumps(original.to_dict()))
    restored = ExcitationConfig.from_dict(payload)

    assert restored == original
    assert restored.occupation is OccupationMethod.MIXED


def test_config_round_trip_through_hdf5_json(tmp_path):
    original = ExcitationConfig(
        occupation="maxvol", scf=SCFConfig(max_cycles=75))
    path = tmp_path / "config.h5"

    with h5.create(path, h5.KIND_CALCULATION) as handle:
        h5.write_json(handle, "excitation_config", original.to_dict())
    with h5.open_read(path, h5.KIND_CALCULATION) as handle:
        restored = ExcitationConfig.from_dict(
            h5.read_json(handle, "excitation_config"))

    assert restored == original


def test_from_dict_rejects_unknown_fields():
    with pytest.raises(ValueError, match="Unknown SCFConfig field"):
        SCFConfig.from_dict({"max_cycle": 100})
    with pytest.raises(ValueError, match="Unknown ExcitationConfig field"):
        ExcitationConfig.from_dict({"occupation_method": "mom"})


def test_direct_excitation_settings_resolve_to_canonical_config():
    scf = SCFConfig(max_cycles=200)
    config = resolve_excitation_config(
        channel="beta", occupation="mixed", mom_warmup_calls=4, scf=scf)

    assert config == ExcitationConfig(
        channel=SpinChannel.BETA,
        occupation=OccupationMethod.MIXED,
        mom_warmup_calls=4,
        scf=scf,
    )


def test_complete_config_cannot_be_mixed_with_direct_settings():
    config = ExcitationConfig()
    assert resolve_excitation_config(config) is config
    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_excitation_config(config, occupation="maxvol")


def test_calculation_and_runtime_configs_round_trip(tmp_path):
    calculation = CalculationConfig(
        charge=1, spin=1, xc="pbe", basis="def2-svpd", method="uks",
        localization="IBO", ground_state_scf=SCFConfig(max_cycles=80))
    runtime = RuntimeConfig(
        work_directory=tmp_path, device="GPU",
        logging=LoggingConfig(
            pymbxas_verbosity=4, pymbxas_logfile="app.log",
            pyscf_verbosity=5, pyscf_logfile="raw.log",
            pyscf_console=False),
        checkpoint=CheckpointConfig(
            filename="calculation.data", pyscf_chkfiles=True))

    assert CalculationConfig.from_dict(calculation.to_dict()) == calculation
    assert RuntimeConfig.from_dict(runtime.to_dict()) == runtime
    assert runtime.device is Device.GPU
    assert runtime.checkpoint.filename == "calculation.h5"


@pytest.mark.parametrize(
    "factory, kwargs",
    [
        (CalculationConfig, {"method": "RKS"}),
        (CalculationConfig, {"localization": "pipek"}),
        (LoggingConfig, {"pyscf_verbosity": True}),
        (CheckpointConfig, {"filename": "nested/calc.h5"}),
        (RuntimeConfig, {"device": "cuda"}),
    ],
)
def test_high_level_configs_reject_invalid_values(factory, kwargs):
    with pytest.raises((TypeError, ValueError)):
        factory(**kwargs)


@pytest.mark.parametrize("value", [True, False, 2, "up", None])
def test_invalid_spin_channels_are_rejected(value):
    with pytest.raises(ValueError, match="channel"):
        ExcitationConfig(channel=value)
