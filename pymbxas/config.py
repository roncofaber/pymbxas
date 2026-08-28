"""Validated, serializable configuration for PyMBXAS calculations."""

from dataclasses import dataclass, field
from enum import Enum
import os
from pathlib import Path
from typing import Mapping, Optional

from pymbxas.calculators.maxvol import (
    normalize_maxvol_warmup_calls,
    normalize_occupation_method,
)
from pymbxas.calculators.scf import normalize_scf_recovery_settings


class SpinChannel(str, Enum):
    """Unrestricted spin channel used for the core excitation."""

    ALPHA = "alpha"
    BETA = "beta"

    @property
    def index(self):
        return 0 if self is SpinChannel.ALPHA else 1

    @classmethod
    def normalize(cls, value):
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            raise ValueError("channel must be 'alpha', 'beta', 0, or 1")
        if value == 0:
            return cls.ALPHA
        if value == 1:
            return cls.BETA
        if isinstance(value, str):
            try:
                return cls(value.strip().lower())
            except ValueError:
                pass
        raise ValueError("channel must be 'alpha', 'beta', 0, or 1")


class OccupationMethod(str, Enum):
    """Available constrained-SCF occupation controllers."""

    MOM = "mom"
    MAXVOL = "maxvol"
    MIXED = "mixed"

    @classmethod
    def normalize(cls, value):
        if isinstance(value, cls):
            return value
        try:
            return cls(normalize_occupation_method(value))
        except ValueError as error:
            raise ValueError(
                "occupation must be 'mom', 'maxvol', or 'mixed'") from error


class Device(str, Enum):
    CPU = "cpu"
    GPU = "gpu"

    @classmethod
    def normalize(cls, value):
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value.strip().lower())
            except ValueError:
                pass
        raise ValueError("device must be 'cpu' or 'gpu'")


def _positive_number(name, value, *, allow_none=False):
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        qualifier = "a positive number or None" if allow_none else "a positive number"
        raise ValueError("{} must be {}".format(name, qualifier))
    if value <= 0:
        qualifier = "positive or None" if allow_none else "positive"
        raise ValueError("{} must be {}".format(name, qualifier))
    return float(value)


def _strict_mapping(cls, values):
    if not isinstance(values, Mapping):
        raise TypeError("{} configuration must be a mapping".format(cls.__name__))
    allowed = set(cls.__dataclass_fields__)
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(
            "Unknown {} field(s): {}".format(
                cls.__name__, ", ".join(sorted(str(key) for key in unknown))))
    return dict(values)


@dataclass(frozen=True)
class SCFConfig:
    """SCF convergence and recovery policy.

    Ground-state calculations use the common convergence fields. Constrained
    FCH/XCH calculations additionally use the DIIS/mixing recovery fields.
    """

    max_cycles: int = 100
    convergence_tolerance: Optional[float] = None
    grid_level: Optional[int] = None
    diis_cycles: int = 50
    mixing_cycles: int = 30
    damping: float = 0.2
    level_shift: float = 0.2
    second_order: bool = False

    def __post_init__(self):
        if isinstance(self.max_cycles, bool) or not isinstance(self.max_cycles, int):
            raise ValueError("max_cycles must be a positive integer")
        if self.max_cycles < 1:
            raise ValueError("max_cycles must be a positive integer")

        tolerance = _positive_number(
            "convergence_tolerance", self.convergence_tolerance,
            allow_none=True)
        if tolerance is not None:
            object.__setattr__(self, "convergence_tolerance", tolerance)

        if self.grid_level is not None:
            if (isinstance(self.grid_level, bool)
                    or not isinstance(self.grid_level, int)
                    or self.grid_level < 0):
                raise ValueError("grid_level must be a non-negative integer or None")

        recovery = normalize_scf_recovery_settings(
            self.diis_cycles, self.mixing_cycles, self.damping,
            self.level_shift, self.second_order)
        object.__setattr__(self, "diis_cycles", recovery["scf_diis_cycles"])
        object.__setattr__(self, "mixing_cycles", recovery["scf_mixing_cycles"])
        object.__setattr__(self, "damping", recovery["scf_damping"])
        object.__setattr__(self, "level_shift", recovery["scf_level_shift"])
        object.__setattr__(self, "second_order", recovery["scf_second_order"])

    def to_dict(self):
        return {
            "max_cycles": self.max_cycles,
            "convergence_tolerance": self.convergence_tolerance,
            "grid_level": self.grid_level,
            "diis_cycles": self.diis_cycles,
            "mixing_cycles": self.mixing_cycles,
            "damping": self.damping,
            "level_shift": self.level_shift,
            "second_order": self.second_order,
        }

    @classmethod
    def from_dict(cls, values):
        return cls(**_strict_mapping(cls, values))


@dataclass(frozen=True)
class CalculationConfig:
    """Immutable settings defining the ground-state calculation."""

    charge: int = 0
    spin: int = 0
    xc: Optional[str] = "b3lyp"
    basis: str = "def2-svpd"
    method: str = "UKS"
    solvent: Optional[float] = None
    pbc: Optional[bool] = None
    localization: str = "ibo"
    ground_state_scf: SCFConfig = field(default_factory=SCFConfig)

    def __post_init__(self):
        for name in ("charge", "spin"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("{} must be an integer".format(name))
        if not isinstance(self.basis, str) or not self.basis.strip():
            raise ValueError("basis must be a non-empty string")
        object.__setattr__(self, "basis", self.basis.strip())
        if self.xc is not None:
            if not isinstance(self.xc, str) or not self.xc.strip():
                raise ValueError("xc must be a non-empty string or None")
            object.__setattr__(self, "xc", self.xc.strip())
        method = self.method.strip().upper() if isinstance(self.method, str) else None
        if method not in {"UKS", "UHF"}:
            raise ValueError("method must be 'UKS' or 'UHF'")
        object.__setattr__(self, "method", method)
        if self.solvent is not None:
            object.__setattr__(
                self, "solvent", _positive_number("solvent", self.solvent))
        if self.pbc is not None and not isinstance(self.pbc, bool):
            raise ValueError("pbc must be a boolean or None")
        localization = (
            self.localization.strip().lower()
            if isinstance(self.localization, str) else None)
        if localization not in {"ibo", "ibom", "boys", "boysm"}:
            raise ValueError(
                "localization must be 'ibo', 'ibom', 'boys', or 'boysm'")
        object.__setattr__(self, "localization", localization)
        if not isinstance(self.ground_state_scf, SCFConfig):
            raise TypeError("ground_state_scf must be an SCFConfig")

    def to_dict(self):
        return {
            "charge": self.charge,
            "spin": self.spin,
            "xc": self.xc,
            "basis": self.basis,
            "method": self.method,
            "solvent": self.solvent,
            "pbc": self.pbc,
            "localization": self.localization,
            "ground_state_scf": self.ground_state_scf.to_dict(),
        }

    @classmethod
    def from_dict(cls, values):
        values = _strict_mapping(cls, values)
        if "ground_state_scf" in values:
            values["ground_state_scf"] = SCFConfig.from_dict(
                values["ground_state_scf"])
        return cls(**values)


@dataclass(frozen=True)
class LoggingConfig:
    """Console and logfile settings for one execution session."""

    pymbxas_verbosity: int = 3
    pymbxas_logfile: Optional[str] = None
    pyscf_verbosity: int = 3
    pyscf_logfile: Optional[str] = None
    pyscf_console: bool = True

    def __post_init__(self):
        for name, lower, upper in (
                ("pymbxas_verbosity", 1, 5), ("pyscf_verbosity", 0, 9)):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, int)
                    or not lower <= value <= upper):
                raise ValueError(
                    "{} must be an integer from {} to {}".format(
                        name, lower, upper))
        if not isinstance(self.pyscf_console, bool):
            raise ValueError("pyscf_console must be a boolean")
        for name in ("pymbxas_logfile", "pyscf_logfile"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, os.fspath(value))

    def to_dict(self):
        return {
            "pymbxas_verbosity": self.pymbxas_verbosity,
            "pymbxas_logfile": self.pymbxas_logfile,
            "pyscf_verbosity": self.pyscf_verbosity,
            "pyscf_logfile": self.pyscf_logfile,
            "pyscf_console": self.pyscf_console,
        }

    @classmethod
    def from_dict(cls, values):
        return cls(**_strict_mapping(cls, values))


@dataclass(frozen=True)
class CheckpointConfig:
    """Checkpoint and optional PySCF artifact policy."""

    enabled: bool = True
    filename: str = "pymbxas.h5"
    pyscf_chkfiles: bool = False
    fchk_files: bool = False

    def __post_init__(self):
        for name in ("enabled", "pyscf_chkfiles", "fchk_files"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError("{} must be a boolean".format(name))
        filename = os.fspath(self.filename)
        if not filename or Path(filename).name != filename:
            raise ValueError("filename must be a file name without a directory")
        if not filename.endswith(".h5"):
            filename = os.path.splitext(filename)[0] + ".h5"
        object.__setattr__(self, "filename", filename)

    def to_dict(self):
        return {
            "enabled": self.enabled,
            "filename": self.filename,
            "pyscf_chkfiles": self.pyscf_chkfiles,
            "fchk_files": self.fchk_files,
        }

    @classmethod
    def from_dict(cls, values):
        return cls(**_strict_mapping(cls, values))


@dataclass(frozen=True)
class RuntimeConfig:
    """Mutable-session concerns kept outside scientific configuration."""

    work_directory: str = "."
    device: Device = Device.CPU
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    def __post_init__(self):
        object.__setattr__(
            self, "work_directory", os.path.abspath(os.fspath(self.work_directory)))
        object.__setattr__(self, "device", Device.normalize(self.device))
        if not isinstance(self.logging, LoggingConfig):
            raise TypeError("logging must be a LoggingConfig")
        if not isinstance(self.checkpoint, CheckpointConfig):
            raise TypeError("checkpoint must be a CheckpointConfig")

    @property
    def is_gpu(self):
        return self.device is Device.GPU

    @property
    def checkpoint_path(self):
        return os.path.join(
            self.work_directory, self.checkpoint.filename)

    def to_dict(self):
        return {
            "work_directory": self.work_directory,
            "device": self.device.value,
            "logging": self.logging.to_dict(),
            "checkpoint": self.checkpoint.to_dict(),
        }

    @classmethod
    def from_dict(cls, values):
        values = _strict_mapping(cls, values)
        if "logging" in values:
            values["logging"] = LoggingConfig.from_dict(values["logging"])
        if "checkpoint" in values:
            values["checkpoint"] = CheckpointConfig.from_dict(
                values["checkpoint"])
        return cls(**values)


@dataclass(frozen=True)
class ExcitationConfig:
    """Settings shared by one or more core-excitation requests.

    ``scf`` supplies the normal FCH and XCH policy. ``fch_scf`` and
    ``xch_scf`` are optional advanced overrides for one state only.
    """

    channel: SpinChannel = SpinChannel.BETA
    xch: bool = True
    occupation: OccupationMethod = OccupationMethod.MAXVOL
    mom_warmup_calls: int = 2
    scf: SCFConfig = field(default_factory=SCFConfig)
    fch_scf: Optional[SCFConfig] = None
    xch_scf: Optional[SCFConfig] = None

    def __post_init__(self):
        object.__setattr__(self, "channel", SpinChannel.normalize(self.channel))
        if not isinstance(self.xch, bool):
            raise ValueError("xch must be a boolean")
        object.__setattr__(
            self, "occupation", OccupationMethod.normalize(self.occupation))
        try:
            warmup_calls = normalize_maxvol_warmup_calls(
                self.mom_warmup_calls)
        except ValueError as error:
            raise ValueError(
                "mom_warmup_calls must be a positive integer") from error
        object.__setattr__(self, "mom_warmup_calls", warmup_calls)
        for name in ("scf", "fch_scf", "xch_scf"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, SCFConfig):
                raise TypeError("{} must be an SCFConfig or None".format(name))

    @property
    def channel_index(self):
        return self.channel.index

    @property
    def resolved_fch_scf(self):
        return self.fch_scf if self.fch_scf is not None else self.scf

    @property
    def resolved_xch_scf(self):
        return self.xch_scf if self.xch_scf is not None else self.scf

    def to_dict(self):
        return {
            "channel": self.channel.value,
            "xch": self.xch,
            "occupation": self.occupation.value,
            "mom_warmup_calls": self.mom_warmup_calls,
            "scf": self.scf.to_dict(),
            "fch_scf": None if self.fch_scf is None else self.fch_scf.to_dict(),
            "xch_scf": None if self.xch_scf is None else self.xch_scf.to_dict(),
        }

    @classmethod
    def from_dict(cls, values):
        values = _strict_mapping(cls, values)
        for name in ("scf", "fch_scf", "xch_scf"):
            if name in values and values[name] is not None:
                values[name] = SCFConfig.from_dict(values[name])
        return cls(**values)


UNSET = object()


def resolve_excitation_config(
        config=None, *, channel=UNSET, xch=UNSET, occupation=UNSET,
        mom_warmup_calls=UNSET, scf=UNSET, fch_scf=UNSET,
        xch_scf=UNSET):
    """Resolve convenience keywords to one canonical excitation config.

    A complete ``config`` and individual overrides are deliberately mutually
    exclusive, so call-site precedence can never be ambiguous.
    """
    overrides = {
        "channel": channel,
        "xch": xch,
        "occupation": occupation,
        "mom_warmup_calls": mom_warmup_calls,
        "scf": scf,
        "fch_scf": fch_scf,
        "xch_scf": xch_scf,
    }
    supplied = {name: value for name, value in overrides.items()
                if value is not UNSET}
    if config is not None:
        if supplied:
            raise ValueError(
                "config cannot be combined with individual excitation settings: {}"
                .format(", ".join(sorted(supplied))))
        if not isinstance(config, ExcitationConfig):
            raise TypeError("config must be an ExcitationConfig")
        return config
    return ExcitationConfig(**supplied)


__all__ = [
    "CalculationConfig", "CheckpointConfig", "Device", "ExcitationConfig",
    "LoggingConfig", "OccupationMethod", "RuntimeConfig", "SCFConfig",
    "SpinChannel", "resolve_excitation_config",
]
