"""Validated, serializable configuration for PyMBXAS calculations."""

from dataclasses import dataclass, fields
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


class _ConfigTemplate:
    """Controlled mutable template with immutable calculator snapshots."""

    __hash__ = None

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        return all(
            getattr(self, item.name) == getattr(other, item.name)
            for item in fields(self))

    def set(self, **changes):
        """Validate and apply changes transactionally, then return ``self``."""
        if getattr(self, "_is_snapshot", False):
            raise TypeError(
                f"{type(self).__name__} is a calculator-owned snapshot; "
                "modify a copy and construct a new calculation instead")
        values = self.to_dict()
        unknown = set(changes) - {item.name for item in fields(self)}
        if unknown:
            raise ValueError(
                "Unknown {} field(s): {}".format(
                    type(self).__name__, ", ".join(sorted(unknown))))
        values.update(changes)
        candidate = type(self).from_dict(values)
        for item in fields(self):
            object.__setattr__(self, item.name, getattr(candidate, item.name))
        return self

    def copy(self, **changes):
        """Return an independent, editable configuration template."""
        result = type(self).from_dict(self.to_dict())
        if changes:
            result.set(**changes)
        return result

    def snapshot(self):
        """Return an immutable copy suitable for calculator ownership."""
        result = self.copy()
        object.__setattr__(result, "_is_snapshot", True)
        return result

    @property
    def is_snapshot(self):
        return bool(getattr(self, "_is_snapshot", False))


def _positive_number(name, value, *, allow_none=False):
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        qualifier = "a positive number or None" if allow_none else "a positive number"
        raise ValueError(f"{name} must be {qualifier}")
    if value <= 0:
        qualifier = "positive or None" if allow_none else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return float(value)


def _strict_mapping(cls, values):
    if not isinstance(values, Mapping):
        raise TypeError(f"{cls.__name__} configuration must be a mapping")
    allowed = {item.name for item in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(
            "Unknown {} field(s): {}".format(
                cls.__name__, ", ".join(sorted(str(key) for key in unknown))))
    return dict(values)


@dataclass(frozen=True, eq=False)
class SCFConfig(_ConfigTemplate):
    """SCF convergence and constrained-state recovery policy."""

    max_cycles: int = 100
    convergence_tolerance: Optional[float] = None
    grid_level: Optional[int] = None
    diis_cycles: int = 50
    mixing_cycles: int = 30
    damping: float = 0.2
    level_shift: float = 0.2
    second_order: bool = False

    def __post_init__(self):
        if (isinstance(self.max_cycles, bool)
                or not isinstance(self.max_cycles, int)
                or self.max_cycles < 1):
            raise ValueError("max_cycles must be a positive integer")
        tolerance = _positive_number(
            "convergence_tolerance", self.convergence_tolerance,
            allow_none=True)
        object.__setattr__(self, "convergence_tolerance", tolerance)
        if self.grid_level is not None and (
                isinstance(self.grid_level, bool)
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
        return {item.name: getattr(self, item.name) for item in fields(self)}

    @classmethod
    def from_dict(cls, values):
        return cls(**_strict_mapping(cls, values))


@dataclass(frozen=True, eq=False)
class CalculationConfig(_ConfigTemplate):
    """Electronic-structure model shared by GS, FCH, and XCH."""

    charge: int = 0
    spin: int = 0
    xc: Optional[str] = "b3lyp"
    basis: str = "def2-svpd"
    method: str = "UKS"
    solvent: Optional[float] = None
    pbc: Optional[bool] = None
    localization: str = "ibo"

    def __post_init__(self):
        for name in ("charge", "spin"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
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

    def to_dict(self):
        return {item.name: getattr(self, item.name) for item in fields(self)}

    @classmethod
    def from_dict(cls, values):
        return cls(**_strict_mapping(cls, values))


@dataclass(frozen=True, eq=False)
class ExcitationConfig(_ConfigTemplate):
    """Physical definition of a core-excitation request."""

    channel: SpinChannel = SpinChannel.BETA
    xch: bool = True
    occupation: OccupationMethod = OccupationMethod.MAXVOL
    mom_warmup_calls: int = 2

    def __post_init__(self):
        object.__setattr__(self, "channel", SpinChannel.normalize(self.channel))
        if not isinstance(self.xch, bool):
            raise ValueError("xch must be a boolean")
        object.__setattr__(
            self, "occupation", OccupationMethod.normalize(self.occupation))
        try:
            warmup_calls = normalize_maxvol_warmup_calls(self.mom_warmup_calls)
        except ValueError as error:
            raise ValueError(
                "mom_warmup_calls must be a positive integer") from error
        object.__setattr__(self, "mom_warmup_calls", warmup_calls)

    @property
    def channel_index(self):
        return self.channel.index

    def to_dict(self):
        return {
            "channel": self.channel.value,
            "xch": self.xch,
            "occupation": self.occupation.value,
            "mom_warmup_calls": self.mom_warmup_calls,
        }

    @classmethod
    def from_dict(cls, values):
        return cls(**_strict_mapping(cls, values))


@dataclass(frozen=True, eq=False)
class RuntimeConfig(_ConfigTemplate):
    """Execution device and working-directory settings for one call."""

    work_directory: str = "."
    device: Device = Device.CPU

    def __post_init__(self):
        object.__setattr__(
            self, "work_directory",
            os.path.abspath(os.fspath(self.work_directory)))
        object.__setattr__(self, "device", Device.normalize(self.device))

    @property
    def is_gpu(self):
        return self.device is Device.GPU

    def to_dict(self):
        return {
            "work_directory": self.work_directory,
            "device": self.device.value,
        }

    @classmethod
    def from_dict(cls, values):
        return cls(**_strict_mapping(cls, values))


@dataclass(frozen=True, eq=False)
class LoggingConfig(_ConfigTemplate):
    """Console and logfile settings for one execution call."""

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
                    f"{name} must be an integer from {lower} to {upper}")
        if not isinstance(self.pyscf_console, bool):
            raise ValueError("pyscf_console must be a boolean")
        for name in ("pymbxas_logfile", "pyscf_logfile"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, os.fspath(value))

    def to_dict(self):
        return {item.name: getattr(self, item.name) for item in fields(self)}

    @classmethod
    def from_dict(cls, values):
        return cls(**_strict_mapping(cls, values))


@dataclass(frozen=True, eq=False)
class CheckpointConfig(_ConfigTemplate):
    """Checkpoint path and optional PySCF artifact policy."""

    path: str = "pymbxas.h5"
    pyscf_chkfiles: bool = False
    fchk_files: bool = False

    def __post_init__(self):
        for name in ("pyscf_chkfiles", "fchk_files"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        path = os.fspath(self.path)
        if not path:
            raise ValueError("path must be a non-empty HDF5 file path")
        if Path(path).suffix.lower() not in {".h5", ".hdf5"}:
            raise ValueError("checkpoint path must end in .h5 or .hdf5")
        object.__setattr__(self, "path", path)

    def to_dict(self):
        return {
            "path": self.path,
            "pyscf_chkfiles": self.pyscf_chkfiles,
            "fchk_files": self.fchk_files,
        }

    @classmethod
    def from_dict(cls, values):
        return cls(**_strict_mapping(cls, values))


def snapshot_config(value, expected_type, name):
    """Validate and copy a public template into calculator-owned state."""
    if value is None:
        value = expected_type()
    if not isinstance(value, expected_type):
        raise TypeError(f"{name} must be a {expected_type.__name__}")
    return value.snapshot()


__all__ = [
    "CalculationConfig", "CheckpointConfig", "Device", "ExcitationConfig",
    "LoggingConfig", "OccupationMethod", "RuntimeConfig", "SCFConfig",
    "SpinChannel", "snapshot_config",
]
