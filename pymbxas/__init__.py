"""Molecular many-body X-ray absorption spectroscopy with PySCF.

PyMBXAS implements an unrestricted GS -> FCH -> optional XCH Delta-SCF
workflow and the one-body core-hole-basis determinant amplitude. Optional
shake-up and spectator-spin arguments add explicit order-resolved determinant
amplitudes and spectator overlaps; see ``dev/shakeup.md`` in the source
distribution for their scientific scope.

The primary calculation entry point is
``pymbxas.calculators.pyscf.PySCFMBXAS``. ``Spectra`` and ``Spectras`` are
re-exported here for post-processing and HDF5 persistence.
"""

import sys

if sys.version_info[0] == 2:
    raise ImportError("Please run with Python 3. This is Python 2.")

__version__ = "0.7.1"
__date__ = "25 Aug. 2026"
__author__ = "Fabrice Roncoroni"
__all__ = [
    "CalculationConfig", "CheckpointConfig", "Device", "ExcitationConfig",
    "LoggingConfig", "OccupationMethod", "RuntimeConfig", "SCFConfig",
    "PySCFMBXAS", "Spectra", "Spectras", "SpinChannel",
]

from pymbxas.config import (
    CalculationConfig, CheckpointConfig, Device, ExcitationConfig,
    LoggingConfig, OccupationMethod, RuntimeConfig, SCFConfig, SpinChannel,
)
from pymbxas.spectra import Spectra
from pymbxas.spectras import Spectras
from pymbxas.calculators.pyscf import PySCFMBXAS
