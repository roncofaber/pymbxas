"""Minimal molecular oxygen K-edge calculation."""

import ase.build
import matplotlib.pyplot as plt

from pymbxas import (
    CalculationConfig, ExcitationConfig, LoggingConfig, PySCFMBXAS,
    RuntimeConfig,
)


structure = ase.build.molecule("H2O")
calculation = PySCFMBXAS(
    structure,
    calculation=CalculationConfig(xc="lda", basis="def2-svpd"),
    checkpoint="pymbxas.h5",
)

calculation.run(
    "O",
    excitation=ExcitationConfig(channel="beta", xch=True),
    runtime=RuntimeConfig(),
    logging=LoggingConfig(
        pymbxas_logfile="pymbxas.log",
        pyscf_verbosity=4,
        pyscf_logfile="pyscf.log",
        pyscf_console=False,
    ),
)
spectrum = calculation.to_spectra(index=0)
energy, intensity = spectrum.get_mbxas_spectra(
    erange=[520, 640], sigma=0.5)

plt.plot(energy, intensity)
plt.xlabel("Photon energy (eV)")
plt.ylabel("Intensity (arb. units)")
plt.show()
