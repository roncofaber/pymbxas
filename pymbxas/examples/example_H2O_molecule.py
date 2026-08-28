"""Minimal molecular oxygen K-edge calculation."""

import ase.build
import matplotlib.pyplot as plt

from pymbxas import (
    CalculationConfig, CheckpointConfig, ExcitationConfig, LoggingConfig,
    PySCFMBXAS, RuntimeConfig,
)


structure = ase.build.molecule("H2O")
calculation = PySCFMBXAS(
    structure,
    config=CalculationConfig(xc="lda", basis="def2-svpd"),
    runtime=RuntimeConfig(
        logging=LoggingConfig(
            pymbxas_logfile="pymbxas.log",
            pyscf_verbosity=4,
            pyscf_logfile="pyscf.log",
            pyscf_console=False,
        ),
        checkpoint=CheckpointConfig(filename="pymbxas.h5"),
    ),
)

calculation.run("O", config=ExcitationConfig(channel="beta", xch=True))
spectrum = calculation.to_spectra(index=0)
energy, intensity = spectrum.get_mbxas_spectra(
    erange=[520, 640], sigma=0.5)

plt.plot(energy, intensity)
plt.xlabel("Photon energy (eV)")
plt.ylabel("Intensity (arb. units)")
plt.show()
