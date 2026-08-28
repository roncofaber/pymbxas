#!/usr/bin/env python3
"""Spin-complete order-resolved spectrum for the 58-atom 6fda structure.

Production uses PBE/def2-SVPD and GPU4PySCF by default; ``--xc b3lyp`` selects
B3LYP with the same basis and workflow. ``--quick`` switches to LDA/STO-3G for
inexpensive validation. Each theory/device pair has its own HDF5 checkpoint.
``--f-order`` selects the cumulative many-body order; the spectator channel
follows it automatically.
"""

import argparse
import logging
from pathlib import Path
import time

import ase.io
import matplotlib
matplotlib.use("Agg")
import numpy as np

from pymbxas import (
    CalculationConfig, CheckpointConfig, ExcitationConfig, LoggingConfig,
    PySCFMBXAS, RuntimeConfig, SCFConfig,
)
from pymbxas.io.config import format_log_fields
from pymbxas.plotting import plot_mbxas_decomposition


SIGMA = 0.5
TOL = 0.01
NPOINTS = 2001
PRE_EDGE = 4.0
POST_EDGE = 25.0
SPECTRUM_XLIM = (525.0, 555.0)
MAX_CONFIGURATIONS = 2_000_000
F_ORDER = 2
STRUCTURE_NAME = "6fda-dam_relaxed_dft.xyz"
GEOMETRY_TAG = "dftgeom"
DEFAULT_XC = "pbe"
DFT_VERBOSE = 4
SCF_DIIS_CYCLES = 50
SCF_MIXING_CYCLES = 150
SCF_DAMPING = 0.2
SCF_LEVEL_SHIFT = 0.2
SCF_SECOND_ORDER = False
ORBITAL_ENERGY_WINDOW = (-10.0, 10.0)
ORBITAL_MIN_OVERLAP = 0.05
ORBITAL_DOS_SIGMA = 0.25


def prepare_output_directories(here):
    """Create and return the output layout grouped by artifact type."""
    root = here / "outputs"
    paths = {
        "root": root,
        "checkpoints": root / "checkpoints",
        "logs": root / "logs",
        "data": root / "data",
        "spectrum_figures": root / "figures" / "spectra",
        "orbital_figures": root / "figures" / "orbitals",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def theory_settings(quick, xc):
    """Return the functional, basis, and filesystem-safe theory tag."""
    if quick:
        return "lda", "sto-3g", "lda_sto3g_quick"
    normalized = xc.strip().lower()
    if normalized not in {"pbe", "b3lyp"}:
        raise ValueError("xc must be 'pbe' or 'b3lyp'")
    return normalized, "def2-svpd", f"{normalized}_def2_svpd"


def output_table(energy, contributions, resolved, total):
    """Return the CSV table and stable column names for any f order."""
    arrays = [energy]
    columns = ["energy_eV"]
    for order in sorted(contributions):
        if order in resolved:
            arrays.extend((
                resolved[order]["shakeup"],
                resolved[order]["shakedown"],
            ))
            columns.extend((f"f{order}_shakeup", f"f{order}_shakedown"))
        arrays.append(contributions[order])
        columns.append(f"f{order}")
    order_tag = "_".join(f"f{order}" for order in sorted(contributions))
    arrays.append(total)
    columns.append(f"total_{order_tag}")
    return np.column_stack(arrays), columns


def validate_checkpoint_geometry(calculation, reference):
    """Reject a checkpoint from a different 6fda geometry."""
    if (calculation.structure.get_chemical_symbols()
            != reference.get_chemical_symbols()):
        raise RuntimeError("Checkpoint atom ordering differs from the input geometry")
    actual = calculation.structure.get_all_distances(mic=False)
    expected = reference.get_all_distances(mic=False)
    maximum = float(np.max(np.abs(actual - expected)))
    if maximum > 1e-6:
        raise RuntimeError(
            f"Checkpoint geometry differs from {STRUCTURE_NAME}: maximum "
            f"pair-distance difference is {maximum:.6f} Angstrom")


def load_or_calculate(here, output_paths, recalculate, use_gpu, xc, basis,
                      tag, quick, occupation_method, mom_warmup_calls):
    device = "gpu" if use_gpu else "cpu"
    method_tag = (f"mixed_w{mom_warmup_calls}"
                  if occupation_method == "mixed" else occupation_method)
    checkpoint_name = f"6fda_{GEOMETRY_TAG}_{tag}_{device}_{method_tag}.h5"
    checkpoint = output_paths["checkpoints"] / checkpoint_name
    scf = SCFConfig(
        max_cycles=100 if quick else 200,
        convergence_tolerance=1e-5 if quick else 1e-6,
        grid_level=1 if quick else 3,
        diis_cycles=SCF_DIIS_CYCLES,
        mixing_cycles=SCF_MIXING_CYCLES,
        damping=SCF_DAMPING,
        level_shift=SCF_LEVEL_SHIFT,
        second_order=SCF_SECOND_ORDER,
    )
    excitation = ExcitationConfig(
        occupation=occupation_method,
        mom_warmup_calls=mom_warmup_calls,
        scf=scf,
    )
    runtime = RuntimeConfig(
        work_directory=output_paths["checkpoints"],
        device=device,
        logging=LoggingConfig(
            pymbxas_verbosity=3,
            pymbxas_logfile=str(
                output_paths["logs"]
                / f"pymbxas_{GEOMETRY_TAG}_{tag}_{device}_{occupation_method}.log"),
            pyscf_verbosity=DFT_VERBOSE,
            pyscf_logfile=str(
                output_paths["logs"]
                / f"pyscf_{GEOMETRY_TAG}_{tag}_{device}_{occupation_method}.log"),
            pyscf_console=False,
        ),
        checkpoint=CheckpointConfig(filename=checkpoint.name),
    )
    if checkpoint.exists() and not recalculate:
        print(f"Loading existing calculation: {checkpoint}")
        obj = PySCFMBXAS.load(checkpoint, runtime=runtime)
        validate_checkpoint_geometry(
            obj, ase.io.read(here / STRUCTURE_NAME))
        expected = {i for i, symbol in enumerate(
            obj.structure.get_chemical_symbols()) if symbol == "O"}
        if expected - set(obj.excited_idxs):
            missing = sorted(expected - set(obj.excited_idxs))
            print(f"Resuming missing oxygen sites: {missing}")
            obj.excite(missing, config=excitation)
        return obj

    structure = ase.io.read(here / STRUCTURE_NAME)
    obj = PySCFMBXAS(
        structure,
        config=CalculationConfig(
            charge=0, spin=0, xc=xc, basis=basis, method="UKS",
            localization="ibo", ground_state_scf=scf),
        runtime=runtime,
    )
    obj.run("O", config=excitation)
    return obj


def save_plot(decomposition, path, xc, basis):
    """Save the package-standard total and f-order contribution plot."""
    figure, axes = plot_mbxas_decomposition(
        decomposition, show_probability=False)
    highest_order = max(decomposition["contributions"])
    axes[0].set_xlim(*SPECTRUM_XLIM)
    axes[0].set_title(
        f"6fda oxygen K-edge: {xc.upper()}/{basis} through f{highest_order}")
    figure.savefig(path, dpi=250)
    matplotlib.pyplot.close(figure)


def save_orbital_plots(spectras, directory, stem):
    """Save one shared-utility GS/FCH orbital diagram per excited site."""
    saved = []
    for spectra in sorted(spectras, key=lambda item: item.exc_idx):
        figure, _ = spectra.plot_orbital_rearrangement(
            energy_window=ORBITAL_ENERGY_WINDOW,
            min_overlap=ORBITAL_MIN_OVERLAP,
            show_indices=False,
            show_dos=True,
            dos_sigma=ORBITAL_DOS_SIGMA,
        )
        path = directory / f"{stem}_O{spectra.exc_idx:02d}_orbitals.png"
        figure.savefig(path, dpi=250)
        matplotlib.pyplot.close(figure)
        saved.append(path)
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recalculate", action="store_true",
                        help="replace the checkpoint and run every SCF again")
    parser.add_argument("--cpu", action="store_true",
                        help="use CPU instead of the default GPU path")
    parser.add_argument("--quick", action="store_true",
                        help="validate with inexpensive LDA/STO-3G settings")
    parser.add_argument(
        "--xc", choices=("pbe", "b3lyp"), default=DEFAULT_XC,
        help=f"production exchange-correlation functional (default: {DEFAULT_XC})")
    parser.add_argument(
        "--occupation-method", choices=("mom", "maxvol", "mixed"),
        default="maxvol",
        help="SCF state tracking for FCH/XCH (default: maxvol)")
    parser.add_argument(
        "--mom-warmup-calls", type=int, default=2,
        help="MOM occupation calls before maxvol in mixed mode (default: 2)")
    parser.add_argument(
        "--f-order", type=int, default=F_ORDER,
        help=(f"highest cumulative many-body order (default: {F_ORDER}); "
              "orders above 2 can be expensive"))
    args = parser.parse_args()
    if args.f_order < 1:
        parser.error("--f-order must be a positive integer")

    here = Path(__file__).resolve().parent
    output_paths = prepare_output_directories(here)
    xc, basis, tag = theory_settings(args.quick, args.xc)
    obj = load_or_calculate(
        here, output_paths, args.recalculate, use_gpu=not args.cpu, xc=xc,
        basis=basis, tag=tag, quick=args.quick,
        occupation_method=args.occupation_method,
        mom_warmup_calls=args.mom_warmup_calls)

    expected = {i for i, symbol in enumerate(
        obj.structure.get_chemical_symbols()) if symbol == "O"}
    completed = obj.to_spectra()
    atom_indices = {sp.exc_idx for sp in completed}
    missing = sorted(expected - atom_indices)
    if missing:
        raise RuntimeError(
            f"Oxygen sites {missing} did not converge; inspect the PySCF and "
            "PyMBXAS logs before generating a partial spectrum")

    onset = min(sp.energies.min() for sp in completed)
    erange = [onset - PRE_EDGE, onset + POST_EDGE]
    common = dict(
        erange=erange,
        sigma=SIGMA,
        npoints=NPOINTS,
        tol=TOL,
        max_configurations=MAX_CONFIGURATIONS,
    )
    postprocess_start = time.perf_counter()

    # Each physical order contains all excited- and spectator-spin
    # constituents. Shake-up/down arrays partition each higher-order
    # contribution; neither is an additional physical spectrum.
    decomposition = completed.get_mbxas_decomposition(
        f_order=args.f_order, average=False, **common)
    energy = decomposition["energy"]
    contributions = decomposition["contributions"]
    resolved = decomposition["decomposition"]
    total = decomposition["total"]

    method_tag = (f"mixed_w{args.mom_warmup_calls}"
                  if args.occupation_method == "mixed"
                  else args.occupation_method)
    order_tag = "_".join(
        f"f{order}" for order in range(1, args.f_order + 1))
    stem_name = f"6fda_{GEOMETRY_TAG}_{tag}_{method_tag}_{order_tag}"
    data_stem = output_paths["data"] / stem_name
    spectrum_figure = output_paths["spectrum_figures"] / f"{stem_name}.png"
    archive = {
        "energy_ev": energy,
        **{f"f{order}": value
           for order, value in contributions.items()},
        **{f"f{order}_{component}": value
           for order, parts in resolved.items()
           for component, value in parts.items()},
        f"total_{order_tag}": total,
        "f_order": args.f_order,
        "xc": xc,
        "basis": basis,
        "sigma_ev": SIGMA,
        "screening_tol": TOL,
        "oxygen_indices": np.array(sorted(atom_indices)),
        "occupation_method": args.occupation_method,
        "mom_warmup_calls": args.mom_warmup_calls,
    }
    np.savez_compressed(data_stem.with_suffix(".npz"), **archive)
    table, columns = output_table(energy, contributions, resolved, total)
    np.savetxt(
        data_stem.with_suffix(".csv"), table, delimiter=",",
        header=",".join(columns), comments="")
    save_plot(decomposition, spectrum_figure, xc, basis)
    orbital_figures = save_orbital_plots(
        completed, output_paths["orbital_figures"], stem_name)

    summary = {
        "completed oxygen sites": sorted(atom_indices),
        "XC / basis": f"{xc} / {basis}",
        "occupation method": args.occupation_method,
        "highest order": f"f{args.f_order}",
        "manifold": "full QE-style energy-windowed FCH space",
    }
    if args.occupation_method == "mixed":
        summary["MOM warm-up calls"] = args.mom_warmup_calls
    for order, contribution in contributions.items():
        summary[f"integrated f{order}"] = (
            f"{np.trapezoid(contribution, energy):.8e}")
    summary[f"integrated total through f{args.f_order}"] = (
        f"{np.trapezoid(total, energy):.8e}")
    summary.update({
        "elapsed": f"{time.perf_counter() - postprocess_start:.1f} s",
        "NPZ file": str(data_stem.with_suffix(".npz").relative_to(here)),
        "CSV file": str(data_stem.with_suffix(".csv").relative_to(here)),
        "spectrum figure": str(spectrum_figure.relative_to(here)),
        "orbital figures": (
            f"{len(orbital_figures)} files in "
            f"{output_paths['orbital_figures'].relative_to(here)}"),
    })
    logging.getLogger("pymbxas.example.6fda").info(
        "6fda decomposition completed\n%s", format_log_fields(summary))


if __name__ == "__main__":
    main()
