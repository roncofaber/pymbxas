#!/usr/bin/env python3
"""Isolated O8 FCH state validation for the 6fda example.

Run one occupation controller at a time. Existing test checkpoints are reused,
and every available state is compared with the production maxvol calculation.
No production checkpoint is modified.
"""

import argparse
import csv
import itertools
import json
import logging
from pathlib import Path

import ase.io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.units import Hartree
from pyscf.scf import uhf

from pymbxas import (
    CalculationConfig, CheckpointConfig, ExcitationConfig, LoggingConfig,
    PySCFMBXAS, RuntimeConfig, SCFConfig,
)
from pymbxas.io.config import configure_logger, format_log_fields
from pymbxas.mbxas.mbxas import core_hole_index


O_INDEX = 8
STRUCTURE_NAME = "6fda-dam_relaxed_dft.xyz"
GEOMETRY_TAG = "dftgeom"
XC = "pbe"
BASIS = "def2-svpd"
DFT_VERBOSE = 4
SCF_MAX_CYCLE = 200
SCF_CONV_TOL = 1e-6
SCF_DIIS_CYCLES = 50
SCF_MIXING_CYCLES = 150
SCF_DAMPING = 0.2
SCF_LEVEL_SHIFT = 0.2
SCF_SECOND_ORDER = False
ORBITAL_ENERGY_WINDOW = (-15.0, 15.0)


def output_directories(here):
    root = here / "outputs"
    paths = {
        "root": root,
        "checkpoints": root / "checkpoints",
        "logs": root / "logs",
        "reports": root / "reports",
        "figures": root / "figures",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def find_excitation(calculation, atom_index=O_INDEX):
    matches = [exc for exc in calculation.excitations
               if exc.ato_idx == atom_index]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one excitation for atom {atom_index}, found {len(matches)}")
    return matches[0]


def validate_geometry(calculation, reference, label):
    """Require identical atom ordering and internal geometry, up to translation."""
    symbols = calculation.structure.get_chemical_symbols()
    if symbols != reference.get_chemical_symbols():
        raise RuntimeError(f"{label} has a different atom ordering")
    calculated = calculation.structure.get_all_distances(mic=False)
    expected = reference.get_all_distances(mic=False)
    maximum = float(np.max(np.abs(calculated - expected)))
    if maximum > 1e-6:
        raise RuntimeError(
            f"{label} geometry differs from {STRUCTURE_NAME}: "
            f"maximum pair-distance difference is {maximum:.6f} Angstrom")


def occupied_coefficients(coefficients, occupations):
    return tuple(coefficients[spin][:, occupations[spin] > 0.5]
                 for spin in range(2))


def determinant_overlap(left, right, overlap):
    """Return stable determinant/subspace diagnostics for two spin spaces."""
    singular_values = []
    for spin in range(2):
        matrix = left[spin].conj().T @ overlap @ right[spin]
        singular_values.append(np.linalg.svd(matrix, compute_uv=False))
    combined = np.concatenate(singular_values)
    clipped = np.clip(combined, np.finfo(float).tiny, None)
    log_abs_det = float(np.log(clipped).sum())
    return {
        "abs_determinant": float(np.exp(log_abs_det)),
        "log_abs_determinant": log_abs_det,
        "minimum_singular_value": float(combined.min()),
        "geometric_mean_singular_value": float(np.exp(np.log(clipped).mean())),
    }


def analyze_state(calculation, label):
    excitation = find_excitation(calculation)
    fch = excitation.data["fch"].to_cpu()
    gs = calculation.gs_data.to_cpu()
    coefficients = np.asarray(fch.mo_coeff)
    occupations = np.asarray(fch.mo_occ)
    gs_coefficients = np.asarray(gs.mo_coeff)
    gs_occupations = np.asarray(gs.mo_occ)
    overlap = fch.mol.intor_symmetric("int1e_ovlp")

    occupied = occupied_coefficients(coefficients, occupations)
    spin_square, multiplicity = uhf.spin_square(occupied, overlap)
    density = np.asarray([
        channel @ channel.conj().T for channel in occupied
    ])
    _, atom_spin = uhf.mulliken_spin_pop(
        fch.mol, density, s=overlap, verbose=0)

    channel = int(excitation.channel)
    gs_core = int(excitation.orb_idx)
    fch_to_gs = coefficients[channel].conj().T @ overlap @ gs_coefficients[channel]
    hole = core_hole_index(fch_to_gs, occupations[channel], gs_core)
    core_overlap = float(abs(fch_to_gs[hole, gs_core]))

    target_occupations = gs_occupations.copy()
    target_occupations[channel, gs_core] = 0
    target = occupied_coefficients(gs_coefficients, target_occupations)
    target_overlap = determinant_overlap(target, occupied, overlap)

    top_spin_atoms = np.argsort(np.abs(atom_spin))[::-1][:10]
    symbols = fch.mol.atom_charges()
    atom_symbols = [fch.mol.atom_symbol(i) for i in range(fch.mol.natm)]
    return {
        "label": label,
        "energy_ha": float(fch.e_tot),
        "s2": float(spin_square),
        "multiplicity": float(multiplicity),
        "spin_contamination": float(spin_square - 0.75),
        "core_hole_mo": int(hole),
        "core_hole_overlap": core_overlap,
        "target_overlap": target_overlap,
        "target_atom_spin": float(atom_spin[O_INDEX]),
        "total_mulliken_spin": float(atom_spin.sum()),
        "top_spin_atoms": [
            {
                "atom_index": int(index),
                "symbol": atom_symbols[index],
                "atomic_number": int(symbols[index]),
                "spin": float(atom_spin[index]),
            }
            for index in top_spin_atoms
        ],
        "_occupied": occupied,
        "_overlap": overlap,
        "_calculation": calculation,
    }


def public_state(state):
    return {key: value for key, value in state.items()
            if not key.startswith("_")}


def compare_states(states):
    comparisons = []
    for left, right in itertools.combinations(states, 2):
        overlap = determinant_overlap(
            left["_occupied"], right["_occupied"], left["_overlap"])
        comparisons.append({
            "left": left["label"],
            "right": right["label"],
            "energy_difference_ev": (
                right["energy_ha"] - left["energy_ha"]) * Hartree,
            "s2_difference": right["s2"] - left["s2"],
            "occupied_overlap": overlap,
        })
    return comparisons


def save_summary(states, comparisons, reports):
    payload = {
        "target_atom": O_INDEX,
        "ideal_fch_s2": 0.75,
        "states": [public_state(state) for state in states],
        "comparisons": comparisons,
    }
    (reports / "o8_state_comparison.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with (reports / "o8_state_comparison.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        columns = [
            "label", "energy_ha", "relative_energy_ev", "s2",
            "spin_contamination", "core_hole_overlap",
            "target_abs_determinant", "target_minimum_singular_value",
            "target_atom_spin", "total_mulliken_spin",
        ]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        minimum = min(state["energy_ha"] for state in states)
        for state in states:
            writer.writerow({
                "label": state["label"],
                "energy_ha": state["energy_ha"],
                "relative_energy_ev": (state["energy_ha"] - minimum) * Hartree,
                "s2": state["s2"],
                "spin_contamination": state["spin_contamination"],
                "core_hole_overlap": state["core_hole_overlap"],
                "target_abs_determinant": state["target_overlap"]["abs_determinant"],
                "target_minimum_singular_value": state["target_overlap"]["minimum_singular_value"],
                "target_atom_spin": state["target_atom_spin"],
                "total_mulliken_spin": state["total_mulliken_spin"],
            })


def save_comparison_plot(states, path):
    labels = [state["label"] for state in states]
    energies = np.array([state["energy_ha"] for state in states])
    relative = (energies - energies.min()) * Hartree
    s2 = [state["s2"] for state in states]
    target_spin = [state["target_atom_spin"] for state in states]

    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].bar(labels, relative)
    axes[0].set_ylabel("Relative FCH energy (eV)")
    axes[1].bar(labels, s2)
    axes[1].axhline(0.75, color="black", linestyle="--", label="pure doublet")
    axes[1].set_ylabel(r"$\langle S^2\rangle$")
    axes[1].legend(frameon=False)
    axes[2].bar(labels, target_spin)
    axes[2].set_ylabel(f"Mulliken spin on O{O_INDEX}")
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def save_orbital_plot(state, path):
    calculation = state["_calculation"]
    excitation_index = next(
        index for index, excitation in enumerate(calculation.excitations)
        if excitation.ato_idx == O_INDEX)
    spectra = calculation.to_spectra(index=excitation_index)
    figure, _ = spectra.plot_orbital_rearrangement(
        energy_window=ORBITAL_ENERGY_WINDOW, min_overlap=0.05,
        show_indices=False)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def load_or_run_variant(here, paths, method, use_gpu, recalculate):
    device = "gpu" if use_gpu else "cpu"
    name = f"o8_{GEOMETRY_TAG}_{XC}_def2_svpd_{device}_{method}"
    checkpoint = paths["checkpoints"] / f"{name}.h5"
    scf = SCFConfig(
        max_cycles=SCF_MAX_CYCLE,
        convergence_tolerance=SCF_CONV_TOL,
        grid_level=3,
        diis_cycles=SCF_DIIS_CYCLES,
        mixing_cycles=SCF_MIXING_CYCLES,
        damping=SCF_DAMPING,
        level_shift=SCF_LEVEL_SHIFT,
        second_order=SCF_SECOND_ORDER)
    excitation = ExcitationConfig(
        xch=False, occupation=method, mom_warmup_calls=2, scf=scf)
    runtime = RuntimeConfig(
        work_directory=paths["checkpoints"], device=device,
        logging=LoggingConfig(
            pymbxas_verbosity=3,
            pymbxas_logfile=str(paths["logs"] / f"pymbxas_{name}.log"),
            pyscf_verbosity=DFT_VERBOSE,
            pyscf_logfile=str(paths["logs"] / f"pyscf_{name}.log"),
            pyscf_console=False),
        checkpoint=CheckpointConfig(filename=checkpoint.name))
    if checkpoint.exists() and not recalculate:
        calculation = PySCFMBXAS.load(checkpoint, runtime=runtime)
        try:
            find_excitation(calculation)
        except RuntimeError:
            calculation.excite(O_INDEX, config=excitation)
        return calculation

    structure_path = here.parent / STRUCTURE_NAME
    calculation = PySCFMBXAS(
        ase.io.read(structure_path),
        config=CalculationConfig(
            xc=XC, basis=BASIS, ground_state_scf=scf),
        runtime=runtime)
    calculation.run(O_INDEX, config=excitation)
    return calculation


def production_baseline(here, analysis_log):
    checkpoint = (
        here.parent / "outputs" / "checkpoints"
        / "6fda_dftgeom_pbe_def2_svpd_gpu_maxvol.h5")
    if not checkpoint.exists():
        return None
    calculation = PySCFMBXAS.load(
        checkpoint,
        runtime=RuntimeConfig(
            work_directory=checkpoint.parent,
            logging=LoggingConfig(
                pymbxas_logfile=str(analysis_log), pyscf_verbosity=0,
                pyscf_console=False)))
    try:
        find_excitation(calculation)
    except RuntimeError:
        return None
    return calculation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", choices=("mom", "maxvol", "mixed"), default="maxvol",
        help="occupation controller to calculate (default: maxvol)")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--recalculate", action="store_true")
    parser.add_argument(
        "--analyze-only", action="store_true",
        help="analyze existing checkpoints without starting SCF")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    paths = output_directories(here)
    states = []

    analysis_log = paths["logs"] / "o8_state_analysis.log"
    reference_structure = ase.io.read(here.parent / STRUCTURE_NAME)
    baseline = production_baseline(here, analysis_log)
    configure_logger(
        3, analysis_log, file_mode="w")
    if baseline is not None:
        validate_geometry(baseline, reference_structure, "production checkpoint")
        states.append(analyze_state(baseline, "production-maxvol"))

    if not args.analyze_only:
        variant = load_or_run_variant(
            here, paths, args.method, not args.cpu, args.recalculate)
        validate_geometry(variant, reference_structure, args.method)
        states.append(analyze_state(variant, args.method))

    for method in ("mom", "maxvol", "mixed"):
        device = "cpu" if args.cpu else "gpu"
        checkpoint = (
            paths["checkpoints"]
            / f"o8_{GEOMETRY_TAG}_{XC}_def2_svpd_{device}_{method}.h5")
        if not checkpoint.exists():
            continue
        if any(state["label"] == method for state in states):
            continue
        calculation = PySCFMBXAS.load(
            checkpoint,
            runtime=RuntimeConfig(
                work_directory=checkpoint.parent,
                logging=LoggingConfig(
                    pyscf_verbosity=0, pyscf_console=False)))
        validate_geometry(calculation, reference_structure, method)
        states.append(analyze_state(calculation, method))

    if not states:
        raise RuntimeError("No completed O8 state is available to analyze")

    # Variant loads restore their own logger; keep the report isolated.
    configure_logger(3, analysis_log, file_mode="a")

    comparisons = compare_states(states)
    save_summary(states, comparisons, paths["reports"])
    save_comparison_plot(states, paths["figures"] / "o8_state_comparison.png")
    for state in states:
        save_orbital_plot(
            state, paths["figures"] / f"o8_orbitals_{state['label']}.png")

    summary = {
        state["label"]: (
            f"E={state['energy_ha']:.10f} Ha, "
            f"<S^2>={state['s2']:.6f}, "
            f"core={state['core_hole_overlap']:.6f}, "
            f"|det target|={state['target_overlap']['abs_determinant']:.6f}")
        for state in states
    }
    logging.getLogger("pymbxas.example.o8-state-test").info(
        "O8 state comparison\n%s", format_log_fields(summary))
    print(f"Wrote diagnostics to {paths['root']}")


if __name__ == "__main__":
    main()
