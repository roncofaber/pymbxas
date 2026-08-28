"""Basis-independent parity fixture for QE's f1/f2/f3 determinant series."""

from itertools import combinations

import numpy as np

from pymbxas.mbxas.shakeup import (
    mbxas_sticks_by_order,
    overlap_sticks_from_K,
    screened_overlap_doubles_from_K,
)
from pymbxas.spectra import _combine_spin_stick_block


def _reference_xas(base_energy, base_amplitude, K, eps_occ, eps_virt):
    """Literal serial translation of QE's MB1/MB2/MB3 formulas."""
    result = {
        0: (base_energy.copy(), base_amplitude.copy(),
            np.zeros(len(base_energy), dtype=bool))
    }

    energy, amplitude, flags = [], [], []
    for v in range(len(eps_occ)):
        for c in range(len(eps_virt)):
            for f in range(c + 1, len(eps_virt)):
                de = eps_virt[c] - eps_occ[v]
                energy.append(base_energy[f] + de)
                amplitude.append(
                    base_amplitude[:, f] * K[c, v]
                    - base_amplitude[:, c] * K[f, v])
                flags.append(de < 0)
    result[1] = (
        np.asarray(energy), np.asarray(amplitude).T,
        np.asarray(flags, dtype=bool))

    energy, amplitude, flags = [], [], []
    for v, vp in combinations(range(len(eps_occ)), 2):
        for cp, c, f in combinations(range(len(eps_virt)), 3):
            de_c = eps_virt[c] - eps_occ[v]
            de_cp = eps_virt[cp] - eps_occ[vp]
            pair_minor = K[c, v] * K[cp, vp] - K[cp, v] * K[c, vp]
            c_minor = K[f, v] * K[cp, vp] - K[cp, v] * K[f, vp]
            cp_minor = K[f, v] * K[c, vp] - K[c, v] * K[f, vp]
            energy.append(base_energy[f] + de_c + de_cp)
            amplitude.append(
                base_amplitude[:, f] * pair_minor
                - base_amplitude[:, c] * c_minor
                + base_amplitude[:, cp] * cp_minor)
            flags.append(de_c < 0 or de_cp < 0)
    result[2] = (
        np.asarray(energy), np.asarray(amplitude).T,
        np.asarray(flags, dtype=bool))
    return result


def _reference_overlap(det_weight, K, eps_occ, eps_virt):
    """Literal QE zero/singles/doubles determinant-overlap formulas."""
    result = {
        0: (np.array([0.0]), np.array([det_weight]),
            np.array([False]))
    }

    energy, weight, flags = [], [], []
    for v in range(len(eps_occ)):
        for c in range(len(eps_virt)):
            de = eps_virt[c] - eps_occ[v]
            energy.append(de)
            weight.append(det_weight * abs(K[c, v]) ** 2)
            flags.append(de < 0)
    result[1] = (
        np.asarray(energy), np.asarray(weight),
        np.asarray(flags, dtype=bool))

    energy, weight, flags = [], [], []
    for v, vp in combinations(range(len(eps_occ)), 2):
        for cp, c in combinations(range(len(eps_virt)), 2):
            de_c = eps_virt[c] - eps_occ[v]
            de_cp = eps_virt[cp] - eps_occ[vp]
            minor = K[c, v] * K[cp, vp] - K[cp, v] * K[c, vp]
            energy.append(de_c + de_cp)
            weight.append(det_weight * abs(minor) ** 2)
            flags.append(de_c < 0 or de_cp < 0)
    result[2] = (
        np.asarray(energy), np.asarray(weight),
        np.asarray(flags, dtype=bool))
    return result


def _canonical_xas(sticks):
    energy, amplitude, flags = sticks
    table = np.column_stack((
        energy,
        amplitude.real.T,
        amplitude.imag.T,
        flags.astype(float),
    ))
    table = np.round(table, 13)
    keys = tuple(table[:, index]
                 for index in range(table.shape[1] - 1, -1, -1))
    return table[np.lexsort(keys)]


def _canonical_overlap(sticks):
    energy, weight, flags = sticks
    table = np.round(np.column_stack((
        energy, weight, flags.astype(float))), 13)
    return table[np.lexsort((table[:, 1], table[:, 0]))]


def _combined_term(xas, spectator):
    energy, amplitude, xas_flags = xas
    shift, probability, spectator_flags = spectator
    oscillator = np.mean(np.abs(amplitude) ** 2, axis=0)
    joint_energy, joint_intensity = _combine_spin_stick_block(
        energy, oscillator, shift, probability)
    flags = (xas_flags[:, None] | spectator_flags[None, :]).ravel()
    return joint_energy.ravel(), joint_intensity.ravel(), flags


def _reference_combined_term(xas, spectator):
    energy, amplitude, xas_flags = xas
    shift, probability, spectator_flags = spectator
    joint_energy, joint_intensity, flags = [], [], []
    for xas_index, xas_energy in enumerate(energy):
        oscillator = np.mean(np.abs(amplitude[:, xas_index]) ** 2)
        for spectator_index, spectator_shift in enumerate(shift):
            final_energy = xas_energy + spectator_shift
            joint_energy.append(final_energy)
            # PyMBXAS's documented molecular cross-section convention adds
            # the final photon-energy prefactor after QE's spin factorization.
            joint_intensity.append(
                final_energy * oscillator * probability[spectator_index])
            flags.append(
                xas_flags[xas_index]
                or spectator_flags[spectator_index])
    return (np.asarray(joint_energy), np.asarray(joint_intensity),
            np.asarray(flags, dtype=bool))


def _canonical_combined(sticks):
    energy, intensity, flags = sticks
    table = np.round(np.column_stack((
        energy, intensity, flags.astype(float))), 13)
    return table[np.lexsort((table[:, 1], table[:, 0]))]


def test_qe_f1_f2_f3_order_resolved_parity():
    # Complex matrices exercise the same algebra used by QE at arbitrary
    # k-points without tying the fixture to plane waves or Gaussian AOs.
    A_excited = np.array([
        [0.93 + 0.02j, 0.04 - 0.01j],
        [-0.03 + 0.02j, 0.89 - 0.03j],
    ])
    Aprime_excited = np.array([
        [0.18 + 0.03j, -0.04 + 0.01j],
        [0.07 - 0.02j, 0.16 + 0.04j],
        [-0.11 + 0.01j, 0.09 - 0.03j],
        [0.05 + 0.02j, -0.13 + 0.01j],
    ])
    K_excited = Aprime_excited @ np.linalg.inv(A_excited)
    determinant_excited = np.linalg.det(A_excited)
    bare_dipole = np.array([
        [0.7 + 0.1j, -0.2 + 0.05j, 0.4 - 0.1j, 0.1 + 0.2j],
        [0.3 - 0.2j, 0.5 + 0.1j, -0.1 + 0.15j, 0.6 - 0.05j],
        [-0.4 + 0.05j, 0.2 - 0.1j, 0.3 + 0.2j, -0.2 + 0.1j],
    ])
    base_amplitude = determinant_excited * bare_dipole
    base_energy = np.array([10.0, 10.8, 11.7, 12.9])
    excited_occ = np.array([-0.65, -0.20])
    excited_virt = np.array([-0.30, 0.10, 0.55, 1.00])

    A_spectator = np.array([
        [0.91 - 0.01j, -0.02 + 0.03j],
        [0.05 + 0.01j, 0.94 + 0.02j],
    ])
    Aprime_spectator = np.array([
        [0.12 - 0.02j, 0.03 + 0.01j],
        [-0.06 + 0.04j, 0.14 - 0.01j],
        [0.08 + 0.02j, -0.09 + 0.03j],
        [0.02 - 0.01j, 0.11 + 0.02j],
    ])
    K_spectator = Aprime_spectator @ np.linalg.inv(A_spectator)
    spectator_weight = abs(np.linalg.det(A_spectator)) ** 2
    spectator_occ = np.array([-0.55, -0.15])
    spectator_virt = np.array([-0.25, 0.05, 0.45, 0.90])

    reference_xas = _reference_xas(
        base_energy, base_amplitude, K_excited,
        excited_occ, excited_virt)
    production_xas = mbxas_sticks_by_order(
        base_energy, base_amplitude, K_excited,
        excited_occ, excited_virt, max_extra_order=2,
        screen_tol=1e-14, pair_energy_max=10.0,
        final_energy_range=(-100.0, 100.0),
        determinant=determinant_excited,
        max_configurations=10_000)

    reference_overlap = _reference_overlap(
        spectator_weight, K_spectator,
        spectator_occ, spectator_virt)
    production_overlap = {}
    for order in (0, 1):
        sticks = overlap_sticks_from_K(
            spectator_weight, K_spectator,
            spectator_occ, spectator_virt, order,
            max_configurations=10_000, pair_energy_max=10.0)
        production_overlap[order] = (
            sticks.energy, sticks.weight, sticks.shakedown)
    doubles = screened_overlap_doubles_from_K(
        spectator_weight, K_spectator,
        spectator_occ, spectator_virt, tol=1e-14,
        max_configurations=10_000, pair_energy_max=10.0)
    production_overlap[2] = (
        doubles.energy, doubles.weight, doubles.shakedown)

    for order in (0, 1, 2):
        assert np.array_equal(
            _canonical_xas(production_xas[order]),
            _canonical_xas(reference_xas[order]))
        assert np.array_equal(
            _canonical_overlap(production_overlap[order]),
            _canonical_overlap(reference_overlap[order]))

    # QE's resolved spin organization: f1=10, f2=20+11,
    # f3=30+21+12. Each named term is compared before any broadening.
    term_orders = {
        "10": (0, 0),
        "20": (1, 0),
        "11": (0, 1),
        "30": (2, 0),
        "21": (1, 1),
        "12": (0, 2),
    }
    assert {name for name, orders in term_orders.items()
            if sum(orders) == 0} == {"10"}
    assert {name for name, orders in term_orders.items()
            if sum(orders) == 1} == {"20", "11"}
    assert {name for name, orders in term_orders.items()
            if sum(orders) == 2} == {"30", "21", "12"}

    for name, (xas_order, spectator_order) in term_orders.items():
        production = _combined_term(
            production_xas[xas_order],
            production_overlap[spectator_order])
        reference = _reference_combined_term(
            reference_xas[xas_order],
            reference_overlap[spectator_order])
        assert np.array_equal(
            _canonical_combined(production),
            _canonical_combined(reference)), name
