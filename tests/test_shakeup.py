import io
import logging

import numpy as np
import pytest

import pymbxas.mbxas.shakeup as shakeup_module
from pymbxas.mbxas.shakeup import (
    _next_qe_element_threshold,
    _next_qe_overlap_doubles_threshold,
    _qe_screening_converged,
    _qe_tolerance_count,
    available_overlap_weight,
    mbxas_sticks_by_order,
    overlap_sticks,
    overlap_sticks_from_K,
    screened_overlap_doubles_from_K,
    shakeup_sticks,
)
from pymbxas.io.config import with_log_context


def test_screening_iterations_are_debug_and_completion_is_info(monkeypatch):
    class Recorder:
        def __init__(self):
            self.debug_messages = []
            self.info_messages = []
            self.warning_messages = []

        def debug(self, message, *args):
            self.debug_messages.append(message % args)

        def info(self, message, *args):
            self.info_messages.append(message % args)

        def warning(self, message, *args):
            self.warning_messages.append(message % args)

    recorder = Recorder()
    monkeypatch.setattr(shakeup_module, "logger", recorder)
    mbxas_sticks_by_order(
        np.array([10.0, 11.0, 12.0, 13.0]),
        np.ones((1, 4)),
        np.array([[0.3, 0.2], [0.2, 0.1], [0.1, 0.05], [0.04, 0.02]]),
        np.array([-0.5, -0.2]), np.array([0.1, 0.3, 0.5, 0.7]), 1,
        screen_tol=0.01, pair_energy_max=2.0,
        final_energy_range=(0.0, 20.0), determinant=1.0,
        max_configurations=10_000)

    assert any("MB2 screening iteration" in message
               for message in recorder.debug_messages)
    assert not any("screening iteration" in message
                   for message in recorder.info_messages)
    assert any("MB2 screening complete" in message
               for message in recorder.info_messages)
    completion = next(
        message for message in recorder.info_messages
        if "MB2 screening complete" in message)
    assert "iterations" in completion
    assert "final threshold" in completion
    assert "final series" in completion
    assert "final delta" in completion
    assert "final derivative" in completion
    assert "final derivative x curvature" in completion


def test_screening_uses_supplied_log_context():
    stream = io.StringIO()
    base_log = logging.Logger("screening-context-test", level=logging.INFO)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    base_log.addHandler(handler)
    contextual_log = with_log_context(
        base_log, site="O:6", stage="f2 full")

    mbxas_sticks_by_order(
        np.array([10.0, 11.0, 12.0, 13.0]),
        np.ones((1, 4)),
        np.array([[0.3, 0.2], [0.2, 0.1], [0.1, 0.05], [0.04, 0.02]]),
        np.array([-0.5, -0.2]), np.array([0.1, 0.3, 0.5, 0.7]), 1,
        screen_tol=0.01, pair_energy_max=2.0,
        final_energy_range=(0.0, 20.0), determinant=1.0,
        max_configurations=10_000, log=contextual_log)

    assert "[O:6 f2 full] MB2 screening complete" in stream.getvalue()


def test_overlap_doubles_are_complete_and_determinant_weighted():
    rng = np.random.default_rng(7)
    n_occ, n_virt = 4, 5
    A = np.eye(n_occ) + 0.03 * rng.normal(size=(n_occ, n_occ))
    Aprime = 0.2 * rng.normal(size=(n_virt, n_occ))
    eps_occ = np.array([-1.0, -0.7, -0.4, -0.1])
    eps_virt = np.array([-0.5, 0.2, 0.4, 0.8, 1.1])
    K = Aprime @ np.linalg.inv(A)

    sticks = overlap_sticks(A, Aprime, eps_occ, eps_virt, order=2)
    assert len(sticks.weight) == 60  # C(4,2) C(5,2)

    expected = []
    for v0 in range(n_occ):
        for v1 in range(v0 + 1, n_occ):
            for c0 in range(n_virt):
                for c1 in range(c0 + 1, n_virt):
                    expected.append(abs(np.linalg.det(
                        K[np.ix_([c0, c1], [v0, v1])])) ** 2
                        * abs(np.linalg.det(A)) ** 2)
    assert np.allclose(np.sort(sticks.weight), np.sort(expected))

    with pytest.raises(ValueError, match="60 configurations"):
        overlap_sticks(
            A, Aprime, eps_occ, eps_virt, order=2,
            max_configurations=10)


def test_available_overlap_matches_complete_cauchy_binet_sum():
    rng = np.random.default_rng(11)
    A = np.eye(2) + 0.05 * (
        rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
    Aprime = 0.2 * (
        rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2)))
    eps_occ = np.array([-0.6, -0.2])
    eps_virt = np.array([-0.1, 0.3, 0.8])

    captured = sum(
        overlap_sticks(
            A, Aprime, eps_occ, eps_virt, order,
            max_configurations=100).weight.sum()
        for order in (0, 1, 2))
    assert captured == pytest.approx(
        available_overlap_weight(A, Aprime), rel=1e-13, abs=1e-15)

    with pytest.raises(ValueError, match="same column count"):
        available_overlap_weight(A, np.empty((3, 3)))


def test_shakedown_uses_any_negative_constituent_not_negative_sum():
    A = np.eye(2)
    Aprime = np.array([[0.3, 0.1], [0.2, 0.4]])
    eps_occ = np.array([0.0, 0.0])
    eps_virt = np.array([-1.0, 2.0])

    sticks = overlap_sticks(A, Aprime, eps_occ, eps_virt, order=2)
    assert sticks.energy.tolist() == [1.0]
    assert sticks.shakedown.tolist() == [True]
    energy, weight = shakeup_sticks(
        A, Aprime, eps_occ, eps_virt, order=2, shakedown_only=True)
    assert energy.tolist() == [1.0]
    assert len(weight) == 1


def test_explicit_mb2_matches_qe_formula():
    # One hole and three virtuals gives one MB2 determinant for each c < f.
    base_energy = np.array([10.0, 11.0, 12.0])
    base_amp = np.array([[1.0, 2.0, 4.0], [0.5, -1.0, 3.0]])
    K = np.array([[0.2], [0.3], [-0.4]])
    eps_occ = np.array([-0.5])
    eps_virt = np.array([0.1, 0.4, 0.9])

    result = mbxas_sticks_by_order(
        base_energy, base_amp, K, eps_occ, eps_virt, max_extra_order=1)
    energy, amplitude, flags = result[1]
    expected_energy = np.array([
        base_energy[1] + eps_virt[0] - eps_occ[0],
        base_energy[2] + eps_virt[0] - eps_occ[0],
        base_energy[2] + eps_virt[1] - eps_occ[0],
    ])
    expected_amp = np.column_stack([
        base_amp[:, 1] * K[0, 0] - base_amp[:, 0] * K[1, 0],
        base_amp[:, 2] * K[0, 0] - base_amp[:, 0] * K[2, 0],
        base_amp[:, 2] * K[1, 0] - base_amp[:, 1] * K[2, 0],
    ])
    assert np.allclose(energy, expected_energy)
    assert np.allclose(amplitude, expected_amp)
    assert not flags.any()


def test_qe_screened_mb2_respects_energy_window_and_count_guard():
    rng = np.random.default_rng(3)
    n_occ, n_virt = 8, 20
    base_energy = np.linspace(10.0, 14.0, n_virt)
    base_amp = rng.normal(size=(3, n_virt))
    K = 0.03 * rng.normal(size=(n_virt, n_occ))
    eps_occ = np.linspace(-0.8, -0.1, n_occ)
    eps_virt = np.linspace(-0.05, 0.8, n_virt)

    screened = mbxas_sticks_by_order(
        base_energy, base_amp, K, eps_occ, eps_virt, 1,
        screen_tol=0.01, pair_energy_max=0.75,
        final_energy_range=(0.0, 12.0), determinant=0.9,
        max_configurations=10_000)[1]
    assert len(screened[0]) < n_occ * (n_virt * (n_virt - 1) // 2)
    assert np.all(screened[0] <= 12.0)
    assert screened[1].shape == (3, len(screened[0]))

    with pytest.raises(ValueError, match="exceeding max_configurations"):
        mbxas_sticks_by_order(
            base_energy, base_amp, K, eps_occ, eps_virt, 1,
            screen_tol=0.01, pair_energy_max=0.75,
            final_energy_range=(0.0, 12.0), determinant=0.9,
            max_configurations=5)


def test_qe_screening_control_flow_matches_fortran_semantics():
    # energy_sequence_tolerance_index returns the first element below the
    # threshold and the Fortran loop includes that endpoint.
    sorted_score = np.array([0.20, 0.08, 0.06, 0.01]) ** 2
    assert _qe_tolerance_count(sorted_score, 0.07) == 3
    assert _qe_tolerance_count(sorted_score, 0.001) == 4

    # Before the first QE spectral buffer fills, the element threshold drops
    # geometrically. Afterwards its reduction is count-adaptive.
    assert _next_qe_element_threshold(
        0.07, 100, spectrum_batch_size=1000) == pytest.approx(0.049)
    assert _next_qe_element_threshold(
        0.07, 1000, spectrum_batch_size=1000) == pytest.approx(
            0.07 / 1.7)
    assert _next_qe_element_threshold(
        0.07, 2000, spectrum_batch_size=1000) == pytest.approx(
            0.07 / 1.35)

    # Spectator doubles use a square root after their first buffer fills
    # because the selection threshold applies to a product of two K entries.
    assert _next_qe_overlap_doubles_threshold(
        0.007, 100, spectrum_batch_size=1000) == pytest.approx(0.0049)
    assert _next_qe_overlap_doubles_threshold(
        0.007, 2000, spectrum_batch_size=1000) == pytest.approx(
            0.007 / np.sqrt(1.35))

    assert not _qe_screening_converged([0.1, 0.11, 0.111], 0.01)
    assert _qe_screening_converged([0.1, 0.11, 0.111, 0.1111], 0.01)


def test_qe_screened_spectator_doubles_match_exhaustive_limit(monkeypatch):
    rng = np.random.default_rng(23)
    n_occ, n_virt = 4, 6
    K = 0.2 * rng.normal(size=(n_virt, n_occ))
    eps_occ = np.array([-0.9, -0.6, -0.3, -0.1])
    eps_virt = np.array([-0.5, -0.05, 0.2, 0.45, 0.8, 1.1])
    det_weight = 0.83
    energy_max = 1.8

    exact = overlap_sticks_from_K(
        det_weight, K, eps_occ, eps_virt, order=2,
        max_configurations=10_000, pair_energy_max=energy_max)
    # doubles_overlap applies a second e_range test to the summed promotion.
    exact_keep = exact.energy <= energy_max
    screened = screened_overlap_doubles_from_K(
        det_weight, K, eps_occ, eps_virt, tol=1e-12,
        max_configurations=10_000, pair_energy_max=energy_max)

    def sorted_table(energy, weight, shakedown):
        table = np.column_stack((energy, weight, shakedown.astype(float)))
        return table[np.lexsort((table[:, 1], table[:, 0]))]

    expected = sorted_table(
        exact.energy[exact_keep], exact.weight[exact_keep],
        exact.shakedown[exact_keep])
    actual = sorted_table(
        screened.energy, screened.weight, screened.shakedown)
    assert actual.shape == expected.shape
    assert np.allclose(actual, expected, rtol=1e-13, atol=1e-15)

    import pymbxas.mbxas.shakeup as shakeup_module
    monkeypatch.setattr(shakeup_module, "_QE_BUILD_BATCH", 2)
    chunked = screened_overlap_doubles_from_K(
        det_weight, K, eps_occ, eps_virt, tol=1e-12,
        max_configurations=10_000, pair_energy_max=energy_max)
    assert np.array_equal(chunked.energy, screened.energy)
    assert np.array_equal(chunked.weight, screened.weight)
    assert np.array_equal(chunked.shakedown, screened.shakedown)


def test_qe_screened_spectator_doubles_prune_and_guard():
    rng = np.random.default_rng(29)
    n_occ, n_virt = 8, 20
    K = 0.12 * rng.normal(size=(n_virt, n_occ))
    eps_occ = np.linspace(-0.8, -0.1, n_occ)
    eps_virt = np.linspace(-0.3, 0.9, n_virt)

    exact = overlap_sticks_from_K(
        0.9, K, eps_occ, eps_virt, order=2,
        max_configurations=20_000, pair_energy_max=1.2)
    screened = screened_overlap_doubles_from_K(
        0.9, K, eps_occ, eps_virt, tol=0.1,
        max_configurations=20_000, pair_energy_max=1.2)
    assert 0 < len(screened.energy) < len(exact.energy)
    assert np.all(screened.energy <= 1.2)
    assert np.all(screened.weight >= 0)

    with pytest.raises(ValueError, match="exceeding max_configurations"):
        screened_overlap_doubles_from_K(
            0.9, K, eps_occ, eps_virt, tol=0.1,
            max_configurations=5, pair_energy_max=1.2)


def test_qe_screened_mb2_build_batch_is_invariant(monkeypatch):
    import pymbxas.mbxas.shakeup as shakeup_module

    rng = np.random.default_rng(19)
    n_occ, n_virt = 3, 9
    base_energy = np.linspace(8.0, 12.0, n_virt)
    base_amp = rng.normal(size=(3, n_virt))
    K = 0.15 * rng.normal(size=(n_virt, n_occ))
    eps_occ = np.linspace(-0.5, -0.1, n_occ)
    eps_virt = np.linspace(-0.2, 0.7, n_virt)
    kwargs = dict(
        screen_tol=1e-12, pair_energy_max=2.0,
        final_energy_range=(0.0, 20.0), determinant=0.9,
        max_configurations=10_000)

    normal = mbxas_sticks_by_order(
        base_energy, base_amp, K, eps_occ, eps_virt, 1, **kwargs)[1]
    exhaustive_kwargs = dict(kwargs)
    exhaustive_kwargs.pop("screen_tol")
    exhaustive = mbxas_sticks_by_order(
        base_energy, base_amp, K, eps_occ, eps_virt, 1,
        screen_tol=None, **exhaustive_kwargs)[1]
    monkeypatch.setattr(shakeup_module, "_QE_BUILD_BATCH", 2)
    chunked = mbxas_sticks_by_order(
        base_energy, base_amp, K, eps_occ, eps_virt, 1, **kwargs)[1]

    assert np.array_equal(chunked[0], normal[0])
    assert np.array_equal(chunked[2], normal[2])
    assert np.allclose(chunked[1], normal[1], rtol=0, atol=0)

    def sorted_sticks(sticks):
        table = np.column_stack((sticks[0], sticks[1].T))
        keys = tuple(table[:, i] for i in range(table.shape[1] - 1, -1, -1))
        return table[np.lexsort(keys)]

    # With a sufficiently tight tolerance the screened QE loop exhausts the
    # relevant pair list and becomes the exact MB2 enumeration.
    assert len(normal[0]) == len(exhaustive[0])
    assert np.allclose(sorted_sticks(normal), sorted_sticks(exhaustive))


def test_mb2_final_manifold_is_independent_of_pair_window():
    # The only strong dipole-final state is the highest virtual. Restricting
    # K(c,v) by pair energy must not truncate that independent f manifold.
    base_energy = np.array([10.0, 11.0, 12.0, 13.0])
    base_amp = np.array([[0.0, 0.0, 0.0, 2.0]])
    K = np.array([[0.5], [0.0], [0.0], [0.0]])
    eps_occ = np.array([0.0])
    eps_virt = np.array([0.1, 0.4, 0.8, 1.2])

    energy, amplitude, _ = mbxas_sticks_by_order(
        base_energy, base_amp, K, eps_occ, eps_virt, 1,
        screen_tol=1e-8, pair_energy_max=0.2,
        final_energy_range=(9.0, 14.0), determinant=1.0)[1]

    assert np.any(np.isclose(energy, 13.1))
    idx = np.argmin(np.abs(energy - 13.1))
    assert amplitude[0, idx] == pytest.approx(1.0)


def test_overlap_pair_energy_window_keeps_all_relevant_singles():
    from pymbxas.mbxas.shakeup import overlap_sticks_from_K

    K = np.arange(1, 7, dtype=float).reshape(3, 2) / 10
    eps_occ = np.array([-0.2, 0.0])
    eps_virt = np.array([-0.1, 0.1, 0.5])
    sticks = overlap_sticks_from_K(
        0.8, K, eps_occ, eps_virt, order=1,
        pair_energy_max=0.35)

    promotion = eps_virt[:, None] - eps_occ[None, :]
    expected = promotion[promotion <= 0.35]
    assert np.array_equal(np.sort(sticks.energy), np.sort(expected))
    assert len(sticks.energy) == len(expected)


def test_explicit_mb3_matches_qe_formula():
    base_energy = np.array([10.0, 11.0, 12.0])
    base_amp = np.array([[1.0, 2.0, 4.0]])
    K = np.array([[0.2, 0.1], [0.3, -0.2], [-0.4, 0.5]])
    eps_occ = np.array([-0.6, -0.2])
    eps_virt = np.array([0.1, 0.4, 0.9])

    energy, amplitude, _ = mbxas_sticks_by_order(
        base_energy, base_amp, K, eps_occ, eps_virt, 2)[2]
    cp, c, f = 0, 1, 2
    v, vp = 0, 1
    expected = (
        base_amp[:, f] * (K[c, v] * K[cp, vp] - K[cp, v] * K[c, vp])
        - base_amp[:, c] * (K[f, v] * K[cp, vp] - K[cp, v] * K[f, vp])
        + base_amp[:, cp] * (K[f, v] * K[c, vp] - K[c, v] * K[f, vp])
    )
    assert energy[0] == pytest.approx(
        base_energy[f] + eps_virt[c] - eps_occ[v]
        + eps_virt[cp] - eps_occ[vp])
    assert np.allclose(amplitude[:, 0], expected)


def test_qe_screened_mb3_matches_exhaustive_limit(monkeypatch):
    rng = np.random.default_rng(37)
    n_occ, n_virt = 3, 6
    base_energy = np.linspace(9.0, 14.0, n_virt)
    base_amp = rng.normal(size=(3, n_virt))
    K = 0.25 * rng.normal(size=(n_virt, n_occ))
    eps_occ = np.array([-0.8, -0.4, -0.1])
    eps_virt = np.array([-0.3, 0.0, 0.25, 0.5, 0.8, 1.1])
    kwargs = dict(
        pair_energy_max=2.0, final_energy_range=(0.0, 20.0),
        determinant=0.87, max_configurations=10_000)

    exact = mbxas_sticks_by_order(
        base_energy, base_amp, K, eps_occ, eps_virt, 2,
        screen_tol=None, **kwargs)[2]
    screened = mbxas_sticks_by_order(
        base_energy, base_amp, K, eps_occ, eps_virt, 2,
        screen_tol=1e-14, **kwargs)[2]

    def sorted_table(sticks):
        energy, amplitude, flags = sticks
        # Equivalent addition orders can differ by the last floating-point
        # bit, so canonicalize before sorting otherwise degenerate-energy
        # rows may be paired in a different order.
        table = np.round(np.column_stack(
            (energy, amplitude.T, flags.astype(float))), 13)
        keys = tuple(table[:, i]
                     for i in range(table.shape[1] - 1, -1, -1))
        return table[np.lexsort(keys)]

    assert len(screened[0]) == len(exact[0])
    assert np.allclose(
        sorted_table(screened), sorted_table(exact),
        rtol=1e-13, atol=1e-15)

    import pymbxas.mbxas.shakeup as shakeup_module
    monkeypatch.setattr(shakeup_module, "_QE_BUILD_BATCH", 2)
    chunked = mbxas_sticks_by_order(
        base_energy, base_amp, K, eps_occ, eps_virt, 2,
        screen_tol=1e-14, **kwargs)[2]
    assert np.array_equal(chunked[0], screened[0])
    assert np.array_equal(chunked[1], screened[1])
    assert np.array_equal(chunked[2], screened[2])


def test_qe_screened_mb3_prunes_and_guards():
    rng = np.random.default_rng(41)
    n_occ, n_virt = 5, 10
    base_energy = np.linspace(9.0, 15.0, n_virt)
    base_amp = rng.normal(size=(3, n_virt))
    K = 0.15 * rng.normal(size=(n_virt, n_occ))
    eps_occ = np.linspace(-0.8, -0.1, n_occ)
    eps_virt = np.linspace(-0.3, 0.9, n_virt)
    kwargs = dict(
        pair_energy_max=1.5, final_energy_range=(9.0, 15.0),
        determinant=0.9, max_configurations=10_000)

    exact = mbxas_sticks_by_order(
        base_energy, base_amp, K, eps_occ, eps_virt, 2,
        screen_tol=None, **kwargs)[2]
    screened = mbxas_sticks_by_order(
        base_energy, base_amp, K, eps_occ, eps_virt, 2,
        screen_tol=0.1, **kwargs)[2]
    assert 0 < len(screened[0]) < len(exact[0])
    assert np.all((screened[0] >= 9.0) & (screened[0] <= 15.0))
    assert screened[1].shape == (3, len(screened[0]))

    with pytest.raises(ValueError, match="exceeding max_configurations"):
        mbxas_sticks_by_order(
            base_energy, base_amp, K, eps_occ, eps_virt, 2,
            screen_tol=0.1, pair_energy_max=1.5,
            final_energy_range=(9.0, 15.0), determinant=0.9,
            max_configurations=5)


def test_broadening_large_stick_set_is_chunk_invariant():
    from pymbxas.mbxas.broaden import broadened_spectrum

    rng = np.random.default_rng(17)
    grid = np.linspace(-5.0, 5.0, 501)
    energies = rng.uniform(-4.0, 4.0, 7000)
    weights = rng.random(7000)

    combined = broadened_spectrum(grid, energies, weights, sigma=0.5)
    split = (
        broadened_spectrum(grid, energies[:3500], weights[:3500], sigma=0.5)
        + broadened_spectrum(grid, energies[3500:], weights[3500:], sigma=0.5)
    )
    assert np.allclose(combined, split, rtol=1e-13, atol=1e-12)


def test_spin_combination_uses_complete_final_photon_energy():
    from pymbxas.spectra import _combine_spin_stick_block

    xas_energy = np.array([10.0, 20.0])
    oscillator = np.array([2.0, 3.0])
    shift = np.array([0.0, 2.0, -1.0])
    probability = np.array([0.5, 0.25, 0.1])

    final_energy, intensity = _combine_spin_stick_block(
        xas_energy, oscillator, shift, probability)
    expected_energy = xas_energy[:, None] + shift[None, :]
    expected_intensity = (
        expected_energy * oscillator[:, None] * probability[None, :])

    assert np.array_equal(final_energy, expected_energy)
    assert np.array_equal(intensity, expected_intensity)
    # A zero spectator shift reduces exactly to the ordinary XAS prefactor.
    assert np.array_equal(
        intensity[:, 0], xas_energy * oscillator * probability[0])
    # Nonzero spectator shifts must not retain the pre-convolution XAS energy.
    old_weighting = (
        xas_energy[:, None] * oscillator[:, None] * probability[None, :])
    assert not np.array_equal(intensity[:, 1:], old_weighting[:, 1:])
