import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pymbxas.plotting import plot_orbital_rearrangement
from pymbxas.spectra import Spectra
from pymbxas.spectras import Spectras
from pymbxas.utils.orbitals import match_orbitals_by_overlap


def _site_decomposition(scale, probability_range=(-1.0, 1.0)):
    energy = np.linspace(10.0, 12.0, 5)
    f1 = scale * np.ones(5)
    f2 = scale * np.linspace(0.0, 1.0, 5)
    total = f1 + f2
    probability_energy = np.linspace(*probability_range, 5)
    return {
        "energy": energy,
        "contributions": {1: f1, 2: f2},
        "decomposition": {
            2: {"shakeup": 0.75 * f2, "shakedown": 0.25 * f2}},
        "cumulative": {1: f1, 2: total},
        "total": total,
        "integrated": {},
        "probability": (
            probability_energy,
            scale * np.ones_like(probability_energy), [0, 1]),
        "overlap": {
            "by_total_order": {0: scale, 1: 0.5 * scale},
            "captured": 1.5 * scale,
            "available": 2.0 * scale,
            "fraction": 0.75,
        },
        "shakedown_fraction": 0.1 * scale,
    }


def _collection():
    collection = Spectras.__new__(Spectras)
    sites = []
    for scale, probability_range in ((1.0, (-1.0, 1.0)),
                                     (3.0, (-2.0, 2.0))):
        site = Spectra.__new__(Spectra)
        data = _site_decomposition(scale, probability_range)
        site.get_mbxas_decomposition = lambda data=data, **kwargs: data
        sites.append(site)
    collection.spectras = sites
    collection.labels = [0, 1]
    collection._erange = [10.0, 12.0]
    return collection


def _orbital_spectra():
    spectra = Spectra.__new__(Spectra)
    spectra._channel = 1
    spectra._core_orb_idx = 0
    spectra._exc_idx = None
    spectra._gs_mo_energy = np.array([
        [-20.0, -1.0, 0.5, 2.0],
        [-20.0, -1.0, 0.5, 2.0],
    ])
    spectra._fch_mo_energy = np.array([
        [-19.8, -0.9, 0.6, 2.1],
        [-19.0, -0.8, 0.4, 2.2],
    ])
    spectra._gs_mo_occ = np.array([[1, 1, 0, 0], [1, 1, 0, 0]])
    spectra._mo_occ = np.array([[1, 1, 0, 0], [0, 1, 1, 0]])
    spectra._mb_overlap = np.array([np.eye(4), np.eye(4)])
    return spectra


def test_collection_decomposition_supports_sum_and_mean():
    collection = _collection()
    summed = collection.get_mbxas_decomposition(
        f_order=2, npoints=7, average=False)
    mean = collection.get_mbxas_decomposition(
        f_order=2, npoints=7, average=True)

    assert summed["site_count"] == 2
    assert summed["aggregation"] == "sum"
    assert np.allclose(summed["contributions"][1], 4.0)
    assert np.allclose(mean["contributions"][1], 2.0)
    assert summed["probability"][0][[0, -1]].tolist() == [-2.0, 2.0]
    assert len(summed["overlap"]["per_site"]) == 2
    assert summed["overlap"]["captured"] == pytest.approx(6.0)
    # Captured-overlap weighted: (0.1*1.5 + 0.3*4.5) / 6.
    assert summed["shakedown_fraction"] == pytest.approx(0.25)


def test_collection_plot_method_forwards_aggregation():
    collection = _collection()
    figure, axes = collection.plot_mbxas_decomposition(
        f_order=2, average=False, npoints=7, show_probability=False)
    assert len(axes) == 1
    assert axes[0].lines[0].get_label() == "Total through f2"
    plt.close(figure)


def test_orbital_matching_is_globally_one_to_one():
    overlap = np.array([[0.90, 0.89], [0.88, 0.10]])
    gs, fch, weights = match_orbitals_by_overlap(overlap)
    assert gs.tolist() == [0, 1]
    assert fch.tolist() == [1, 0]
    assert np.allclose(weights, [0.88**2, 0.89**2])


def test_orbital_rearrangement_tracks_frontiers_and_core_hole():
    spectra = _orbital_spectra()
    data = spectra.get_orbital_rearrangement(
        energy_window=None, min_overlap=0.5)
    excited = data["channels"][1]

    assert data["reference_label"] == "GS HOMO"
    assert excited["gs_homo"] == 1
    assert excited["gs_lumo"] == 2
    assert excited["fch_homo"] == 2
    # The constrained core hole is tracked separately and is not the
    # chemically useful FCH LUMO.
    assert excited["fch_lumo"] == 3
    assert excited["core_gs"] == 0
    assert excited["core_fch"] == 0
    assert excited["matches"]["gs_index"].tolist() == [0, 1, 2, 3]


def test_orbital_plot_has_two_spin_panels_and_occupancy_connectors():
    data = _orbital_spectra().get_orbital_rearrangement(
        energy_window=None, min_overlap=0.5)
    figure, axes = plot_orbital_rearrangement(data, show_indices=True)

    assert len(axes) == 2
    assert "spectator" in axes[0].get_title()
    assert "excited" in axes[1].get_title()
    assert len(axes[1].patches) == 0
    horizontal_levels = [
        line for line in axes[1].lines
        if len(line.get_ydata()) == 2
        and line.get_ydata()[0] == line.get_ydata()[1]
    ]
    assert len(horizontal_levels) == 8
    assert {line.get_color() for line in horizontal_levels} >= {
        "#303030", "#a8a8a8"}
    assert any(line.get_color() == "crimson" for line in axes[1].lines)
    plt.close(figure)


def test_orbital_dos_uses_existing_outer_margins():
    data = _orbital_spectra().get_orbital_rearrangement(
        energy_window=None, min_overlap=0.5)
    figure, axes = plot_orbital_rearrangement(
        data, show_dos=True, dos_sigma=0.2)

    assert axes[0].get_xlim() == (-0.55, 1.55)
    assert len(axes[0].collections) == 2
    density_lines = [
        line for line in axes[0].lines if len(line.get_xdata()) == 600]
    assert len(density_lines) == 2
    assert np.max(density_lines[0].get_xdata()) <= -0.225
    assert np.min(density_lines[1].get_xdata()) >= 1.225
    plt.close(figure)

    with pytest.raises(ValueError, match="dos_sigma must be positive"):
        plot_orbital_rearrangement(data, show_dos=True, dos_sigma=0)


def test_orbital_plot_uses_requested_energy_window_as_fixed_limits():
    data = _orbital_spectra().get_orbital_rearrangement(
        energy_window=(-30.0, 30.0), min_overlap=0.5)
    figure, axes = plot_orbital_rearrangement(data, show_dos=True)

    assert axes[0].get_ylim() == (-30.0, 30.0)
    assert axes[1].get_ylim() == (-30.0, 30.0)
    density_lines = [
        line for line in axes[0].lines if len(line.get_xdata()) == 600]
    assert density_lines[0].get_ydata()[[0, -1]].tolist() == [-30.0, 30.0]
    plt.close(figure)


def test_historical_spectra_explains_missing_ground_state_energies():
    spectra = _orbital_spectra()
    spectra._gs_mo_energy = None
    with pytest.raises(RuntimeError, match="historical Spectra"):
        spectra.get_orbital_rearrangement()
