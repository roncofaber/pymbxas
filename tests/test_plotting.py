import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pymbxas.plotting import plot_mbxas_decomposition
from pymbxas.spectra import Spectra


def _decomposition(order=3):
    energy = np.linspace(1.0, 4.0, 7)
    contributions = {
        current: np.full_like(energy, 1.0 / current)
        for current in range(1, order + 1)
    }
    cumulative = {}
    running = np.zeros_like(energy)
    for current, contribution in contributions.items():
        running = running + contribution
        cumulative[current] = running.copy()
    resolved = {
        current: {
            "shakeup": 0.8 * contributions[current],
            "shakedown": 0.2 * contributions[current],
        }
        for current in range(2, order + 1)
    }
    probability_energy = np.linspace(-1.0, 1.0, 5)
    return {
        "energy": energy,
        "contributions": contributions,
        "decomposition": resolved,
        "cumulative": cumulative,
        "total": cumulative[order],
        "probability": (
            probability_energy, np.ones_like(probability_energy),
            list(range(1, order))),
    }


def test_decomposition_plot_retains_default_two_panel_api():
    figure, axes = plot_mbxas_decomposition(_decomposition(order=2))
    assert len(axes) == 2
    assert [line.get_label() for line in axes[0].lines] == [
        "Total through f2", "f1 contribution", "f2 contribution"]
    plt.close(figure)


def test_decomposition_plot_supports_resolved_and_cumulative_panels():
    figure, axes = plot_mbxas_decomposition(
        _decomposition(order=3), show_probability=True,
        show_resolved=True, show_cumulative=True)
    assert len(axes) == 3
    assert "Cumulative through f2" in {
        line.get_label() for line in axes[0].lines}
    assert [line.get_label() for line in axes[1].lines] == [
        "f2 shake-up", "f2 shake-down",
        "f3 shake-up", "f3 shake-down",
    ]
    plt.close(figure)


def test_resolved_panel_requires_a_higher_order():
    with pytest.raises(ValueError, match="at least f2"):
        plot_mbxas_decomposition(
            _decomposition(order=1), show_probability=False,
            show_resolved=True)


def test_spectra_plot_method_forwards_scientific_and_plot_options():
    spectra = Spectra.__new__(Spectra)
    requested = {}

    def get_decomposition(**kwargs):
        requested.update(kwargs)
        return _decomposition(order=2)

    spectra.get_mbxas_decomposition = get_decomposition
    figure, axes = spectra.plot_mbxas_decomposition(
        f_order=2, sigma=0.7, show_probability=False,
        show_resolved=True, figsize=(6.0, 5.0))

    assert requested["f_order"] == 2
    assert requested["sigma"] == 0.7
    assert len(axes) == 2
    assert tuple(figure.get_size_inches()) == pytest.approx((6.0, 5.0))
    plt.close(figure)
