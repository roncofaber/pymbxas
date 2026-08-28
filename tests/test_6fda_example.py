import importlib.util
from pathlib import Path
from types import SimpleNamespace

import ase
import numpy as np
import pytest


def _load_example():
    path = Path(__file__).parents[1] / "examples" / "6fda_shakeup_compare.py"
    spec = importlib.util.spec_from_file_location("pymbxas_6fda_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_state_test_example():
    path = Path(__file__).parents[1] / "examples" / "6fda_o8_state_test.py"
    spec = importlib.util.spec_from_file_location("pymbxas_6fda_o8_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _decomposition(scale):
    energy = np.array([1.0, 2.0, 3.0])
    contributions = {
        1: scale * np.array([1.0, 2.0, 3.0]),
        2: scale * np.array([0.4, 0.5, 0.6]),
        3: scale * np.array([0.07, 0.08, 0.09]),
    }
    return {
        "energy": energy,
        "contributions": contributions,
        "decomposition": {
            2: {
                "shakeup": 0.75 * contributions[2],
                "shakedown": 0.25 * contributions[2],
            },
            3: {
                "shakeup": 0.8 * contributions[3],
                "shakedown": 0.2 * contributions[3],
            },
        },
        "cumulative": {
            order: sum(contributions[current]
                       for current in range(1, order + 1))
            for order in contributions},
        "total": sum(contributions.values()),
    }


def test_6fda_example_handles_arbitrary_f_order(tmp_path):
    example = _load_example()
    assert example.theory_settings(False, "PBE") == (
        "pbe", "def2-svpd", "pbe_def2_svpd")
    assert example.theory_settings(False, "B3LYP") == (
        "b3lyp", "def2-svpd", "b3lyp_def2_svpd")
    assert example.theory_settings(True, "b3lyp") == (
        "lda", "sto-3g", "lda_sto3g_quick")
    with pytest.raises(ValueError, match="xc must"):
        example.theory_settings(False, "not-a-functional")
    output_paths = example.prepare_output_directories(tmp_path)
    assert all(path.is_dir() for path in output_paths.values())
    assert output_paths["data"] == tmp_path / "outputs" / "data"
    decomposition = _decomposition(3.0)
    energy = decomposition["energy"]
    contributions = decomposition["contributions"]
    resolved = decomposition["decomposition"]
    total = decomposition["total"]
    assert set(contributions) == {1, 2, 3}
    assert set(resolved) == {2, 3}
    assert np.array_equal(total, sum(contributions.values()))

    table, columns = example.output_table(
        energy, contributions, resolved, total)
    assert columns == [
        "energy_eV", "f1", "f2_shakeup", "f2_shakedown", "f2",
        "f3_shakeup", "f3_shakedown", "f3", "total_f1_f2_f3",
    ]
    assert table.shape == (len(energy), len(columns))

    figure = tmp_path / "f3.png"
    example.save_plot(
        decomposition, figure, xc="pbe", basis="def2-svpd")
    assert figure.is_file() and figure.stat().st_size > 0

    import matplotlib.pyplot as plt
    plotted, axes = example.plot_mbxas_decomposition(
        decomposition, show_probability=False)
    plt.close(plotted)
    # The shared plotting helper remains data-driven; the fixed display window
    # belongs specifically to this comparison example.
    assert axes[0].get_xlim() == (energy[0], energy[-1])

    captured = {}
    original = example.plot_mbxas_decomposition

    def recording_plot(*args, **kwargs):
        plotted, axes = original(*args, **kwargs)
        captured["axes"] = axes
        return plotted, axes

    example.plot_mbxas_decomposition = recording_plot
    example.save_plot(
        decomposition, tmp_path / "windowed.png",
        xc="pbe", basis="def2-svpd")
    assert captured["axes"][0].get_xlim() == example.SPECTRUM_XLIM


def test_6fda_example_saves_one_shared_orbital_plot_per_site(tmp_path):
    example = _load_example()

    class FakeSpectra:
        def __init__(self, exc_idx):
            self.exc_idx = exc_idx
            self.request = None

        def plot_orbital_rearrangement(self, **kwargs):
            import matplotlib.pyplot as plt
            self.request = kwargs
            return plt.subplots()

    sites = [FakeSpectra(8), FakeSpectra(2)]
    paths = example.save_orbital_plots(sites, tmp_path, "test_f1_f2")

    assert [path.name for path in paths] == [
        "test_f1_f2_O02_orbitals.png", "test_f1_f2_O08_orbitals.png"]
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
    assert sites[0].request["energy_window"] == (
        example.ORBITAL_ENERGY_WINDOW)
    assert sites[0].request["min_overlap"] == example.ORBITAL_MIN_OVERLAP
    assert sites[0].request["show_dos"] is True
    assert sites[0].request["dos_sigma"] == example.ORBITAL_DOS_SIGMA


def test_6fda_example_rejects_a_checkpoint_from_another_geometry():
    example = _load_example()
    reference = ase.Atoms("HO", positions=[(0, 0, 0), (1, 0, 0)])
    translated = reference.copy()
    translated.translate((3, 2, 1))
    example.validate_checkpoint_geometry(
        SimpleNamespace(structure=translated), reference)

    distorted = reference.copy()
    distorted.positions[1, 0] += 0.01
    with pytest.raises(RuntimeError, match="Checkpoint geometry differs"):
        example.validate_checkpoint_geometry(
            SimpleNamespace(structure=distorted), reference)


def test_o8_state_test_uses_stable_spin_resolved_determinant_overlap():
    example = _load_state_test_example()
    overlap = np.eye(4)
    left = (np.eye(4)[:, :2], np.eye(4)[:, 2:])
    angle = 0.31
    rotation = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])
    right = (left[0] @ rotation, left[1] @ rotation)

    identical_subspace = example.determinant_overlap(left, right, overlap)
    np.testing.assert_allclose(
        identical_subspace["abs_determinant"], 1.0, atol=1e-14)
    assert identical_subspace["minimum_singular_value"] > 1 - 1e-14

    changed = (np.eye(4)[:, [0, 2]], left[1])
    different_subspace = example.determinant_overlap(left, changed, overlap)
    assert different_subspace["abs_determinant"] < 1e-300
    assert different_subspace["minimum_singular_value"] == 0.0


def test_o8_state_test_rejects_a_different_internal_geometry():
    example = _load_state_test_example()
    reference = ase.Atoms("HOH", positions=[(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    translated = reference.copy()
    translated.translate((5, -2, 3))
    example.validate_geometry(
        SimpleNamespace(structure=translated), reference, "translated")

    distorted = reference.copy()
    distorted.positions[1, 0] += 0.02
    with pytest.raises(RuntimeError, match="geometry differs"):
        example.validate_geometry(
            SimpleNamespace(structure=distorted), reference, "distorted")
