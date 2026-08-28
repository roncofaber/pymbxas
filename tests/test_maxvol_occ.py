import itertools
import sys

import numpy as np
import pytest

import pymbxas.calculators.maxvol as maxvol_module
from pymbxas.calculators.maxvol import (
    apply_occupation_method, maxvol_occ, maxvol_select, mixed_occ,
    normalize_maxvol_warmup_calls, normalize_occupation_method,
)


COMPLEX_OVERLAP = np.array([
    [0.20 + 0.00j, 0.00 + 0.00j],
    [0.00 + 0.00j, 0.30 + 0.00j],
    [1.00 + 0.00j, 0.00 + 0.10j],
    [0.00 + 0.20j, 1.00 + 0.00j],
])


def _exact_maximum_determinant(matrix):
    determinants = {
        rows: abs(np.linalg.det(matrix[np.asarray(rows)]))
        for rows in itertools.combinations(range(matrix.shape[0]), matrix.shape[1])
    }
    return max(determinants.values())


def test_maxvol_select_matches_small_exhaustive_complex_search():
    result = maxvol_select(
        COMPLEX_OVERLAP, initial_rows=[0, 1], tol=1e-13)

    assert result.determinant == pytest.approx(
        _exact_maximum_determinant(COMPLEX_OVERLAP), rel=1e-13)
    assert set(result.pivots) == {2, 3}
    assert result.iterations == 2
    assert result.max_coefficient <= 1 + 1e-13

    selected = COMPLEX_OVERLAP[result.pivots]
    coefficients = np.linalg.solve(selected.T, COMPLEX_OVERLAP.T).T
    coefficients[result.pivots] = 0
    assert np.max(abs(coefficients)) <= 1 + 1e-13


def test_maxvol_select_recovers_from_singular_requested_seed():
    matrix = COMPLEX_OVERLAP.copy()
    matrix[1] = 2 * matrix[0]
    result = maxvol_select(matrix, initial_rows=[0, 1], tol=1e-13)
    assert result.determinant > 1.0
    assert len(np.unique(result.pivots)) == matrix.shape[1]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"matrix": np.ones(3)}, "two-dimensional"),
        ({"matrix": np.ones((1, 2))}, "tall or square"),
        ({"matrix": np.ones((3, 2)), "initial_rows": [0, 0]}, "distinct"),
        ({"matrix": np.ones((3, 2)), "tol": -1}, "non-negative"),
        ({"matrix": np.ones((3, 2)), "max_iter": 0}, "positive"),
    ],
)
def test_maxvol_select_validates_inputs(kwargs, match):
    with pytest.raises(ValueError, match=match):
        maxvol_select(**kwargs)


class _FakeUHF:
    def __init__(self, overlap, nelec):
        self._overlap = overlap
        self.nelec = nelec
        self.stdout = sys.stdout
        self.verbose = 0
        self.mo_energy = None
        self.mo_coeff = None

    def istype(self, name):
        return name == "UHF"

    def get_ovlp(self):
        return self._overlap


def _occupation_fixture(array_module):
    reference = array_module.eye(4)
    permutation = [3, 2, 0, 1]
    current = reference[:, permutation]
    reference_coeff = array_module.stack((reference, reference))
    target_occ = array_module.asarray([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ])
    mo_coeff = array_module.stack((current, current))
    mo_energy = array_module.zeros((2, 4))
    return reference_coeff, target_occ, mo_coeff, mo_energy


def test_maxvol_occ_tracks_unrestricted_reference_subspaces():
    reference, target_occ, current, energies = _occupation_fixture(np)
    mf = _FakeUHF(np.eye(4), nelec=(2, 1))
    maxvol_occ(mf, reference, target_occ, tol=1e-13)

    occupation = mf.get_occ(energies, current)

    assert np.array_equal(np.flatnonzero(occupation[0]), [1, 2])
    assert np.array_equal(np.flatnonzero(occupation[1]), [3])
    assert len(mf.get_occ.maxvol_history) == 1
    for diagnostic in mf.get_occ.maxvol_history[0]:
        assert diagnostic["determinant"] == pytest.approx(1.0)
        assert diagnostic["occupation_changes"] == 0
        assert diagnostic["elapsed_seconds"] >= 0
    assert len(mf.get_occ.maxvol_call_times) == 1
    assert mf.get_occ.maxvol_call_times[0] >= sum(
        diagnostic["elapsed_seconds"]
        for diagnostic in mf.get_occ.maxvol_history[0])

    mf.get_occ(energies, current)
    assert len(mf.get_occ.maxvol_history) == 2
    assert all(diagnostic["occupation_changes"] == 0
               for diagnostic in mf.get_occ.maxvol_history[1])


def test_occupation_method_normalization_and_dispatch(monkeypatch):
    assert normalize_occupation_method(" MOM ") == "mom"
    assert normalize_occupation_method("MAXVOL") == "maxvol"
    assert normalize_occupation_method(" MIXED ") == "mixed"
    with pytest.raises(ValueError, match="occupation_method"):
        normalize_occupation_method("imom")

    calls = []
    monkeypatch.setattr(
        maxvol_module, "mom_occ",
        lambda mf, coeff, occ: calls.append("mom") or mf)
    monkeypatch.setattr(
        maxvol_module, "maxvol_occ",
        lambda mf, coeff, occ: calls.append("maxvol") or mf)
    monkeypatch.setattr(
        maxvol_module, "mixed_occ",
        lambda mf, coeff, occ, warmup_calls: calls.append(
            ("mixed", warmup_calls)) or mf)
    marker = object()
    assert apply_occupation_method(marker, None, None, "mom") is marker
    assert apply_occupation_method(marker, None, None, "maxvol") is marker
    assert apply_occupation_method(
        marker, None, None, "mixed", maxvol_warmup_calls=4) is marker
    assert calls == ["mom", "maxvol", ("mixed", 4)]


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "2"])
def test_maxvol_warmup_calls_validation(value):
    with pytest.raises(ValueError, match="positive integer"):
        normalize_maxvol_warmup_calls(value)


def test_mixed_occ_hands_off_from_mom_to_fixed_reference(monkeypatch):
    mf = _FakeUHF(np.eye(4), nelec=(2, 1))
    calls = []

    def fake_mom(obj, reference, occupation):
        def get_occ(mo_energy=None, mo_coeff=None):
            calls.append("mom")
            return np.full((2, 4), 1.0)
        obj.get_occ = get_occ
        return obj

    def fake_maxvol(obj, reference, occupation, tol=0.01, max_iter=100):
        def get_occ(mo_energy=None, mo_coeff=None):
            calls.append("maxvol")
            return np.zeros((2, 4))
        get_occ.maxvol_history = []
        get_occ.maxvol_call_times = []
        obj.get_occ = get_occ
        return obj

    monkeypatch.setattr(maxvol_module, "mom_occ", fake_mom)
    monkeypatch.setattr(maxvol_module, "maxvol_occ_", fake_maxvol)
    mixed_occ(mf, np.ones((2, 4, 4)), np.ones((2, 4)), warmup_calls=2)

    assert np.all(mf.get_occ() == 1)
    assert np.all(mf.get_occ() == 1)
    assert np.all(mf.get_occ() == 0)
    assert calls == ["mom", "mom", "maxvol"]
    assert mf.get_occ.mixed_phases == ["mom", "mom", "maxvol"]
    assert mf.get_occ.maxvol_warmup_calls == 2


def _run_maxvol_pymbxas(tmp_path, gpu, occupation_method="maxvol"):
    import ase.build
    from pymbxas.calculators.pyscf import PySCFMBXAS
    from pymbxas.config import (
        CalculationConfig, CheckpointConfig, LoggingConfig, RuntimeConfig,
        SCFConfig,
    )

    obj = PySCFMBXAS(
        ase.build.molecule("H2O"),
        config=CalculationConfig(
            xc="lda", basis="sto-3g",
            ground_state_scf=SCFConfig(
                max_cycles=50, convergence_tolerance=1e-8)),
        runtime=RuntimeConfig(
            work_directory=tmp_path, device="gpu" if gpu else "cpu",
            logging=LoggingConfig(
                pymbxas_verbosity=3,
                pymbxas_logfile=f"{occupation_method}.log",
                pyscf_verbosity=3, pyscf_console=False),
            checkpoint=CheckpointConfig(enabled=False)),
    )
    obj.run("O", occupation=occupation_method, mom_warmup_calls=2,
            scf=SCFConfig(max_cycles=50, convergence_tolerance=1e-8))
    assert obj.excitations[0].config.occupation.value == occupation_method
    assert obj.excitations[0].config.mom_warmup_calls == 2
    assert len(obj.excitations) == 1
    assert set(obj.excitations[0].data) == {"fch", "xch"}
    # Per-call details are DEBUG-level raw output; normal verbosity gets one
    # aggregate high-level record for each constrained SCF instead.
    assert "MAXVOL call" not in obj.excitations[0].output["fch"]
    assert "MAXVOL call" not in obj.excitations[0].output["xch"]
    assert "MIXED call" not in obj.excitations[0].output["fch"]
    app_log = (tmp_path / f"{occupation_method}.log").read_text()
    assert app_log.count("Occupation tracking") == 2
    assert "[O:0 FCH] Converged" in app_log
    assert "[O:0 XCH] Converged" in app_log
    assert "\n\tcycles" in app_log
    assert "\n\tspin square" in app_log
    assert "\n\toccupied determinant" in app_log
    if occupation_method == "mixed":
        assert "MOM warm-up calls" in app_log
    for label in ("fch", "xch"):
        data = obj.excitations[0].data[label]
        assert tuple(np.count_nonzero(data.to_cpu().mo_occ == 1, axis=1)) == data.nelec
    return obj


def test_maxvol_pymbxas_cpu_and_persistence(tmp_path):
    from pymbxas.calculators.pyscf import PySCFMBXAS

    obj = _run_maxvol_pymbxas(tmp_path, gpu=False)
    path = obj.save(tmp_path / "maxvol.h5")
    restored = PySCFMBXAS.load(path)
    assert restored.excitations[0].config.occupation.value == "maxvol"


def test_mixed_pymbxas_cpu_and_persistence(tmp_path):
    from pymbxas.calculators.pyscf import PySCFMBXAS

    obj = _run_maxvol_pymbxas(tmp_path, gpu=False, occupation_method="mixed")
    path = obj.save(tmp_path / "mixed.h5")
    restored = PySCFMBXAS.load(path)
    assert restored.excitations[0].config.occupation.value == "mixed"
    assert restored.excitations[0].config.mom_warmup_calls == 2


def test_maxvol_gpu_selector_and_occupation_parity():
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception as error:
        pytest.skip(f"CUDA unavailable: {error}")

    cpu_result = maxvol_select(
        COMPLEX_OVERLAP, initial_rows=[0, 1], tol=1e-13)
    gpu_result = maxvol_select(
        cupy.asarray(COMPLEX_OVERLAP), initial_rows=[0, 1], tol=1e-13)
    assert np.array_equal(gpu_result.pivots, cpu_result.pivots)
    assert gpu_result.determinant == pytest.approx(cpu_result.determinant, rel=1e-13)

    reference, target_occ, current, energies = _occupation_fixture(cupy)
    mf = _FakeUHF(cupy.eye(4), nelec=(2, 1))
    maxvol_occ(mf, reference, target_occ, tol=1e-13)
    occupation = mf.get_occ(energies, current)

    assert isinstance(occupation, cupy.ndarray)
    assert np.array_equal(cupy.flatnonzero(occupation[0]).get(), [1, 2])
    assert np.array_equal(cupy.flatnonzero(occupation[1]).get(), [3])


@pytest.mark.parametrize("occupation_method", ["maxvol", "mixed"])
def test_maxvol_pymbxas_gpu(tmp_path, occupation_method):
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception as error:
        pytest.skip(f"CUDA unavailable: {error}")
    _run_maxvol_pymbxas(
        tmp_path, gpu=True, occupation_method=occupation_method)


def test_adaptive_constrained_solver_gpu(tmp_path):
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception as error:
        pytest.skip(f"CUDA unavailable: {error}")

    import ase.build
    from pymbxas.calculators.pyscf import PySCFMBXAS
    from pymbxas.config import (
        CalculationConfig, CheckpointConfig, LoggingConfig, RuntimeConfig,
        SCFConfig,
    )

    obj = PySCFMBXAS(
        ase.build.molecule("H2O"),
        config=CalculationConfig(
            xc="lda", basis="sto-3g",
            ground_state_scf=SCFConfig(
                max_cycles=30, convergence_tolerance=1e-8)),
        runtime=RuntimeConfig(
            work_directory=tmp_path, device="gpu",
            logging=LoggingConfig(
                pymbxas_verbosity=3,
                pymbxas_logfile="adaptive-gpu.log",
                pyscf_verbosity=0, pyscf_console=False),
            checkpoint=CheckpointConfig(enabled=False)))
    obj.run(
        "O", xch=False, occupation="mixed",
        scf=SCFConfig(
            max_cycles=30, convergence_tolerance=1e-8,
            diis_cycles=1, mixing_cycles=1, second_order=True))

    assert len(obj.excitations) == 1
    log = (tmp_path / "adaptive-gpu.log").read_text()
    assert "solver path" in log
    assert "DIIS -> stabilized DIIS -> second-order SCF" in log
