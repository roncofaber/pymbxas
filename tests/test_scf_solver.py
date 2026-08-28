import logging
from types import SimpleNamespace

import numpy as np
import pytest

from pymbxas.calculators.scf import (
    normalize_scf_recovery_settings, run_constrained_scf,
)
from pymbxas.calculators.excitation import _fch_state_diagnostics


class _FakeSCF:
    def __init__(self, converge_first=False):
        self.callback = None
        self.converge_first = converge_first
        self.kernel_calls = []
        self.newton_calls = 0
        self.converged = False
        self.cycles = 0
        self.damp = 0.0
        self.level_shift = 0.0
        self.diis_start_cycle = 1
        self.mo_coeff = np.zeros((2, 2, 2))
        self.mo_occ = np.ones((2, 2))

    def kernel(self, dm0=None):
        call = len(self.kernel_calls) + 1
        self.kernel_calls.append(np.asarray(dm0).copy())
        self.cycles = self.max_cycle
        self.converged = self.converge_first and call == 1
        coeff = np.full((2, 2, 2), float(call))
        occ = np.ones((2, 2))
        self.mo_coeff, self.mo_occ = coeff, occ
        self.callback({
            "norm_gorb": 1.0 / call,
            "mo_coeff": coeff,
            "mo_occ": occ,
        })

    def make_rdm1(self, mo_coeff=None, mo_occ=None):
        coeff = self.mo_coeff if mo_coeff is None else mo_coeff
        return np.asarray(coeff).copy()

    def newton(self):
        self.newton_calls += 1
        return _FakeNewton(self)


class _FakeNewton:
    def __init__(self, parent):
        self.parent = parent
        self.callback = None
        self.converged = False
        self.cycles = 0
        self.inputs = None

    def kernel(self, mo_coeff=None, mo_occ=None):
        self.inputs = (np.asarray(mo_coeff).copy(), np.asarray(mo_occ).copy())
        self.cycles = 4
        self.converged = True
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.callback({
            "imacro": self.cycles - 1,
            "norm_gorb": 0.01,
            "mo_coeff": mo_coeff,
            "mo_occ": mo_occ,
        })


def test_adaptive_scf_uses_stabilized_diis_then_second_order():
    calculator = _FakeSCF()
    result = run_constrained_scf(
        calculator, np.zeros((2, 2, 2)), logging.getLogger("test"),
        max_cycle=100, diis_cycles=20, mixing_cycles=10,
        damping=0.25, level_shift=0.15, second_order=True)

    assert result.converged
    assert [item.solver for item in result._pymbxas_scf_attempts] == [
        "DIIS", "stabilized DIIS", "second-order SCF"]
    assert [item.cycles for item in result._pymbxas_scf_attempts] == [20, 10, 4]
    assert calculator.damp == pytest.approx(0.25)
    assert calculator.level_shift == pytest.approx(0.15)
    assert calculator.diis_start_cycle == 3
    assert np.all(calculator.kernel_calls[1] == 1.0)
    assert np.all(result.inputs[0] == 2.0)
    assert result.max_cycle == 70


def test_adaptive_scf_leaves_easy_convergence_on_original_path():
    calculator = _FakeSCF(converge_first=True)
    result = run_constrained_scf(
        calculator, np.zeros((2, 2, 2)), logging.getLogger("test"),
        max_cycle=100)

    assert result is calculator
    assert calculator.newton_calls == 0
    assert calculator.damp == 0.0
    assert calculator.level_shift == 0.0
    assert [item.solver for item in result._pymbxas_scf_attempts] == ["DIIS"]


def test_adaptive_scf_does_not_bypass_constraint_by_default():
    calculator = _FakeSCF()
    result = run_constrained_scf(
        calculator, np.zeros((2, 2, 2)), logging.getLogger("test"),
        max_cycle=100, diis_cycles=20, mixing_cycles=10)

    assert result is calculator
    assert not result.converged
    assert calculator.newton_calls == 0
    assert [item.solver for item in result._pymbxas_scf_attempts] == [
        "DIIS", "stabilized DIIS"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_cycle": 0},
        {"max_cycle": 10, "diis_cycles": 0},
        {"max_cycle": 10, "mixing_cycles": -1},
        {"max_cycle": 10, "damping": -0.1},
        {"max_cycle": 10, "level_shift": -0.1},
    ],
)
def test_adaptive_scf_validates_settings(kwargs):
    with pytest.raises(ValueError):
        run_constrained_scf(
            _FakeSCF(), np.zeros((2, 2, 2)), logging.getLogger("test"),
            **kwargs)


def test_recovery_setting_normalization():
    assert normalize_scf_recovery_settings(40, 20, 0.3, 0.1) == {
        "scf_diis_cycles": 40,
        "scf_mixing_cycles": 20,
        "scf_damping": 0.3,
        "scf_level_shift": 0.1,
        "scf_second_order": False,
    }
    for args in ((True, 20, 0.2, 0.2), (40, -1, 0.2, 0.2),
                 (40, 20, True, 0.2), (40, 20, 0.2, -0.1)):
        with pytest.raises(ValueError):
            normalize_scf_recovery_settings(*args)
    with pytest.raises(ValueError, match="scf_second_order"):
        normalize_scf_recovery_settings(40, 20, 0.2, 0.2, "yes")


def test_fch_state_diagnostics_recognizes_a_pure_target_doublet():
    coefficients = np.stack((np.eye(4), np.eye(4)))
    occupations = np.array([
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ])
    calculator = SimpleNamespace(
        mo_coeff=coefficients, mo_occ=occupations,
        mol=SimpleNamespace(spin=1))

    diagnostics = _fch_state_diagnostics(
        calculator, coefficients, occupations, np.eye(4))

    assert diagnostics["spin_square"] == pytest.approx(0.75)
    assert diagnostics["spin_contamination"] == pytest.approx(0.0)
    assert diagnostics["occupied_determinant"] == pytest.approx(1.0)
    assert diagnostics["minimum_singular_value"] == pytest.approx(1.0)
