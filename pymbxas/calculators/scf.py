"""SCF execution helpers for constrained core-excited calculations."""

from dataclasses import dataclass

from pymbxas.io.config import format_log_fields


@dataclass(frozen=True)
class SCFAttempt:
    """One stage in an adaptive constrained-SCF calculation."""

    solver: str
    cycles: int
    converged: bool
    best_gradient: float


class _BestOrbitalState:
    """Retain the lowest-gradient orbitals without moving GPU arrays to CPU."""

    def __init__(self, previous_callback=None):
        self.previous_callback = previous_callback
        self.gradient = float("inf")
        self.mo_coeff = None
        self.mo_occ = None
        self.macro_cycles = 0

    def __call__(self, envs):
        gradient = envs.get("norm_gorb")
        mo_coeff = envs.get("mo_coeff")
        mo_occ = envs.get("mo_occ")
        if gradient is not None and mo_coeff is not None and mo_occ is not None:
            value = float(gradient.item() if hasattr(gradient, "item") else gradient)
            if value < self.gradient:
                self.gradient = value
                self.mo_coeff = mo_coeff.copy()
                self.mo_occ = mo_occ.copy()
        if "imacro" in envs:
            self.macro_cycles = max(
                self.macro_cycles, int(envs["imacro"]) + 1)
        if callable(self.previous_callback):
            self.previous_callback(envs)

    def density(self, calculator):
        if self.mo_coeff is None:
            return calculator.make_rdm1()
        return calculator.make_rdm1(self.mo_coeff, self.mo_occ)


def normalize_scf_recovery_settings(diis_cycles=50, mixing_cycles=30,
                                    damping=0.2, level_shift=0.2,
                                    second_order=False):
    """Validate and normalize the public adaptive-SCF controls."""
    for name, value, allow_zero in (
            ("scf_diis_cycles", diis_cycles, False),
            ("scf_mixing_cycles", mixing_cycles, True)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if value < (0 if allow_zero else 1):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be {qualifier}")
    for name, value in (
            ("scf_damping", damping), ("scf_level_shift", level_shift)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a non-negative number")
        if value < 0:
            raise ValueError(f"{name} must be a non-negative number")
    if not isinstance(second_order, bool):
        raise ValueError("scf_second_order must be a boolean")
    return {
        "scf_diis_cycles": int(diis_cycles),
        "scf_mixing_cycles": int(mixing_cycles),
        "scf_damping": float(damping),
        "scf_level_shift": float(level_shift),
        "scf_second_order": second_order,
    }


def _record_attempt(calculator, solver, tracker, cycles=None):
    return SCFAttempt(
        solver=solver,
        cycles=(int(getattr(calculator, "cycles", 0))
                if cycles is None else int(cycles)),
        converged=bool(calculator.converged),
        best_gradient=tracker.gradient,
    )


def _log_retry(log, attempt, next_solver, **settings):
    details = {
        "solver": attempt.solver,
        "cycles": attempt.cycles,
        "best orbital gradient": f"{attempt.best_gradient:.3g}",
        "next solver": next_solver,
    }
    details.update(settings)
    log.warning("SCF stage did not converge\n%s", format_log_fields(details))


def _log_exhausted(log, attempt):
    log.warning(
        "Constrained SCF recovery exhausted\n%s",
        format_log_fields({
            "solver": attempt.solver,
            "cycles": attempt.cycles,
            "best orbital gradient": f"{attempt.best_gradient:.3g}",
            "second-order SCF": "disabled to preserve state tracking",
        }))


def run_constrained_scf(calculator, dm0, log, *, max_cycle,
                        diis_cycles=50, mixing_cycles=30,
                        damping=0.2, level_shift=0.2,
                        second_order=False):
    """Run constrained SCF with DIIS, stabilized mixing, then SOSCF.

    All stages retain the occupation pattern installed on ``calculator``.
    Failed stages restart from the lowest-gradient orbitals observed so far.
    The total number of conventional and second-order macro iterations never
    exceeds ``max_cycle``.
    """
    max_cycle = int(max_cycle)
    diis_cycles = int(diis_cycles)
    mixing_cycles = int(mixing_cycles)
    if max_cycle < 1 or diis_cycles < 1 or mixing_cycles < 0:
        raise ValueError("SCF cycle budgets must be positive (mixing may be zero)")
    if damping < 0 or level_shift < 0:
        raise ValueError("SCF damping and level shift must be non-negative")
    if not isinstance(second_order, bool):
        raise ValueError("second_order must be a boolean")

    tracker = _BestOrbitalState(getattr(calculator, "callback", None))
    calculator.callback = tracker
    attempts = []
    used_cycles = 0

    # Stage 1: ordinary DIIS. Easy sites retain their historical path.
    calculator.max_cycle = min(diis_cycles, max_cycle)
    calculator.kernel(dm0=dm0)
    attempt = _record_attempt(calculator, "DIIS", tracker)
    attempts.append(attempt)
    used_cycles += attempt.cycles
    if calculator.converged or used_cycles >= max_cycle:
        calculator._pymbxas_scf_attempts = tuple(attempts)
        return calculator

    # Stage 2: delay DIIS briefly so damping acts, and shift the virtual block.
    stabilized_cycles = min(mixing_cycles, max_cycle - used_cycles)
    if stabilized_cycles:
        _log_retry(
            log, attempt, "stabilized DIIS",
            damping=damping, level_shift=f"{level_shift:.3g} Ha")
        calculator.damp = float(damping)
        calculator.level_shift = float(level_shift)
        calculator.diis_start_cycle = max(3, int(calculator.diis_start_cycle))
        calculator.max_cycle = stabilized_cycles
        calculator.kernel(dm0=tracker.density(calculator))
        attempt = _record_attempt(calculator, "stabilized DIIS", tracker)
        attempts.append(attempt)
        used_cycles += attempt.cycles
        if calculator.converged or used_cycles >= max_cycle:
            calculator._pymbxas_scf_attempts = tuple(attempts)
            return calculator

    # Stage 3: second-order orbital optimization is effective near a DIIS
    # plateau, but its continuous orbital rotations do not reapply MOM/maxvol.
    # It is therefore an explicit diagnostic opt-in for constrained states.
    if not second_order:
        _log_exhausted(log, attempt)
        calculator._pymbxas_scf_attempts = tuple(attempts)
        return calculator

    _log_retry(log, attempt, "second-order SCF")
    second_order = calculator.newton()
    second_order.callback = tracker
    second_order.max_cycle = max_cycle - used_cycles
    tracker.macro_cycles = 0
    second_order.kernel(mo_coeff=tracker.mo_coeff, mo_occ=tracker.mo_occ)
    attempts.append(_record_attempt(
        second_order, "second-order SCF", tracker,
        cycles=tracker.macro_cycles))
    second_order._pymbxas_scf_attempts = tuple(attempts)
    return second_order


__all__ = [
    "SCFAttempt", "normalize_scf_recovery_settings", "run_constrained_scf",
]
