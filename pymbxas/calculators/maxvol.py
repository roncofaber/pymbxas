"""Maximum-volume occupation selection for PySCF SCF calculations.

This module is deliberately independent of the MBXAS spectral maxvol helper.
It provides the occupation-selection role played by ``reference_method =
'maxvol'`` in the Prendergast Quantum ESPRESSO fork: select the current
occupied subspace collectively by maximizing its overlap determinant with a
target occupied subspace.

The array operations stay on the backend of the supplied MO coefficients.
NumPy is used for CPU PySCF objects and CuPy for GPU4PySCF objects; CuPy is not
an import-time dependency.
"""

from dataclasses import dataclass
import time

import numpy as np
from pyscf.lib import logger
from pyscf.scf.addons import mom_occ


OCCUPATION_METHODS = ("mom", "maxvol", "mixed")


@dataclass(frozen=True)
class MaxvolResult:
    """Result of a locally maximum-volume row selection."""

    pivots: np.ndarray
    determinant: float
    log_determinant: float
    iterations: int
    max_coefficient: float


def _array_module(array):
    """Return NumPy or CuPy without importing CuPy on CPU-only systems."""
    if type(array).__module__.split(".", 1)[0] == "cupy":
        import cupy
        return cupy
    return np


def _host_array(array):
    """Copy a small backend array to the host."""
    return np.asarray(array.get() if hasattr(array, "get") else array)


def _rank_revealing_rows(matrix, xp, relative_tol):
    """Construct a nonsingular row seed by pivoted Gram--Schmidt."""
    nrow, ncol = matrix.shape
    residual = matrix.copy()
    pivots = []
    initial_norm = None

    for _ in range(ncol):
        norms = xp.sum(xp.abs(residual) ** 2, axis=1).real
        if pivots:
            norms[xp.asarray(pivots, dtype=int)] = -xp.inf
        row = int(xp.argmax(norms).item())
        norm = float(norms[row].item())
        if initial_norm is None:
            initial_norm = norm
        if not np.isfinite(norm) or norm <= relative_tol * initial_norm:
            raise np.linalg.LinAlgError(
                "Cannot construct a full-rank maxvol seed from the overlap matrix")

        pivots.append(row)
        vector = residual[row].copy()
        denominator = xp.vdot(vector, vector).real
        projection = residual @ vector.conj() / denominator
        residual -= projection[:, None] * vector[None, :]

    if len(set(pivots)) != ncol or any(row < 0 or row >= nrow for row in pivots):
        raise RuntimeError("Internal error while constructing maxvol row seed")
    return np.asarray(pivots, dtype=int)


def _validate_initial_rows(initial_rows, nrow, ncol):
    rows = np.asarray(initial_rows, dtype=int)
    if rows.shape != (ncol,):
        raise ValueError(
            f"initial_rows must have shape ({ncol},), got {rows.shape}")
    if np.any(rows < 0) or np.any(rows >= nrow):
        raise ValueError("initial_rows contains an out-of-range row index")
    if np.unique(rows).size != ncol:
        raise ValueError("initial_rows must contain distinct row indices")
    return rows.copy()


def maxvol_select(matrix, initial_rows=None, tol=0.01, max_iter=100,
                  seed_relative_tol=1e-12):
    """Select a locally maximum-volume square row submatrix.

    Parameters
    ----------
    matrix
        A real or complex tall matrix with shape ``(n_candidates, n_occ)``.
    initial_rows
        Optional distinct candidate rows used to seed the search. If that
        submatrix is singular, a rank-revealing seed is constructed instead.
    tol
        Stop when every replacement coefficient has magnitude at most
        ``1 + tol``. QE's SCF default is 0.01.
    max_iter
        Maximum number of determinant-increasing row replacements.
    seed_relative_tol
        Relative rank threshold used only while constructing a fallback seed.

    Returns
    -------
    MaxvolResult
        Host pivot indices and determinant diagnostics. Matrix algebra remains
        on the input array's NumPy or CuPy backend.
    """
    xp = _array_module(matrix)
    matrix = xp.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be two-dimensional, got ndim={matrix.ndim}")
    nrow, ncol = matrix.shape
    if ncol == 0:
        raise ValueError("matrix must contain at least one column")
    if nrow < ncol:
        raise ValueError(
            f"maxvol requires a tall or square matrix, got shape {matrix.shape}")
    if tol < 0:
        raise ValueError("tol must be non-negative")
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    if seed_relative_tol <= 0:
        raise ValueError("seed_relative_tol must be positive")
    if matrix.dtype.kind not in "fc":
        matrix = matrix.astype(float)

    if initial_rows is None:
        pivots = _rank_revealing_rows(matrix, xp, seed_relative_tol)
    else:
        pivots = _validate_initial_rows(initial_rows, nrow, ncol)

    used_fallback = initial_rows is None
    max_coefficient = np.inf
    replacements = 0
    while True:
        selected = matrix[xp.asarray(pivots, dtype=int)]
        try:
            # B @ selected = matrix. Solving is more stable than forming the
            # inverse explicitly and is supported by NumPy and CuPy.
            coefficients = xp.linalg.solve(selected.T, matrix.T).T
        except xp.linalg.LinAlgError:
            if used_fallback:
                raise np.linalg.LinAlgError(
                    "Maxvol overlap matrix is rank deficient") from None
            pivots = _rank_revealing_rows(matrix, xp, seed_relative_tol)
            used_fallback = True
            continue

        coefficients[xp.asarray(pivots, dtype=int), :] = 0
        flat_index = int(xp.argmax(xp.abs(coefficients)).item())
        candidate, replaced_column = divmod(flat_index, ncol)
        max_coefficient = float(xp.abs(
            coefficients[candidate, replaced_column]).item())

        if max_coefficient <= 1.0 + tol:
            break
        if replacements >= max_iter:
            raise RuntimeError(
                f"Maxvol did not converge after {max_iter} row replacements")
        pivots[replaced_column] = candidate
        replacements += 1

    selected = matrix[xp.asarray(pivots, dtype=int)]
    sign, log_determinant = xp.linalg.slogdet(selected)
    log_determinant = float(log_determinant.item())
    determinant = (0.0 if not np.isfinite(log_determinant)
                   else float(xp.exp(log_determinant).item()))
    if float(xp.abs(sign).item()) == 0.0:
        determinant = 0.0

    return MaxvolResult(
        pivots=pivots.copy(),
        determinant=determinant,
        log_determinant=log_determinant,
        iterations=replacements,
        max_coefficient=max_coefficient,
    )


def maxvol_occ_(mf, reference_coeff, target_occ, tol=0.01, max_iter=100):
    """Attach fixed-reference maximum-volume occupations to a UHF/UKS object.

    This follows the extension pattern of :func:`pyscf.scf.addons.mom_occ`.
    The requested occupied reference subspace remains fixed throughout the
    SCF. An evolving previous-iteration reference and QE's multi-degree state
    selection are intentionally outside this isolated first implementation.

    The per-call diagnostics are available as ``mf.get_occ.maxvol_history``.
    """
    if not mf.istype("UHF"):
        raise TypeError("maxvol_occ currently supports unrestricted UHF/UKS objects only")

    target_occ_host = _host_array(target_occ)
    if target_occ_host.ndim != 2 or target_occ_host.shape[0] != 2:
        raise ValueError(
            "target_occ must have shape (2, nmo) for an unrestricted calculation")
    if len(reference_coeff) != 2:
        raise ValueError("reference_coeff must contain alpha and beta coefficients")
    if not np.all((target_occ_host == 0) | (target_occ_host == 1)):
        raise ValueError("target_occ must contain only integer occupations 0 or 1")

    references = []
    occupation_counts = []
    for spin in range(2):
        xp = _array_module(reference_coeff[spin])
        coefficients = xp.asarray(reference_coeff[spin])
        if coefficients.ndim != 2 or coefficients.shape[1] != target_occ_host.shape[1]:
            raise ValueError(
                "reference_coeff and target_occ must describe the same MO manifold")
        occupied = target_occ_host[spin] > 0
        references.append(coefficients[:, occupied])
        occupation_counts.append(int(np.count_nonzero(occupied)))

    expected_counts = tuple(int(value) for value in mf.nelec)
    if tuple(occupation_counts) != expected_counts:
        raise ValueError(
            f"target_occ contains {tuple(occupation_counts)} electrons, "
            f"but the SCF object requires {expected_counts}")

    log = logger.Logger(mf.stdout, mf.verbose)
    history = []
    call_times = []
    previous_pivots = [None, None]

    def get_occ(mo_energy=None, mo_coeff=None):
        call_start = time.perf_counter()
        if mo_energy is None:
            mo_energy = mf.mo_energy
        if mo_coeff is None:
            mo_coeff = mf.mo_coeff
        xp = _array_module(mo_coeff[0])
        occupations = xp.zeros_like(xp.asarray(mo_energy), dtype=float)
        overlap_metric = xp.asarray(mf.get_ovlp())
        call_diagnostics = []

        for spin in range(2):
            spin_start = time.perf_counter()
            nocc = occupation_counts[spin]
            if nocc == 0:
                call_diagnostics.append({
                    "pivots": np.empty(0, dtype=int),
                    "determinant": 1.0,
                    "log_determinant": 0.0,
                    "iterations": 0,
                    "max_coefficient": 0.0,
                    "occupation_changes": 0,
                    "elapsed_seconds": time.perf_counter() - spin_start,
                })
                continue

            current = xp.asarray(mo_coeff[spin])
            reference = xp.asarray(references[spin])
            overlap = current.conj().T @ overlap_metric @ reference

            # MOM projection ranking gives an all-electron-safe initial set:
            # unlike QE's pseudopotential bands, PySCF still contains the
            # deliberately emptied deep core orbital among its lowest MOs.
            scores = xp.sum(xp.abs(overlap) ** 2, axis=1).real
            seed = _host_array(xp.argsort(scores)[-nocc:]).astype(int)
            result = maxvol_select(
                overlap, initial_rows=seed, tol=tol, max_iter=max_iter)
            occupations[spin, xp.asarray(result.pivots, dtype=int)] = 1.0
            previous = previous_pivots[spin]
            occupation_changes = (0 if previous is None else
                                  len(set(previous) ^ set(result.pivots)) // 2)
            previous_pivots[spin] = result.pivots.copy()
            call_diagnostics.append({
                "pivots": result.pivots.copy(),
                "determinant": result.determinant,
                "log_determinant": result.log_determinant,
                "iterations": result.iterations,
                "max_coefficient": result.max_coefficient,
                "occupation_changes": occupation_changes,
                "elapsed_seconds": time.perf_counter() - spin_start,
            })
            log.debug(
                "Maxvol spin %d\n"
                "\t|det| = %.8g\n"
                "\tswaps = %d\n"
                "\tmax|B| = %.8g\n"
                "\toccupied = %s",
                spin, result.determinant, result.iterations,
                result.max_coefficient, result.pivots)

        history.append(tuple(call_diagnostics))
        elapsed = time.perf_counter() - call_start
        call_times.append(elapsed)
        changed = [item["occupation_changes"] for item in call_diagnostics]
        if any(changed):
            log.info(
                "MAXVOL occupation changed at call %d: alpha=%d, beta=%d",
                len(history), changed[0], changed[1])
        log.debug(
            "MAXVOL call %d\n"
            "\telapsed = %.6f s\n"
            "\talpha: swaps=%d, changes=%d\n"
            "\tbeta : swaps=%d, changes=%d",
            len(history), elapsed,
            call_diagnostics[0]["iterations"],
            call_diagnostics[0]["occupation_changes"],
            call_diagnostics[1]["iterations"],
            call_diagnostics[1]["occupation_changes"])
        return occupations

    get_occ.maxvol_history = history
    get_occ.maxvol_call_times = call_times
    mf.get_occ = get_occ
    return mf


maxvol_occ = maxvol_occ_


def normalize_maxvol_warmup_calls(calls):
    """Validate the number of MOM occupation calls before mixed maxvol."""
    if isinstance(calls, (bool, np.bool_)) or not isinstance(calls, (int, np.integer)):
        raise ValueError("maxvol_warmup_calls must be a positive integer")
    calls = int(calls)
    if calls < 1:
        raise ValueError("maxvol_warmup_calls must be a positive integer")
    return calls


def mixed_occ_(mf, reference_coeff, target_occ, warmup_calls=2,
               tol=0.01, max_iter=100):
    """Use MOM briefly, then fixed-reference maximum-volume occupations.

    The reference is never replaced by an intermediate SCF snapshot. For an
    FCH calculation it is the GS orbital set with the requested core hole;
    for XCH it is the converged FCH set with the spectator electron added.
    MOM establishes that all-electron target before maxvol begins collective
    determinant tracking against the same fixed physical reference.
    """
    warmup_calls = normalize_maxvol_warmup_calls(warmup_calls)

    mom_occ(mf, reference_coeff, target_occ)
    mom_get_occ = mf.get_occ
    maxvol_occ_(mf, reference_coeff, target_occ, tol=tol, max_iter=max_iter)
    maxvol_get_occ = mf.get_occ
    log = logger.Logger(mf.stdout, mf.verbose)
    phases = []

    def get_occ(mo_energy=None, mo_coeff=None):
        call_number = len(phases) + 1
        if call_number <= warmup_calls:
            phases.append("mom")
            log.debug(
                "MIXED call %d/%d: MOM warm-up against fixed target reference",
                call_number, warmup_calls)
            return mom_get_occ(mo_energy, mo_coeff)

        phases.append("maxvol")
        return maxvol_get_occ(mo_energy, mo_coeff)

    get_occ.mixed_phases = phases
    get_occ.maxvol_warmup_calls = warmup_calls
    get_occ.maxvol_history = maxvol_get_occ.maxvol_history
    get_occ.maxvol_call_times = maxvol_get_occ.maxvol_call_times
    mf.get_occ = get_occ
    return mf


mixed_occ = mixed_occ_


def normalize_occupation_method(method):
    """Validate and normalize the public SCF occupation-method setting."""
    if not isinstance(method, str):
        raise ValueError(
            f"occupation_method must be one of {OCCUPATION_METHODS}, got {method!r}")
    normalized = method.strip().lower()
    if normalized not in OCCUPATION_METHODS:
        raise ValueError(
            f"occupation_method must be one of {OCCUPATION_METHODS}, got {method!r}")
    return normalized


def apply_occupation_method(mf, reference_coeff, target_occ, method="mom",
                            maxvol_warmup_calls=2):
    """Attach the requested reversible SCF occupation controller."""
    method = normalize_occupation_method(method)
    if method == "mom":
        return mom_occ(mf, reference_coeff, target_occ)
    if method == "maxvol":
        return maxvol_occ(mf, reference_coeff, target_occ)
    return mixed_occ(
        mf, reference_coeff, target_occ, warmup_calls=maxvol_warmup_calls)


__all__ = [
    "MaxvolResult", "OCCUPATION_METHODS", "maxvol_select", "maxvol_occ",
    "maxvol_occ_", "mixed_occ", "mixed_occ_", "normalize_occupation_method",
    "normalize_maxvol_warmup_calls", "apply_occupation_method",
]
