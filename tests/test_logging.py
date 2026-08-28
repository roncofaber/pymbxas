import ast
import builtins
import io
import logging
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import ase
import pytest
from pyscf import gto

import pymbxas.calculators.pyscf as pyscf_module
import pymbxas.build.input_pyscf as input_pyscf_module
import pymbxas.io.write as write_module
from pymbxas.calculators.pyscf import PySCFMBXAS
from pymbxas.config import (
    CalculationConfig, CheckpointConfig, ExcitationConfig, LoggingConfig,
    RuntimeConfig, SCFConfig,
)
from pymbxas.io.config import (
    configure_logger, format_log_fields, log_scf_completion,
    MultilineFormatter, occupation_tracking_diagnostics, with_log_context,
)
from pymbxas.io.logger import Logger
from pymbxas.utils.metrics import zmatlike


def _memory_logger():
    stream = io.StringIO()
    log = logging.Logger("pymbxas-test", level=logging.DEBUG)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(MultilineFormatter("%(levelname)s %(message)s"))
    log.addHandler(handler)
    return log, stream


def test_log_context_combines_site_and_stage():
    log, stream = _memory_logger()
    site_log = with_log_context(log, site="O:8")
    with_log_context(site_log, stage="FCH").info("Starting calculation")

    assert stream.getvalue() == "INFO [O:8 FCH] Starting calculation\n"


def test_scf_completion_aggregates_occupation_tracking():
    log, stream = _memory_logger()
    get_occ = lambda: None
    get_occ.maxvol_warmup_calls = 2
    get_occ.maxvol_call_times = [0.1, 0.2, 0.3]
    get_occ.maxvol_history = [
        ({"iterations": 1, "occupation_changes": 0},
         {"iterations": 0, "occupation_changes": 0}),
        ({"iterations": 0, "occupation_changes": 2},
         {"iterations": 0, "occupation_changes": 1}),
        ({"iterations": 1, "occupation_changes": 0},
         {"iterations": 0, "occupation_changes": 1}),
    ]
    calculator = SimpleNamespace(
        get_occ=get_occ, cycles=7, e_tot=-10.25,
        mol=SimpleNamespace(nelec=(5, 4)))

    diagnostics = occupation_tracking_diagnostics(calculator)
    assert diagnostics == {
        "warmup_calls": 2,
        "maxvol_calls": 3,
        "swaps": 2,
        "changed_calls": 2,
        "orbital_changes": 4,
        "last_change_call": 3,
        "selector_seconds": pytest.approx(0.6),
    }

    log_scf_completion(
        with_log_context(log, site="O:8", stage="FCH"), calculator, 4.2,
        occupation_method="mixed", core_hole_overlap="0.99871")
    output = stream.getvalue()
    assert "INFO [O:8 FCH] Converged" in output
    assert "\n\tcycles                  : 7" in output
    assert "\n\tenergy                  : -10.250000000000 Ha" in output
    assert "\n\tcore hole overlap       : 0.99871" in output
    assert "\n\telapsed                 : 4.2 s" in output
    assert "INFO [O:8 FCH] Occupation tracking (mixed)" in output
    assert "\n\tMOM warm-up calls             : 2" in output
    assert "\n\tmaxvol row replacements       : 2" in output
    assert "\n\tcalls with occupation changes : 2" in output
    assert "\n\tlast changing call            : 3" in output


def test_excite_reports_success_failure_and_skip(monkeypatch):
    log, stream = _memory_logger()
    obj = PySCFMBXAS(
        ase.Atoms("OOO"), calculation=CalculationConfig(basis="sto-3g"),
        checkpoint=None)
    obj._ran_GS = True
    default = ExcitationConfig()
    obj._excitations = [SimpleNamespace(
        ato_idx=2, channel=default.channel_index, config=default,
        fch_scf=SCFConfig(), xch_scf=SCFConfig())]
    obj._last_excitation_outcomes = ()
    obj.gs_data = object()
    obj.df_obj = None
    obj.logger = log

    class FakeExcitation:
        def __init__(self, structure, gs_data, ato_idx, config,
                     fch_scf, xch_scf):
            self.ato_idx = ato_idx
            self.channel = config.channel_index
            self.config = config
            self.fch_scf = fch_scf
            self.xch_scf = xch_scf

        def run(self, *args):
            if self.ato_idx == 1:
                raise RuntimeError("FCH SCF did not converge after 100 cycles")
            return self

    monkeypatch.setattr(pyscf_module, "Excitation", FakeExcitation)
    outcomes = obj.excite([0, 1, 2])

    assert [item.status for item in outcomes] == [
        "succeeded", "failed", "skipped"]
    assert obj.last_excitation_outcomes == outcomes
    output = stream.getvalue()
    assert "[O:1] Excitation failed: FCH SCF did not converge" in output
    assert "[O:2] Equivalent excitation already exists; skipping" in output
    assert "Run completed with failures" in output
    assert "\n\tsucceeded : 1" in output
    assert "\n\tfailed    : 1" in output
    assert "\n\tskipped   : 1" in output
    assert "[O:1] FCH SCF did not converge after 100 cycles" in output
    assert "finished successfully" not in output.lower()


def test_excite_requires_ground_state():
    log, stream = _memory_logger()
    obj = PySCFMBXAS(ase.Atoms("H"), checkpoint=None)
    obj.logger = log

    with pytest.raises(RuntimeError, match="before the ground state"):
        obj.excite(0)
    assert "ERROR Cannot excite atoms before the ground state has run" in stream.getvalue()


def test_configure_logger_preserves_external_handler_and_closes_owned(tmp_path):
    package_log = logging.getLogger("pymbxas")
    external_stream = io.StringIO()
    external = logging.StreamHandler(external_stream)
    package_log.addHandler(external)
    first_path = tmp_path / "first.log"
    second_path = tmp_path / "second.log"

    try:
        configure_logger(3, first_path, file_mode="w")
        first_owned = [
            handler for handler in package_log.handlers
            if getattr(handler, "_pymbxas_owned_handler", False)]
        first_file = next(
            handler for handler in first_owned
            if isinstance(handler, logging.FileHandler))
        assert external in package_log.handlers
        assert len(first_owned) == 2

        configure_logger(3, second_path, file_mode="w")
        second_owned = [
            handler for handler in package_log.handlers
            if getattr(handler, "_pymbxas_owned_handler", False)]
        assert external in package_log.handlers
        assert len(second_owned) == 2
        assert first_file.stream is None
    finally:
        package_log.removeHandler(external)
        external.close()
        configure_logger(3)


def test_configure_logger_file_modes_are_explicit(tmp_path):
    path = tmp_path / "pymbxas.log"
    package_log = logging.getLogger("pymbxas")

    configure_logger(3, path, file_mode="w")
    package_log.info("first run")
    configure_logger(3, path, file_mode="w")
    package_log.info("replacement run")
    configure_logger(3, path, file_mode="a")
    package_log.info("restart run")

    text = path.read_text(encoding="utf-8")
    assert "first run" not in text
    assert text.count("replacement run") == 1
    assert text.count("restart run") == 1
    assert re.search(
        r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \| INFO\s+\| "
        r"pymbxas \| replacement run", text)
    configure_logger(3)


def test_multiline_formatter_indents_and_wraps_fields():
    record = logging.LogRecord(
        "pymbxas.test", logging.INFO, __file__, 1,
        "Section\n%s", (format_log_fields({
            "short": 1,
            "long field": "a deliberately long value " * 6,
        }, width=48),), None)
    formatter = MultilineFormatter("%(levelname)s %(message)s")
    text = formatter.format(record)

    lines = text.splitlines()
    assert lines[0] == "INFO Section"
    assert all(line.startswith("\t") for line in lines[1:])
    assert any("long field" in line for line in lines)
    assert len(lines) > 4
    blank = logging.LogRecord(
        "pymbxas.test", logging.INFO, __file__, 1, "", (), None)
    assert formatter.format(blank) == ""


def test_calculation_log_paths_are_relative_to_target_directory(tmp_path):
    target = tmp_path / "calculation"
    obj = PySCFMBXAS(
        ase.Atoms("H"),
        calculation=CalculationConfig(spin=1, basis="sto-3g"),
        checkpoint=None)
    obj._prepare_execution(
        RuntimeConfig(work_directory=target),
        LoggingConfig(
            pymbxas_verbosity=1,
            pymbxas_logfile="application.log",
            pyscf_verbosity=0,
            pyscf_logfile="pyscf.log",
            pyscf_console=False))

    assert obj.logging.pymbxas_logfile == str(target / "application.log")
    assert obj.logging.pyscf_logfile == str(target / "pyscf.log")
    assert (target / "application.log").exists()
    configure_logger(3)


def test_raw_logger_flushes_utf8_file(tmp_path):
    path = tmp_path / "raw.log"
    stream = Logger(print_to_terminal=False, log_file=path)
    stream.write("finished ✓\n")
    stream.flush()
    assert path.read_text(encoding="utf-8") == "finished ✓\n"
    stream.close()


def test_raw_logger_does_not_silently_drop_requested_file(tmp_path):
    path = tmp_path / "missing" / "raw.log"
    with pytest.raises(FileNotFoundError):
        Logger(print_to_terminal=False, log_file=path)


def test_invalid_zmatlike_input_raises_without_printing(capsys):
    with pytest.raises(ValueError, match="At least 4 atoms"):
        zmatlike(ase.Atoms("H2"))

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_hf_ignored_xc_uses_logging(monkeypatch):
    fake_logger = Mock()
    monkeypatch.setattr(input_pyscf_module, "logger", fake_logger)
    mol = gto.M(
        atom="H 0 0 0", basis="sto-3g", spin=1, verbose=0)

    input_pyscf_module.make_pyscf_calculator(
        mol, xc="pbe", calc_type="UHF")

    fake_logger.warning.assert_called_once_with(
        "HF calculator ignores the XC setting %r", "pbe")


def test_missing_mokit_uses_logging(monkeypatch):
    original_import = builtins.__import__

    def import_without_mokit(name, *args, **kwargs):
        if name == "mokit" or name.startswith("mokit."):
            raise ImportError("test-controlled missing dependency")
        return original_import(name, *args, **kwargs)

    fake_logger = Mock()
    monkeypatch.setattr(builtins, "__import__", import_without_mokit)
    write_module.write_data_to_fchk(
        None, oname="orbitals.fchk", logger=fake_logger)

    fake_logger.warning.assert_called_once_with(
        "MOKIT is unavailable; FCHK file was not written: %s",
        "orbitals.fchk")


def test_library_prints_are_limited_to_presentation_apis():
    package_root = Path(__file__).parents[1] / "pymbxas"
    allowed = {
        ("cli/pyscf.py", "main"),
        ("spectra.py", "print_mbxas_summary"),
    }
    found = []

    class PrintVisitor(ast.NodeVisitor):
        def __init__(self, relative_path):
            self.relative_path = relative_path
            self.function_stack = []

        def visit_FunctionDef(self, node):
            self.function_stack.append(node.name)
            self.generic_visit(node)
            self.function_stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == "print":
                function = self.function_stack[-1] if self.function_stack else None
                found.append((self.relative_path, function))
            self.generic_visit(node)

    for path in package_root.rglob("*.py"):
        relative = str(path.relative_to(package_root))
        PrintVisitor(relative).visit(
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))

    assert set(found) <= allowed


def test_raw_logger_writes_structured_scf_boundary(tmp_path):
    path = tmp_path / "raw-section.log"
    stream = Logger(
        print_to_terminal=False, log_file=path,
        section_context={
            "site": "O:8", "stage": "FCH", "channel": "beta",
            "occupation": "mixed", "note": "line one\nline two",
        })
    stream.write("converged SCF energy = -10.0\n")
    stream.close()

    text = path.read_text(encoding="utf-8")
    assert "BEGIN PyMBXAS SCF" in text
    assert "timestamp  : " in text
    assert "site       : O:8" in text
    assert "stage      : FCH" in text
    assert "channel    : beta" in text
    assert "occupation : mixed" in text
    assert "note       : line one line two" in text
    assert text.index("BEGIN PyMBXAS SCF") < text.index("converged SCF energy")


@pytest.mark.parametrize("mode", ["x", "append", None])
def test_configure_logger_rejects_unknown_file_mode(mode):
    with pytest.raises(ValueError, match="file_mode"):
        configure_logger(3, file_mode=mode)
