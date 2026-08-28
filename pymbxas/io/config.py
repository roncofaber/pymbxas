#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 14:35:55 2025

@author: roncofaber
"""

import sys
import logging
import textwrap

#%%

# below DEBUG (10): fine-grained per-quantity numerical trace data, too voluminous
# to want at the ordinary DEBUG level but useful when actively diagnosing
# a specific numerical question. Registered once here so any module can
# `from pymbxas.io.config import TRACE` and `logger.log(TRACE, ...)`.
TRACE = 5
logging.addLevelName(TRACE, "TRACE")

_OWNED_HANDLER_ATTRIBUTE = "_pymbxas_owned_handler"


class MultilineFormatter(logging.Formatter):
    """Indent continuation lines while leaving empty separator records blank."""

    def format(self, record):
        if not record.getMessage() and not record.exc_info:
            return ""
        rendered = super().format(record)
        lines = rendered.splitlines()
        if len(lines) < 2:
            return rendered
        return lines[0] + "".join(f"\n\t{line}" for line in lines[1:])


def format_log_fields(fields, width=88):
    """Format ordered name/value pairs as an aligned multiline block."""
    items = [(str(name), value) for name, value in fields.items()
             if value is not None]
    if not items:
        return ""
    name_width = max(len(name) for name, _ in items)
    value_width = max(24, width - name_width - 3)
    lines = []
    for name, value in items:
        wrapped = textwrap.wrap(
            str(value), width=value_width, break_long_words=True,
            break_on_hyphens=False) or [""]
        lines.append(f"{name:<{name_width}} : {wrapped[0]}")
        continuation = " " * (name_width + 3)
        lines.extend(f"{continuation}{line}" for line in wrapped[1:])
    return "\n".join(lines)


class ContextLoggerAdapter(logging.LoggerAdapter):
    """Prefix records with stable scientific workflow context."""

    def process(self, msg, kwargs):
        labels = [self.extra.get("site"), self.extra.get("stage")]
        context = " ".join(str(label) for label in labels if label)
        return (f"[{context}] {msg}" if context else msg), kwargs


def with_log_context(log, *, site=None, stage=None):
    """Return a logger adapter carrying optional site and stage labels."""
    extra = {}
    if isinstance(log, logging.LoggerAdapter):
        extra.update(log.extra)
        log = log.logger
    if site is not None:
        extra["site"] = site
    if stage is not None:
        extra["stage"] = stage
    return ContextLoggerAdapter(log, extra)


def occupation_tracking_diagnostics(calculator):
    """Aggregate maxvol callback history into one SCF-level diagnostic."""
    get_occ = getattr(calculator, "get_occ", None)
    history = getattr(get_occ, "maxvol_history", None)
    if history is None:
        return None

    swaps = 0
    orbital_changes = 0
    changed_calls = 0
    last_change_call = None
    for call_number, call in enumerate(history, start=1):
        call_changes = sum(item["occupation_changes"] for item in call)
        swaps += sum(item["iterations"] for item in call)
        orbital_changes += call_changes
        if call_changes:
            changed_calls += 1
            last_change_call = call_number

    return {
        "warmup_calls": getattr(get_occ, "maxvol_warmup_calls", 0),
        "maxvol_calls": len(history),
        "swaps": swaps,
        "changed_calls": changed_calls,
        "orbital_changes": orbital_changes,
        "last_change_call": last_change_call,
        "selector_seconds": float(sum(
            getattr(get_occ, "maxvol_call_times", ()))),
    }


def log_scf_completion(log, calculator, elapsed_seconds, *,
                       occupation_method=None, **validation):
    """Log one SCF result and one optional aggregate tracking diagnostic."""
    energy = getattr(calculator, "e_tot", float("nan"))
    if hasattr(energy, "item"):
        energy = energy.item()
    cycles = int(getattr(calculator, "cycles", 0))
    electrons = tuple(int(value) for value in calculator.mol.nelec)
    details = {
        "cycles": cycles,
        "energy": f"{float(energy):.12f} Ha",
        "electrons (alpha, beta)": electrons,
    }
    attempts = getattr(calculator, "_pymbxas_scf_attempts", ())
    if attempts:
        details["solver path"] = " -> ".join(item.solver for item in attempts)
        details["cycles by stage"] = " + ".join(
            str(item.cycles) for item in attempts)
    details.update({name.replace("_", " "): value
                    for name, value in validation.items()})
    details["elapsed"] = f"{elapsed_seconds:.1f} s"
    log.info("Converged\n%s", format_log_fields(details))

    tracking = occupation_tracking_diagnostics(calculator)
    if tracking is None:
        return
    parts = {}
    if tracking["warmup_calls"]:
        parts["MOM warm-up calls"] = tracking["warmup_calls"]
    parts.update({
        "maxvol calls": tracking["maxvol_calls"],
        "maxvol row replacements": tracking["swaps"],
        "calls with occupation changes": tracking["changed_calls"],
        "occupied-orbital changes": tracking["orbital_changes"],
        "last changing call": tracking["last_change_call"] or "none",
        "selector time": f"{tracking['selector_seconds']:.3f} s",
    })
    log.info("Occupation tracking (%s)\n%s", occupation_method,
             format_log_fields(parts))

def _replace_owned_handlers(log):
    """Remove and close handlers installed by :func:`configure_logger`."""
    for handler in tuple(log.handlers):
        if getattr(handler, _OWNED_HANDLER_ATTRIBUTE, False):
            log.removeHandler(handler)
            handler.close()


def _mark_owned(handler):
    setattr(handler, _OWNED_HANDLER_ATTRIBUTE, True)
    return handler


def configure_logger(level, log_file=None, file_mode="a"):
    """
    Configures the pymbxas logger with the specified logging level.

    Args:
        level (int): Logging level (1-5).
        log_file (str): Optional path to a log file.
        file_mode (str): ``"w"`` to replace or ``"a"`` to append.
    """
    # Define a mapping from user input to logging levels
    level_mapping = {
        1: logging.ERROR,
        2: logging.WARNING,
        3: logging.INFO,
        4: logging.DEBUG,
        5: TRACE,
    }

    if level not in level_mapping:
        raise ValueError("Invalid logging level: {}. Choose a level between 1 and 5.".format(level))
    if file_mode not in ("a", "w"):
        raise ValueError("file_mode must be 'a' or 'w'")

    # Map the user input level to logging level
    level = level_mapping[level]

    # Get the named logger for pymbxas
    pymbxas_logger = logging.getLogger("pymbxas")

    # Replace only handlers installed by this function. Applications may
    # attach their own handlers to the package logger and retain ownership.
    _replace_owned_handlers(pymbxas_logger)

    # Set the logging level
    pymbxas_logger.setLevel(level)

    # Create a console handler using stdout
    console_handler = _mark_owned(logging.StreamHandler(sys.stdout))
    console_handler.setLevel(level)

    console_handler.setFormatter(MultilineFormatter(
        "%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"))

    # Add the handler to the pymbxas logger
    pymbxas_logger.addHandler(console_handler)

    # Prevent records from also going to the root logger
    pymbxas_logger.propagate = False

    # If log_file is specified, create a file handler
    if log_file:
        file_handler = _mark_owned(logging.FileHandler(
            log_file, mode=file_mode, encoding="utf-8"))
        file_handler.setLevel(level)
        file_handler.setFormatter(MultilineFormatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"))
        pymbxas_logger.addHandler(file_handler)

    return
