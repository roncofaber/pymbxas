#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 15:50:13 2025

@author: roncofaber
"""

# some libraries
import sys
import json
import logging
import argparse

# ASE
from ase.io import read

# my stuff
from pymbxas.drivers.acquisitor import pyscf_acquire
from pymbxas.config import (
    CalculationConfig, ExcitationConfig, LoggingConfig, RuntimeConfig,
    SCFConfig,
)


#%%

def main():
    """Main function for the pymbxas command-line interface."""
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(
        description="Run PySCF calculations and save spectra."
    )
    parser.add_argument(
        "input_file", help="Path to input structure file "
                           "(e.g., structure.xyz)"
    )
    parser.add_argument(
        "-o", "--output_file", default="spectrum.h5",
        help="Path to save the spectrum (default: spectrum.h5)"
    )
    parser.add_argument(
        "-e", "--sites", required=True,
        help="Atom index(es)/symbol(s) to excite (required)"
    )
    parser.add_argument(
        "--calculation-config", type=str, default="{}",
        help="JSON object defining the ground-state calculation"
    )
    parser.add_argument(
        "--runtime-config", type=str, default="{}",
        help="JSON object defining work directory and device"
    )
    parser.add_argument(
        "--logging-config", type=str, default="{}",
        help="JSON object defining application and PySCF logging"
    )
    parser.add_argument(
        "--excitation-config", type=str, default="{}",
        help="JSON object defining FCH/XCH and occupation tracking"
    )
    parser.add_argument(
        "--gs-scf-config", type=str, default="{}",
        help="JSON object defining ground-state SCF controls")
    parser.add_argument(
        "--fch-scf-config", type=str, default="{}",
        help="JSON object defining FCH SCF controls")
    parser.add_argument(
        "--xch-scf-config", type=str, default="{}",
        help="JSON object defining XCH SCF controls")
    parser.add_argument(
        "--checkpoint", default="pymbxas.h5",
        help="Calculation checkpoint path; use 'none' to disable")
    args = parser.parse_args()

    try:
        calculation = CalculationConfig.from_dict(
            json.loads(args.calculation_config))
        runtime = RuntimeConfig.from_dict(json.loads(args.runtime_config))
        logging_config = LoggingConfig.from_dict(
            json.loads(args.logging_config))
        excitation = ExcitationConfig.from_dict(
            json.loads(args.excitation_config))
        gs_scf = SCFConfig.from_dict(json.loads(args.gs_scf_config))
        fch_scf = SCFConfig.from_dict(json.loads(args.fch_scf_config))
        xch_scf_values = json.loads(args.xch_scf_config)
        if not excitation.xch and xch_scf_values:
            raise ValueError(
                "--xch-scf-config cannot be used when XCH is disabled")
        xch_scf = (
            SCFConfig.from_dict(xch_scf_values) if excitation.xch else None)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}", file=sys.stderr)
        return 1  # Indicate an error

    try:
        atoms = read(args.input_file)
        spectra = pyscf_acquire(
            atoms, sites=args.sites, calculation=calculation,
            gs_scf=gs_scf,
            checkpoint=(None if args.checkpoint.lower() == "none"
                        else args.checkpoint),
            runtime=runtime, logging=logging_config,
            excitation=excitation, fch_scf=fch_scf, xch_scf=xch_scf,
        )

        if spectra is None:
            print("Calculation failed.", file=sys.stderr)
            return 1

        spectra.save(args.output_file)
        print(f"Spectrum saved to {args.output_file}")
        return 0  # Indicate success
    except FileNotFoundError:
        print(f"Error: File '{args.input_file}' not found.",
              file=sys.stderr)
        return 1
    except Exception:
        logger.exception("An unexpected error occurred")
        return 1
