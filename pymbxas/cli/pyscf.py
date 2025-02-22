#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 15:50:13 2025

@author: roncofaber
"""

# some libraries
import sys
import json
import argparse

# ASE
from ase.io import read

# my stuff
from pymbxas.drivers.acquisitor import pyscf_acquire


#%%

def main():
    """Main function for the pymbxas command-line interface."""
    parser = argparse.ArgumentParser(
        description="Run PySCF calculations and save spectra."
    )
    parser.add_argument(
        "input_file", help="Path to input structure file "
                           "(e.g., structure.xyz)"
    )
    parser.add_argument(
        "-o", "--output_file", default="spectrum.pkl",
        help="Path to save the spectrum (default: spectrum.pkl)"
    )
    parser.add_argument(
        "-e", "--to_excite", required=True,
        help="Atom index(es)/symbol(s) to excite (required)"
    )
    parser.add_argument(
        "-k", "--kernel_kwargs", type=str, default="{}",
        help="JSON string of kwargs for pyscf_acquire"
    )
    args = parser.parse_args()

    try:
        kernel_kwargs = json.loads(args.kernel_kwargs)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}", file=sys.stderr)
        return 1  # Indicate an error
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1

    try:
        atoms = read(args.input_file)
        spectra = pyscf_acquire(
            atoms, to_excite=args.to_excite, **kernel_kwargs
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
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        return 1
