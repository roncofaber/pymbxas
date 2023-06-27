#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 16:59:25 2023

@author: roncofaber
"""

import argparse
import os, sys

import pymbxas as pym
import pymbxas.io

#%%

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

#%%
    
parser = argparse.ArgumentParser()

parser.add_argument("-np", "--npoints", dest="gridP", nargs='?',
                    help="Number of grid points.",
                    default=301, type=int)

parser.add_argument("-he", "--highE", dest="highE", nargs='?',
                    help="High energy limit.",
                    default=0.0, type=float)

parser.add_argument("-le", "--lowE", dest="lowE", nargs='?',
                    help="Low energy limit.",
                    default=0.0, type=float)

parser.add_argument("-s", "--sigma", dest="sigma", nargs='?',
                    help="Gaussian broadening RIXS",
                    default=0.3, type=float)

parser.add_argument("-r", "--rixs", dest="DoRIXS", nargs='?',
                    help="Do RIXS.",
                    default=False, type=str2bool, const=True)

parser.add_argument("-a", "--align", dest="do_align", nargs='?',
                    help="Do alignment.",
                    default=False, type=str2bool, const=True)


def main(argv=sys.argv[1:]):
    
    # initialize variables
    run_path = os.getcwd()
    
    # default params
    mbxas_params = {
        "gridP"    : 100,
        "highE"    : 0,
        "lowE"     : 0,
        "sigma"    : 0.3,
        "do_align" : False,
        "DoRIXS"   : False,

        "Gamma"         : "0+0.3j",
        "check_amp"     : True,
        "printsticks"   : True,
        "printspec"     : True,
        "Dodebug"       : False,
        "calc_abs"      : True,
        "printinteg"    : False,
        "input_file"    : "{}/qchem.input".format(run_path),
        "output_file"   : "{}/qchem.output".format(run_path),
        "printanalysis" : True
    }
    
    # parse arguments
    attributes = parser.parse_args(argv)
    
    # update default parameters
    mbxas_params.update(attributes.__dict__)
    
    pym.io.write.write_mbxas_input(mbxas_params, run_path)

    pym.io.run.submit_mbxas_job(run_path)
    