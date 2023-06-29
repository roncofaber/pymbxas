"""PyMBXAS"""

import os
import sys

if sys.version_info[0] == 2:
    raise ImportError('Please run with Python3. This is Python2.')


# package info
__all__ = ["spectra"]
__version__ = '0.0.1a'

# change those accordingly TODO: maybe set them up when installing pyMBXAS?
__mbxasdir__   = '/global/home/groups/nano/share/software/electrolyte_machine/gitlab_repo/CleaRIXS/'
__qcdir__      = "/clusterfs/etna/pscratch/subhayan/QCHEM_CODE/qchem-trunk-34979"
__qcauxdir__   = "/clusterfs/etna/pscratch/subhayan/QCHEM_CODE/qcaux-trunk"

test_path = "/clusterfs/etna/pscratch/{}".format(os.getlogin())

if os.path.exists(test_path):
    __scratchdir__ = test_path
else:
    __scratchdir__ = "/tmp"

# set environment
from .utils.environment import set_qchem_environment
set_qchem_environment()