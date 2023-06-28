"""PyMBXAS"""

import sys
from pymbxas.utils.environment import set_qchem_environment

if sys.version_info[0] == 2:
    raise ImportError('Please run with Python3. This is Python2.')


# print("Welcome to pyMBXAS")

# set env
set_qchem_environment()


__all__ = ["spectra"]
__version__ = '0.0.1a'
__mbxasdir__ = '/global/home/groups/nano/share/software/electrolyte_machine/gitlab_repo/CleaRIXS/'

from pymbxas.spectra import XAS_spectra