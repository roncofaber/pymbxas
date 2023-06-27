"""PyMBXAS"""

import sys
import pymbxas.io.write

if sys.version_info[0] == 2:
    raise ImportError('Please run with Python3. This is Python2.')


# print("Welcome to pyMBXAS")

# set env
pymbxas.io.write.set_qchem_environment()


__version__ = '0.0.1a'
__mbxasdir__ = '/global/home/groups/nano/share/software/electrolyte_machine/gitlab_repo/CleaRIXS/'