"""PyMBXAS"""

import sys
import pymbxas.io.write

if sys.version_info[0] == 2:
    raise ImportError('Please run with Python3. This is Python2.')

__version__ = '0.0.1a'

print("Dummy test")

# set env
pymbxas.io.write.set_qchem_environment()
