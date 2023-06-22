"""PyMBXAS"""

import sys


if sys.version_info[0] == 2:
    raise ImportError('Electrolyte machine requires Python3. This is Python2.')

__version__ = '0.0.1a'
