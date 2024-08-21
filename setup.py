#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 17:07:19 2024

@author: roncofaber
"""

import os
import re
from setuptools import setup, find_packages
#%%

# metadata
def read_metadata(file_path, variable_name):
    with open(file_path, 'r') as file:
        content = file.read()
        match = re.search(rf"^__{variable_name}__ = ['\"]([^'\"]*)['\"]", content, re.M)
        if match:
            return match.group(1)
        raise RuntimeError(f"Unable to find __{variable_name}__ string.")

init_file_path = os.path.join('pymbxas', '__init__.py')

version = read_metadata(init_file_path, 'version')
author = read_metadata(init_file_path, 'author')

# readme
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
setup(
    name             = 'pymbxas',
    version          = version,
    
    author           = author,
    author_email     = "fabrice.roncoroni@gmail.com",
    url              = "https://github.com/roncofaber/pymbxas",
    
    description      = "PyMBXAS is a package for setting up, manipulating, running and visualizing MBXAS calculations using Python.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    
    python_requires  = '>=3.6',
    packages         = find_packages(),
    install_requires = [
        'pyscf>=2.6.2',
        'ase>=3.23.0',
        'dill>=0.3.8',
        'psutil'
    ],
)