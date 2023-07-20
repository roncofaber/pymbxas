# CleaRIXS
CleaRIXS for Q-chem. This codes reads the electronic-structure output from a Q-Chem calculation (single-particle transition matrix elements and single-particle overlap matrix elements between the ground and the core-excited state) and computes the RIXS spectrum using the CleaRIXS method (https://doi.org/10.1103/PhysRevB.106.115115).

By setting DoRIXS=False in the input file (INP_FILE), it is possible to run an x-ray absorption calculation within the CHB-MBXAS framework (https://doi.org/10.1103/PhysRevB.107.035146) without running CleaRIXS.

Run code as:
python RIXS_main.py INP_FILE > OUTPUT
