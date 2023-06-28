#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:42:44 2023

@author: roncofaber
"""

import os

#%%

def set_qchem_environment(run_path=os.getcwd()):

    #OnDemand:
    # os.environ[CONFIG_FILE] = "/global/home/users/asanzmatias/ondemand/data/sys/dashboard/batch_connect/sys/lrc_jupyter/output/eef5b63f-f943-46a1-ac7c-f361e549e800/config.py"

    QC    = "/clusterfs/etna/pscratch/subhayan/QCHEM_CODE/qchem-trunk-34979"
    QCAUX = "/clusterfs/etna/pscratch/subhayan/QCHEM_CODE/qchem-trunk-34979"

    # MBXASDIR = '/global/home/groups/nano/share/software/electrolyte_machine/gitlab_repo/CleaRIXS/'

    #Clearixs qchem
    os.environ["QC"]    = QC
    os.environ["QCAUX"] = QCAUX

    os.environ["QCSCRATCH"] = run_path + "/tmp_scratch"
    os.environ["PATH"] += "{}/bin:{}/bin/perl".format(QC, QC)

    return