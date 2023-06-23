#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 12:01:06 2023

@author: roncofaber
"""

import os

from pyqchem.parsers.basic import basic_parser_qchem
from pyqchem import Structure, QchemInput, get_output_from_qchem
from pyqchem.parsers.parser_optimization import basic_optimization
from pyqchem.qc_input import CustomSection

#%%

def set_qchem_environment(run_path):
    
    #OnDemand:
    # os.environ[CONFIG_FILE] = "/global/home/users/asanzmatias/ondemand/data/sys/dashboard/batch_connect/sys/lrc_jupyter/output/eef5b63f-f943-46a1-ac7c-f361e549e800/config.py"
    
    QC    = "/clusterfs/etna/pscratch/subhayan/QCHEM_CODE/qchem-trunk-34979"
    QCAUX = "/clusterfs/etna/pscratch/subhayan/QCHEM_CODE/qchem-trunk-34979"
   
    #Clearixs qchem
    os.environ["QC"]    = QC
    os.environ["QCAUX"] = QCAUX
   
    os.environ["QCSCRATCH"] = run_path + "/tmp_scratch"
    os.environ["PATH"] += "{}/bin:{}/bin/perl".format(QC, QC)
    
    return



def write_mbxas_input(my_job, mbxas_parameters):
    
    default_mbxas_parameters = { 'gridP' : 100,
    'highE' :  1127.5,
    'lowE' : 20,
    'sigma' : 0.3,
    'do_align' : False,
    'DoRIXS' : False,

    'Gamma' : '0+0.3j',
    'check_amp' : True,
    'printsticks' : True,
    'printspec' : True,
    'Dodebug' : False,
    'calc_abs' : True,
    'printinteg' : False,
    'input_file' : my_job+'qchem.input',
    'output_file' : my_job+'output',
    'printanalysis' : True}
    
    mbxas_f = my_job+'/INP_FILE'
    
    with open(mbxas_f, "w") as f:
    
        for key, value in default_mbxas_parameters.items():
              if key in mbxas_parameters.keys():
                    value = mbxas_parameters[key]
            
              f.write(key + '=' + str(value) + "\n")
    return



def submit_qchem_job(my_job, partition, account, procs, time):
    
        file = my_job + '/qchem.sh'
        
        if os.path.isfile(file) == True:
            os.remove(file)
        
        with open(file, 'a') as the_file:
            the_file.write('#!/bin/bash\n')
            the_file.write('#SBATCH --job-name=mbxas.qchem\n')
            the_file.write('#SBATCH --partition='+partition+'\n')
            the_file.write('#SBATCH --time='+time+'\n')
            the_file.write('#SBATCH -N 1 \n')
            the_file.write('#SBATCH --account='+account+'\n')
            the_file.write('#SBATCH --chdir='+my_job+'\n')
            
            the_file.write('export  QC=/clusterfs/etna/pscratch/subhayan/QCHEM_CODE/qchem-trunk-34979\n')
            
            the_file.write('export QCSCRATCH='+my_job+'\n')
            the_file.write('export SCRATCH='+my_job+'\n')
            
            the_file.write('export QCTHREADS=1\n')
            the_file.write('export QCPARALLEL=True\n')
            the_file.write('export QCAUX=/clusterfs/etna/pscratch/subhayan/QCHEM_CODE/qcaux-trunk\n')
            the_file.write('export PATH=$PATH:$QC/bin:$QC/bin/perl\n')
            the_file.write('module purge\n')
            the_file.write('module load gcc/6.3.0 fftw/3.3.6-gcc openmpi/3.0.1-gcc mkl/2016.4.072 cmake/3.15.0\n')
            the_file.write('module load openmpi\n')
            the_file.write('cd '+my_job+' \n')
            
            
            the_file.write('qchem -nt '+ str(procs) + '  -save qchem.input      > input.qchem.out output saved_files')
        
        print("submitting... "+my_job)
        
        os.chdir(my_job)
        os.system("sbatch {}/qchem.sh".format(my_job))
            
        return

    
def qchem_job_write(molecule, charge, multiplicity,
                    qchem_params, run_path, from_scratch = True):
    
    if os.path.isdir(run_path) == False:
        os.mkdir(run_path)
    
    if from_scratch:
        if os.path.isfile(run_path + 'qchem.input') == True:
            os.remove(run_path + 'qchem.input')
    
    # make molecule
    molecule_str = Structure(
        coordinates  = molecule.get_positions(),
        symbols      = molecule.get_chemical_symbols(),
        charge       = charge,
        multiplicity = multiplicity)

    # generate input
    molecule_str_input = QchemInput(
            molecule_str,
            **qchem_params,
            )
    
    # write input in target path (append mode)
    with open(run_path + "qchem.input", 'a') as fout:
        fout.write( molecule_str_input.get_txt())
        
    return


def qchem_write_input(clusters, job_params):
    
    my_dir, label, exchange, basis, charge, multiplicity, unrestricted, myjob, dielectric,  sym_ignore, symmetry, extra_rem_keywords, extra_sections = job_params
    for i, cluster in enumerate(clusters):
        
        i_label = str(i) + label
        print(i_label)
        qchem_job_write(my_dir, cluster, i_label, exchange, basis, charge, 
                                         multiplicity, unrestricted, myjob, dielectric, sym_ignore, symmetry,  extra_rem_keywords, extra_sections )
        
    return 



#Read outputs

def qchem_job_read(my_job, jobtype):
    
    if os.path.isdir(my_job) == False:
        print('doesnt exist!')
    print(my_job + "output")
    with open(my_job + "output",'r') as f:
        output = f.read()
        
    if jobtype == 'sp':
        parsed_data = basic_parser_qchem(output)
        
    elif jobtype == 'opt':

        parsed_data = basic_optimization(output)
        
    return output, parsed_data