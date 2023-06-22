#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 12:01:06 2023

@author: roncofaber
"""

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
        ! sbatch {my_job}'/qchem.sh'
            
        return

    
def qchem_job_write(my_job, molecule, qchem_params, from_scratch = True):
    
    my_dir, label, exchange, basis, charge, multiplicity, unrestricted, myjob, dielectric,  sym_ignore, symmetry, extra_rem_keywords, extra_sections = qchem_params
    
    if os.path.isdir(my_job) == False:
        os.mkdir(my_job)
    
    if from_scratch == True:
        if os.path.isfile(my_job+'qchem.input') == True:
            os.remove(my_job+'qchem.input')

    molecule_str = Structure(coordinates=molecule.positions,
                      symbols=molecule.get_chemical_symbols(),
                      charge=charge,
                      multiplicity=multiplicity)

    
    if dielectric == None:
        
        
            molecule_str_input = QchemInput(molecule_str,
                      jobtype=jobtype,
                      exchange=exchange,
                      basis=basis,
                      unrestricted=unrestricted, 
                      max_scf_cycles=500, 
                      scf_convergence=5,
                      extra_rem_keywords = extra_rem_keywords,
                      extra_sections = extra_sections,
                      geom_opt_max_cycles=500,
                      sym_ignore = sym_ignore, 
                      symmetry =  symmetry        
                                         
                                )
    else:
    
        molecule_str_input = QchemInput(molecule_str,
                          jobtype=jobtype,
                          exchange=exchange,
                          basis=basis,
                          unrestricted=unrestricted, 
                          max_scf_cycles=500, 
                          scf_convergence=5,
                          solvent_method='pcm',
                          solvent_params={'Dielectric': dielectric},  # Cl2CH2
                          extra_rem_keywords = extra_rem_keywords,
                          extra_sections = extra_sections,
                          geom_opt_max_cycles=500
                                    )

    txt = molecule_str_input.get_txt()
    with open(my_job + "qchem.input",'a') as f:
        f.write(txt)
        
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