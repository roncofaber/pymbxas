#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 10:25:44 2023

@author: roncofaber
"""

import os
#%%

def submit_qchem_job(run_path, partition, account, procs, time,
                     output="qchem.output"):
    
        file = run_path + "/qchemjob.sh"
        
        if os.path.isfile(file) == True:
            os.remove(file)
        
        with open(file, "a") as fout:
            fout.write("#!/bin/bash\n\n")
            fout.write("#SBATCH --job-name=mbxas.qchem\n")
            fout.write("#SBATCH --partition="+partition+"\n")
            fout.write("#SBATCH --time="+time+"\n")
            fout.write("#SBATCH -N 1 \n")
            fout.write("#SBATCH --account="+account+"\n")
            fout.write("#SBATCH --chdir="+run_path+"\n")
            
            fout.write("export  QC=/clusterfs/etna/pscratch/subhayan/QCHEM_CODE/qchem-trunk-34979\n")
            
            fout.write("export QCSCRATCH="+run_path+"\n")
            fout.write("export SCRATCH="+run_path+"\n")
            
            fout.write("export QCTHREADS=1\n")
            fout.write("export QCPARALLEL=True\n")
            fout.write("export QCAUX=/clusterfs/etna/pscratch/subhayan/QCHEM_CODE/qcaux-trunk\n")
            fout.write("export PATH=$PATH:$QC/bin:$QC/bin/perl\n")
            
            fout.write("module purge\n")
            
            for module in ["gcc/6.3.0", "fftw/3.3.6-gcc",
                           "openmpi/3.0.1-gcc", "mkl/2016.4.072"]:
                fout.write("module load {}\n".format(module))
                
            # fout.write("module load openmpi\n")
            fout.write("cd {}\n".format(run_path))
            
            fout.write("qchem -nt {} -save qchem.input > input.qchem.out {} saved_files".format(
                procs, output))
        
        print("Job submitted in: {}".format(run_path))
        
        os.chdir(run_path)
        os.system("sbatch {}/qchemjob.sh".format(run_path))
            
        return
    
    
def submit_mbxas_job(run_path, mbxas_dir):
    
    os.chdir(run_path)
    with open("{}/clearixsjob.sh".format(run_path), "w") as fout:
        fout.write("#!/bin/bash\n\n")
        fout.write("python {}/RIXS_main.py {}/INP_FILE\n".format(
            mbxas_dir, run_path))

    os.system("bash {}/clearixsjob.sh".format(run_path))
    
    return
             