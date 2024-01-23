""" Read Q-Chem outputs """
from __future__ import print_function


import sys
import os
import bisect

import numpy as sp

def CISreader(FileName,check_amp):
    should_read = False
    ReadFile=open(FileName)
    indx = 0
    CI_exp = []
    LowestExcitation = True
    for line in ReadFile:
     if ("excitation energy (eV)" in line):
      should_read = True
      if not LowestExcitation:
       CI_exp.append([indx, ener, entry])
       if check_amp:
        print("Cumulative amplitude for indx = "+str(indx)+" is "+str(cumulamp))
      if check_amp:
       cumulamp = 0
      ener = float(line.split()[-1])
      LowestExcitation = False
      entry = []
      indx = indx+1
     if should_read:
      if (" --> " in line):
       V = line.split()[1].replace(')','')
       C = line.split()[4].replace(')','')
       AMP = line.split()[7]
       if check_amp:
        cumulamp = cumulamp + (float(AMP))**2
       if (abs(float(AMP)) > 0):
        entry += [[V, C, AMP]]
      if ("SETman timing summary" in line):
       CI_exp.append([indx, ener, entry])
       break
    return CI_exp

def Unres_CISreader(FileName,check_amp):
    should_read = False
    ReadFile=open(FileName)
    indx = 0
    CI_exp = []
    LowestExcitation = True
    for line in ReadFile:
     if ("excitation energy (eV)" in line):
      should_read = True
      if not LowestExcitation:
       CI_exp.append([indx, ener, entry])
       if check_amp:
        print("Cumulative amplitude for indx = "+str(indx)+" is "+str(cumulamp))
      if check_amp:
       cumulamp = 0
      ener = float(line.split()[-1])
      LowestExcitation = False
      entry = []
      indx = indx+1
     if should_read:
      if (" --> " in line and "beta" in line):
       V = line.split()[1].replace(')','')
       C = line.split()[4].replace(')','')
       AMP = line.split()[7]
       if check_amp:
        cumulamp = cumulamp + (float(AMP))**2
       if (abs(float(AMP)) > 0):
        entry += [[V, C, AMP]]
    return CI_exp


def occuFinder(FileName):
    should_read = False
    line_index = 0
    ReadFile=open(FileName)
    for line in ReadFile:
     if ("$occupied" in line):
      should_read = True
     if should_read:
      line_index = line_index +1
     if (line_index == 2):
      nocc = int(line.split()[-1])-1
      list1 = line.split()
     if (line_index == 3):
      list2 = line.split()
      core = sp.setdiff1d(list1,list2)
      core=int(core[0])
      break
    return nocc,core

def tot_energy_reader(FileName,do_align):
    #FileName = 'qchem.out'
    should_read = False
    line_index = 0
    ReadFile=open(FileName)
    eigenvalue_list=[]
    which_tot_en = 0
    for line in ReadFile:
      if "Total energy in the final basis set" in line:
       which_tot_en = which_tot_en +1
       if (which_tot_en == 1):
        GS_tot_en = float(line.split()[-1])
       if (which_tot_en == 2):
        FCH_tot_en = float(line.split()[-1])
    if do_align:
     AlignFile = "./AlignDir/align_calc.out"
     ReadAl=open(AlignFile)
     for line in reversed(list(ReadAl)):
      if "Total energy in the final basis set" in line:
       Align_tot_en = float(line.split()[-1])
       break
     print("GS total energy = "+str(27.2114 * GS_tot_en))
     print("FCH total energy = "+str(27.2114 * FCH_tot_en))
     print("XCH total energy = "+str(27.2114 * Align_tot_en))
     return (27.2114 * GS_tot_en, 27.2114 * FCH_tot_en, 27.2114 * Align_tot_en)
    else:
     print("GS total energy = "+str(27.2114 * GS_tot_en))
     print("FCH total energy = "+str(27.2114 * FCH_tot_en))
     return (27.2114 * GS_tot_en, 27.2114 * FCH_tot_en, 0)

def energy_reader(nocc,core):
    FileName = 'qchem.out'
    should_read = False
    line_index = 0
    ReadFile=open(FileName)
    eigenvalue_list=[]
    which_tot_en = 0
    for line in ReadFile:
      if ("Beta MOs" in line):
       should_read = True
      if should_read:
       line_index = line_index + 1
      if should_read:
       if ("-- Virtual --" in line):
        break
      if (line_index > 2):
       list1 = line.split()
       eigenvalue_list = eigenvalue_list + list1
    #para.print("eigenvalue_list = "+str(eigenvalue_list))
    eigenvalue_list = sp.delete(eigenvalue_list,int(core-1))
    ener = sp.asarray(eigenvalue_list, dtype = sp.float64,order ='C')
    ener = 27.2114 * ener
    return(ener)

def FCHunoccupied_energy_reader(nocc,core,FileName):
    should_read = 0
    ReadFile=open(FileName)
    eigenvalue_list=[]
    virtual=False
    for line in ReadFile:
      if ("Beta MOs" in line):
       should_read = should_read + 1
      if (should_read == 2):
       if ("-- Virtual --" in line):
        virtual=True
       if (virtual):
        if("-------------------------------" in line):
         break
        if ("-- Virtual --" not in line):
         list1 = line.split()
         eigenvalue_list = eigenvalue_list + list1
    eigenvalue_list.pop(0)
    #print("eigenvalue_list = "+str(eigenvalue_list))
    eigenvalue_list = sp.array(eigenvalue_list)
    FCH_ener = sp.asarray(eigenvalue_list, dtype = sp.float64,order ='C')
    conv_fac = 27.2114
    FCH_ener = conv_fac * FCH_ener
    return(FCH_ener)

def for_ixmat(nelec):
        FileName="dipole_beta_mom.dat"
        ReadFile=open(FileName)
        dipole_mom=[]
        for line in ReadFile:
          dumm=[]
          for i in range(3):
            dumm.append(float(line.split()[i]))
          dipole_mom.append(dumm)
        FileName2="dipole_beta_gs.dat"
        ReadFile2=open(FileName2)
        dipole_gs=[]
        for line in ReadFile2:
          dumm=[]
          for i in range(3):
            dumm.append(float(line.split()[i]))
          dipole_gs.append(dumm)
        os_gs=(sp.array(dipole_gs))**2
        os_mom=(sp.array(dipole_mom))**2
        return dipole_mom, dipole_gs, os_gs, os_mom
def read_full_matrices(nelec, core_ind_gs, channel="beta", path ="./"):
        X_FILE= path + "/{}_dipole_x_mom.txt".format(channel)
        Y_FILE= path + "/{}_dipole_y_mom.txt".format(channel)
        Z_FILE= path + "/{}_dipole_z_mom.txt".format(channel)
        ########
        full_mom_dipole=[]
        ReadFile=open(X_FILE)
        dum_mom_dipole=[]
        indx=0
        for line in ReadFile:
          indx = indx + 1
          if (indx != nelec+1):
           dumm=float(line.split()[nelec])
           dum_mom_dipole.append(dumm)
        full_mom_dipole.append(dum_mom_dipole)
        ########
        ReadFile=open(Y_FILE)
        dum_mom_dipole=[]
        indx=0
        for line in ReadFile:
          indx = indx + 1
          if (indx != nelec+1):
           dumm=float(line.split()[nelec])
           dum_mom_dipole.append(dumm)
        full_mom_dipole.append(dum_mom_dipole)
        ########
        ReadFile=open(Z_FILE)
        dum_mom_dipole=[]
        indx=0
        for line in ReadFile:
          indx = indx + 1
          if (indx != nelec+1):
           dumm=float(line.split()[nelec])
           dum_mom_dipole.append(dumm)
        full_mom_dipole.append(dum_mom_dipole)
        ########
        FileName= path + "/{}_ovlp_gs_es.txt".format(channel)
        ReadFile=open(FileName)
        gs_indx=0
        ovlp_dum=[]
        for line in ReadFile:
          gs_indx=gs_indx+1
          if (gs_indx != core_ind_gs):#Skipping the core level to be emptied
            dumm = line.split()
            dumm2 = [float(i) for i in dumm]
            ovlp_dum.append(dumm2)
        ovlp=sp.delete(ovlp_dum,nelec,1)
        ########
        X_FILE= path + "/{}_dipole_x_gs.txt".format(channel)
        Y_FILE= path + "/{}_dipole_y_gs.txt".format(channel)
        Z_FILE= path + "/{}_dipole_z_gs.txt".format(channel)
        ########
        full_gs_dipole=[]
        ReadFile=open(X_FILE)
        dum_gs_dipole=[]
        indx=0
        for line in ReadFile:
          indx = indx + 1
          if (indx != core_ind_gs):#Skipping the core level to be emptied
           dumm=float(line.split()[int(core_ind_gs-1)])#read dipole element for core
           dum_gs_dipole.append(dumm)
        full_gs_dipole.append(dum_gs_dipole)
        ########
        ReadFile=open(Y_FILE)
        dum_gs_dipole=[]
        indx=0
        for line in ReadFile:
          indx = indx + 1
          if (indx != core_ind_gs):#Skipping the core level to be emptied
           dumm=float(line.split()[int(core_ind_gs-1)])#read dipole element for core
           dum_gs_dipole.append(dumm)
        full_gs_dipole.append(dum_gs_dipole)
        ########
        ReadFile=open(Z_FILE)
        dum_gs_dipole=[]
        indx=0
        for line in ReadFile:
          indx = indx + 1
          if (indx != core_ind_gs):#Skipping the core level to be emptied
           dumm=float(line.split()[int(core_ind_gs-1)])#read dipole element for core
           dum_gs_dipole.append(dumm)
        full_gs_dipole.append(dum_gs_dipole)
        return full_mom_dipole, full_gs_dipole, ovlp


def for_energy_eigenvalue(nelec):
        FileName="GS_KS_eigenvalues.dat"
        ReadFile=open(FileName)
        ener=[]
        for line in ReadFile:
          pass
        arr1 = sp.array(line.split())
        ener = sp.asarray(arr1, dtype = sp.float64,order ='C')
        #para.print("Kohn-Sham Energy Eigenvalues")
        #para.print(str(ener))
        return(ener)
def DipoleMomentCalculator(core_orb_ind,nelec):
        gs_matrix = sp.genfromtxt("beta_dipole_x_gs.txt")
        mom_matrix = sp.genfromtxt("beta_dipole_x_mom.txt")
        gs_dipole_x = 0
        mom_dipole_x = 0
        for i in range(nelec):
         mom_dipole_x += mom_matrix[i,i]
        for i in range(nelec+1):
         gs_dipole_x += gs_matrix[i,i]
        gs_dipole_x = gs_dipole_x - gs_matrix[core_orb_ind-1 , core_orb_ind-1]
        para.print("mom_dipole_x = "+str(mom_dipole_x))
        para.print("gs_dipole_x = "+str(gs_dipole_x))
        Difference_x = mom_dipole_x-gs_dipole_x
        para.print("Difference = "+str(Difference_x))
        gs_matrix = sp.genfromtxt("beta_dipole_y_gs.txt")
        mom_matrix = sp.genfromtxt("beta_dipole_y_mom.txt")
        gs_dipole_x = 0
        mom_dipole_x = 0
        for i in range(nelec):
         mom_dipole_x += mom_matrix[i,i]
        for i in range(nelec+1):
         gs_dipole_x += gs_matrix[i,i]
        gs_dipole_x = gs_dipole_x - gs_matrix[core_orb_ind-1 , core_orb_ind-1]
        para.print("mom_dipole_y = "+str(mom_dipole_x))
        para.print("gs_dipole_y = "+str(gs_dipole_x))
        Difference_y = mom_dipole_x-gs_dipole_x
        para.print("Difference = "+str(Difference_y))
        gs_matrix = sp.genfromtxt("beta_dipole_z_gs.txt")
        mom_matrix = sp.genfromtxt("beta_dipole_z_mom.txt")
        gs_dipole_x = 0
        mom_dipole_x = 0
        for i in range(nelec):
         mom_dipole_x += mom_matrix[i,i]
        for i in range(nelec+1):
         gs_dipole_x += gs_matrix[i,i]
        gs_dipole_x = gs_dipole_x - gs_matrix[core_orb_ind-1 , core_orb_ind-1]
        para.print("mom_dipole_z = "+str(mom_dipole_x))
        para.print("gs_dipole_z = "+str(gs_dipole_x))
        Difference_z = mom_dipole_x-gs_dipole_x
        para.print("Difference = "+str(Difference_z))
        RMS_Difference = ((Difference_x)**2 + (Difference_y)**2 + (Difference_z)**2)**(0.5)
        para.print("RMS Difference = "+str(RMS_Difference))
        para.print("RMS Difference in Debye = "+str(RMS_Difference/0.3934303))
        return mom_dipole_x