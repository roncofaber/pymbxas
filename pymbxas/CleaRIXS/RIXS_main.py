import time
import numpy as sp
from rixs_modules import *
from read_modules import *
from Classdefs import *

time0 = time.time()
###########
userin = sys.stdin
userin = open(sys.argv[1], 'r')
lines=userin.read()
variables = {}
for block in lines.split('\n'):
 if (len(block) > 0):
  param = block.split('=')[0]
  val = block.split('=')[1]
  variables.update({param:val})
userinp=inp_data()
for P in set(vars(userinp)):
 for Q in set(variables):
  if (P.strip() == Q.strip()):
   if ("false" in str(variables[Q]).lower()):
    variables[Q]=False
   if ("true" in str(variables[Q]).lower()):
    variables[Q]=True
   setattr(userinp, P, variables[Q])
############

highE = float(userinp.highE)
lowE = float(userinp.lowE)
gridP = float(userinp.gridP)
sigma= float(userinp.sigma)
Gamma=complex(userinp.Gamma)
input_file=userinp.input_file
output_file=userinp.output_file
Dodebug=userinp.Dodebug
check_amp=userinp.check_amp
do_align=userinp.do_align
printsticks=userinp.printsticks
printspec=userinp.printspec
calc_abs=userinp.calc_abs
printinteg=userinp.printinteg
printanalysis=userinp.printanalysis
DoRIXS=userinp.DoRIXS
print("check_amp = "+str(check_amp))
print("Will read electronic structure data from "+str(input_file)+" and "+str(output_file))
GS_tot_en, FCH_tot_en, Align_tot_en = tot_energy_reader(output_file,do_align)
CI_Expansion=Unres_CISreader(output_file,check_amp)
E_f = []
E_f.append(0)
for i in CI_Expansion:
 E_f.append(i[1]) 
nocc,core_ind_gs=occuFinder(input_file)
full_mom_matrix, full_gs_matrix, full_ovlp_matrix = read_full_matrices(nocc,core_ind_gs)
ener = FCHunoccupied_energy_reader(nocc,core_ind_gs,output_file)
if do_align:
 align_shift = Align_tot_en - sp.min(ener)
 ener = ener + align_shift
else:
 ener = ener + FCH_tot_en
#
#print("Printing the intermediate state total energies")
#print(ener)
xi=(sp.array(full_ovlp_matrix)).T
norb=min(xi.shape)
#
Absorption=[]
for ixyz in [0,1,2]:
 chb_xmat=sp.array(full_mom_matrix)[ixyz,:]
 Absorption.append(AbsCalc(ixyz,xi,nocc,norb,chb_xmat))
inp_freq = SuggestWin(GS_tot_en, ener, lowE, highE, gridP) 
if calc_abs:
 for ixyz in [0,1,2]:
  AbsSpecFile = "Abs_spec."+str(ixyz)+".dat"
  #print("Absorption[ixyz] = "+str(Absorption[ixyz]))
  if (ixyz == 0):
   AbsSpectrum = abs_spec(Absorption[ixyz], inp_freq, ener, Gamma, GS_tot_en, AbsSpecFile) 
  else:
   AbsSpectrum += abs_spec(Absorption[ixyz], inp_freq, ener, Gamma, GS_tot_en, AbsSpecFile)
 AbsSpectrum = AbsSpectrum/3.0
 sp.savetxt("Abs_spec.dat",sp.array([inp_freq , AbsSpectrum]).T) 
print("Done with absorption calculation")
time1 = time.time()
print("Time elapsed = "+str(time1-time0))
if DoRIXS:
 T_f = [[0,0,0],[0,0,0],[0,0,0]]
 if Dodebug:
  DebugRIXS(full_gs_matrix, inp_freq, CI_Expansion, nocc, norb, ener, GS_tot_en,Absorption,Gamma,core_ind_gs,xi)
 for emm_idir in [0,1,2]:
  gb_xmat=sp.array(full_gs_matrix)[emm_idir,:]
  emm_xi,emm_xi_c_,eta_mat,zeta,Dmy1_DET,absp_ref_det = RefMats(emm_idir,nocc,norb,xi,gb_xmat)
  EmmAmp=[[0 for col in range(norb-nocc)] for row in range(len(CI_Expansion)+1)] #Emmission amplitude. Row for 'F'. Column for 'X_k'.
  gb_xmat_ = sp.array([gb_xmat[:nocc]])
  for k in range(nocc,norb):
   print("Going in with emission-direction = "+str(emm_idir)+" and k = "+str(k))
   Denom = sp.array([[1/((w_inGS+Gamma+GS_tot_en)-ener[k-nocc])] for w_inGS in inp_freq]) #sp.array corresponding to different values of w_in
   D1_det, D2_det = K_DepDets(k,emm_idir,nocc,norb,emm_xi,emm_xi_c_,eta_mat,zeta,Dmy1_DET,absp_ref_det)
   T2 = la.kron(gb_xmat_,D2_det)
   Yk= D1_det + T2
   EmmAmp[0][k-nocc] = Absorption[emm_idir][k-nocc] # Emission term for elastic scattering 
   for fstate in range(len(CI_Expansion)):
    E_fstate = float(CI_Expansion[fstate][1])
    for single in CI_Expansion[fstate][2]:
     if (int(single[0]) < core_ind_gs):
      V_orb = int(single[0])-1
     if (int(single[0]) > core_ind_gs):
      V_orb = int(single[0])-2
     if (int(single[0]) == core_ind_gs):
      continue
     C_orb = int(single[1])-1
     single_contrib = float(single[2])
     EmmAmp[fstate+1][k-nocc] += single_contrib.conjugate() * Yk[C_orb,V_orb] #Quantity in Eq. 9
     ##
   EmmAmp_ = sp.array(EmmAmp)
   EmmAmp_k = sp.array([EmmAmp_[:,k-nocc]])
   for ixyz in [0,1,2]:
    Absp = Absorption[ixyz][k-nocc] * Denom #sp.array corresponding to different values of w_in
    T_f[emm_idir][ixyz] += la.kron(Absp , EmmAmp_k) #Quantity in Eq. 2. This is a (L X F) matrix. L is number of input frequencies and F is number of final states including GS
 sp.set_printoptions(threshold=sys.maxsize)
 print("RIXS amplitudes calculated")
 time2=time.time()
 print("Time elapsed = "+str(time2-time0))
 if printsticks or printanalysis:
  outp_freq = inp_freq
 if printanalysis:
  write_anly = open("Local_RIXS_Maxima.dat","w")
 for emm_idir in [0,1,2]:
  for ixyz in [0,1,2]:
   S = SpectrumPlot(emm_idir, ixyz, T_f[emm_idir][ixyz], inp_freq)
   SumOverF = 0
   if printsticks:
    S.sticks()
   if printspec or printanalysis:
    if (emm_idir==0 and ixyz==0):
     S_new = S.spec(E_f, sigma, outp_freq)
     specarray = S_new 
    else:
     S_new = S.spec(E_f, sigma, outp_freq)
     specarray += S_new
   if printanalysis:
    numb_in_w = len(inp_freq)
    numb_out_w = len(outp_freq)
    spec_analysis = SpectrumAnalyze(S_new,numb_in_w,numb_out_w)
    local_maxima = spec_analysis.twod_local_maxima(inp_freq)
    write_anly.write("#################################################################"+"\n")
    write_anly.write("Local Maxima for emm_idir, ixyz = "+str([emm_idir, ixyz])+"\n")
    thres_intens = 0.1 * sp.max(sp.array(local_maxima)[:,2])
    write_anly.write("threshold intensity = "+str(thres_intens)+"\n")
    write_anly.write("....................................................................."+"\n")
    write_anly.write("CH orb index  ,   w_in  ,  Final State index  ,  w_out  ,  intensity"+"\n")
    write_anly.write("....................................................................."+"\n")
    for d1 in range(len(local_maxima)):
     if (local_maxima[d1][2] > thres_intens):
      in_w = local_maxima[d1][0] 
      out_w = local_maxima[d1][1]
      Xorb, Forb = FindOrbs(E_f, ener, GS_tot_en, in_w, out_w, thres_orb=0.1)
      write_anly.write(str(Xorb[0].tolist())+"  ,  "+str(in_w) + " ,  " +str(Forb[0].tolist())+"  ,  "+str(out_w)+"  ,  "+str(local_maxima[d1][2])+ "\n")
   if printinteg:
    if (emm_idir==0 and ixyz==0):
     integ_array = S.integ_spec()
    else:
     integ_array += S.integ_spec()
 if printspec:
  FinalFile = "FinalSpectrum.dat"
  specarray = specarray/9.0
  sp.savetxt(FinalFile,specarray)
 if printinteg:
  IntegFile = "IntegratedSpectrum.dat"
  integ_array = integ_array/9.0
  sp.savetxt(IntegFile,integ_array)
time3 = time.time()
print("All done")
print("Time elapsed = "+str(time3-time0))
