import sys
import os
import numpy as sp
from rixs_modules import *

class inp_data():
      def __init__(self):
          self.lowE= 0
          self.sigma= 0.3
          self.gridP = 100
          self.highE = 0
          self.Gamma = 0.3
          self.Dodebug = False 
          self.check_amp = False
          self.do_align = False
          self.printsticks = True
          self.printspec = True
          self.calc_abs = False
          self.printinteg = False
          self.input_file ="qche.in"
          self.output_file ="qche.out"
          self.printanalysis=False
          self.DoRIXS=False

class SpectrumAnalyze():
      def __init__(self,S,len_inw,len_outw):
          self.intensity = S[:,2].reshape(len_outw,len_inw)
          self.outw = S[:,1]

      def twod_local_maxima(self,inw):
          local_maxima=[]
          Bool_arr = local_bool(self.intensity)
          #print("self.inw = "+str(self.inw))
          for i in range(Bool_arr.shape[0]):
           for j in range(Bool_arr.shape[1]):
            if Bool_arr[i,j]:
             local_maxima.append([inw[i] , self.outw[j], self.intensity[i,j]])
          return(local_maxima)

class SpectrumPlot():
      def __init__(self,emm_idir, ixyz, T_f,inp_freq):
          self.emm_idir = emm_idir
          self.ixyz = ixyz
          self.T_f = T_f
          self.inp_freq = inp_freq

      def sticks(self):
          StickFile = "STICK."+str(self.emm_idir)+"."+str(self.ixyz)+".dat" 
          Stick_Write = open(StickFile,"w")
          StickSize = (self.T_f).shape
          for i in range(StickSize[0]):
           for j in range(StickSize[1]):
            Stick_Write.write(str(i)+"  "+str(self.inp_freq[i]) +"  "+str(j)+"  "+str(self.T_f[i,j])+"   "+str(abs(self.T_f[i,j])) + "\n")

      def integ_spec(self):
          IntegFile = "F_integ."+str(self.emm_idir)+"."+str(self.ixyz)+".dat"
          Integ_Write = open(IntegFile,"w")
          win_numb = len(self.inp_freq)
          I_ampl = sp.zeros(win_numb)
          F_numb = (self.T_f).shape[1]
          for f in range(F_numb):
           I_ampl += abs(self.T_f[:,f])**2
          for w_in in range(win_numb):
           Integ_Write.write(str(self.inp_freq[w_in])+"  "+str(I_ampl[w_in]) + "\n")
          specarray = sp.array([self.inp_freq , I_ampl]).T
          return(specarray)
          
      def spec(self,e_f, sigma, wout_axis):
          SpecFile = "Spectrum."+str(self.emm_idir)+"."+str(self.ixyz)+".dat"
          if os.path.exists(SpecFile):
           os.remove(SpecFile)
          win_numb = len(self.inp_freq)
          broadnd_spec = [0] * win_numb #Make this an array of dimensions len(w_in)
          numb = len(e_f)
          w_in_index = -1
          with open(SpecFile, "ab") as f:
           for w_in in self.inp_freq:
            dum_array = sp.array([w_in] * len(wout_axis))
            w_in_index = 1+w_in_index
            OS = sp.zeros(numb)
            for i in range(numb):
             OS[i] = abs((self.T_f[w_in_index,i])) **2
             broadnd_spec[w_in_index] +=  OS[i] * sp.exp(-( w_in - wout_axis - e_f[i] )**2/(2*sigma**2))
            sp.savetxt(f,sp.array([ dum_array, wout_axis , broadnd_spec[w_in_index]]).T)#, fmt="%9.6f") 
            if (w_in_index == 0):
             specarray = sp.array([ dum_array, wout_axis , broadnd_spec[w_in_index]]).T
            if (w_in_index > 0):
             specarray = sp.append(specarray,sp.array([ dum_array, wout_axis , broadnd_spec[w_in_index]]).T, axis=0)
          return specarray
