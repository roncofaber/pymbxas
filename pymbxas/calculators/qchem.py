#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 12:28:13 2023

@author: roncofaber
"""

import os

# good ol' numpy
import numpy as np

# self module utilities
import pymbxas
from pymbxas.io.copy import copy_output_files
from pymbxas.build.input import make_qchem_input

# pyqchem stuff
from pyqchem import get_output_from_qchem
from pyqchem.file_io import write_to_fchk

#%%

class Qchem_mbxas():

    def __init__(self,
                 structure,
                 charge,
                 multiplicity,
                 qchem_params   = None,
                 fch_occ        = None,
                 xch_occ        = None,
                 scratch_dir    = None,
                 print_fchk     = False,
                 run_calc       = True,
                 use_mpi        = False, # somewhat MPI is not working atm
                 ):

        # initialize environment (set env variables)
        pymbxas.utils.environment.set_qchem_environment()

        # set up internal variables
        self.__is_mpi = use_mpi
        self.__nprocs = os.cpu_count()
        self.__pid    = os.getpid()
        self.__cdir   = os.getcwd()
        self.__sdir   = os.getcwd() if scratch_dir is None else scratch_dir
        self.__wdir   = "{}/pyqchem_{}/".format(os.getcwd(), self.__pid)
        self.__print_fchk = print_fchk
        # delete scratch earlier if not XCH calc
        self.__is_xch = True if xch_occ is not None else False

        # store data
        self.structure    = structure
        self.charge       = charge
        self.multiplicity = multiplicity
        self.qchem_params = qchem_params
        self.fch_occ      = fch_occ
        self.xch_occ      = xch_occ

        # initialize empty stuff
        self.output = {}

        # run MBXAS calculation
        if run_calc:
            self.run_all_calculations()

        return

    # Function to run all calc (GS, FCH, XCH) in sequence
    def run_all_calculations(self):

        # run ground state
        gs_output, gs_data = self.run_ground_state()

        # run FCH
        fch_output, fch_data  = self.run_fch(gs_data["coefficients"])

        # only run XCH if there is input
        if self.__is_xch:
            xch_output, xch_data = self.run_xch(fch_data["coefficients"])

        return

    # run the GS calculation
    def run_ground_state(self):

        structure = self.structure
        charge = self.charge
        multiplicity = self.multiplicity
        qchem_params = self.qchem_params

        # GS input
        gs_input = make_qchem_input(structure, charge, multiplicity,
                             qchem_params, "gs", occupation = None)

        # run calculation
        gs_output, gs_data = get_output_from_qchem(
            gs_input, processors = self.__nprocs, use_mpi = self.__is_mpi,
            return_electronic_structure = True, scratch = self.__sdir,
            delete_scratch = False)

        # obtain number of electrons #TODO make a function that stores relevant output (but not too heavy stuff)
        self.n_alpha = gs_data["number_of_electrons"]["alpha"]
        self.n_beta  = gs_data["number_of_electrons"]["beta"]
        self.n_electrons = self.n_alpha + self.n_beta

        # store output
        self.output["gs"] = gs_output

        # do boys postprocessing to understand orbital occupations
        # self.__boys_postprocess(gs_data)

        # write output file #TODO change in the future to be more flexible
        with open("qchem.output", "w") as fout:
            fout.write(gs_output)

        if self.__print_fchk:
            write_to_fchk(gs_data, 'output_gs.fchk')

        return gs_output, gs_data

    # run the FCH calculation
    def run_fch(self, scf_guess=None):

        structure = self.structure
        charge = self.charge + 1 # +1 cause we kick out one lil electron
        multiplicity = abs(self.n_alpha - self.n_beta) + 1
        qchem_params = self.qchem_params
        fch_occ = self.fch_occ

        # FCH input
        fch_input = make_qchem_input(structure, charge, multiplicity,
                                     qchem_params, "fch", occupation=fch_occ,
                                     scf_guess=scf_guess)

        # run calculation
        fch_output, fch_data = get_output_from_qchem(
            fch_input, processors = self.__nprocs, use_mpi = self.__is_mpi,
            return_electronic_structure = True, scratch = self.__sdir,
            delete_scratch = not self.__is_xch)

        # store output
        self.output["fch"] = fch_output

        # write input and output
        with open("qchem.input", "w") as fout:
            fout.write(fch_input.get_txt())
        with open("qchem.output", "a") as fout:
            fout.write(fch_output)
        # copy MOM files in relevant directory
        copy_output_files(self.__wdir, self.__cdir)

        if self.__print_fchk:
            write_to_fchk(fch_data, 'output_fch.fchk')

        return fch_output, fch_data

    # run the XCH calculation
    def run_xch(self, scf_guess=None):

        structure = self.structure
        charge = self.charge
        multiplicity = self.multiplicity
        qchem_params = self.qchem_params
        xch_occ = self.xch_occ

        # XCH input
        xch_input = make_qchem_input(structure, charge, multiplicity,
                                     qchem_params, "xch", occupation=xch_occ,
                                     scf_guess=scf_guess)

        # run calculation
        xch_output, xch_data = get_output_from_qchem(
            xch_input, processors = self.__nprocs, use_mpi = self.__is_mpi,
            return_electronic_structure = True, scratch = self.__sdir,
            delete_scratch = self.__is_xch)

        # store output
        self.output["xch"] = xch_output

        if self.__print_fchk:
            write_to_fchk(xch_data, 'output_xch.fchk')

        # generate AlignDir directory #TODO change for more flex
        os.mkdir("AlignDir")

        # write XCH output file
        with open("AlignDir/align_calc.out", "w") as fout:
            fout.write(xch_output)

        return xch_output, xch_data

    # do Boys postprocessing
    def __boys_postprocess(self, gs_electronic_structure):

        symbols = self.structure.get_chemical_symbols()

        #for boys_orb in gs_electronic_structure['localized_coefficients']['alpha']:
        indices={}
        variance={}
        for i,a in enumerate(gs_electronic_structure['basis']['atoms']):
            print(a['symbol'],a['atomic_number'])
            c=0
            for s in a['shells']:
                st=s['shell_type']
                sf=s['functions']
                print(s)
                if st in indices:
                    indices[st]=indices[st] + list(range(c,c+sf))
                    variance[st]= variance[st] + list(np.repeat(np.sum(np.multiply(s['con_coefficients'],1.0/np.array(s['p_exponents']))),sf))
                else:
                    indices[st] = list(range(c,c+sf))
                    variance[st] = list(np.repeat(np.sum(np.multiply(s['con_coefficients'],1.0/np.array(s['p_exponents']))),sf))
                c+=sf

        print(indices,variance)

        for iboys,boys_orb in enumerate(gs_electronic_structure['localized_coefficients']['alpha']):
            w={}
            var={}
            for st,ind in indices.items():
                #print(st,ind)
                v = np.array(list(boys_orb[i] for i in ind))
                w[st] = np.dot(v,v)
                var[st] = np.dot(np.abs(v),variance[st])
            print(iboys,w,var)

        atom_coeffs=[]
        satom=0
        for i,a in enumerate(gs_electronic_structure['basis']['atoms']):
            print(a['symbol'],a['atomic_number'])
            istart=satom
            for s in a['shells']:
                #print(s)
                #print(s['functions'])
                satom+=s['functions']
            atom_coeffs.append(slice(istart,satom))
            print('atom_coeffs',atom_coeffs[i])

        #print(np.sum([bo*bo for bo in gs_electronic_structure['localized_coefficients']['alpha'][0][atom_coeffs[1]]]))

        # loop over atoms
        num_1s_cores = len([atom for atom in symbols if atom!='H'])

        atom_boys_alpha=[]
        atom_boys_beta=[]
        core_orbital_of_atom={'alpha':{}, 'beta':{}}
        core_orbital_alpha_of_atom={}
        core_orbital_beta_of_atom={}
        for iorb in range(num_1s_cores):
            # where is Boys orbital i localized?
            boys_orb = gs_electronic_structure['localized_coefficients']['alpha'][iorb]
            atom_weight=[]
            for iatom,atom in enumerate(gs_electronic_structure['basis']['atoms']):
                atom_weight.append(np.sum([bo*bo for bo in boys_orb[atom_coeffs[iatom]]]))
            iatom=np.argmax(atom_weight)
            atom_boys_alpha.append([iatom,symbols[iatom]])
            core_orbital_alpha_of_atom[iatom] = iorb
            core_orbital_of_atom['alpha'][iatom] = iorb
            print('Boys alpha orbital',iorb,'localized on atom',atom_boys_alpha[iorb])

            # Now beta
            boys_orb = gs_electronic_structure['localized_coefficients']['beta'][iorb]
            atom_weight=[]
            for iatom,atom in enumerate(gs_electronic_structure['basis']['atoms']):
                atom_weight.append(np.sum([bo*bo for bo in boys_orb[atom_coeffs[iatom]]]))
            iatom=np.argmax(atom_weight)
            atom_boys_beta.append([iatom,symbols[iatom]])
            core_orbital_beta_of_atom[iatom] = iorb
            core_orbital_of_atom['beta'][iatom] = iorb
            print('Boys  beta orbital',iorb,'localized on atom',atom_boys_beta[iorb])

        # Build occupied orbital list
        nalpha=gs_electronic_structure['number_of_electrons']['alpha']
        nbeta=gs_electronic_structure['number_of_electrons']['beta']
        gs_orblist={'alpha':[str(i+1) for i in range(nalpha)],
                    'beta' :[str(i+1) for i in range(nbeta)]}
        print(" ".join(gs_orblist['alpha'])+"\n"+" ".join(gs_orblist['beta']))

        occupations=[]
        for iatom,atom in enumerate(symbols):
            if(atom=='H'): continue
            for ch in gs_orblist.keys():
                if(ch=='alpha'): continue # only excite beta - is this a requirement that Nalpha>Nbeta?
                ix=core_orbital_of_atom[ch][iatom]
                print('core exciting',atom,'1s of atom',iatom,'in channel',ch,'( orbital',ix+1,')')
                orblist=gs_orblist.copy()
                orblist['alpha']=gs_orblist['alpha'].copy()
                orblist['beta']=gs_orblist['beta'].copy()
                del orblist[ch][ix]
                occupations.append(" ".join(orblist['alpha'])+"\n"+" ".join(orblist['beta']))
                print(occupations[-1])