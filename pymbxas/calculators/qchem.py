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

    def __init__(self, structure,
                 qchem_params   = None,
                 charge         = 0,
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

        # store data
        self.structure    = structure
        self.charge       = charge
        self.qchem_params = qchem_params

        # run MBXAS calculation
        if run_calc:
            self.run_calculations(structure, qchem_params, charge, fch_occ,
                                  xch_occ, print_fchk)

        return

    def run_calculations(self, structure, qchem_params, charge, fch_occ,
                         xch_occ, print_fchk):

        # delete scratch earlier if not XCH calc
        is_xch = True if xch_occ is not None else False

        # GS input
        multiplicity = 1
        gs_input = make_qchem_input(structure, charge, multiplicity,
                             qchem_params, "gs", occupation = None)

        # run GS
        gs_output, gs_data = get_output_from_qchem(
            gs_input, processors = self.__nprocs, use_mpi = self.__is_mpi,
            return_electronic_structure = True, scratch = self.__sdir,
            delete_scratch = False)

        self.__boys_postprocess(gs_data)

        # write output file #TODO change in the future to be more flexible
        with open("qchem.output", "w") as fout:
            fout.write(gs_output)

        if print_fchk:
            write_to_fchk(gs_data, 'output_gs.fchk')

        # update input with guess and run FCH
        # FCH input
        multiplicity = 2
        fch_input = make_qchem_input(structure, charge+1, multiplicity,
                                     qchem_params, "fch", occupation=fch_occ,
                                     scf_guess=gs_data["coefficients"])

        fch_output, fch_data = get_output_from_qchem(
            fch_input, processors = self.__nprocs, use_mpi = self.__is_mpi,
            return_electronic_structure = True, scratch = self.__sdir,
            delete_scratch = not is_xch)

        # write input and output
        with open("qchem.input", "w") as fout:
            fout.write(fch_input.get_txt())
        with open("qchem.output", "a") as fout:
            fout.write(fch_output)
        # copy MOM files in relevant directory
        copy_output_files(self.__wdir, self.__cdir)

        if print_fchk:
            write_to_fchk(fch_data, 'output_fch.fchk')

        # only run XCH if there is input
        if is_xch:
            multiplicity = 1
            xch_input = make_qchem_input(structure, charge, multiplicity,
                                         qchem_params, "xch", occupation=xch_occ,
                                         scf_guess=fch_data["coefficients"])

            xch_output, xch_data = get_output_from_qchem(
                xch_input, processors = self.__nprocs, use_mpi = self.__is_mpi,
                return_electronic_structure = True, scratch = self.__sdir,
                delete_scratch = is_xch)

            if print_fchk:
                write_to_fchk(xch_data, 'output_xch.fchk')

            # generate AlignDir directory #TODO change for more flex
            os.mkdir("AlignDir")

            # write XCH output file
            with open("AlignDir/align_calc.out", "w") as fout:
                fout.write(xch_output)

        return


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