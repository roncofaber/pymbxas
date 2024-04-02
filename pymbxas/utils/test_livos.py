#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 14:06:44 2024

@author: roncoroni
"""

    def _calculate_livvo(self, locmethod='IBO', iaos=None, s=None,
              exponent=4, grad_tol=1e-8, max_iter=200, verbose=None): #TODO only for channel 1 (but should be same)
        
        mo_coeff = self.data.mo_coeff[1].copy()
        mo_occ   = self.data.mo_occ[1].copy()
        
        occ_idxs = np.where(mo_occ > 0)[0]
        uno_idxs = np.where(mo_occ == 0)[0]
        
        mo_vvo = lo.vvo.livvo(self.mol,
                              mo_coeff[:, occ_idxs], mo_coeff[:, uno_idxs],
                              locmethod=locmethod,
                              iaos=iaos, s=s,
                              exponent=exponent, grad_tol=grad_tol,
                              max_iter=max_iter, verbose=verbose)
        
        return mo_vvo
    
    
    def get_excited_orbital_character(self, ato_idx, livvo=None):
        
        if livvo is None:
            livvo = self.data.mo_livvo
            
        basis_ovlp = self.mol.intor("int1e_ovlp")
        
        ato_idxs = atoms_to_indexes(self.structure, ato_idx)
        
        projections = []
        for excitation in self.excitations:
            if excitation.ato_idx not in ato_idxs:
                continue
            
            data = excitation.data["fch"]
            
            # get unoccupied orbitals
            uno_idxs = np.where(data.mo_occ[excitation.channel] == 0)[0][1:]
            CMO      = data.mo_coeff[excitation.channel][:, uno_idxs]
            
            # calculate projection of CMOs on LIVVOs basis
            projection = (CMO.conj().T@basis_ovlp@livvo)**2
            
            projections.append(projection)
        
        if len(ato_idxs) == 1:
            return projections[0]
        else:
            return projections