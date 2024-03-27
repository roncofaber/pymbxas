#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 17:19:24 2024

@author: roncoroni
"""

import numpy as np

from pymbxas.build.structure import mole_to_ase

from pyscf.md.integrators import _toframe, _Integrator

#%%
class NVTBerendson(_Integrator):
    '''Berendsen (constant N, V, T) molecular dynamics

    Args:
        method : lib.GradScanner or rhf.GradientsMixin instance, or
        has nuc_grad_method method.
            Method by which to compute the energy gradients and energies
            in order to propagate the equations of motion. Realistically,
            it can be any callable object such that it returns the energy
            and potential energy gradient when given a mol.

        T      : float
            Target temperature for the NVT Ensemble. Given in K.

        taut   : float
            Time constant for Berendsen temperature coupling.
            Given in atomic units.

    Attributes:
        accel : ndarray
            Current acceleration for the simulation. Values are given
            in atomic units (Bohr/a.u.^2). Dimensions is (natm, 3) such as

             [[x1, y1, z1],
             [x2, y2, z2],
             [x3, y3, z3]]
    '''

    def __init__(self, method, T, taut, **kwargs):
        self.T = T
        self.taut = taut
        self.accel = None
        super().__init__(method, **kwargs)

    def _next(self):
        '''Computes the next frame of the simulation and sets all internal
         variables to this new frame. First computes the new geometry,
         then the next acceleration, and finally the velocity, all according
         to the Velocity Verlet algorithm.

        Returns:
            The next frame of the simulation.
        '''

        # If no acceleration, compute that first, and then go
        # onto the next step
        if self.accel is None:
            next_epot, next_accel = self._compute_accel()

        else:
            self._zero_translation()
            self._zero_rotation()
            
            self._scale_velocities()
            self.mol.set_geom_(self._next_geometry(), unit='B')
            self.mol.build()
            next_epot, next_accel = self._compute_accel()
            self.veloc = self._next_velocity(next_accel)

        self.epot = next_epot
        self.ekin = self.compute_kinetic_energy()
        self.accel = next_accel

        return _toframe(self)

    def _compute_accel(self):
        '''Given the current geometry, computes the acceleration
        for each atom.'''
        e_tot, grad = self.scanner(self.mol)
        if not self.scanner.converged:
            raise RuntimeError('Gradients did not converge!')

        a = -1 * grad / self._masses.reshape(-1, 1)
        return e_tot, a

    def _scale_velocities(self):
        '''NVT Berendsen velocity scaling
        v_rescale(t) = v(t) * (1 + ((T_target/T - 1)
                            * (/delta t / taut)))^(0.5)
        '''
        tautscl = self.dt / self.taut
        scl_temp = np.sqrt(1.0 + (self.T / self.temperature() - 1.0) * tautscl)

        # Limit the velocity scaling to reasonable values
        # (taken from ase md/nvtberendson.py)
        if scl_temp > 1.1:
            scl_temp = 1.1
        if scl_temp < 0.9:
            scl_temp = 0.9

        self.veloc = self.veloc * scl_temp
        return
    
    def _zero_rotation(self):
        "Sets the total angular momentum to zero by counteracting rigid rotations."

        # Save initial temperature
        temp0 = self.temperature()
        
        atoms = mole_to_ase(self.mol, units="Bohr", velocities=self.veloc)

        # Find the principal moments of inertia and principal axes basis vectors
        Ip, basis = atoms.get_moments_of_inertia(vectors=True)
        
        # Calculate the total angular momentum and transform to principal basis
        Lp = np.dot(basis, atoms.get_angular_momentum())
        
        # Calculate the rotation velocity vector in the principal basis, avoiding
        # zero division, and transform it back to the cartesian coordinate system
        omega = np.dot(np.linalg.inv(basis), np.select([Ip > 0], [Lp / Ip]))
        
        # We subtract a rigid rotation corresponding to this rotation vector
        com = atoms.get_center_of_mass()
        positions = atoms.get_positions()
        positions -= com  # translate center of mass to origin
        velocities = atoms.get_velocities()
        
        self.veloc = velocities - np.cross(omega, positions)
        
        return
    
    def _zero_translation(self):
        
        atoms = mole_to_ase(self.mol, units="Bohr", velocities=self.veloc)
        
        mean_v = atoms.get_velocities().mean(axis=0)
        
        self.veloc -= mean_v
        
        return
        
        
    def _next_geometry(self):
        '''Computes the next geometry using the Velocity Verlet algorithm. The
        necessary equations of motion for the position is
            r(t_i+1) = r(t_i) + /delta t * v(t_i) + 0.5(/delta t)^2 a(t_i)
        '''
        return self.mol.atom_coords() + self.dt * self.veloc + \
            0.5 * (self.dt ** 2) * self.accel

    def _next_velocity(self, next_accel):
        '''Compute the next velocity using the Velocity Verlet algorithm. The
        necessary equations of motion for the velocity is
            v(t_i+1) = v(t_i) + /delta t * 0.5(a(t_i+1) + a(t_i))'''
        return self.veloc + 0.5 * self.dt * (self.accel + next_accel)