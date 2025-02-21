#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 15:16:30 2025

@author: roncofaber
"""

import gpflow
import numpy as np
from gpflow.kernels import Matern32, LinearCoregionalization
from gpflow.mean_functions import Zero
from gpflow.inducing_variables import SharedIndependentInducingVariables, InducingPoints
from gpflow.models import SVGP
import tensorflow as tf
from scipy.cluster.vq import kmeans

def optimize_model_with_scipy(model, data):
    optimizer = gpflow.optimizers.Scipy()
    optimizer.minimize(
        model.training_loss_closure(data),
        variables=model.trainable_variables,
        method="l-bfgs-b",
        options={"disp": True, "maxiter": 1000},
    )



def _multi_output_fit( Xs, Ys):
    
    num_outputs = Ys.shape[1]
    num_inducing = int(Xs.shape[0]/10)  # Number of inducing points
    input_dim = Xs.shape[1]
    
    lgts = 3 * np.ones(input_dim)
    vras = 5
    
    # Create individual kernels for each output
    kernels = gpflow.kernels.SharedIndependent(
        Matern32(variance=vras, lengthscales=lgts)
        , output_dim=num_outputs)
    
    # Combine the kernels into a LinearCoregionalization kernel
    # coreg_kernel = LinearCoregionalization(kernels, W=np.random.randn(num_outputs, num_outputs))

    Xs_ind, _ = kmeans(Xs, num_inducing)
    # Create inducing points
    # Z = Xs
    inducing_variable = SharedIndependentInducingVariables(InducingPoints(Xs_ind))
    
    # Create the SVGP model with the multi-output kernel
    model_A = SVGP(kernel=kernels,
                   likelihood=gpflow.likelihoods.Gaussian(),
                   inducing_variable=inducing_variable,
                   num_latent_gps=num_outputs)
    
    
    optimize_model_with_scipy(model_A, (Xs, Ys))
    
    return model_A
