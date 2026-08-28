#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 10:58:36 2025

@author: roncofaber
"""

import logging

import gpflow
import numpy as np
import gpytorch
import torch
torch.set_default_dtype(torch.float64)

from gpytorch.mlls import SumMarginalLogLikelihood
from gpytorch.constraints.constraints import Interval

import tensorflow_probability as tfp
import tensorflow as tf

logger = logging.getLogger(__name__)
#%%

class BaseGPmodel(object):
    
    def __init__(self):
        
        self._model = None
        
        return
    
    @property
    def model(self):
        return self._model
    
#%% GPflow

class GPflow_model(BaseGPmodel):
    
    def __init__(self, Xs, Ys, parameters=None):
        
        super().__init__()
        
        # Estimate variance and lengthscales
        vras = 1.#np.var(Ys)
        lgts = (Xs.max(axis=0) - Xs.min(axis=0))/2
        
        # make kernel
        my_kernel = gpflow.kernels.Matern12(variance = vras, lengthscales = lgts)
        
        # assign prior to model lengthscales
        mu_0    = 2#gpflow.Parameter(0.0, trainable=True)
        sigma_0 = 1#gpflow.Parameter(1.0, transform=tfp.bijectors.Exp(), trainable=True)
        
        lprior = tfp.distributions.Normal(
            loc=mu_0 + np.log(len(lgts))/2, scale=sigma_0)
        
        my_kernel.lengthscales.prior = lprior
        
        # create GP model
        model = gpflow.models.GPR(
            (Xs, Ys),
            kernel         = my_kernel,
            mean_function  = gpflow.functions.Constant(),
            noise_variance = 1e-2
        )
        
        # reassign parameters
        if parameters is not None:
            gpflow.utilities.multiple_assign(model, parameters)
            
        # set variance as NOT trainable --> check this
        gpflow.utilities.set_trainable(model.likelihood, False)
        
        self._model = model
        
        return
    
    def train(self, mode="scipy"):
        
        if mode == "scipy":
            # run optimizer
            opt = gpflow.optimizers.Scipy()
            opt.minimize(
                self.model.training_loss,
                self.model.trainable_variables,
                options = dict(maxiter=5000),
                method  = "l-bfgs-b",
            )
            
                    
    def predict(self, Xtest):
        Y_pre, Y_var = self.model.predict_f(Xtest)
        return Y_pre.numpy(), Y_var.numpy()
    
#%% GPyTorch
    
class MTExactGPModelWithLogNormalPrior(gpytorch.models.ExactGP):
    
    def __init__(self, train_x, train_y, likelihood):
        
        # initialize model
        super().__init__(train_x, train_y, likelihood)
        
        _, n_dimensions = train_x.shape
        _, n_tasks      = train_y.shape

        # With a little elbow grease, these
        # could be trainable parameters as well.
        mu_0    = 0.0
        sigma_0 = 1.0
        
        # define mean
        self.mean_module  = gpytorch.means.ConstantMean(batch_shape = torch.Size([n_tasks]))
        
        # define kernel
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims = n_dimensions,
                # THE ONLY CHANGE IS THE FOLLOWING LINE:
                lengthscale_prior=gpytorch.priors.LogNormalPrior(mu_0 + np.log(n_dimensions) / 2, sigma_0),
                # lengthscale_prior=gpytorch.priors.NormalPrior(mu_0 + np.log(n_dimensions) / 2, sigma_0),
                batch_shape=torch.Size([n_tasks])
                ),
            batch_shape=torch.Size([n_tasks])
        )

    def forward(self, x):
        mean_x  = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )
    
class MTGPyTorch_model(BaseGPmodel):
    
    def __init__(self, Xs, Ys, parameters=None):
        
        super().__init__()
    
        # convert from numpy
        Xs = torch.from_numpy(Xs)
        Ys = torch.stack([torch.from_numpy(ys) for ys in Ys.T], -1)
        
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks = Ys.shape[1],
            # noise_constraint = Interval(1e-6, 1e-3)
            )

        # Disable the gradient for the noise parameters to make them non-trainable
        # likelihood.task_noises.requires_grad = False    
        
        # initialize model
        model = MTExactGPModelWithLogNormalPrior(Xs, Ys, likelihood)
            
        # store GP model
        self._model      = model
        self._likelihood = likelihood
        self._mll        = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        
        # store monitoring
        self._loss = []
        
        return
    
    def train(self, niter=300, lr = 0.05):
        
        # Find optimal model hyperparameters
        self._model.train()
        self._likelihood.train()

        # Use the Adam optimizer
        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)

        for ii in range(niter):
            
            # forward pass
            output = self._model(*self._model.train_inputs)
            loss   = -self._mll(output, self._model.train_targets)
            
            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            logger.debug(
                "GP training iteration %d/%d; loss=%.6g",
                ii + 1, niter, loss.item())
            self._loss.append(loss.item())

        if niter:
            logger.info(
                "GP training completed\n\titerations : %d\n\tfinal loss : %.6g",
                niter, self._loss[-1])

        return
    
    def to_cuda(self):
        self._model = self._model.cuda()
        self._likelihood = self._likelihood.cuda()
        self._mll = self._mll.cuda()
        return
    
    def to_cpu(self):
        self._model = self._model.cpu()
        self._likelihood = self._likelihood.cpu()
        self._mll = self._mll.cpu()
        return
    
    def predict(self, Xtest):
        
        # set eval mode
        self.model.eval()
        self._likelihood.eval()
        
        Xtest = torch.from_numpy(Xtest).cuda()
        
        # Make predictions (use the same test points)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
    
            # This contains predictions for both outcomes as a list
            prediction = self.model(Xtest)
        
            f_mean = prediction.mean
            f_var = prediction.variance
        
        return np.array(f_mean.cpu().numpy()), np.array(f_var.cpu().numpy())
