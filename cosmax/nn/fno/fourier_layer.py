# from __future__ import annotations
import equinox as eqx
from typing import Callable
import jax
from .spectral_convolution import SpectralConvolution

class FourierLayer(eqx.Module):
    """
    Paper by Li et. al:
    FOURIER NEURAL OPERATOR FOR PARAMETRIC PARTIAL DIFFERENTIAL EQUATIONS

    Implementation inspired by:
    Felix Köhler : https://github.com/Ceyron/machine-learning-and-simulation/
    NeuralOperator: https://github.com/neuraloperator/neuraloperator
    """
    spectral_conv : SpectralConvolution
    bypass_conv : eqx.nn.Conv
    activation : Callable

    def __init__(
            self, 
            modes : int,
            n_channels : int,
            activation: Callable,
            key):
    
        self.activation = activation

        k1, k2 = jax.random.split(key)
        self.spectral_conv = SpectralConvolution(
            modes=modes,
            n_channels=n_channels, 
            key=k1)
        
        self.bypass_conv = eqx.nn.Conv(
            num_spatial_dims = 3,
            in_channels = n_channels,
            out_channels = n_channels,
            kernel_size = 1,
            padding = 'SAME',
            padding_mode = 'CIRCULAR',
            key = k2)
        
    def __call__(self, x):
        y = self.spectral_conv(x)
        z = self.bypass_conv(x)
        return self.activation(y + z)