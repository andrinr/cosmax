# Implementation of Fast Furier Convolution
# from __future__ import annotations
import equinox as eqx
import jax
from .fourier_layer import FourierLayer
from ..base_module import BaseModule

class FNO(BaseModule):
    """
    Paper by Li et. al:
    FOURIER NEURAL OPERATOR FOR PARAMETRIC PARTIAL DIFFERENTIAL EQUATIONS

    Implementation inspired by:
    Felix Köhler : https://github.com/Ceyron/machine-learning-and-simulation/
    NeuralOperator: https://github.com/neuraloperator/neuraloperator
    """
    
    lift : eqx.nn.Conv
    fourier_layers : list[FourierLayer]
    project : eqx.nn.Conv

    def __init__(
            self,
            activation: str,
            modes : int,
            input_channels : int,
            hidden_channels : int,
            output_channels : int,
            n_fourier_layers : int,
            increasing_modes : bool,
            key):
        
        super().__init__(activation=activation)
         
        k1, k2, k3 = jax.random.split(key, 3)

        self.lift = eqx.nn.Conv(
            num_spatial_dims = 3,
            in_channels = input_channels,
            out_channels = hidden_channels, 
            kernel_size = 3,
            padding = 'SAME',
            padding_mode = 'CIRCULAR',
            key=k1)
        
        self.fourier_layers = []
        furier_keys = jax.random.split(k2, n_fourier_layers)

        for i in range(n_fourier_layers):
            self.fourier_layers.append(FourierLayer(
                modes = min(2**(i+3), modes) if increasing_modes else modes,
                n_channels = hidden_channels,
                activation = self.activation,
                key = furier_keys[i]))
            
        self.project = eqx.nn.Conv(
            num_spatial_dims = 3,
            in_channels = hidden_channels,
            out_channels = output_channels,
            kernel_size = 3,
            padding = 'SAME',
            padding_mode = 'CIRCULAR',
            key=k3)

    def __call__(self, x : jax.Array):
            
        x = self.lift(x)

        for layer in self.fourier_layers:
            x = layer(x)

        x = self.project(x)

        return x