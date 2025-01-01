# from __future__ import annotations
import equinox as eqx
from typing import Callable
import jax
import jax.numpy as jnp

class SpectralConvolution(eqx.Module):
    """
    Paper by Li et. al:
    FOURIER NEURAL OPERATOR FOR PARAMETRIC PARTIAL DIFFERENTIAL EQUATIONS

    Implementation inspired by:
    Felix Köhler : https://github.com/Ceyron/machine-learning-and-simulation/
    NeuralOperator: https://github.com/neuraloperator/neuraloperator
    """

    modes : int
    weights_real : list[jax.Array]
    weights_imag : list[jax.Array]

    def __init__(
            self, 
            modes : int,
            n_channels : int,
            key):
        
        self.modes = modes
        keys = jax.random.split(key, 8)
        
        scale = 1.0 / (n_channels ** 2)

        self.weights_real = []
        self.weights_imag = []

        for i in range(4):
            real = jax.random.uniform(
                keys[i], 
                (n_channels, n_channels, modes, modes, modes),
                minval=-scale, maxval=scale)
            
            imag = jax.random.uniform(
                keys[i + 4], 
                (n_channels, n_channels, modes, modes, modes),
                minval=-scale, maxval=scale)
            
            self.weights_real.append(real)
            self.weights_imag.append(imag)
        
    def complex_mul3d(self, a, b):
        return jnp.einsum("ixyz,ioxyz->oxyz", a, b)

    def __call__(self, x : jax.Array):
        N = x.shape[1]
        # x shape : n_channels, N, N, N
        # x_fs shape : n_channels, N, N, N // 2, 2

        x_fs = jnp.fft.rfftn(x, s=(N, N, N), axes=(1, 2, 3))

        out_fs = jnp.zeros_like(x_fs)

        # low pass filter for all dimensions
        weights = self.weights_real[0] + 1j * self.weights_imag[0]
        out_fs = out_fs.at[:, :self.modes, :self.modes, :self.modes].set(
            self.complex_mul3d(x_fs[:, :self.modes, :self.modes, :self.modes], weights))
        
        # high pass dim 1, low pass else
        weights = self.weights_real[1] + 1j * self.weights_imag[1]
        out_fs = out_fs.at[:, -self.modes:, :self.modes, :self.modes].set(
            self.complex_mul3d(x_fs[:, -self.modes:, :self.modes, :self.modes], weights))
        
        # high pass dim 2, low pass else
        weights = self.weights_real[2] + 1j * self.weights_imag[2]
        out_fs = out_fs.at[:, :self.modes, -self.modes:, :self.modes].set(
            self.complex_mul3d(x_fs[:, :self.modes, -self.modes:, :self.modes], weights))
        
        # low pass dim 3, high else
        weights = self.weights_real[3] + 1j * self.weights_imag[3]
        out_fs = out_fs.at[:, -self.modes:, -self.modes:, :self.modes].set(
            self.complex_mul3d(x_fs[:, -self.modes:, -self.modes:, :self.modes], weights))
        
        #  # low pass filter for all dimensions
        # weights = self.weights_real[0] + 1j * self.weights_imag[0]
        # out_fs = out_fs.at[:, -self.modes:, -self.modes:, -self.modes:].set(
        #     self.complex_mul3d(x_fs[:, -self.modes:, -self.modes:, -self.modes:], weights))
        
        # # high pass dim 1, low pass else
        # weights = self.weights_real[1] + 1j * self.weights_imag[1]
        # out_fs = out_fs.at[:, :self.modes, -self.modes:, -self.modes:].set(
        #     self.complex_mul3d(x_fs[:, :self.modes, -self.modes:, -self.modes:], weights))
        
        # # high pass dim 2, low pass else
        # weights = self.weights_real[2] + 1j * self.weights_imag[2]
        # out_fs = out_fs.at[:, -self.modes:, :self.modes, -self.modes:].set(
        #     self.complex_mul3d(x_fs[:, -self.modes:, :self.modes, -self.modes:], weights))
        
        # # low pass dim 3, high else
        # weights = self.weights_real[3] + 1j * self.weights_imag[3]
        # out_fs = out_fs.at[:, :self.modes, :self.modes, -self.modes:].set(
        #     self.complex_mul3d(x_fs[:, :self.modes, :self.modes, -self.modes:], weights))
           
        return jnp.fft.irfftn(out_fs, s=(N, N, N), axes=(1, 2, 3))