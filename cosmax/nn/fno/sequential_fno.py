# Implementation of Fast Furier Convolution
# from __future__ import annotations
import equinox as eqx
from typing import Callable
import jax
from typing import Callable
from .fno import FNO
import jax.numpy as jnp

class SequentialFNO(eqx.Module):
    """
    Sequential Fourier Neural Operator
    """

    fno_operators : list[FNO]

    def __init__(
            self,
            sequence_length : int,
            modes : int,
            input_channels : int,
            hidden_channels : int,
            output_channels : int,
            activation: Callable,
            n_fourier_layers : int,
            key):
        
        keys = jax.random.split(key, sequence_length)

        self.fno_operators = []

        for i in range(sequence_length):
            self.fno_operators.append(
                FNO(modes = modes,
                    input_channels = input_channels,
                    hidden_channels = hidden_channels,
                    output_channels = output_channels,
                    activation = activation,
                    n_fourier_layers=n_fourier_layers,
                    key=keys[i]))
        return

    def __call__(self, x : jax.Array, sequential_mode : bool):
        """
        shape of x:
        [Frames, Channels, Depth, Height, Width]
        """
        f, c, d, h, w = x.shape
        y = jnp.zeros((f-1, c, d, h, w))

        if sequential_mode:
            carry = x[0]
            for i, operator in enumerate(self.fno_operators):
                carry = operator(carry)
                y = y.at[i].set(carry)

        else:
            for i, operator in enumerate(self.fno_operators):
                y = y.at[i].set(operator(x[i]))

        return y