import jax.numpy as jnp
import jax
from typing import Tuple
from .spectral_op import SpectralOperation

class PowerSpectrum(SpectralOperation):
    """
    Power spectrum from a 3D density field.

    Args:
        n_grid : number of grid points in each dimension
        n_bins : number of bins for the power spectrum
        boxlength : length of the box in real space

    Attributes:
        n_bins : number of bins for the power spectrum
        index_grid : index of the bin for each wavenumber
        n_modes : number of modes in each bin
        boxlength : length of the box in real space
    """

    n_bins : int
    index_grid : jax.Array
    n_modes : jax.Array

    def __init__(self, n_grid : int, n_bins : int, boxlength : float):
        super().__init__(n_grid=n_grid)
        self.n_bins = n_bins
        self.boxlength = boxlength

        self.index_grid = jnp.digitize(self.k, jnp.linspace(0, self.nyquist, self.n_bins), right=True) - 1

        self.n_modes = jnp.zeros(self.n_bins)
        self.n_modes = self.n_modes.at[self.index_grid].add(1)

    def __call__(self, delta : jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        Compute the power spectrum from a 3D density field
        
        Args:
            delta : 3D density field

        Returns:
            wavenumber and power spectrum
        """
        # get the density field in fourier space
        delta_k = jnp.fft.rfftn(delta)

        power = jnp.zeros(self.n_bins)
        power = power.at[self.index_grid].add(delta_k * jnp.conj(delta_k))  

        # compute the average power
        power = power / self.n_modes
        power = power / (self.boxlength ** 3 ** 2)

        power = jnp.where(jnp.isnan(power), 0, power)

        k = jnp.linspace(0, 0.5, self.n_bins)

        return k, power

