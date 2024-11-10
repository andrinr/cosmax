import jax.numpy as jnp
import jax
from typing import Tuple
from .spectral_op import SpectralOperation

class SpectralLoss(SpectralOperation):
    """
    Compute the MSE in spectral space for each wavenumber

    Args:
        n_grid : number of grid points in each dimension
        n_bins : number of bins for the power spectrum

    Attributes:
        n_bins : number of bins for the power spectrum
        index_grid : index of the bin for each wavenumber
        n_modes : number of modes in each bin
    """
    n_bins : int
    index_grid : jax.Array
    n_modes : jax.Array

    def __init__(self, n_grid : int, n_bins : int):
        super().__init__(n_grid=n_grid)
        self.n_bins = n_bins

        self.index_grid = jnp.digitize(self.k, jnp.linspace(0, self.nyquist, self.n_bins), right=True) - 1

        self.n_modes = jnp.zeros(self.n_bins)
        self.n_modes = self.n_modes.at[self.index_grid].add(1)

    def __call__(self, pred : jax.Array, true : jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        Compute the power spectrum from a 3D density field

        Args:
            pred : 3D density field
            true : 3D density field

        Returns:
            wavenumber and power spectrum

        """
        # get the density field in fourier space
        pred_k = jnp.fft.rfftn(pred)
        true_k = jnp.fft.rfftn(true)

        loss = jnp.zeros(self.n_bins)
        power = jnp.zeros(self.n_bins)
        loss = loss.at[self.index_grid].add((abs(pred_k)**2 - abs(true_k)**2)**2)
        power = power.at[self.index_grid].add(abs(true_k)**2)

        # compute the average power
        loss = loss / self.n_modes
        loss = loss / power

        loss = jnp.where(jnp.isnan(loss), 0, loss)

        k = jnp.linspace(0, 0.5, self.n_bins)

        return k, loss
    


