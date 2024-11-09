import jax.numpy as jnp
import jax
from typing import Tuple
from .spectral_op import SpectralOperation

class SpectralLoss(SpectralOperation):
    """
    Compute the MSE in spectral space for each wavenumber

    :parm n_grid: number of grid points in each dimension
    :parm n_bins: number of bins for the power spectrum
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

        :param pred: predicted density field
        :param true: true density field

        :return: The wavenumber
        :return: The loss for each wavenumber
        """
        # get the density field in fourier space
        pred_k = jnp.fft.rfftn(pred)
        true_k = jnp.fft.rfftn(true)

        complex_loss = jnp.zeros(self.n_bins)
        power = jnp.zeros(self.n_bins)
        complex_loss = complex_loss.at[self.index_grid].add((abs(pred_k)**2 - abs(true_k)**2)**2)
        power = power.at[self.index_grid].add(abs(true_k)**2)

        # compute the average power
        complex_loss = complex_loss / self.n_modes
        complex_loss = complex_loss / power

        complex_loss = jnp.where(jnp.isnan(complex_loss), 0, complex_loss)

        k = jnp.linspace(0, 0.5, self.n_bins)

        return k, complex_loss
    


