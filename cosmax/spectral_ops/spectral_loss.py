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

    def __init__(self, n_grid : int, grid_size : float, n_bins : int):
        super().__init__(n_grid=n_grid, grid_size=grid_size)
        self.n_bins = n_bins

        self.bin_edges = jnp.linspace(0, self.k_mag.max(), self.n_bins + 1, endpoint=True)[1:]

        bins_pad = jnp.pad(self.bin_edges, (1, 0), mode='constant', constant_values=0)
        self.k = (bins_pad[1:] + bins_pad[:-1]) / 2

        self.index_grid = jnp.digitize(
            self.k_mag, 
            self.bin_edges,
            right=False)

        self.n_modes = jnp.zeros(self.n_bins)
        self.n_modes = self.n_modes.at[self.index_grid].add(1)

    def __call__(self, delta_a : jax.Array, delta_b : jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        Compute the power spectrum from a 3D density field

        Args:
            pred : 3D density field
            true : 3D density field

        Returns:
            wavenumber and power spectrum

        """

        # volume of the box
        V = float(self.grid_size ** 3)
        # volume of each grid cell
        Vx = V / self.n_grid ** 3

        # get the density field in fourier space
        delta_k_a = jnp.fft.rfftn(delta_a, norm="backward")  
        delta_k_a = Vx * delta_k_a

        delta_k_b = jnp.fft.rfftn(delta_b, norm="backward")
        delta_k_b = Vx * delta_k_b

        power = jnp.real(delta_k_a * jnp.conj(delta_k_b) / V)

        power_ensemble = jnp.zeros(self.n_bins)
        power_ensemble = power_ensemble.at[self.index_grid].add(power)

        power_ensemble_avg = power_ensemble / self.n_modes
        power_ensemble_avg = jnp.where(jnp.isnan(power_ensemble_avg), 0, power_ensemble_avg)
    
        return self.k, power_ensemble_avg