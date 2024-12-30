import jax.numpy as jnp
import jax
from typing import Tuple
from .spectral_op import SpectralOperation

class Generator(SpectralOperation):
    """
    Generates initial conditions given a power spectrum

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

    def __init__(self, n_grid : int, grid_size : float):
        super().__init__(n_grid=n_grid, grid_size=grid_size)

        self.bin_edges = jnp.linspace(0, self.k_mag.max(), self.grid_size + 1, endpoint=True)[1:]

        bins_pad = jnp.pad(self.bin_edges, (1, 0), mode='constant', constant_values=0)
        self.k = (bins_pad[1:] + bins_pad[:-1]) / 2

        self.index_grid = jnp.digitize(
            self.k_mag, 
            self.bin_edges,
            right=False)

    def __call__(self, seed : jax.Array, Pk : jax.Array) -> jax.Array:
        """
        Compute the power spectrum from a 3D density field

        Args:
            pred : 3D density field
            true : 3D density field

        Returns:
            wavenumber and power spectrum

        """

        assert Pk.shape == (self.n_grid,)

        # Generate the correlation kernel
        Ax = jnp.sqrt(Pk)
        Ax = Ax.at[self.index_grid].get()

        # Generate the random field
        delta_k = jax.random.normal(seed, shape=(self.n_grid, self.n_grid, self.n_grid))
        delta_k = jnp.fft.rfftn(delta_k)

        # Multiply the random field by the correlation kernel
        delta_k = delta_k * Ax

        # Transform back to real space
        delta = jnp.fft.rfftn(delta_k)

        return delta