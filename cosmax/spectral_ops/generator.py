import jax.numpy as jnp
import jax
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
    """
    n_bins : int
    index_grid : jax.Array

    def __init__(self, n_grid : int, grid_size : float):
        super().__init__(n_grid=n_grid, grid_size=grid_size)

        self.bin_edges = jnp.linspace(0, self.k_mag.max(), self.n_grid + 1, endpoint=True)[1:]

        bins_pad = jnp.pad(self.bin_edges, (1, 0), mode='constant', constant_values=0)
        self.k = (bins_pad[1:] + bins_pad[:-1]) / 2

        self.index_grid = jnp.digitize(
            self.k_mag, 
            self.bin_edges,
            right=False)

    def __call__(self, seed : jax.Array, Pk : jax.Array) -> jax.Array:
        """
        Generate a 3D density for a given power spectrum.

        Args:
            seed : random seed
            Pk : power spectrum

        Returns:
            3D density field

        """

        assert Pk.shape == (self.n_grid,)

        # Generate the correlation kernel
        Ax = jnp.sqrt(Pk)
        Ax = Ax.at[self.index_grid].get()

        # generate random key
        key = jax.random.PRNGKey(seed)

        # volume of the box
        V = float(self.grid_size ** 3)
        # volume of each grid cell
        Vx = V / self.n_grid ** 3

        # Generate the random field
        delta = jax.random.normal(key, shape=(self.n_grid, self.n_grid, self.n_grid))
        delta = delta / jnp.sqrt(Vx)
        delta_k = jnp.fft.rfftn(delta)

        # Multiply the random field by the correlation kernel
        delta_k = delta_k * Ax

        # Transform back to real space
        delta = jnp.fft.irfftn(delta_k)

        return delta