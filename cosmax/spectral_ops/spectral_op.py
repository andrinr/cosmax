import jax
import jax.numpy as jnp

class SpectralOperation:
    """
    Base class for spectral operations

    Args:
        n_grid : number of grid points in each dimension

    Attributes:
        k : wavenumber
        frequencies : frequency grid
        n_grid : number of grid points in each dimension
        nyquist : nyquist frequency
    """
    k = jax.Array
    frequencies : jax.Array
    n_grid : int
    grid_size : float
    nyquist : int

    def __init__(self, n_grid : int, grid_size : float = 1):
        self.n_grid = n_grid
        self.grid_size = grid_size
        self.frequencies = jnp.fft.fftfreq(n_grid, d=1/self.grid_size)
        self.real_frequencies = jnp.fft.rfftfreq(n_grid, d=1/self.grid_size)

        self.nyquist_index = jnp.ceil(n_grid / 2).astype(int)

        kx, ky, kz = jnp.meshgrid(self.frequencies, self.frequencies, self.real_frequencies)
        self.k = jnp.sqrt(kx**2 + ky**2 + kz**2)